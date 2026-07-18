# Copyright 2025 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
from typing import TYPE_CHECKING, Optional, Union

from omegaconf.dictconfig import DictConfig
from tqdm import tqdm

from rlinf.data.replay_buffer import SACReplayBuffer
from rlinf.scheduler import Channel
from rlinf.scheduler import WorkerGroupFuncResult as Handle
from rlinf.utils.distributed import ScopedTimer
from rlinf.utils.metric_logger import MetricLogger
from rlinf.utils.metric_utils import compute_evaluate_metrics
from rlinf.utils.resource_monitor import ResourceMonitor
from rlinf.utils.runner_utils import check_progress

if TYPE_CHECKING:
    from rlinf.workers.actor.async_fsdp_sac_policy_worker import (
        AsyncEmbodiedSACFSDPPolicy,
    )
    from rlinf.workers.actor.fsdp_actor_worker import EmbodiedFSDPActor
    from rlinf.workers.actor.fsdp_sac_policy_worker import EmbodiedSACFSDPPolicy
    from rlinf.workers.env.async_env_worker import AsyncEnvWorker
    from rlinf.workers.env.env_worker import EnvWorker
    from rlinf.workers.rollout.hf.async_huggingface_worker import (
        AsyncMultiStepRolloutWorker,
    )
    from rlinf.workers.rollout.hf.huggingface_worker import MultiStepRolloutWorker

import torch 


class EmbodiedRunner:
    def __init__(
        self,
        cfg: DictConfig,
        actor: Union[
            "EmbodiedFSDPActor", "EmbodiedSACFSDPPolicy", "AsyncEmbodiedSACFSDPPolicy"
        ],
        rollout: Union["MultiStepRolloutWorker", "AsyncMultiStepRolloutWorker"],
        env: Union["EnvWorker", "AsyncEnvWorker"],
        demo_buffer: Optional[SACReplayBuffer] = None,
        critic=None,
        reward=None,
        run_timer=None,
    ):
        self.cfg = cfg
        self.actor = actor
        self.rollout = rollout
        self.env = env
        self.demo_buffer = demo_buffer
        self.critic = critic
        self.reward = reward

        # Data channels
        self.env_channel = Channel.create("Env")
        self.rollout_channel = Channel.create("Rollout")
        self.actor_channel = Channel.create("Actor")
        if self.demo_buffer is not None:
            self.demo_data_channel = Channel.create("DemoBufferChannel")

        # this timer checks if we should stop training
        self.run_timer = run_timer

        self.consumed_samples = 0
        # the step here is GRPO step
        self.global_step = 0

        # compute `max_steps`
        self.set_max_steps()

        self.timer = ScopedTimer(reduction="max", sync_cuda=False)

        self.metric_logger = MetricLogger(cfg)
        self.resource_monitor = ResourceMonitor(cfg, self.metric_logger)

    def init_workers(self):
        # create worker in order to decrease the maximum memory usage
        self.actor.init_worker().wait()
        self.rollout.init_worker().wait()
        self.env.init_worker().wait()

        resume_dir = self.cfg.runner.get("resume_dir", None)
        if resume_dir is None:
            return

        actor_checkpoint_path = os.path.join(resume_dir, "actor")
        assert os.path.exists(actor_checkpoint_path), (
            f"resume_dir {actor_checkpoint_path} does not exist."
        )
        self.actor.load_checkpoint(actor_checkpoint_path).wait()
        self.global_step = int(resume_dir.split("global_step_")[-1])

    def send_demo_buffer(self):
        if self.demo_buffer is not None:
            sub_demo_buffer_ls = self.demo_buffer.split_to_dict(self.actor._world_size)

            for sub_demo_buffer in sub_demo_buffer_ls:
                self.demo_data_channel.put(sub_demo_buffer, async_op=True)
            self.actor.recv_demo_data(self.demo_data_channel).wait()

    def update_rollout_weights(self):
        rollout_handle: Handle = self.rollout.sync_model_from_actor()
        actor_handle: Handle = self.actor.sync_model_to_rollout()
        actor_handle.wait()
        rollout_handle.wait()

    def evaluate(self):
        env_handle: Handle = self.env.evaluate(
            input_channel=self.rollout_channel,
            output_channel=self.env_channel,
        )
        rollout_handle: Handle = self.rollout.evaluate(
            input_channel=self.env_channel,
            output_channel=self.rollout_channel,
        )
        env_results = env_handle.wait()
        rollout_handle.wait()
        eval_metrics_list = [results for results in env_results if results is not None]
        eval_metrics = compute_evaluate_metrics(eval_metrics_list)
        return eval_metrics

    def run(self):
        """
        Main training loop. Dispatches to sync or async mode based on pipeline_mode config.

        pipeline_mode:
            - "sync": Traditional synchronous mode - all rollout epochs complete before training
            - "async": Asynchronous pipeline mode - training starts as soon as data is available
        """
        pipeline_mode = self.cfg.algorithm.get("pipeline_mode", "sync")
        self.resource_monitor.start()
        try:
            if pipeline_mode == "async":
                self._run_async()
            else:
                self._run_sync()
        finally:
            self.resource_monitor.stop()
            self.metric_logger.finish()

    def _run_sync(self):
        """
        Synchronous training mode.

        Flow: Rollout (all epochs) -> Receive all data -> Compute advantages -> Train
        """
        start_step = self.global_step
        global_pbar = tqdm(
            initial=start_step,
            total=self.max_steps,
            desc="Global Step",
            ncols=800,
        )
        self.send_demo_buffer()
        for _step in range(start_step, self.max_steps):
            self.actor.set_global_step(self.global_step)
            self.rollout.set_global_step(self.global_step)

            with self.timer("step"):
                with self.timer("sync_weights"):
                    self.update_rollout_weights()
                with self.timer("generate_rollouts"):
                    env_handle: Handle = self.env.interact(
                        input_channel=self.rollout_channel,
                        output_channel=self.env_channel,
                    )
                    rollout_handle: Handle = self.rollout.generate(
                        input_channel=self.env_channel,
                        output_channel=self.rollout_channel,
                        actor_channel=self.actor_channel,
                    )
                    self.actor.recv_rollout_batch(
                        input_channel=self.actor_channel
                    ).wait()
                    rollout_timing_metrics = rollout_handle.wait()
                    # Collect timing metrics from rollout workers
                    rollout_time_data = {}
                    if rollout_timing_metrics:
                        for metrics in rollout_timing_metrics:
                            if metrics:
                                for key, value in metrics.items():
                                    rollout_time_data[f"rollout_{key}"] = value

                # Recompute old_logprobs using actor if enabled
                if self.cfg.algorithm.get("recompute_old_logprobs", False):
                    with self.timer("recompute_old_logprobs"):
                        self.actor.recompute_old_logprobs().wait()

                with self.timer("cal_adv_and_returns"):
                    actor_rollout_metrics = (
                        self.actor.compute_advantages_and_returns().wait()
                    )

                with self.timer("actor_training"):
                    actor_training_metrics = self.actor.run_training().wait()

                self.global_step += 1

                run_val, save_model, is_train_end = check_progress(
                    self.global_step,
                    self.max_steps,
                    self.cfg.runner.val_check_interval,
                    self.cfg.runner.save_interval,
                    1.0,
                    run_time_exceeded=False,
                )

                eval_metrics = {}
                if run_val:
                    with self.timer("eval"):
                        self.update_rollout_weights()
                        eval_metrics = self.evaluate()
                        eval_metrics = {f"eval/{k}": v for k, v in eval_metrics.items()}
                        self.metric_logger.log(data=eval_metrics, step=_step)

                if save_model:
                    self._save_checkpoint()

            time_metrics = self.timer.consume_durations()
            # Add rollout timing metrics (env_wait, generate)
            time_metrics.update(rollout_time_data)
            env_results_list = [
                results for results in env_handle.wait() if results is not None
            ]
            env_metrics = compute_evaluate_metrics(env_results_list)

            _EPISODE_TIME_KEYS = ("episode_time", "episode_env_step_time", "episode_action_wait_time")
            for _ekey in _EPISODE_TIME_KEYS:
                _all_vals = [r[_ekey] for r in env_results_list if _ekey in r]
                if _all_vals:
                    # 兼容：标量列表 或 张量列表（每个张量可能含多个值，如 vectorized env）
                    parts = []
                    for v in _all_vals:
                        if torch.is_tensor(v):
                            parts.append(v.flatten())
                        else:
                            parts.append(torch.tensor([v], dtype=torch.float))
                    _vals_tensor = torch.cat(parts)
                    env_metrics[f"{_ekey}_max"] = _vals_tensor.max().item()
                    env_metrics[f"{_ekey}_min"] = _vals_tensor.min().item()

            time_metrics = {f"time/{k}": v for k, v in time_metrics.items()}
            rollout_metrics = {
                f"rollout/{k}": v for k, v in actor_rollout_metrics[0].items()
            }
            env_metrics = {f"env/{k}": v for k, v in env_metrics.items()}
            training_metrics = {
                f"train/{k}": v for k, v in actor_training_metrics[0].items()
            }
            self.metric_logger.log(env_metrics, _step)
            self.metric_logger.log(rollout_metrics, _step)
            self.metric_logger.log(time_metrics, _step)
            self.metric_logger.log(training_metrics, _step)

            logging_metrics = time_metrics
            logging_metrics.update(eval_metrics)
            logging_metrics.update(env_metrics)
            logging_metrics.update(rollout_metrics)
            logging_metrics.update(training_metrics)

            global_pbar.set_postfix(logging_metrics, refresh=False)
            global_pbar.update(1)

    def _run_async(self):
        """
        Asynchronous pipeline training mode.

        Flow: Rollout and Training run in parallel, data sent per epoch
        """
        start_step = self.global_step
        global_pbar = tqdm(
            initial=start_step,
            total=self.max_steps,
            desc="Global Step",
            ncols=800,
        )
        self.send_demo_buffer()
        for _step in range(start_step, self.max_steps):
            self.actor.set_global_step(self.global_step)
            self.rollout.set_global_step(self.global_step)

            with self.timer("step"):
                with self.timer("sync_weights"):
                    self.update_rollout_weights()
                with self.timer("async_pipeline"):
                    # Start rollout and training in parallel
                    import time as time_module

                    env_handle: Handle = self.env.interact(
                        input_channel=self.rollout_channel,
                        output_channel=self.env_channel,
                    )

                    # Record rollout start time
                    rollout_start = time_module.time()
                    rollout_handle: Handle = self.rollout.generate(
                        input_channel=self.env_channel,
                        output_channel=self.rollout_channel,
                        actor_channel=self.actor_channel,
                    )

                    rollout_epoch = self.cfg.algorithm.rollout_epoch
                    actor_handle: Handle = self.actor.async_train_loop(
                        input_channel=self.actor_channel,
                        num_epochs=rollout_epoch,
                    )

                    # Wait for both to complete and record rollout time
                    rollout_handle.wait()
                    rollout_time = time_module.time() - rollout_start

                    all_epoch_metrics = actor_handle.wait()

                self.global_step += 1

                run_val, save_model, is_train_end = check_progress(
                    self.global_step,
                    self.max_steps,
                    self.cfg.runner.val_check_interval,
                    self.cfg.runner.save_interval,
                    1.0,
                    run_time_exceeded=False,
                )

                eval_metrics = {}
                if run_val:
                    with self.timer("eval"):
                        self.update_rollout_weights()
                        eval_metrics = self.evaluate()
                        eval_metrics = {f"eval/{k}": v for k, v in eval_metrics.items()}
                        self.metric_logger.log(data=eval_metrics, step=_step)

                if save_model:
                    self._save_checkpoint()

            time_metrics = self.timer.consume_durations()
            env_results_list = [
                results for results in env_handle.wait() if results is not None
            ]
            env_metrics = compute_evaluate_metrics(env_results_list)

            # # 添加max/min统计
            # _EPISODE_TIME_KEYS = ("episode_time", "episode_env_step_time", "episode_action_wait_time")
            # for _ekey in _EPISODE_TIME_KEYS:
            #     _all_vals = [r[_ekey] for r in env_results_list if _ekey in r]
            #     if _all_vals:
            #         _vals_tensor = torch.tensor(_all_vals)
            #         env_metrics[f"{_ekey}_max"] = _vals_tensor.max().item()
            #         env_metrics[f"{_ekey}_min"] = _vals_tensor.min().item()

            actor_rollout_metrics, actor_training_metrics, actor_training_time = self._aggregate_async_metrics(
                all_epoch_metrics
            )

            # Add rollout and training time to metrics
            time_metrics["generate_rollouts"] = rollout_time
            time_metrics["actor_training"] = actor_training_time
            time_metrics = {f"time/{k}": v for k, v in time_metrics.items()}
            rollout_metrics = {
                f"rollout/{k}": v for k, v in actor_rollout_metrics.items()
            }
            env_metrics = {f"env/{k}": v for k, v in env_metrics.items()}
            training_metrics = {
                f"train/{k}": v for k, v in actor_training_metrics.items()
            }
            self.metric_logger.log(env_metrics, _step)
            self.metric_logger.log(rollout_metrics, _step)
            self.metric_logger.log(time_metrics, _step)
            self.metric_logger.log(training_metrics, _step)

            logging_metrics = time_metrics
            logging_metrics.update(eval_metrics)
            logging_metrics.update(env_metrics)
            logging_metrics.update(rollout_metrics)
            logging_metrics.update(training_metrics)

            global_pbar.set_postfix(logging_metrics, refresh=False)
            global_pbar.update(1)

    def _aggregate_async_metrics(
        self, all_epoch_metrics: list[list[dict]]
    ) -> tuple[dict, dict, float]:
        """
        聚合异步流水线中多个 epoch 的 metrics。

        Args:
            all_epoch_metrics: 来自各个 worker 的 epoch metrics 列表
                每个 worker 返回 [{rollout: {...}, training: {...}, epoch: n, training_time: t}, ...]

        Returns:
            tuple: (aggregated_rollout_metrics, aggregated_training_metrics, total_training_time)
        """
        # 从第一个有效的 worker 获取 metrics（假设所有 worker 返回相同结构）
        worker_metrics = None
        for wm in all_epoch_metrics:
            if wm is not None and len(wm) > 0:
                worker_metrics = wm
                break

        if worker_metrics is None or len(worker_metrics) == 0:
            return {}, {}, 0.0

        # 聚合 rollout metrics
        rollout_keys = worker_metrics[0]["rollout"].keys()
        aggregated_rollout = {}
        for key in rollout_keys:
            values = [m["rollout"][key] for m in worker_metrics if "rollout" in m]
            if len(values) > 0:
                aggregated_rollout[key] = sum(values) / len(values)

        # 聚合 training metrics
        training_keys = worker_metrics[0]["training"].keys()
        aggregated_training = {}
        for key in training_keys:
            values = [m["training"][key] for m in worker_metrics if "training" in m]
            if len(values) > 0:
                aggregated_training[key] = sum(values) / len(values)

        # 提取训练时间（所有 epoch 的总和）
        total_training_time = sum(
            m.get("training_time", 0.0) for m in worker_metrics if "training_time" in m
        )

        return aggregated_rollout, aggregated_training, total_training_time
        
    def _save_checkpoint(self):
        base_output_dir = os.path.join(
            self.cfg.runner.logger.log_path,
            self.cfg.runner.logger.experiment_name,
            f"checkpoints/global_step_{self.global_step}",
        )
        actor_save_path = os.path.join(base_output_dir, "actor")
        os.makedirs(actor_save_path, exist_ok=True)
        self.actor.save_checkpoint(actor_save_path, self.global_step).wait()

    def set_max_steps(self):
        self.num_steps_per_epoch = 1
        self.max_steps = self.num_steps_per_epoch * self.cfg.runner.max_epochs

        if (max_steps := self.cfg.runner.get("max_steps", -1)) >= 0:
            self.max_steps = min(self.max_steps, max_steps)

    @property
    def epoch(self):
        return self.global_step // self.num_steps_per_epoch
