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

"""
TruePerEnvAsyncEnvWorker: True Per-Env Async Pipeline implementation.

This is the REAL per-env async implementation where:
- Each EnvManager runs its OWN epoch loop INDEPENDENTLY
- An env can start the next rollout_epoch immediately after finishing the current one
- No waiting for other environments at epoch boundaries
- Uses env_id-based Channel routing for independent message flow

Architecture:
    ┌─────────────────────────────────────────────────────────────────────────┐
    │                     TruePerEnvAsyncEnvWorker                             │
    │                                                                          │
    │  ┌────────────────┐  ┌────────────────┐       ┌────────────────┐        │
    │  │  EnvManager 0  │  │  EnvManager 1  │  ...  │  EnvManager N  │        │
    │  │  (Independent  │  │  (Independent  │       │  (Independent  │        │
    │  │   epoch loop)  │  │   epoch loop)  │       │   epoch loop)  │        │
    │  └───────┬────────┘  └───────┬────────┘       └───────┬────────┘        │
    │          │                   │                        │                  │
    │          │ Key: env_0_train  │ Key: env_1_train       │ Key: env_N_train│
    │          │                   │                        │                  │
    │          └───────────────────┴────────────────────────┘                  │
    │                              │                                           │
    │                    RequestQueue (per env_id)                             │
    └──────────────────────────────┬───────────────────────────────────────────┘
                                   │
                                   ▼
    ┌─────────────────────────────────────────────────────────────────────────┐
    │                  PerEnvAsyncRolloutWorker                                │
    │                                                                          │
    │   DynamicBatchingEngine:                                                 │
    │   - Receives requests from ANY env_id                                    │
    │   - Batches dynamically across all envs                                  │
    │   - Returns results to specific env_id immediately                       │
    └─────────────────────────────────────────────────────────────────────────┘

Key improvements over PerEnvAsyncEnvWorker:
1. TRUE per-env independence - each env runs its own epoch loop
2. NO epoch-level synchronization - fast envs don't wait for slow ones
3. env_id-based routing - each env has its own Channel key
"""

import asyncio
import logging
import threading
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, wait
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch
from omegaconf import DictConfig

from rlinf.data.io_struct import EnvOutput
from rlinf.envs import get_env_cls
from rlinf.envs.action_utils import prepare_actions
from rlinf.envs.env_manager import EnvManager
from rlinf.scheduler import Channel, Cluster, Worker
from rlinf.utils.placement import HybridComponentPlacement
import time
import random

logger = logging.getLogger(__name__)


class SerializedChannelReceiver:
    """
    Wrapper around Channel that uses a global lock to serialize recv() calls.

    Channel.get()/get_any_nowait() eventually call Worker.recv(), which is not
    thread-safe.  Keep the lock only around the actual recv attempt so other
    env-manager threads are not blocked while one key is still empty.
    """

    def __init__(self, input_channel: Channel, poll_keys: Optional[List[str]] = None):
        self.input_channel = input_channel
        self._lock = threading.Lock()

    def get(
        self,
        key: str,
        stop_event: Optional[threading.Event] = None,
    ) -> Any:
        """Get from channel with global lock and short polling.

        If ``stop_event`` is set while waiting, returns ``None`` (caller should
        exit; used by flexible per-env async rollout puller threads).
        """
        while True:
            if stop_event is not None and stop_event.is_set():
                return None
            if not self.input_channel.empty(key=key):
                with self._lock:
                    if not self.input_channel.empty(key=key):
                        return self.input_channel.get(key=key)

            time.sleep(0.0001)

    def get_many(
        self,
        keys: List[str],
        stop_event: Optional[threading.Event] = None,
    ) -> Optional[List[Any]]:
        """Receive one item for each key without head-of-line polling."""
        remaining = list(keys)
        results: Dict[str, Any] = {}
        while remaining:
            if stop_event is not None and stop_event.is_set():
                return None

            got_item = False
            with self._lock:
                try:
                    key, item = self.input_channel.get_any_nowait(remaining)
                except asyncio.QueueEmpty:
                    key = None
                    item = None

            if key is not None:
                results[key] = item
                remaining.remove(key)
                got_item = True

            if not got_item:
                time.sleep(0.0001)

        return [results[key] for key in keys]


class SerializedChannelSender:
    """Serialize Channel.put() calls because Worker.send() is not thread-safe."""

    def __init__(self, output_channel: Channel):
        self.output_channel = output_channel
        self._lock = threading.Lock()

    def put(
        self,
        item: Any,
        key: str,
        weight: int = 0,
        async_op: bool = False,
    ) -> Any:
        with self._lock:
            return self.output_channel.put(
                item=item,
                weight=weight,
                key=key,
                async_op=async_op,
            )

    def put_many(
        self,
        items: list[tuple[Any, Any, int]],
        async_op: bool = False,
    ) -> Any:
        with self._lock:
            return self.output_channel.put_many(
                items=items,
                async_op=async_op,
            )


@dataclass
class EnvLoopState:
    """State for a single environment's rollout loop."""
    env_id: int
    stage_id: int
    completed_epochs: int = 0
    total_steps: int = 0
    is_running: bool = False


class TruePerEnvAsyncEnvWorker(Worker):
    """
    True Per-Env Async Environment Worker.

    Key architectural differences from PerEnvAsyncEnvWorker:
    1. Each EnvManager runs its OWN independent epoch loop
    2. NO epoch-level synchronization - envs don't wait for each other
    3. Uses env_id-based Channel routing (not gather_id)
    4. Envs can start next epoch immediately after finishing current one

    This enables the target behavior from the design doc:
    "当一个样本在环境 step 后立马发送给 rollout 模型进行生成，
     然后立马发送给 env 进行环境交互 step。完全不需要等待其他样本。"

    Configuration:
        env.per_env_async.enabled: true
        env.per_env_async.thread_pool_size: 16  # One per env ideally
    """

    def __init__(self, cfg: DictConfig):
        Worker.__init__(self)

        self.cfg = cfg
        self.train_video_cnt = 0
        self.eval_video_cnt = 0
        self.should_stop = False

        self.env_list: List[EnvManager] = []
        self.eval_env_list: List[EnvManager] = []

        # Per-env state tracking
        self.env_states: Dict[int, EnvLoopState] = {}

        self._component_placement = HybridComponentPlacement(cfg, Cluster())

        # For true per-env async, we don't use gather_num in the traditional sense
        # Each env has its own channel key
        self.stage_num = self.cfg.rollout.pipeline_stage_num

        # Env configurations
        self.only_eval = getattr(self.cfg.runner, "only_eval", False)
        self.enable_eval = self.cfg.runner.val_check_interval > 0 or self.only_eval
        if not self.only_eval:
            self.train_num_envs_per_stage = (
                self.cfg.env.train.total_num_envs // self._world_size // self.stage_num
            )
        if self.enable_eval:
            self.eval_num_envs_per_stage = (
                self.cfg.env.eval.total_num_envs // self._world_size // self.stage_num
            )

        # Per-env async specific configuration
        per_env_cfg = cfg.env.get("per_env_async", {})
        # Default: one thread per env to maximize parallelism
        total_envs = self.stage_num * getattr(self, 'train_num_envs_per_stage', 1)
        self.thread_pool_size = per_env_cfg.get("thread_pool_size", max(total_envs, 4))
        self.thread_pool: Optional[ThreadPoolExecutor] = None
        self._serialized_env_output_sender: Optional[SerializedChannelSender] = None

        # Lock for thread-safe metrics collection
        self._metrics_lock = threading.Lock()
        self._all_metrics: Dict[str, List[torch.Tensor]] = defaultdict(list)

    def init_worker(self):
        """Initialize worker with environments and thread pool."""
        enable_offload = self.cfg.env.enable_offload

        train_env_cls = get_env_cls(self.cfg.env.train.env_type, self.cfg.env.train)
        eval_env_cls = get_env_cls(self.cfg.env.eval.env_type, self.cfg.env.eval)

        # Barrier for initial setup
        self.broadcast(True, list(range(self._world_size)))

        if not self.only_eval:
            for stage_id in range(self.stage_num):
                self.env_list.append(
                    EnvManager(
                        self.cfg.env.train,
                        rank=self._rank,
                        num_envs=self.train_num_envs_per_stage,
                        seed_offset=self._rank * self.stage_num + stage_id,
                        total_num_processes=self._world_size * self.stage_num,
                        env_cls=train_env_cls,
                        worker_info=self.worker_info,
                        enable_offload=enable_offload,
                    )
                )

        if self.enable_eval:
            for stage_id in range(self.stage_num):
                self.eval_env_list.append(
                    EnvManager(
                        self.cfg.env.eval,
                        rank=self._rank,
                        num_envs=self.eval_num_envs_per_stage,
                        seed_offset=self._rank * self.stage_num + stage_id,
                        total_num_processes=self._world_size * self.stage_num,
                        env_cls=eval_env_cls,
                        worker_info=self.worker_info,
                        enable_offload=enable_offload,
                    )
                )

        # Initialize thread pool
        self.thread_pool = ThreadPoolExecutor(
            max_workers=self.thread_pool_size,
            thread_name_prefix="TruePerEnvAsync"
        )

        # Initialize env states
        for stage_id in range(self.stage_num):
            env_id = self._get_global_env_id(stage_id)
            self.env_states[env_id] = EnvLoopState(
                env_id=env_id,
                stage_id=stage_id,
                completed_epochs=0,
                total_steps=0,
                is_running=False,
            )

        # Pre-initialize PyTorch linalg module to avoid "lazy wrapper should be called
        # at most once" when multiple threads first call torch.inverse/torch.linalg
        # (e.g., in ManiSkill env step). See: https://github.com/pytorch/pytorch/issues/90613
        if torch.cuda.is_available():
            try:
                device_id = self._rank % torch.cuda.device_count()
                torch.inverse(torch.ones((1, 1), device=f"cuda:{device_id}"))
            except Exception:
                pass

        logger.info(f"TruePerEnvAsyncEnvWorker initialized: rank={self._rank}, "
                   f"stages={self.stage_num}, envs_per_stage={self.train_num_envs_per_stage}, "
                   f"thread_pool_size={self.thread_pool_size}")

    def _get_global_env_id(self, stage_id: int) -> int:
        """Get global env_id from stage_id."""
        return self._rank * self.stage_num + stage_id

    def _get_env_channel_key(self, stage_id: int, mode: str = "train") -> str:
        """
        Get the Channel key for a specific environment.

        Key format: "perenv_{global_env_id}_{mode}"
        Uses global env_id to support flexible env-rollout worker mapping.
        This allows env and rollout workers to have different counts.
        """
        global_env_id = self._get_global_env_id(stage_id)
        return f"perenv_{global_env_id}_{mode}"

    def interact(self, input_channel: Channel, output_channel: Channel):
        """
        Main interaction loop with TRUE per-env async.

        Each EnvManager runs its own independent epoch loop.
        No synchronization between environments at epoch boundaries.

        IMPORTANT: Uses SerializedChannelReceiver to avoid race condition where
        multiple threads calling Channel.get() can receive each other's messages.
        """
        # Start all environments
        for env in self.env_list:
            env.start_env()

        # Reset metrics
        with self._metrics_lock:
            self._all_metrics.clear()

        # Create serialized receiver to avoid race condition when receiving actions
        self._serialized_action_receiver = SerializedChannelReceiver(input_channel)
        self._serialized_env_output_sender = SerializedChannelSender(output_channel)

        # Launch independent loops for each env using ThreadPoolExecutor
        futures = []
        for stage_id in range(self.stage_num):
            future = self.thread_pool.submit(
                self._run_single_env_loop,
                stage_id,
                output_channel,
            )
            futures.append(future)

        # Wait for ALL envs to complete ALL their epochs
        wait(futures)

        # Check for exceptions
        for future in futures:
            if future.exception() is not None:
                raise future.exception()

        # Stop all environments
        for env in self.env_list:
            env.stop_env()

        # Aggregate metrics
        with self._metrics_lock:
            result_metrics = {}
            for key, values in self._all_metrics.items():
                if values:
                    result_metrics[key] = torch.cat(values, dim=0).contiguous().cpu()

        return result_metrics

    def _run_single_env_loop(
        self,
        stage_id: int,
        output_channel: Channel,
    ):
        """
        Run the rollout loop for a SINGLE environment.

        This is the core of true per-env async:
        - This method runs independently in its own thread
        - Completes ALL rollout_epochs for this env
        - Does NOT wait for other envs at any point

        IMPORTANT: Uses self._serialized_action_receiver to avoid race condition.
        """
        env_id = self._get_global_env_id(stage_id)
        env_key = self._get_env_channel_key(stage_id, "train")

        # DEBUG: 检查 rank, stage_id, env_id, env_key
        # logger.info(f"[DEBUG] EnvWorker handler start: rank={self._rank}, stage_id={stage_id}, "
        #            f"env_id={env_id}, env_key={env_key}")

        n_chunk_steps = (
            self.cfg.env.train.max_steps_per_rollout_epoch
            // self.cfg.actor.model.num_action_chunks
        )

        env_metrics = defaultdict(list)
        # 新增：存储每个episode的时间统计
        episode_times: List[float] = []
        episode_env_step_times: List[float] = []
        episode_action_wait_times: List[float] = []

        state = self.env_states[env_id]
        state.is_running = True

        # logger.debug(f"Env {env_id} starting independent loop: "
        #             f"{self.cfg.algorithm.rollout_epoch} epochs, "
        #             f"{n_chunk_steps} steps each")

        # Initialize last obs (for auto_reset mode)
        last_obs = None
        last_dones = None
        last_terminations = None
        last_truncations = None
        last_intervened_info = (None, None)

        if self.cfg.env.train.auto_reset:
            # Initial reset to get first obs
            self.env_list[stage_id].is_start = True
            extracted_obs, _ = self.env_list[stage_id].reset()
            dones = (
                torch.zeros((self.train_num_envs_per_stage,), dtype=bool)
                .unsqueeze(1)
                .repeat(1, self.cfg.actor.model.num_action_chunks)
            )
            last_obs = extracted_obs
            last_dones = dones
            last_terminations = dones.clone()
            last_truncations = dones.clone()

        try:
            for epoch in range(self.cfg.algorithm.rollout_epoch):
                # ============================================================
                # EPOCH START - 开始计时
                # ============================================================
                epoch_start_time = time.time()
                epoch_env_step_time = 0.0
                epoch_action_wait_time = 0.0

                if not self.cfg.env.train.auto_reset:
                    # Manual reset at start of each epoch
                    self.env_list[stage_id].is_start = True
                    extracted_obs, infos = self.env_list[stage_id].reset()
                    dones = (
                        torch.zeros((self.train_num_envs_per_stage,), dtype=bool)
                        .unsqueeze(1)
                        .repeat(1, self.cfg.actor.model.num_action_chunks)
                    )
                    env_output = EnvOutput(
                        obs=extracted_obs,
                        dones=dones,
                        terminations=dones.clone(),
                        truncations=dones.clone(),
                        final_obs=infos.get("final_observation"),
                        intervene_actions=None,
                        intervene_flags=None,
                    )
                else:
                    # Use last obs from previous epoch (auto_reset mode)
                    env_output = EnvOutput(
                        obs=last_obs,
                        rewards=None,
                        dones=last_dones,
                        terminations=last_terminations,
                        truncations=last_truncations,
                        intervene_actions=last_intervened_info[0],
                        intervene_flags=last_intervened_info[1],
                    )

                # Send initial observation for this epoch
                # Debug: log initial observation
                # logger.info(f"[DEBUG EnvWorker] Env {env_id} epoch {epoch}: Sending initial obs, "
                #            f"rewards is None: {env_output.rewards is None}, env_key={env_key}")
                self._send_env_output(output_channel, env_key, env_output, step_idx=-1, epoch_idx=epoch)

                # ============================================================
                # STEP LOOP - Independent per-env
                # ============================================================
                for step in range(n_chunk_steps):
                    # Receive action for THIS env only
                    action_wait_start = time.time()
                    raw_chunk_actions = self._recv_chunk_actions(env_key)
                    epoch_action_wait_time += time.time() - action_wait_start

                    # random_sleep = random.random() ** 2
                    # time.sleep(random_sleep)
                    # # Execute env step

                    env_step_start = time.time()
                    env_output, env_info = self._env_interact_step(
                        raw_chunk_actions, stage_id
                    )
                    epoch_env_step_time += time.time() - env_step_start

                    # Send observation immediately - no waiting for other envs!
                    self._send_env_output(output_channel, env_key, env_output, step_idx=step, epoch_idx=epoch)

                    # # DEBUG: Log metrics collection
                    # if step == 0 or step == n_chunk_steps - 1 or len(env_info) > 0:
                    #     logger.info(f"[DEBUG EnvWorker] Env {env_id} epoch {epoch} step {step}: "
                    #                f"env_info keys={list(env_info.keys())}, "
                    #                f"has_episode_len={'episode_len' in env_info}")
                    #     if 'episode_len' in env_info:
                    #         logger.info(f"[DEBUG EnvWorker] Env {env_id} epoch {epoch} step {step}: "
                    #                    f"episode_len={env_info['episode_len']}")

                    # Collect metrics
                    # Match sync version behavior: in auto_reset=false + ignore_terminations=false mode,
                    # overwrite metrics for the same epoch instead of appending
                    for key, value in env_info.items():
                        if (
                            not self.cfg.env.train.auto_reset
                            and not self.cfg.env.train.ignore_terminations
                        ):
                            if key in env_metrics and len(env_metrics[key]) > epoch:
                                env_metrics[key][epoch] = value  # Overwrite
                            else:
                                env_metrics[key].append(value)
                        else:
                            env_metrics[key].append(value)

                    state.total_steps += 1


                # 计算总episode时间
                episode_total_time = time.time() - epoch_start_time
                episode_times.append(episode_total_time)
                episode_env_step_times.append(epoch_env_step_time)
                episode_action_wait_times.append(epoch_action_wait_time)
                
                # 计算其他开销时间
                episode_other_time = episode_total_time - epoch_env_step_time - epoch_action_wait_time
                
                # 记录每个epoch的详细时间信息
                logger.info(
                    f"Env {env_id} epoch {epoch}: "
                    f"episode_time={episode_total_time:.3f}s "
                    f"(env_step={epoch_env_step_time:.3f}s, "
                    f"action_wait={epoch_action_wait_time:.3f}s, "
                    f"other={episode_other_time:.3f}s)"
                )
                
                # 将时间指标添加到env_metrics中，以便后续上报
                if epoch < len(env_metrics.get('episode_len', [1])):
                    # 如果有episode_len，在对应位置添加时间指标
                    if 'episode_time' not in env_metrics:
                        env_metrics['episode_time'] = torch.zeros(self.cfg.algorithm.rollout_epoch, 1)
                    if 'episode_env_step_time' not in env_metrics:
                        env_metrics['episode_env_step_time'] = torch.zeros(self.cfg.algorithm.rollout_epoch, 1)
                    if 'episode_action_wait_time' not in env_metrics:
                        env_metrics['episode_action_wait_time'] = torch.zeros(self.cfg.algorithm.rollout_epoch, 1)
                    
                    env_metrics['episode_time'][epoch][0] = episode_total_time
                    env_metrics['episode_env_step_time'][epoch][0] = epoch_env_step_time
                    env_metrics['episode_action_wait_time'][epoch][0] = epoch_action_wait_time
                    
                # ============================================================
                # EPOCH END - Update state and IMMEDIATELY continue
                # ============================================================
                last_obs = env_output.obs
                last_dones = env_output.dones
                last_terminations = env_output.terminations
                last_truncations = env_output.truncations
                last_intervened_info = (
                    env_output.intervene_actions,
                    env_output.intervene_flags,
                )

                # Finish rollout for this env
                self._finish_env_rollout(stage_id)

                state.completed_epochs += 1
                # logger.info(f"[DEBUG EnvWorker] Env {env_id} completed epoch {epoch}, total completed: {state.completed_epochs}")

                # # DEBUG: Log metrics summary for this epoch
                # logger.info(f"[DEBUG EnvWorker] Env {env_id} completed epoch {epoch + 1}/{self.cfg.algorithm.rollout_epoch}")
                # for key in env_metrics.keys():
                #     logger.info(f"[DEBUG EnvWorker] Env {env_id} epoch {epoch}: "
                #                f"metric '{key}' collected {len(env_metrics[key])} times")

                # Wait for RolloutWorker's epoch completion acknowledgment
                # This prevents message misalignment at epoch boundaries
                # This is needed in BOTH sync and async modes
                if epoch < self.cfg.algorithm.rollout_epoch - 1:
                    logger.info(f"[DEBUG EnvWorker] Env {env_id} epoch {epoch}: Waiting for ack from RolloutWorker")
                    ack = self._serialized_action_receiver.get(env_key)
                    if not ack.get("__epoch_done__"):
                        raise RuntimeError(f"Env {env_id} epoch {epoch}: Expected epoch done ack, got {ack}")
                    logger.info(f"[DEBUG EnvWorker] Env {env_id} epoch {epoch}: Received ack, starting next epoch")


            if episode_times:
                avg_time = sum(episode_times) / len(episode_times)
                min_time = min(episode_times)
                max_time = max(episode_times)
                logger.info(
                    f"Env {env_id} completed {len(episode_times)} episodes: "
                    f"avg={avg_time:.3f}s, min={min_time:.3f}s, max={max_time:.3f}s, "
                    f"total_steps={state.total_steps}"
                )

        except Exception as e:
            logger.error(f"Error in env {env_id} loop: {e}")
            raise
        finally:
            state.is_running = False

        # Aggregate metrics for this env
        # Note: env_metrics can contain both lists (of tensors) and raw tensors
        # (e.g. episode_time, episode_env_step_time). Use len()/numel() to avoid
        # "Boolean value of Tensor with more than one value is ambiguous".
        with self._metrics_lock:
            for key, values in env_metrics.items():
                has_values = (
                    len(values) > 0
                    if isinstance(values, (list, tuple))
                    else (isinstance(values, torch.Tensor) and values.numel() > 0)
                )
                if has_values:
                    if isinstance(values, torch.Tensor):
                        self._all_metrics[key].append(values)
                    else:
                        self._all_metrics[key].extend(values)

        # DEBUG: Log final metrics summary
        # logger.info(f"[DEBUG EnvWorker] Env {env_id} completed all epochs. "
        #            f"Total steps: {state.total_steps}")
        # for key, values in env_metrics.items():
        #     if values:
        #         logger.info(f"[DEBUG EnvWorker] Env {env_id} final: "
        #                    f"metric '{key}' total count={len(values)}")

    def _env_interact_step(
        self, chunk_actions: torch.Tensor, stage_id: int
    ) -> Tuple[EnvOutput, Dict[str, Any]]:
        """Execute a single step for a single env."""
        chunk_actions = prepare_actions(
            raw_chunk_actions=chunk_actions,
            env_type=self.cfg.env.train.env_type,
            model_type=self.cfg.actor.model.model_type,
            num_action_chunks=self.cfg.actor.model.num_action_chunks,
            action_dim=self.cfg.actor.model.action_dim,
            policy=self.cfg.actor.model.get("policy_setup", None),
        )
        env_info = {}

        extracted_obs, chunk_rewards, chunk_terminations, chunk_truncations, infos = (
            self.env_list[stage_id].chunk_step(chunk_actions)
        )
        chunk_dones = torch.logical_or(chunk_terminations, chunk_truncations)

        # Collect episode info
        # Match sync version behavior (env_worker.py:152-162)
        if not self.cfg.env.train.auto_reset:
            if self.cfg.env.train.ignore_terminations:
                if chunk_truncations[:, -1].any():
                    if "episode" in infos:
                        for key in infos["episode"]:
                            env_info[key] = infos["episode"][key].cpu()
            else:
                # In auto_reset=false + ignore_terminations=false mode,
                # sync version collects metrics at EVERY step and uses overwrite strategy
                # to keep only the last value per epoch
                if "episode" in infos:
                    for key in infos["episode"]:
                        env_info[key] = infos["episode"][key].cpu()
        elif chunk_dones.any():
            if "final_info" in infos:
                final_info = infos["final_info"]
                for key in final_info["episode"]:
                    env_info[key] = final_info["episode"][key][chunk_dones[:, -1]].cpu()

        intervene_actions = infos.get("intervene_action")
        intervene_flags = infos.get("intervene_flag")
        if self.cfg.env.train.auto_reset and chunk_dones.any():
            if "intervene_action" in infos.get("final_info", {}):
                intervene_actions = infos["final_info"]["intervene_action"]
                intervene_flags = infos["final_info"]["intervene_flag"]

        env_output = EnvOutput(
            obs=extracted_obs,
            final_obs=infos.get("final_observation"),
            rewards=chunk_rewards,
            dones=chunk_dones,
            terminations=chunk_terminations,
            truncations=chunk_truncations,
            intervene_actions=intervene_actions,
            intervene_flags=intervene_flags,
        )
        return env_output, env_info

    def _send_env_output(
        self, output_channel: Channel, env_key: str, env_output: EnvOutput,
        step_idx: int = -1, epoch_idx: int = 0
    ):
        """Send env output using env-specific key.

        Includes _debug_env_key in the message for the dispatcher to route correctly.
        """
        env_output_dict = env_output.to_dict()
        # Add routing info for PerEnvMessageDispatcher
        env_output_dict["_debug_step_idx"] = step_idx
        env_output_dict["_debug_epoch_idx"] = epoch_idx
        env_output_dict["_debug_env_key"] = env_key
        sender = getattr(self, "_serialized_env_output_sender", None)
        if sender is not None:
            sender.put(
                item=env_output_dict,
                key=env_key,
            )
        else:
            output_channel.put(
                item=env_output_dict,
                key=env_key,
            )

    def _recv_chunk_actions(self, env_key: str) -> torch.Tensor:
        """Receive actions for a specific env using serialized receiver."""
        return self._serialized_action_receiver.get(env_key)

    def _finish_env_rollout(self, stage_id: int):
        """Finish rollout for a single env."""
        if self.cfg.env.train.video_cfg.save_video:
            self.env_list[stage_id].flush_video()
        self.env_list[stage_id].update_reset_state_ids()

    def evaluate(self, input_channel: Channel, output_channel: Channel):
        """Evaluation loop with true per-env async."""
        # Start all eval environments
        for stage_id in range(self.stage_num):
            self.eval_env_list[stage_id].start_env()

        # Reset metrics
        with self._metrics_lock:
            self._all_metrics.clear()

        # Create serialized receiver for eval actions
        self._serialized_action_receiver = SerializedChannelReceiver(input_channel)
        self._serialized_env_output_sender = SerializedChannelSender(output_channel)

        # Launch independent eval loops for each env using ThreadPoolExecutor
        futures = []
        for stage_id in range(self.stage_num):
            future = self.thread_pool.submit(
                self._run_single_env_eval_loop,
                stage_id,
                output_channel,
            )
            futures.append(future)

        # Wait for all eval handlers to complete
        wait(futures)

        # Check for exceptions
        for future in futures:
            if future.exception() is not None:
                raise future.exception()

        # Stop all eval environments
        for stage_id in range(self.stage_num):
            self.eval_env_list[stage_id].stop_env()

        # Aggregate metrics
        with self._metrics_lock:
            result_metrics = {}
            for key, values in self._all_metrics.items():
                if values:
                    result_metrics[key] = torch.cat(values, dim=0).contiguous().cpu()

        return result_metrics

    def _run_single_env_eval_loop(
        self,
        stage_id: int,
        output_channel: Channel,
    ):
        """Run eval loop for a single environment."""
        env_key = self._get_env_channel_key(stage_id, "eval")

        n_chunk_steps = (
            self.cfg.env.eval.max_steps_per_rollout_epoch
            // self.cfg.actor.model.num_action_chunks
        )

        eval_metrics = defaultdict(list)

        for epoch in range(self.cfg.algorithm.eval_rollout_epoch):
            # Reset env
            self.eval_env_list[stage_id].is_start = True
            extracted_obs, infos = self.eval_env_list[stage_id].reset()

            env_output = EnvOutput(
                obs=extracted_obs,
                final_obs=infos.get("final_observation"),
            )
            self._send_env_output(output_channel, env_key, env_output)

            for eval_step in range(n_chunk_steps):
                # Receive action using serialized receiver
                raw_chunk_actions = self._recv_chunk_actions(env_key)

                # Step
                env_output, env_info = self._env_evaluate_step(
                    raw_chunk_actions, stage_id
                )

                for key, value in env_info.items():
                    eval_metrics[key].append(value)

                if eval_step < n_chunk_steps - 1:
                    self._send_env_output(output_channel, env_key, env_output)

            self._finish_env_rollout_eval(stage_id)

        # Aggregate metrics
        with self._metrics_lock:
            for key, values in eval_metrics.items():
                if values:
                    self._all_metrics[key].extend(values)

    def _env_evaluate_step(
        self, raw_actions: torch.Tensor, stage_id: int
    ) -> Tuple[EnvOutput, Dict[str, Any]]:
        """Execute a single eval step."""
        chunk_actions = prepare_actions(
            raw_chunk_actions=raw_actions,
            env_type=self.cfg.env.train.env_type,
            model_type=self.cfg.actor.model.model_type,
            num_action_chunks=self.cfg.actor.model.num_action_chunks,
            action_dim=self.cfg.actor.model.action_dim,
            policy=self.cfg.actor.model.get("policy_setup", None),
        )
        env_info = {}

        extracted_obs, chunk_rewards, chunk_terminations, chunk_truncations, infos = (
            self.eval_env_list[stage_id].chunk_step(chunk_actions)
        )
        chunk_dones = torch.logical_or(chunk_terminations, chunk_truncations)

        if chunk_dones.any():
            if "episode" in infos:
                for key in infos["episode"]:
                    env_info[key] = infos["episode"][key].cpu()
            if "final_info" in infos:
                final_info = infos["final_info"]
                for key in final_info["episode"]:
                    env_info[key] = final_info["episode"][key][chunk_dones[:, -1]].cpu()

        env_output = EnvOutput(
            obs=extracted_obs,
            final_obs=infos.get("final_observation"),
        )
        return env_output, env_info

    def _finish_env_rollout_eval(self, stage_id: int):
        """Finish eval rollout for a single env."""
        if self.cfg.env.eval.video_cfg.save_video:
            self.eval_env_list[stage_id].flush_video()
        if not self.cfg.env.eval.auto_reset:
            self.eval_env_list[stage_id].update_reset_state_ids()

    def __del__(self):
        """Cleanup thread pool."""
        if getattr(self, 'thread_pool', None):
            self.thread_pool.shutdown(wait=False)