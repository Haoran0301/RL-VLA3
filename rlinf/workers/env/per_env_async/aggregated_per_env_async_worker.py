import logging
import time
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

import torch
from omegaconf import DictConfig

from rlinf.data.io_struct import EnvOutput
from rlinf.scheduler import Channel
from rlinf.workers.env.per_env_async.true_per_env_async_worker import (
    EnvLoopState,
    SerializedChannelReceiver,
    TruePerEnvAsyncEnvWorker,
)

logger = logging.getLogger(__name__)


class AggregatedPerEnvAsyncEnvWorker(TruePerEnvAsyncEnvWorker):
    """
    Experimental env worker for slot-based aggregation.

    Difference from TruePerEnvAsyncEnvWorker:
    - One env loop uses a larger env batch.
    - The env output is split into `aggregate_slots_per_env` slot chunks.
    - Each slot communicates with rollout via key: perenv_slot_{slot_id}_{mode}.
    - The env loop gathers all slot actions, concatenates them, then steps once.
    """

    def __init__(self, cfg: DictConfig):
        super().__init__(cfg)
        per_env_cfg = cfg.env.get("per_env_async", {})
        rollout_cfg = cfg.rollout.get("per_env_async", {})
        self.aggregate_slots_per_env = int(
            per_env_cfg.get(
                "aggregate_slots_per_env",
                rollout_cfg.get("aggregate_slots_per_env", 1),
            )
        )
        if self.aggregate_slots_per_env < 1:
            raise ValueError("aggregate_slots_per_env must be >= 1")

    def init_worker(self):
        super().init_worker()
        if self.aggregate_slots_per_env == 1:
            logger.info("AggregatedPerEnvAsyncEnvWorker runs in compatibility mode (slots=1).")
            return

        if not self.only_eval and self.train_num_envs_per_stage % self.aggregate_slots_per_env != 0:
            raise ValueError(
                "train_num_envs_per_stage must be divisible by aggregate_slots_per_env. "
                f"Got train_num_envs_per_stage={self.train_num_envs_per_stage}, "
                f"aggregate_slots_per_env={self.aggregate_slots_per_env}"
            )
        logger.info(
            "AggregatedPerEnvAsyncEnvWorker initialized: "
            f"rank={self._rank}, slots_per_env={self.aggregate_slots_per_env}, "
            f"train_num_envs_per_stage={getattr(self, 'train_num_envs_per_stage', None)}"
        )

    def _get_slot_id(self, global_env_id: int, slot_idx: int) -> int:
        return global_env_id * self.aggregate_slots_per_env + slot_idx

    def _get_slot_channel_key(self, global_env_id: int, slot_idx: int, mode: str = "train") -> str:
        slot_id = self._get_slot_id(global_env_id, slot_idx)
        return f"perenv_slot_{slot_id}_{mode}"

    def _split_tensor(self, value: Optional[torch.Tensor], n_chunks: int) -> List[Optional[torch.Tensor]]:
        if value is None:
            return [None for _ in range(n_chunks)]
        return list(torch.chunk(value, n_chunks, dim=0))

    def _split_nested(self, value: Any, n_chunks: int) -> List[Any]:
        if value is None:
            return [None for _ in range(n_chunks)]
        if isinstance(value, torch.Tensor):
            return list(torch.chunk(value, n_chunks, dim=0))
        if isinstance(value, dict):
            per_key_chunks = {k: self._split_nested(v, n_chunks) for k, v in value.items()}
            merged = []
            for i in range(n_chunks):
                merged.append({k: per_key_chunks[k][i] for k in per_key_chunks})
            return merged
        if isinstance(value, list):
            if len(value) % n_chunks != 0:
                return [value for _ in range(n_chunks)]
            chunk_size = len(value) // n_chunks
            return [value[i * chunk_size : (i + 1) * chunk_size] for i in range(n_chunks)]
        return [value for _ in range(n_chunks)]

    def _split_env_output(self, env_output: EnvOutput, n_chunks: int) -> List[EnvOutput]:
        if env_output.obs is None:
            raise ValueError("EnvOutput.obs must not be None")
        obs_chunks = self._split_nested(env_output.obs, n_chunks)
        final_obs_chunks = self._split_nested(env_output.final_obs, n_chunks)
        rewards_chunks = self._split_tensor(env_output.rewards, n_chunks)
        dones_chunks = self._split_tensor(env_output.dones, n_chunks)
        term_chunks = self._split_tensor(env_output.terminations, n_chunks)
        trunc_chunks = self._split_tensor(env_output.truncations, n_chunks)
        intervene_actions_chunks = self._split_tensor(env_output.intervene_actions, n_chunks)
        intervene_flags_chunks = self._split_tensor(env_output.intervene_flags, n_chunks)

        outputs = []
        for i in range(n_chunks):
            outputs.append(
                EnvOutput(
                    obs=obs_chunks[i],
                    final_obs=final_obs_chunks[i],
                    rewards=rewards_chunks[i],
                    dones=dones_chunks[i],
                    terminations=term_chunks[i],
                    truncations=trunc_chunks[i],
                    intervene_actions=intervene_actions_chunks[i],
                    intervene_flags=intervene_flags_chunks[i],
                )
            )
        return outputs

    def _recv_slot_actions(self, global_env_id: int) -> List[torch.Tensor]:
        actions = []
        for slot_idx in range(self.aggregate_slots_per_env):
            slot_key = self._get_slot_channel_key(global_env_id, slot_idx, "train")
            actions.append(self._serialized_action_receiver.get(slot_key))
        return actions

    def _send_slot_outputs(
        self,
        output_channel: Channel,
        global_env_id: int,
        env_output: EnvOutput,
        step_idx: int,
        epoch_idx: int,
    ):
        split_outputs = self._split_env_output(env_output, self.aggregate_slots_per_env)
        for slot_idx, split_output in enumerate(split_outputs):
            slot_key = self._get_slot_channel_key(global_env_id, slot_idx, "train")
            self._send_env_output(output_channel, slot_key, split_output, step_idx=step_idx, epoch_idx=epoch_idx)

    def _run_single_env_loop(
        self,
        stage_id: int,
        output_channel: Channel,
    ):
        if self.aggregate_slots_per_env == 1:
            return super()._run_single_env_loop(stage_id, output_channel)

        env_id = self._get_global_env_id(stage_id)
        n_chunk_steps = (
            self.cfg.env.train.max_steps_per_rollout_epoch
            // self.cfg.actor.model.num_action_chunks
        )

        env_metrics = defaultdict(list)
        episode_times: List[float] = []
        episode_env_step_times: List[float] = []
        episode_action_wait_times: List[float] = []

        state: EnvLoopState = self.env_states[env_id]
        state.is_running = True

        last_obs = None
        last_dones = None
        last_terminations = None
        last_truncations = None
        last_intervened_info = (None, None)

        if self.cfg.env.train.auto_reset:
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
                epoch_start_time = time.time()
                epoch_env_step_time = 0.0
                epoch_action_wait_time = 0.0

                if not self.cfg.env.train.auto_reset:
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
                    env_output = EnvOutput(
                        obs=last_obs,
                        rewards=None,
                        dones=last_dones,
                        terminations=last_terminations,
                        truncations=last_truncations,
                        intervene_actions=last_intervened_info[0],
                        intervene_flags=last_intervened_info[1],
                    )

                self._send_slot_outputs(
                    output_channel,
                    env_id,
                    env_output,
                    step_idx=-1,
                    epoch_idx=epoch,
                )

                for step in range(n_chunk_steps):
                    action_wait_start = time.time()
                    slot_actions = self._recv_slot_actions(env_id)
                    raw_chunk_actions = torch.cat(slot_actions, dim=0)
                    epoch_action_wait_time += time.time() - action_wait_start

                    env_step_start = time.time()
                    env_output, env_info = self._env_interact_step(raw_chunk_actions, stage_id)
                    epoch_env_step_time += time.time() - env_step_start

                    self._send_slot_outputs(
                        output_channel,
                        env_id,
                        env_output,
                        step_idx=step,
                        epoch_idx=epoch,
                    )

                    for key, value in env_info.items():
                        if (
                            not self.cfg.env.train.auto_reset
                            and not self.cfg.env.train.ignore_terminations
                        ):
                            if key in env_metrics and len(env_metrics[key]) > epoch:
                                env_metrics[key][epoch] = value
                            else:
                                env_metrics[key].append(value)
                        else:
                            env_metrics[key].append(value)
                    state.total_steps += 1

                episode_total_time = time.time() - epoch_start_time
                episode_times.append(episode_total_time)
                episode_env_step_times.append(epoch_env_step_time)
                episode_action_wait_times.append(epoch_action_wait_time)

                last_obs = env_output.obs
                last_dones = env_output.dones
                last_terminations = env_output.terminations
                last_truncations = env_output.truncations
                last_intervened_info = (
                    env_output.intervene_actions,
                    env_output.intervene_flags,
                )

                self._finish_env_rollout(stage_id)
                state.completed_epochs += 1

                if epoch < self.cfg.algorithm.rollout_epoch - 1:
                    for slot_idx in range(self.aggregate_slots_per_env):
                        slot_key = self._get_slot_channel_key(env_id, slot_idx, "train")
                        ack = self._serialized_action_receiver.get(slot_key)
                        if not ack.get("__epoch_done__"):
                            raise RuntimeError(
                                f"Env {env_id} epoch {epoch}, slot {slot_idx}: "
                                f"Expected epoch done ack, got {ack}"
                            )

            if episode_times:
                avg_time = sum(episode_times) / len(episode_times)
                logger.info(
                    f"Aggregated env {env_id} completed {len(episode_times)} episodes: "
                    f"avg={avg_time:.3f}s, total_steps={state.total_steps}"
                )
        except Exception as e:
            logger.error(f"Error in aggregated env {env_id} loop: {e}")
            raise
        finally:
            state.is_running = False

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

    def evaluate(self, input_channel: Channel, output_channel: Channel):
        if self.aggregate_slots_per_env > 1:
            raise NotImplementedError(
                "AggregatedPerEnvAsyncEnvWorker currently supports training interact only. "
                "Please set runner.val_check_interval=0 for this experimental mode."
            )
        return super().evaluate(input_channel, output_channel)
