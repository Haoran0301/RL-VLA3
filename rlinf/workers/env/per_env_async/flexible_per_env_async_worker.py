import json
import logging
import os
import threading
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, wait
from typing import Any, Dict, List, Optional

import torch
from omegaconf import DictConfig

from rlinf.data.io_struct import EnvOutput
from rlinf.envs import get_env_cls
from rlinf.envs.env_manager import EnvManager
from rlinf.scheduler import Channel
from rlinf.workers.env.per_env_async.true_per_env_async_worker import (
    EnvLoopState,
    SerializedChannelReceiver,
    SerializedChannelSender,
    TruePerEnvAsyncEnvWorker,
)
from rlinf.workers.per_env_flex_plan import FlexPlan, SlotSpec, build_flex_plan

logger = logging.getLogger(__name__)


class FlexiblePerEnvAsyncEnvWorker(TruePerEnvAsyncEnvWorker):
    """
    Flexible env worker with configurable per-rank manager sizes and rollout routing.

    Key behavior:
    - Per env-rank can have variable number of managers with variable batch sizes.
    - Managers are processed sequentially within one env worker.
    - Each manager output can be split to arbitrary rollout workers via slot plan.
    """

    def __init__(self, cfg: DictConfig):
        super().__init__(cfg)
        self.flex_plan: Optional[FlexPlan] = None
        self.local_manager_batch_sizes: List[int] = []
        self.local_global_env_ids: List[int] = []
        # If enabled, send all slot outputs with async put then wait once.
        self._slot_put_async_enabled: bool = True
        # How to wait async slot put works: "strict" (wait all) or "none" (fire-and-forget).
        self._slot_put_wait_mode: str = "strict"
        # Clone split slot tensors into compact storage before Channel.put().
        # This avoids serializing the original manager-sized storage behind torch.split views.
        self._compact_slot_chunks: bool = True
        # Skip per-epoch rollout ack waits when rollout frontier checks can absorb prefetch.
        self._epoch_prefetch_enabled: bool = False
        # [DEBUG_TIMING] debug timing state
        self._debug_t0: float = 0.0
        self._debug_log: Optional[List[Dict[str, Any]]] = None
        self._debug_log_lock: Optional[threading.Lock] = None
        self._debug_timing_enabled: bool = False
        self._debug_timing_dir: str = "."

    def _get_slot_channel_key(self, global_env_id: int, local_slot_index: int, mode: str = "train") -> str:
        """Channel key by (global_env_id, local_slot_index) so each env manager's slots are locally numbered."""
        return f"perenv_slot_{global_env_id}_{local_slot_index}_{mode}"

    def _get_rollout_input_channel_key(self, rollout_rank: int, slot_id: int, mode: str = "train") -> str:
        """Channel key for env->rollout data path grouped by rollout rank and slot id."""
        return f"perenv_rollout_input_{rollout_rank}_{slot_id}_{mode}"

    def _write_debug_timing_log(self, entry: Dict[str, Any]) -> None:
        """[DEBUG_TIMING] Append one timing entry to in-memory log (thread-safe)."""
        if not self._debug_timing_enabled or self._debug_log is None or self._debug_log_lock is None:
            return
        with self._debug_log_lock:
            self._debug_log.append(entry)

    def _build_env_output_dict(
        self,
        env_key: str,
        env_output: EnvOutput,
        step_idx: int = -1,
        epoch_idx: int = 0,
        *,
        global_env_id: Optional[int] = None,
        batch_ts: Optional[float] = None,
        route_slot_id: Optional[int] = None,
        route_local_slot_index: Optional[int] = None,
        async_op: bool = False,
    ) -> tuple[Dict[str, Any], Optional[float]]:
        """Build one env message and emit pre-submit timing logs."""
        env_output_dict = env_output.to_dict()
        env_output_dict["_debug_step_idx"] = step_idx
        env_output_dict["_debug_epoch_idx"] = epoch_idx
        env_output_dict["_debug_env_key"] = env_key
        env_output_dict["_batch_ts"] = batch_ts if batch_ts is not None else time.time()
        env_output_dict["_route_slot_id"] = route_slot_id
        env_output_dict["_route_local_slot_index"] = route_local_slot_index

        put_start_ts: Optional[float] = None
        if self._debug_timing_enabled:
            put_start_ts = time.time()
            env_output_dict["_debug_put_start_ts"] = put_start_ts
            self._write_debug_timing_log(
                {
                    "event": "env_put_start",
                    "ts_rel": put_start_ts - self._debug_t0,
                    "ts_abs": put_start_ts,
                    "env_rank": self._rank,
                    "global_env_id": global_env_id,
                    "env_key": env_key,
                    "step_idx": step_idx,
                    "epoch_idx": epoch_idx,
                    "route_slot_id": route_slot_id,
                    "async_op": async_op,
                }
            )

            t_now = time.time()
            env_output_dict["_debug_env_rank"] = self._rank
            env_output_dict["_debug_global_env_id"] = global_env_id
            env_output_dict["_debug_ts_send"] = t_now
            env_output_dict["_debug_ts_since_start"] = t_now - self._debug_t0
            self._write_debug_timing_log(
                {
                    "event": "env_send",
                    "ts_rel": t_now - self._debug_t0,
                    "ts_abs": t_now,
                    "env_rank": self._rank,
                    "global_env_id": global_env_id,
                    "env_key": env_key,
                    "step_idx": step_idx,
                    "epoch_idx": epoch_idx,
                }
            )

        return env_output_dict, put_start_ts

    def _record_env_put_submit(
        self,
        *,
        put_start_ts: Optional[float],
        put_submit_ts: Optional[float],
        put_return_ts: Optional[float],
        env_key: str,
        step_idx: int,
        epoch_idx: int,
        global_env_id: Optional[int],
        route_slot_id: Optional[int],
        async_op: bool,
    ) -> None:
        if not self._debug_timing_enabled:
            return
        self._write_debug_timing_log(
            {
                "event": "env_put_submit",
                "ts_rel": put_submit_ts - self._debug_t0 if put_submit_ts is not None else None,
                "ts_abs": put_submit_ts,
                "env_rank": self._rank,
                "global_env_id": global_env_id,
                "env_key": env_key,
                "step_idx": step_idx,
                "epoch_idx": epoch_idx,
                "route_slot_id": route_slot_id,
                "async_op": async_op,
                "put_prepare_delay": (
                    put_submit_ts - put_start_ts
                    if put_submit_ts is not None and put_start_ts is not None
                    else None
                ),
                "put_call_return_delay": (
                    put_return_ts - put_submit_ts
                    if put_return_ts is not None and put_submit_ts is not None
                    else None
                ),
            }
        )

    def _send_env_output(
        self,
        output_channel: Channel,
        env_key: str,
        env_output: EnvOutput,
        step_idx: int = -1,
        epoch_idx: int = 0,
        *,
        global_env_id: Optional[int] = None,
        batch_ts: Optional[float] = None,
        route_slot_id: Optional[int] = None,
        route_local_slot_index: Optional[int] = None,
        async_op: bool = False,
    ):
        """[DEBUG_TIMING] Override: add debug fields and optional timing log."""
        env_output_dict, put_start_ts = self._build_env_output_dict(
            env_key=env_key,
            env_output=env_output,
            step_idx=step_idx,
            epoch_idx=epoch_idx,
            global_env_id=global_env_id,
            batch_ts=batch_ts,
            route_slot_id=route_slot_id,
            route_local_slot_index=route_local_slot_index,
            async_op=async_op,
        )

        put_submit_ts = time.time() if self._debug_timing_enabled else None
        if put_submit_ts is not None:
            env_output_dict["_debug_put_submit_ts"] = put_submit_ts

        sender = getattr(self, "_serialized_env_output_sender", None)
        if sender is not None:
            put_work = sender.put(item=env_output_dict, key=env_key, async_op=async_op)
        else:
            put_work = output_channel.put(item=env_output_dict, key=env_key, async_op=async_op)

        put_return_ts = time.time() if self._debug_timing_enabled else None
        self._record_env_put_submit(
            put_start_ts=put_start_ts,
            put_submit_ts=put_submit_ts,
            put_return_ts=put_return_ts,
            env_key=env_key,
            step_idx=step_idx,
            epoch_idx=epoch_idx,
            global_env_id=global_env_id,
            route_slot_id=route_slot_id,
            async_op=async_op,
        )
        return put_work

    def _format_timing_log_dir(self, timing_dir: str) -> str:
        """Format timing_log_dir with current timestamp.

        Supports placeholders:
        - {timestamp}: replaced with current time in format MMDDHHMMSS
        - {datetime}: replaced with current time in format YYYYMMDD_HHMMSS

        Examples:
            "./debug_timing_logs_{timestamp}" -> "./debug_timing_logs_0313235200"
            "./logs/{datetime}_experiment" -> "./logs/20250313_235200_experiment"
        """
        from datetime import datetime

        now = datetime.now()
        formatted = timing_dir.replace("{timestamp}", now.strftime("%m%d%H%M%S"))
        formatted = formatted.replace("{datetime}", now.strftime("%Y%m%d_%H%M%S"))
        return formatted

    def init_worker(self):
        enable_offload = self.cfg.env.enable_offload
        train_env_cls = get_env_cls(self.cfg.env.train.env_type, self.cfg.env.train)
        eval_env_cls = get_env_cls(self.cfg.env.eval.env_type, self.cfg.env.eval)
        self.broadcast(True, list(range(self._world_size)))

        # Use the same calculation logic as rollout side to ensure consistency
        rollout_world_size = self._component_placement.get_world_size("rollout")
        
        # When flex config is provided, derive defaults from config instead of stage_num
        flex_cfg = self.cfg.env.get("per_env_async", {}).get("flex", None)
        if flex_cfg is not None and flex_cfg.get("enabled", False):
            # Use config's manager_batch_sizes_by_env_rank to determine defaults
            mgr_batch_sizes = flex_cfg.get("manager_batch_sizes_by_env_rank", [])
            if mgr_batch_sizes:
                # default_stage_num = max number of managers per env rank from config
                default_stage_num = max(len(sizes) for sizes in mgr_batch_sizes)
                # default_batch_size = average batch size from config
                total_envs = sum(sum(sizes) for sizes in mgr_batch_sizes)
                default_batch_size = total_envs // max(1, self._world_size) // max(1, default_stage_num)
            else:
                default_stage_num = self.stage_num
                default_batch_size = (
                    self.cfg.env.train.total_num_envs // max(1, self._world_size) // max(1, default_stage_num)
                )
        else:
            default_stage_num = self.stage_num
            default_batch_size = (
                self.cfg.env.train.total_num_envs // max(1, self._world_size) // max(1, default_stage_num)
            )
        
        self.flex_plan = build_flex_plan(
            cfg=self.cfg,
            env_world_size=self._world_size,
            rollout_world_size=rollout_world_size,
            default_stage_num=default_stage_num,
            default_batch_size=default_batch_size,
        )
        per_env_cfg = self.cfg.env.get("per_env_async", {})
        self._slot_put_async_enabled = per_env_cfg.get("slot_put_async", True)
        self._slot_put_wait_mode = per_env_cfg.get("slot_put_wait_mode", "strict")
        self._compact_slot_chunks = per_env_cfg.get("compact_slot_chunks", True)
        rollout_flex_cfg = self.cfg.rollout.get("per_env_async", {}).get("flex", {})
        self._epoch_prefetch_enabled = bool(
            rollout_flex_cfg.get("epoch_prefetch_enabled", False)
        )
        if self._slot_put_wait_mode not in ("strict", "none"):
            logger.warning(
                "Unknown env.per_env_async.slot_put_wait_mode=%s, fallback to strict",
                self._slot_put_wait_mode,
            )
            self._slot_put_wait_mode = "strict"
        self.local_manager_batch_sizes = list(self.flex_plan.manager_batch_sizes_by_env_rank[self._rank])
        self.local_global_env_ids = list(self.flex_plan.global_env_id_by_env_rank[self._rank])
        # In flex mode, total env managers = global_env_id 个数，不再使用 pipeline_stage_num
        # self._total_flex_managers = len(self.flex_plan.manager_dispatch_order_global_env_ids)
        self._total_flex_managers = sum(len(s) for s in self.flex_plan.manager_batch_sizes_by_env_rank)

        self.env_list = []
        if not self.only_eval:
            for local_manager_idx, manager_batch_size in enumerate(self.local_manager_batch_sizes):
                self.env_list.append(
                    EnvManager(
                        self.cfg.env.train,
                        rank=self._rank,
                        num_envs=int(manager_batch_size),
                        seed_offset=self.local_global_env_ids[local_manager_idx],
                        total_num_processes=self._total_flex_managers,
                        env_cls=train_env_cls,
                        worker_info=self.worker_info,
                        enable_offload=enable_offload,
                    )
                )

        if self.enable_eval:
            self.eval_env_list = []
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

        self.env_states = {}
        for stage_id, global_env_id in enumerate(self.local_global_env_ids):
            self.env_states[global_env_id] = EnvLoopState(
                env_id=global_env_id,
                stage_id=stage_id,
                completed_epochs=0,
                total_steps=0,
                is_running=False,
            )

        if torch.cuda.is_available():
            try:
                device_id = self._rank % torch.cuda.device_count()
                torch.inverse(torch.ones((1, 1), device=f"cuda:{device_id}"))
            except Exception:
                pass

        # [DEBUG_TIMING] read debug timing config (enable via cfg.debug.timing_log_enabled / timing_log_dir)
        debug_cfg = getattr(self.cfg, "debug", None)
        if debug_cfg is not None:
            self._debug_timing_enabled = getattr(debug_cfg, "timing_log_enabled", False)
            timing_dir = getattr(debug_cfg, "timing_log_dir", ".")
            # Auto-format timestamp placeholders in the path
            self._debug_timing_dir = self._format_timing_log_dir(timing_dir)
        else:
            self._debug_timing_dir = "."
        # [DEBUG_FLEX_PLAN] Log slot mapping for verification
        for global_env_id in self.local_global_env_ids:
            slots = self.flex_plan.slot_specs_by_global_env_id[global_env_id]
            slot_info = [(s.slot_id, s.local_slot_index, s.rollout_rank) for s in slots]
            # logger.info(
            #     f"[DEBUG_FLEX_PLAN] env_rank={self._rank} global_env_id={global_env_id}: "
            #     f"slots={slot_info}"
            # )
        
        # logger.info(
        #     "FlexiblePerEnvAsyncEnvWorker initialized: "
        #     f"rank={self._rank}, local_managers={len(self.local_manager_batch_sizes)}, "
        #     f"local_manager_batch_sizes={self.local_manager_batch_sizes}, "
        #     f"local_global_env_ids={self.local_global_env_ids}, "
        #     f"total_slots={self.flex_plan.total_slot_count}"
        # )

    def _compact_slot_tensor(self, value: torch.Tensor) -> torch.Tensor:
        """Return a tensor with storage limited to this slot's payload."""
        if not self._compact_slot_chunks:
            return value

        try:
            storage_bytes = value.untyped_storage().nbytes()
        except Exception:
            storage_bytes = value.numel() * value.element_size()
        payload_bytes = value.numel() * value.element_size()

        if (
            storage_bytes > payload_bytes
            or value.storage_offset() != 0
            or not value.is_contiguous()
        ):
            return value.clone(memory_format=torch.contiguous_format)
        return value

    def _split_tensor_by_sizes(
        self, value: Optional[torch.Tensor], split_sizes: List[int]
    ) -> List[Optional[torch.Tensor]]:
        if value is None:
            return [None for _ in split_sizes]
        return [
            self._compact_slot_tensor(chunk)
            for chunk in torch.split(value, split_sizes, dim=0)
        ]

    def _split_nested_by_sizes(self, value: Any, split_sizes: List[int]) -> List[Any]:
        if value is None:
            return [None for _ in split_sizes]
        if isinstance(value, torch.Tensor):
            return [
                self._compact_slot_tensor(chunk)
                for chunk in torch.split(value, split_sizes, dim=0)
            ]
        if isinstance(value, dict):
            split_dict = {k: self._split_nested_by_sizes(v, split_sizes) for k, v in value.items()}
            merged: List[Dict[str, Any]] = []
            for i in range(len(split_sizes)):
                merged.append({k: split_dict[k][i] for k in split_dict})
            return merged
        return [value for _ in split_sizes]

    def _split_env_output_by_slots(self, env_output: EnvOutput, slots: List[SlotSpec]) -> List[EnvOutput]:
        split_sizes = [s.batch_size for s in slots]
        obs_chunks = self._split_nested_by_sizes(env_output.obs, split_sizes)
        final_obs_chunks = self._split_nested_by_sizes(env_output.final_obs, split_sizes)
        rewards_chunks = self._split_tensor_by_sizes(env_output.rewards, split_sizes)
        dones_chunks = self._split_tensor_by_sizes(env_output.dones, split_sizes)
        term_chunks = self._split_tensor_by_sizes(env_output.terminations, split_sizes)
        trunc_chunks = self._split_tensor_by_sizes(env_output.truncations, split_sizes)
        intervene_actions_chunks = self._split_tensor_by_sizes(env_output.intervene_actions, split_sizes)
        intervene_flags_chunks = self._split_tensor_by_sizes(env_output.intervene_flags, split_sizes)

        outputs: List[EnvOutput] = []
        for i in range(len(slots)):
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

    def _send_slot_outputs(
        self,
        output_channel: Channel,
        global_env_id: int,
        env_output: EnvOutput,
        step_idx: int,
        epoch_idx: int,
        slots: Optional[List[SlotSpec]] = None,
        slot_keys: Optional[List[str]] = None,
    ):
        if slots is None:
            slots = self.flex_plan.slot_specs_by_global_env_id[global_env_id]
        if slot_keys is None:
            slot_keys = [
                self._get_rollout_input_channel_key(slot.rollout_rank, slot.slot_id, "train")
                for slot in slots
            ]
        split_outputs = self._split_env_output_by_slots(env_output, slots)
        batch_ts = time.time()
        if self._slot_put_async_enabled:
            sender = getattr(self, "_serialized_env_output_sender", None)
            put_target = sender if sender is not None else output_channel
            can_batch_put = (
                len(split_outputs) > 1
                and hasattr(put_target, "put_many")
            )

            if can_batch_put:
                batch_items = []
                batch_meta = []
                for slot, key, split_output in zip(slots, slot_keys, split_outputs):
                    env_output_dict, put_start_ts = self._build_env_output_dict(
                        env_key=key,
                        env_output=split_output,
                        step_idx=step_idx,
                        epoch_idx=epoch_idx,
                        global_env_id=global_env_id,
                        batch_ts=batch_ts,
                        route_slot_id=slot.slot_id,
                        route_local_slot_index=slot.local_slot_index,
                        async_op=True,
                    )
                    batch_items.append((key, env_output_dict, 0))
                    batch_meta.append((slot, key, put_start_ts, env_output_dict))

                put_submit_ts = time.time() if self._debug_timing_enabled else None
                if put_submit_ts is not None:
                    for _slot, _key, _put_start_ts, env_output_dict in batch_meta:
                        env_output_dict["_debug_put_submit_ts"] = put_submit_ts

                put_work = put_target.put_many(items=batch_items, async_op=True)
                put_return_ts = time.time() if self._debug_timing_enabled else None
                for slot, key, put_start_ts, _env_output_dict in batch_meta:
                    self._record_env_put_submit(
                        put_start_ts=put_start_ts,
                        put_submit_ts=put_submit_ts,
                        put_return_ts=put_return_ts,
                        env_key=key,
                        step_idx=step_idx,
                        epoch_idx=epoch_idx,
                        global_env_id=global_env_id,
                        route_slot_id=slot.slot_id,
                        async_op=True,
                    )

                if self._slot_put_wait_mode == "strict" and put_work is not None:
                    wait_start = time.time() if self._debug_timing_enabled else None
                    put_work.wait()
                    if self._debug_timing_enabled:
                        wait_done = time.time()
                        wait_time = wait_done - wait_start if wait_start is not None else None
                        for slot, key, _put_start_ts, _env_output_dict in batch_meta:
                            self._write_debug_timing_log(
                                {
                                    "event": "env_put_done",
                                    "ts_rel": wait_done - self._debug_t0,
                                    "ts_abs": wait_done,
                                    "env_rank": self._rank,
                                    "global_env_id": global_env_id,
                                    "env_key": key,
                                    "step_idx": step_idx,
                                    "epoch_idx": epoch_idx,
                                    "route_slot_id": slot.slot_id,
                                    "put_wait_time": wait_time,
                                    "batched_put": True,
                                }
                            )
                elif self._debug_timing_enabled:
                    t_now = time.time()
                    self._write_debug_timing_log(
                        {
                            "event": "env_put_wait_skipped",
                            "ts_rel": t_now - self._debug_t0,
                            "ts_abs": t_now,
                            "env_rank": self._rank,
                            "global_env_id": global_env_id,
                            "step_idx": step_idx,
                            "epoch_idx": epoch_idx,
                            "slot_count": len(batch_items),
                            "wait_mode": self._slot_put_wait_mode,
                            "batched_put": True,
                        }
                    )
                return

            pending_works = []
            for slot, key, split_output in zip(slots, slot_keys, split_outputs):
                work = self._send_env_output(
                    output_channel, key, split_output,
                    step_idx=step_idx, epoch_idx=epoch_idx,
                    global_env_id=global_env_id,
                    batch_ts=batch_ts,
                    route_slot_id=slot.slot_id,
                    route_local_slot_index=slot.local_slot_index,
                    async_op=True,
                )
                if work is not None:
                    pending_works.append((work, slot, key))
            if self._slot_put_wait_mode == "strict":
                for work, slot, key in pending_works:
                    wait_start = time.time() if self._debug_timing_enabled else None
                    work.wait()
                    if self._debug_timing_enabled:
                        wait_done = time.time()
                        self._write_debug_timing_log(
                            {
                                "event": "env_put_done",
                                "ts_rel": wait_done - self._debug_t0,
                                "ts_abs": wait_done,
                                "env_rank": self._rank,
                                "global_env_id": global_env_id,
                                "env_key": key,
                                "step_idx": step_idx,
                                "epoch_idx": epoch_idx,
                                "route_slot_id": slot.slot_id,
                                "put_wait_time": (
                                    wait_done - wait_start if wait_start is not None else None
                                ),
                            }
                        )
            elif self._debug_timing_enabled:
                t_now = time.time()
                self._write_debug_timing_log(
                    {
                        "event": "env_put_wait_skipped",
                        "ts_rel": t_now - self._debug_t0,
                        "ts_abs": t_now,
                        "env_rank": self._rank,
                        "global_env_id": global_env_id,
                        "step_idx": step_idx,
                        "epoch_idx": epoch_idx,
                        "slot_count": len(pending_works),
                        "wait_mode": self._slot_put_wait_mode,
                    }
                )
        else:
            for slot, key, split_output in zip(slots, slot_keys, split_outputs):
                self._send_env_output(
                    output_channel, key, split_output,
                    step_idx=step_idx, epoch_idx=epoch_idx,
                    global_env_id=global_env_id,
                    batch_ts=batch_ts,
                    route_slot_id=slot.slot_id,
                    route_local_slot_index=slot.local_slot_index,
                )

    def _recv_slot_actions(
        self,
        global_env_id: int,
        slot_keys: Optional[List[str]] = None,
    ) -> List[torch.Tensor]:
        """Receive actions for slots. Note: slots share the same env manager,
        so actions are received and processed together."""
        if slot_keys is None:
            slots = self.flex_plan.slot_specs_by_global_env_id[global_env_id]
            slot_keys = [
                self._get_slot_channel_key(global_env_id, slot.local_slot_index, "train")
                for slot in slots
            ]
        # Wait for all slot actions as a set so a ready later key is consumed
        # immediately instead of sitting behind an earlier slow key.
        t_before = time.time()
        results = self._serialized_action_receiver.get_many(slot_keys)
        t_after = time.time()
        if results is None:
            raise RuntimeError(f"action receiver stopped for global_env_id={global_env_id}")
        if self._debug_timing_enabled:
            for key in slot_keys:
                self._write_debug_timing_log(
                    {
                        "event": "env_recv",
                        "ts_rel": t_after - self._debug_t0,
                        "ts_abs": t_after,
                        "env_rank": self._rank,
                        "global_env_id": global_env_id,
                        "env_key": key,
                        "wait_time": t_after - t_before,
                    }
                )
        return results

    def interact(self, input_channel: Channel, output_channel: Channel):
        # [DEBUG_TIMING] start phase clock and init log buffer for this interact().
        self._debug_t0 = time.time()
        if self._debug_timing_enabled:
            self._debug_log = []
            self._debug_log_lock = threading.Lock()
            self._write_debug_timing_log(
                {
                    "event": "env_interact_start",
                    "ts_rel": 0.0,
                    "ts_abs": self._debug_t0,
                    "env_rank": self._rank,
                    "manager_count": len(self.env_list),
                }
            )

        env_start_t0 = time.time()
        for env in self.env_list:
            env.start_env()
        if self._debug_timing_enabled:
            t_now = time.time()
            self._write_debug_timing_log(
                {
                    "event": "env_start_done",
                    "ts_rel": t_now - self._debug_t0,
                    "ts_abs": t_now,
                    "env_rank": self._rank,
                    "manager_count": len(self.env_list),
                    "elapsed": t_now - env_start_t0,
                }
            )

        with self._metrics_lock:
            self._all_metrics.clear()

        self._serialized_action_receiver = SerializedChannelReceiver(input_channel)
        self._serialized_env_output_sender = SerializedChannelSender(output_channel)

        # Run manager loops in parallel to avoid deadlock in rollout router
        # Each manager (global_env_id) runs in its own thread so that all
        # slots of the same global_env_id can be sent simultaneously
        max_workers = len(self.env_list)
        with ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix=f"FlexEnvMgr-r{self._rank}") as executor:
            futures = [
                executor.submit(self._run_single_manager_loop, local_stage_idx, output_channel)
                for local_stage_idx in range(len(self.env_list))
            ]
            wait(futures)
            # Check for exceptions
            for future in futures:
                if future.exception() is not None:
                    raise future.exception()

        for env in self.env_list:
            env.stop_env()

        # [DEBUG_TIMING] flush timing log to file (one jsonl per env rank)
        if self._debug_timing_enabled and self._debug_log is not None and self._debug_log_lock is not None:
            os.makedirs(self._debug_timing_dir, exist_ok=True)
            log_path = os.path.join(self._debug_timing_dir, f"debug_timing_env_rank{self._rank}.jsonl")
            with self._debug_log_lock:
                entries = list(self._debug_log)
            with open(log_path, "a") as f:
                for entry in entries:
                    f.write(json.dumps(entry, default=str) + "\n")
            self._debug_log = None

        with self._metrics_lock:
            result_metrics = {}
            for key, values in self._all_metrics.items():
                if values:
                    result_metrics[key] = torch.cat(values, dim=0).contiguous().cpu()
        return result_metrics

    def _run_single_manager_loop(self, local_stage_idx: int, output_channel: Channel):
        global_env_id = self.local_global_env_ids[local_stage_idx]
        manager_batch_size = self.local_manager_batch_sizes[local_stage_idx]
        slots = self.flex_plan.slot_specs_by_global_env_id[global_env_id]
        rollout_input_keys = [
            self._get_rollout_input_channel_key(slot.rollout_rank, slot.slot_id, "train")
            for slot in slots
        ]
        slot_action_keys = [
            self._get_slot_channel_key(global_env_id, slot.local_slot_index, "train")
            for slot in slots
        ]
        # [DEBUG_CHANNEL_KEY] Log channel keys for this manager at start
        # logger.info(
        #     f"[DEBUG_CHANNEL_KEY] env_rank={self._rank} global_env_id={global_env_id} START: "
        #     f"slots={[(s.slot_id, s.local_slot_index, s.rollout_rank) for s in slots]}, "
        #     f"slot_keys={slot_keys}"
        # )
        # logger.info(
        #     "[DEBUG FLEX ENV] env_rank=%s global_env_id=%s interact() starting, will send epochs 0..%s (step -1 init then steps 0..%s per epoch)",
        #     self._rank,
        #     global_env_id,
        #     self.cfg.algorithm.rollout_epoch - 1,
        #     (self.cfg.env.train.max_steps_per_rollout_epoch // self.cfg.actor.model.num_action_chunks) - 1,
        # )
        n_chunk_steps = (
            self.cfg.env.train.max_steps_per_rollout_epoch
            // self.cfg.actor.model.num_action_chunks
        )

        env_metrics = defaultdict(list)
        state = self.env_states[global_env_id]
        state.is_running = True

        last_obs = None
        last_dones = None
        last_terminations = None
        last_truncations = None
        last_intervened_info = (None, None)

        if self.cfg.env.train.auto_reset:
            self.env_list[local_stage_idx].is_start = True
            reset_t0 = time.time()
            extracted_obs, _ = self.env_list[local_stage_idx].reset()
            if self._debug_timing_enabled:
                t_now = time.time()
                self._write_debug_timing_log(
                    {
                        "event": "env_reset_done",
                        "ts_rel": t_now - self._debug_t0,
                        "ts_abs": t_now,
                        "env_rank": self._rank,
                        "global_env_id": global_env_id,
                        "local_stage_idx": local_stage_idx,
                        "epoch_idx": -1,
                        "elapsed": t_now - reset_t0,
                        "auto_reset": True,
                    }
                )
            dones = (
                torch.zeros((manager_batch_size,), dtype=bool)
                .unsqueeze(1)
                .repeat(1, self.cfg.actor.model.num_action_chunks)
            )
            last_obs = extracted_obs
            last_dones = dones
            last_terminations = dones.clone()
            last_truncations = dones.clone()

        previous_epoch_final_put_done_ts: Optional[float] = None

        try:
            for epoch in range(self.cfg.algorithm.rollout_epoch):
                if not self.cfg.env.train.auto_reset:
                    self.env_list[local_stage_idx].is_start = True
                    reset_t0 = time.time()
                    extracted_obs, infos = self.env_list[local_stage_idx].reset()
                    if self._debug_timing_enabled:
                        t_now = time.time()
                        self._write_debug_timing_log(
                            {
                                "event": "env_reset_done",
                                "ts_rel": t_now - self._debug_t0,
                                "ts_abs": t_now,
                                "env_rank": self._rank,
                                "global_env_id": global_env_id,
                                "local_stage_idx": local_stage_idx,
                                "epoch_idx": epoch,
                                "elapsed": t_now - reset_t0,
                                "auto_reset": False,
                            }
                        )
                    dones = (
                        torch.zeros((manager_batch_size,), dtype=bool)
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

                if self._debug_timing_enabled and previous_epoch_final_put_done_ts is not None:
                    t_now = time.time()
                    self._write_debug_timing_log(
                        {
                            "event": "env_epoch_next_init_ready",
                            "ts_rel": t_now - self._debug_t0,
                            "ts_abs": t_now,
                            "env_rank": self._rank,
                            "global_env_id": global_env_id,
                            "epoch_idx": epoch,
                            "prev_epoch_idx": epoch - 1,
                            "delay_from_prev_final_put_done": t_now - previous_epoch_final_put_done_ts,
                            "epoch_prefetch_enabled": self._epoch_prefetch_enabled,
                        }
                    )

                self._send_slot_outputs(
                    output_channel,
                    global_env_id,
                    env_output,
                    step_idx=-1,
                    epoch_idx=epoch,
                    slots=slots,
                    slot_keys=rollout_input_keys,
                )

                for step in range(n_chunk_steps):
                    slot_actions = self._recv_slot_actions(global_env_id, slot_keys=slot_action_keys)
                    # Channel may deliver numpy arrays; ensure tensors for torch.cat
                    slot_tensors = [
                        a if isinstance(a, torch.Tensor) else torch.as_tensor(a)
                        for a in slot_actions
                    ]
                    raw_chunk_actions = torch.cat(slot_tensors, dim=0)
                    env_output, env_info = self._env_interact_step(raw_chunk_actions, local_stage_idx)
                    self._send_slot_outputs(
                        output_channel,
                        global_env_id,
                        env_output,
                        step_idx=step,
                        epoch_idx=epoch,
                        slots=slots,
                        slot_keys=rollout_input_keys,
                    )
                    if step == n_chunk_steps - 1:
                        previous_epoch_final_put_done_ts = time.time()

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

                last_obs = env_output.obs
                last_dones = env_output.dones
                last_terminations = env_output.terminations
                last_truncations = env_output.truncations
                last_intervened_info = (
                    env_output.intervene_actions,
                    env_output.intervene_flags,
                )
                self._finish_env_rollout(local_stage_idx)
                state.completed_epochs += 1

                if epoch < self.cfg.algorithm.rollout_epoch - 1:
                    ack_wait_start = time.time()
                    if self._epoch_prefetch_enabled:
                        if self._debug_timing_enabled:
                            t_now = time.time()
                            self._write_debug_timing_log(
                                {
                                    "event": "env_epoch_ack_wait_skipped",
                                    "ts_rel": t_now - self._debug_t0,
                                    "ts_abs": t_now,
                                    "env_rank": self._rank,
                                    "global_env_id": global_env_id,
                                    "epoch_idx": epoch,
                                    "next_epoch_idx": epoch + 1,
                                    "slot_count": len(slot_action_keys),
                                    "epoch_prefetch_enabled": True,
                                }
                            )
                    else:
                        # logger.info(
                        #     "[DEBUG_EPOCH] global_env_id=%s epoch=%s: waiting for __epoch_done__ acks on %s slot_keys",
                        #     global_env_id, epoch, len(slot_action_keys)
                        # )
                        for i, ack_key in enumerate(slot_action_keys):
                            # logger.info(
                            #     "[DEBUG_EPOCH] global_env_id=%s epoch=%s: waiting for ack %s/%s on key=%s",
                            #     global_env_id, epoch, i+1, len(slot_action_keys), ack_key
                            # )
                            ack = self._serialized_action_receiver.get(ack_key)
                            if not ack.get("__epoch_done__"):
                                raise RuntimeError(
                                    f"global_env_id={global_env_id}, epoch={epoch}, "
                                    f"ack_key={ack_key}: expected ack, got {ack}"
                                )
                            # logger.info(
                            #     "[DEBUG_EPOCH] global_env_id=%s epoch=%s: received ack %s/%s on key=%s",
                            #     global_env_id, epoch, i+1, len(slot_keys), ack_key
                            # )
                        if self._debug_timing_enabled:
                            t_now = time.time()
                            self._write_debug_timing_log(
                                {
                                    "event": "env_epoch_ack_wait",
                                    "ts_rel": t_now - self._debug_t0,
                                    "ts_abs": t_now,
                                    "env_rank": self._rank,
                                    "global_env_id": global_env_id,
                                    "epoch_idx": epoch,
                                    "next_epoch_idx": epoch + 1,
                                    "slot_count": len(slot_action_keys),
                                    "wait_time": t_now - ack_wait_start,
                                    "epoch_prefetch_enabled": False,
                                }
                            )
                    # logger.info(
                    #     "[DEBUG_EPOCH] global_env_id=%s epoch=%s: all %s acks handled, proceeding to epoch %s",
                    #     global_env_id, epoch, len(slot_keys), epoch + 1
                    # )
        finally:
            state.is_running = False

        with self._metrics_lock:
            for key, values in env_metrics.items():
                if values:
                    self._all_metrics[key].extend(values)

    def evaluate(self, input_channel: Channel, output_channel: Channel):
        raise NotImplementedError(
            "FlexiblePerEnvAsyncEnvWorker currently supports training interact only. "
            "Please set runner.val_check_interval=0 for this experimental mode."
        )
