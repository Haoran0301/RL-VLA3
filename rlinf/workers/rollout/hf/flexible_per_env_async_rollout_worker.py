import asyncio
import json
import logging
import os
import queue
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional

from omegaconf import DictConfig
import torch

from rlinf.data.io_struct import EmbodiedRolloutResult
from rlinf.scheduler import Channel
from rlinf.utils.nested_dict_process import put_tensor_device
from rlinf.workers.per_env_flex_plan import FlexPlan, SlotSpec, build_flex_plan
from rlinf.workers.rollout.hf.per_env_async_rollout_worker import (
    EnvRolloutBuffer,
    PerEnvAsyncRolloutWorker,
    PerEnvDynamicBatchingEngine,
    SerializedChannelSender,
    PerEnvInferenceRequest,
    PerEnvInferenceResult,
)

logger = logging.getLogger(__name__)


class FlexiblePerEnvDynamicBatchingEngine(PerEnvDynamicBatchingEngine):
    """
    Dynamic batching by real sample count, not request count.

    Behavior change:
    - `max_batch_size` means target sample size.
    - If current sample count is 80 and next request has 40 while target is 100,
      the full 40 is accepted and the batch runs with 120 (no truncation).
    """

    def configure_debug_timing(
        self,
        *,
        enabled: bool,
        writer,
        debug_t0: float,
        rollout_rank: int,
    ) -> None:
        self._debug_timing_enabled = enabled
        self._debug_timing_writer = writer
        self._debug_t0 = debug_t0
        self._debug_rollout_rank = rollout_rank
        self._debug_first_batch_logged = False

    def _write_debug_timing(self, entry: Dict[str, Any]) -> None:
        if not getattr(self, "_debug_timing_enabled", False):
            return
        writer = getattr(self, "_debug_timing_writer", None)
        if writer is None:
            return
        writer(entry)

    def _request_sample_size(self, request: PerEnvInferenceRequest) -> int:
        return max(1, self._get_batch_size(request.env_output.get("obs", {})))

    def _inference_loop(self):
        while not self.should_stop:
            batch: List[PerEnvInferenceRequest] = []
            batch_start_time = time.time()
            sample_count = 0

            while True:
                elapsed = time.time() - batch_start_time
                remaining_time = self.batch_timeout_ms / 1000 - elapsed
                if remaining_time <= 0 and batch:
                    break
                try:
                    timeout = max(0.001, remaining_time) if batch else None
                    request = self.request_queue.get(timeout=timeout)
                    if request is self._stop_sentinel:
                        self.should_stop = True
                        break
                    batch.append(request)
                    sample_count += self._request_sample_size(request)
                    if sample_count >= self.max_batch_size:
                        break
                except queue.Empty:
                    if batch:
                        break
                    continue

            if not batch:
                continue
            if self.should_stop:
                exc = RuntimeError("PerEnvDynamicBatchingEngine stopped before inference")
                for request in batch:
                    future = self.result_futures.pop(request.request_id, None)
                    if future is not None and not future.done():
                        future.set_exception(exc)
                break

            try:
                inference_start = time.time()
                if not getattr(self, "_debug_first_batch_logged", False):
                    self._debug_first_batch_logged = True
                    self._write_debug_timing(
                        {
                            "event": "rollout_engine_first_batch",
                            "ts_rel": inference_start - getattr(self, "_debug_t0", inference_start),
                            "ts_abs": inference_start,
                            "rollout_rank": getattr(self, "_debug_rollout_rank", None),
                            "batch_request_count": len(batch),
                            "batch_sample_count": sample_count,
                        }
                    )
                results = self._run_batch_inference(batch)
                self.total_generate_time += time.time() - inference_start
                for request, result in zip(batch, results):
                    future = self.result_futures.pop(request.request_id, None)
                    if future is not None:
                        future.set_result(result)
                self.total_batches += 1
                self.total_batch_size += sample_count
            except Exception as e:
                logger.error(f"Flexible batch inference error: {e}")
                for request in batch:
                    future = self.result_futures.pop(request.request_id, None)
                    if future is not None:
                        future.set_exception(e)


class FlexiblePerEnvAsyncRolloutWorker(PerEnvAsyncRolloutWorker):
    """Flexible slot-routed rollout worker."""

    def __init__(self, cfg: DictConfig):
        super().__init__(cfg)
        self.flex_plan: Optional[FlexPlan] = None
        self.my_slot_specs: List[SlotSpec] = []
        self.slot_buffer_map = {}
        self._input_channel: Optional[Channel] = None
        self._output_sender: Optional[SerializedChannelSender] = None
        self._actor_sender: Optional[SerializedChannelSender] = None
        self._slot_spec_by_slot_id: Dict[int, SlotSpec] = {}
        self._slot_queue_map: Dict[int, queue.Queue] = {}
        self._slot_inflight: Dict[int, bool] = {}
        self._slot_inflight_frontier: Dict[int, Optional[tuple]] = {}
        self._slot_done_frontier: Dict[int, Optional[tuple]] = {}
        self._env_slot_ids_local: Dict[int, List[int]] = {}
        self._slot_id_by_env_local: Dict[tuple, int] = {}
        self._env_output_router_thread: Optional[threading.Thread] = None
        self._router_stop_event: Optional[threading.Event] = None
        self._router_lock: Optional[threading.Lock] = None
        self._router_cv: Optional[threading.Condition] = None
        # When True: use global round-robin order (minimize env wait).
        # When False: use router_mode_when_no_global_order.
        self._use_global_round_robin: bool = True
        self._router_mode_when_no_global_order: str = "greedy"  # "greedy", "env_group_greedy", or "direct"
        self._pull_timeout_s: float = 0.001
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
        super().init_worker()
        env_world_size = self.placement.get_world_size("env")
        
        # When flex config is provided, derive defaults from config instead of pipeline_stage_num
        flex_cfg = self.cfg.env.get("per_env_async", {}).get("flex", None)
        if flex_cfg is not None and flex_cfg.get("enabled", False):
            # Use config's manager_batch_sizes_by_env_rank to determine defaults
            mgr_batch_sizes = flex_cfg.get("manager_batch_sizes_by_env_rank", [])
            if mgr_batch_sizes:
                # default_stage_num = max number of managers per env rank from config
                default_stage_num = max(len(sizes) for sizes in mgr_batch_sizes)
                # default_batch_size = average batch size from config
                total_envs = sum(sum(sizes) for sizes in mgr_batch_sizes)
                default_batch_size = total_envs // max(1, env_world_size) // max(1, default_stage_num)
            else:
                default_stage_num = self.num_pipeline_stages
                default_batch_size = (
                    self.cfg.env.train.total_num_envs // max(1, env_world_size) // max(1, default_stage_num)
                )
        else:
            default_stage_num = self.num_pipeline_stages
            default_batch_size = (
                self.cfg.env.train.total_num_envs // max(1, env_world_size) // max(1, default_stage_num)
            )
        
        self.flex_plan = build_flex_plan(
            cfg=self.cfg,
            env_world_size=env_world_size,
            rollout_world_size=self.rollout_world_size,
            default_stage_num=default_stage_num,
            default_batch_size=default_batch_size,
        )
        self.my_slot_specs = list(self.flex_plan.slot_specs_by_rollout_rank.get(self._rank, []))
        self._slot_spec_by_slot_id = {s.slot_id: s for s in self.my_slot_specs}
        self._slot_id_by_env_local = {
            (s.global_env_id, s.local_slot_index): s.slot_id for s in self.my_slot_specs
        }

        # Memory probe: print only a tiny number of samples to pinpoint offload issues.
        # We intentionally keep it ultra-sparse to avoid slowing down the rollout.
        self._mem_probe_enabled = bool(torch.cuda.is_available() and self._rank == 0)
        self._mem_probe_generate_cnt = 0
        self._mem_probe_slot_id = (
            min((s.slot_id for s in self.my_slot_specs), default=None)
            if self._mem_probe_enabled
            else None
        )

        # [DEBUG_FLEX_PLAN] Log slot mapping for verification
        # for spec in self.my_slot_specs:
        #     logger.info(
        #         f"[DEBUG_FLEX_PLAN] rollout_rank={self._rank} slot_id={spec.slot_id}: "
        #         f"global_env_id={spec.global_env_id}, local_slot_index={spec.local_slot_index}, "
        #         f"env_key={self._get_slot_channel_key(spec.global_env_id, spec.local_slot_index, 'train')}"
        #     )
        
        # logger.info(f"[DEBUG] FlexiblePerEnvAsyncRolloutWorker initialized: rank={self._rank}, my_slots={[s.slot_id for s in self.my_slot_specs]}, total_slots={self.flex_plan.total_slot_count}, target_max_batch={self.max_batch_size}, self.my_slot_specs={self.my_slot_specs}")

        self.batching_engine = FlexiblePerEnvDynamicBatchingEngine(
            hf_model=self.hf_model,
            cfg=self.cfg,
            max_batch_size=self.max_batch_size,
            batch_timeout_ms=self.batch_timeout_ms,
            device=self.device,
        )

        if self.handler_pool is not None:
            self.handler_pool.shutdown(wait=False)
        self.handler_pool = ThreadPoolExecutor(
            max_workers=max(1, len(self.my_slot_specs)),
            thread_name_prefix="FlexSlotHandler",
        )

        self.env_buffers = {}
        self.slot_buffer_map = {}
        self._slot_queue_map = {}
        self._slot_inflight = {}
        self._slot_inflight_frontier = {}
        self._slot_done_frontier = {}
        self._env_slot_ids_local = {}
        for slot_spec in self.my_slot_specs:
            self.env_buffers[slot_spec.slot_id] = EnvRolloutBuffer(
                env_id=slot_spec.slot_id,
                stage_id=slot_spec.slot_id,
                rollout_epoch=self.cfg.algorithm.rollout_epoch,
            )
            self.slot_buffer_map[slot_spec.slot_id] = self.env_buffers[slot_spec.slot_id]
            self._slot_queue_map[slot_spec.slot_id] = queue.Queue()
            self._slot_inflight[slot_spec.slot_id] = False
            self._slot_inflight_frontier[slot_spec.slot_id] = None
            self._slot_done_frontier[slot_spec.slot_id] = None
            self._env_slot_ids_local.setdefault(slot_spec.global_env_id, []).append(slot_spec.slot_id)

        self._router_lock = threading.Lock()
        self._router_cv = threading.Condition(self._router_lock)
        self._router_stop_event = threading.Event()
        self._env_output_router_thread = None

        # Optional: disable global round-robin. Then router_mode_when_no_global_order:
        # - "greedy" (default): dispatch any slot when its next message is ready (no env_id order).
        # - "env_group_greedy": dispatch env group only when all slots of that env are ready;
        #   no fixed global env order (env that becomes ready first is dispatched first).
        # - "direct": no router, each slot recv directly from channel like parent.
        rollout_flex = self.cfg.rollout.get("per_env_async", {}).get("flex", {})
        self._use_global_round_robin = rollout_flex.get("use_global_round_robin", True)
        self._router_mode_when_no_global_order = rollout_flex.get("router_mode_when_no_global_order", "greedy")
        pull_timeout_s = rollout_flex.get("pull_timeout_s", 0.001)
        self._pull_timeout_s = 0.001 if pull_timeout_s is None else max(0.0, float(pull_timeout_s))
        self._epoch_prefetch_enabled = bool(
            rollout_flex.get("epoch_prefetch_enabled", False)
        )
        if not self._use_global_round_robin and self._router_mode_when_no_global_order == "direct":
            logger.warning(
                "router_mode_when_no_global_order=direct is incompatible with routed rollout input keys. "
                "Falling back to greedy router."
            )
            self._router_mode_when_no_global_order = "greedy"

        # [DEBUG_TIMING] read debug timing config (enable via cfg.debug.timing_log_enabled / timing_log_dir)
        debug_cfg = getattr(self.cfg, "debug", None)
        if debug_cfg is not None:
            self._debug_timing_enabled = getattr(debug_cfg, "timing_log_enabled", False)
            timing_dir = getattr(debug_cfg, "timing_log_dir", ".")
            # Auto-format timestamp placeholders in the path
            self._debug_timing_dir = self._format_timing_log_dir(timing_dir)
        else:
            self._debug_timing_dir = "."

        # logger.info(
        #     "FlexiblePerEnvAsyncRolloutWorker initialized: "
        #     f"rank={self._rank}, my_slots={[s.slot_id for s in self.my_slot_specs]}, "
        #     f"total_slots={self.flex_plan.total_slot_count}, target_max_batch={self.max_batch_size}"
        # )

    def _write_debug_timing_log(self, entry: Dict[str, Any]) -> None:
        """[DEBUG_TIMING] Append one timing entry to in-memory log (thread-safe)."""
        if not self._debug_timing_enabled or self._debug_log is None or self._debug_log_lock is None:
            return
        with self._debug_log_lock:
            self._debug_log.append(entry)

    def get_actor_split_num(self):
        from rlinf.utils.metric_utils import compute_split_num

        send_num = self.flex_plan.total_slot_count
        recv_num = self.placement.get_world_size("actor")
        return compute_split_num(recv_num, send_num)

    def _send_rollout_batch(
        self, channel: Channel, slot_id: int, use_key: bool = True
    ):
        """
        Send rollout batch to actor workers for flexible mode.

        This method overrides the parent class to use slot_id-based routing
        instead of env_id-based routing, which is required for flexible
        env-rollout mapping.

        Args:
            channel: Channel to send data to actors
            slot_id: Slot ID (used as the basic unit in flexible mode)
            use_key: Whether to use deterministic key routing (for async mode)
                     If False, send without key (for sync mode, faster)
        """
        buffer = self.slot_buffer_map[slot_id].buffer

        split_num = self.get_actor_split_num()
        splitted_rollout_result = buffer.to_splitted_dict(split_num)
        for item in splitted_rollout_result:
            item["__behavior_policy_version__"] = self.behavior_policy_version

        sender = self._actor_sender
        if use_key:
            # Async mode: use deterministic key routing based on slot_id
            actor_world_size = self.placement.get_world_size("actor")
            for i in range(split_num):
                # Use slot_id for deterministic routing in flexible mode
                # This matches actor's recv_key = f"actor_{rank}" logic
                global_msg_id = slot_id * split_num + i
                target_actor = global_msg_id % actor_world_size
                if sender is not None:
                    sender.put(
                        item=splitted_rollout_result[i],
                        key=f"actor_{target_actor}",
                        async_op=True,
                    )
                else:
                    channel.put(
                        item=splitted_rollout_result[i],
                        key=f"actor_{target_actor}",
                        async_op=True,
                    )
        else:
            # Sync mode: send without key (faster, no message competition in sync mode)
            # Actor will receive all messages and concatenate them
            for i in range(split_num):
                if sender is not None:
                    sender.put(item=splitted_rollout_result[i], key="default_queue", async_op=True)
                else:
                    channel.put(item=splitted_rollout_result[i], async_op=True)

    async def generate(
        self, input_channel: Channel, output_channel: Channel, actor_channel: Channel
    ):
        if self.enable_offload:
            self.reload_model()

        do_peak_probe = False
        if self._mem_probe_enabled:
            # Only probe the first generate() call to avoid excessive logs.
            do_peak_probe = self._mem_probe_generate_cnt == 0
            if do_peak_probe:
                torch.cuda.reset_peak_memory_stats()
            self._mem_probe_generate_cnt += 1

        self._generate_start_time = time.time()
        self._env_handler_wait_times = {}
        self.batching_engine.total_generate_time = 0.0
        # [DEBUG_TIMING] start phase clock and init log buffer for this generate().
        self._debug_t0 = self._generate_start_time
        if self._debug_timing_enabled:
            self._debug_log = []
            self._debug_log_lock = threading.Lock()
            self._write_debug_timing_log(
                {
                    "event": "rollout_generate_start",
                    "ts_rel": 0.0,
                    "ts_abs": self._generate_start_time,
                    "rollout_rank": self._rank,
                    "slot_count": len(self.my_slot_specs),
                    "epoch_prefetch_enabled": self._epoch_prefetch_enabled,
                }
            )

        for slot_id, buffer in self.env_buffers.items():
            buffer.buffer = EmbodiedRolloutResult(rollout_epoch=self.cfg.algorithm.rollout_epoch)
            buffer.last_extracted_obs = None
            buffer.last_forward_inputs = None
            buffer.completed_epochs = 0

        # Reset router slot state so each generate() expects (epoch=0, step=-1) from env.
        # Env sends epoch 0..rollout_epoch-1 per interact(); without this, second step
        # would expect (4,-1) and drop env's (0,-1) as stale, causing deadlock.
        for slot_id in self._slot_done_frontier:
            self._slot_done_frontier[slot_id] = None
            self._slot_inflight[slot_id] = False
            self._slot_inflight_frontier[slot_id] = None
        # logger.info(
        #     "[DEBUG FLEX ROUTER] rollout_rank=%s generate() started, reset slot state for slot_ids=%s",
        #     self._rank,
        #     list(self._slot_done_frontier.keys()),
        # )

        if hasattr(self.batching_engine, "configure_debug_timing"):
            self.batching_engine.configure_debug_timing(
                enabled=self._debug_timing_enabled,
                writer=self._write_debug_timing_log,
                debug_t0=self._debug_t0,
                rollout_rank=self._rank,
            )
        if self._debug_timing_enabled:
            t_now = time.time()
            self._write_debug_timing_log(
                {
                    "event": "rollout_engine_start",
                    "ts_rel": t_now - self._debug_t0,
                    "ts_abs": t_now,
                    "rollout_rank": self._rank,
                }
            )
        self.batching_engine.start()
        self._input_channel = input_channel
        self._output_sender = SerializedChannelSender(output_channel)
        self._actor_sender = SerializedChannelSender(actor_channel)

        assert self._router_stop_event is not None
        self._router_stop_event.clear()
        if self._use_global_round_robin:
            self._env_output_router_thread = threading.Thread(
                target=self._run_env_output_router_loop,
                name=f"FlexEnvOutputRouter-r{self._rank}",
                daemon=True,
            )
            self._env_output_router_thread.start()
        elif self._router_mode_when_no_global_order == "greedy":
            self._env_output_router_thread = threading.Thread(
                target=self._run_greedy_env_output_router_loop,
                name=f"FlexGreedyRouter-r{self._rank}",
                daemon=True,
            )
            self._env_output_router_thread.start()
        elif self._router_mode_when_no_global_order == "env_group_greedy":
            self._env_output_router_thread = threading.Thread(
                target=self._run_env_group_greedy_router_loop,
                name=f"FlexEnvGroupGreedyRouter-r{self._rank}",
                daemon=True,
            )
            self._env_output_router_thread.start()
        else:
            raise RuntimeError(
                f"unsupported router mode: {self._router_mode_when_no_global_order}"
            )

        try:
            pipeline_mode = self.cfg.algorithm.get("pipeline_mode", "sync")
            loop = asyncio.get_event_loop()
            tasks = []
            for slot_spec in self.my_slot_specs:
                task = loop.run_in_executor(
                    self.handler_pool,
                    self._run_slot_handler,
                    slot_spec,
                    output_channel,
                    actor_channel,
                    pipeline_mode,
                )
                tasks.append(task)

            await asyncio.gather(*tasks)

            if pipeline_mode == "sync":
                for slot_spec in self.my_slot_specs:
                    self._send_rollout_batch(actor_channel, slot_spec.slot_id, use_key=False)
            elif pipeline_mode == "async":
                sender = self._actor_sender
                if sender is not None:
                    sender.put(
                        item={"__done__": True},
                        key=f"rollout_{self._rank}",
                        async_op=True,
                    )
                else:
                    actor_channel.put(
                        item={"__done__": True}, key=f"rollout_{self._rank}", async_op=True
                    )
        finally:
            self.batching_engine.stop()
            if self._router_stop_event is not None:
                self._router_stop_event.set()
                if self._router_cv is not None:
                    with self._router_cv:
                        self._router_cv.notify_all()
                if (
                    self._use_global_round_robin
                    or self._router_mode_when_no_global_order in ("greedy", "env_group_greedy")
                ) and self._env_output_router_thread is not None:
                    self._env_output_router_thread.join(timeout=5.0)
                    self._env_output_router_thread = None
            # [DEBUG_TIMING] flush timing log to file (one jsonl per rollout rank)
            if self._debug_timing_enabled and self._debug_log is not None and self._debug_log_lock is not None:
                os.makedirs(self._debug_timing_dir, exist_ok=True)
                log_path = os.path.join(self._debug_timing_dir, f"debug_timing_rollout_rank{self._rank}.jsonl")
                with self._debug_log_lock:
                    entries = list(self._debug_log)
                with open(log_path, "a") as f:
                    for entry in entries:
                        f.write(json.dumps(entry, default=str) + "\n")
                self._debug_log = None

        if self.enable_offload:
            self.offload_model()

        if do_peak_probe:
            # Ensure memory stats are up-to-date.
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            alloc = torch.cuda.memory_allocated() if torch.cuda.is_available() else -1
            reserved = torch.cuda.memory_reserved() if torch.cuda.is_available() else -1
            peak = torch.cuda.max_memory_allocated() if torch.cuda.is_available() else -1
            print(
                f"[MEM_PROBE] generate_end rank={self._rank} "
                f"alloc={alloc/2**30:.3f}GiB reserved={reserved/2**30:.3f}GiB "
                f"peak_alloc={peak/2**30:.3f}GiB"
            )

        max_env_wait = max(self._env_handler_wait_times.values()) if self._env_handler_wait_times else 0.0
        return {
            "env_wait": max_env_wait,
            "generate": self.batching_engine.total_generate_time,
        }

    def _count_cuda_tensors(self, obj: Any) -> int:
        """Count torch.Tensors inside a nested structure that are still on CUDA."""
        cnt = 0
        if obj is None:
            return 0
        if isinstance(obj, torch.Tensor):
            return 1 if obj.is_cuda else 0
        if isinstance(obj, dict):
            for v in obj.values():
                cnt += self._count_cuda_tensors(v)
            return cnt
        if isinstance(obj, (list, tuple)):
            for v in obj:
                cnt += self._count_cuda_tensors(v)
            return cnt
        return 0

    def _mem_probe_should_log(self, *, epoch_idx: int, step_idx: int, slot_id: int) -> bool:
        return (
            self._mem_probe_enabled
            and self._mem_probe_generate_cnt <= 1
            and epoch_idx == 0
            and step_idx == 1
            and self._mem_probe_slot_id is not None
            and slot_id == self._mem_probe_slot_id
        )

    def _run_slot_handler(
        self,
        slot_spec: SlotSpec,
        output_channel: Channel,
        actor_channel: Channel,
        pipeline_mode: str,
    ):
        slot_id = slot_spec.slot_id
        env_key = self._get_slot_channel_key(slot_spec.global_env_id, slot_spec.local_slot_index, "train")
        buffer = self.slot_buffer_map[slot_id]
        n_chunk_steps = (
            self.cfg.env.train.max_steps_per_rollout_epoch
            // self.cfg.actor.model.num_action_chunks
        )
        request_counter = 0

        # [DEBUG_CHANNEL_KEY] Log slot handler start
        # logger.info(
        #     f"[DEBUG_CHANNEL_KEY] rollout_rank={self._rank} slot_id={slot_id} START: "
        #     f"global_env_id={slot_spec.global_env_id}, local_slot_index={slot_spec.local_slot_index}, "
        #     f"env_key={env_key}"
        # )

        for epoch_idx in range(self.cfg.algorithm.rollout_epoch):
            if pipeline_mode == "async":
                buffer.buffer = EmbodiedRolloutResult(rollout_epoch=1)
                buffer.last_extracted_obs = None
                buffer.last_forward_inputs = None

            last_extracted_obs = None
            last_forward_inputs = None

            for step_idx in range(n_chunk_steps):
                env_wait_start = time.time()
                env_output = self._recv_env_output_for_slot(slot_id)
                env_wait_elapsed = time.time() - env_wait_start
                # [DEBUG dones/rewards] 第一步应为 init (step=-1, rewards=None)；step>0 应为 step_idx-1 且 has_rewards=True
                msg_step = env_output.get("_debug_step_idx", "?")
                msg_epoch = env_output.get("_debug_epoch_idx", "?")
                from_global_env_id = env_output.get("_debug_global_env_id", "?")
                msg_env_key = env_output.get("_debug_env_key", "?")
                has_rewards = env_output.get("rewards") is not None
                
                # [DEBUG_CHANNEL_KEY] Verify message is from correct channel
                # if msg_env_key != env_key:
                #     logger.error(
                #         f"[DEBUG_CHANNEL_KEY MISMATCH] slot_id={slot_id} expected env_key={env_key} "
                #         f"but got msg_env_key={msg_env_key} from_global_env_id={from_global_env_id}"
                #     )
                
                # if step_idx == 0:
                #     logger.info(
                #         "[DEBUG dones/rewards] slot_id=%s epoch_idx=%s first_recv: "
                #         "msg_step=%s msg_epoch=%s from_global_env_id=%s msg_env_key=%s expected_env_key=%s has_rewards=%s (expect step=-1, has_rewards=False)",
                #         slot_id, epoch_idx, msg_step, msg_epoch, from_global_env_id, msg_env_key, env_key, has_rewards,
                #     )
                # else:
                #     logger.info(
                #         "[DEBUG dones/rewards] slot_id=%s epoch_idx=%s step_recv loop_step=%s: "
                #         "msg_step=%s msg_epoch=%s from_global_env_id=%s msg_env_key=%s expected_env_key=%s has_rewards=%s (expect msg_step=loop_step-1, has_rewards=True)",
                #         slot_id, epoch_idx, step_idx, msg_step, msg_epoch, from_global_env_id, msg_env_key, env_key, has_rewards,
                #     )
    
                # [DEBUG_TIMING] log when rollout received env data and from which env worker
                if self._debug_timing_enabled:
                    t_recv = time.time()
                    self._write_debug_timing_log(
                        {
                            "event": "rollout_recv",
                            "ts_rel": t_recv - self._debug_t0,
                            "ts_abs": t_recv,
                            "rollout_rank": self._rank,
                            "slot_id": slot_id,
                            "env_key": env_key,
                            "step_idx": step_idx,
                            "epoch_idx": epoch_idx,
                            "wait_time": env_wait_elapsed,
                            "from_env_rank": env_output.get("_debug_env_rank"),
                            "from_global_env_id": env_output.get("_debug_global_env_id"),
                            "env_ts_send": env_output.get("_debug_ts_send"),
                            "env_put_start_ts": env_output.get("_debug_put_start_ts"),
                            "env_put_submit_ts": env_output.get("_debug_put_submit_ts"),
                            "env_ts_since_start": env_output.get("_debug_ts_since_start"),
                            "pull_ts_abs": env_output.get("_debug_pull_ts"),
                            "pending_ts_abs": env_output.get("_debug_pending_ts"),
                            "dispatch_ts_abs": env_output.get("_debug_dispatch_ts"),
                            "slot_dequeue_ts_abs": env_output.get("_debug_slot_dequeue_ts"),
                            "transport_delay": (
                                t_recv - env_output.get("_debug_ts_send")
                                if env_output.get("_debug_ts_send") is not None
                                else None
                            ),
                            "send_to_pull_delay": (
                                env_output.get("_debug_pull_ts") - env_output.get("_debug_ts_send")
                                if env_output.get("_debug_pull_ts") is not None
                                and env_output.get("_debug_ts_send") is not None
                                else None
                            ),
                            "put_submit_to_pull_delay": (
                                env_output.get("_debug_pull_ts") - env_output.get("_debug_put_submit_ts")
                                if env_output.get("_debug_pull_ts") is not None
                                and env_output.get("_debug_put_submit_ts") is not None
                                else None
                            ),
                            "pull_to_dispatch_delay": (
                                env_output.get("_debug_dispatch_ts") - env_output.get("_debug_pull_ts")
                                if env_output.get("_debug_dispatch_ts") is not None
                                and env_output.get("_debug_pull_ts") is not None
                                else None
                            ),
                        }
                    )
                with self._env_wait_lock:
                    self._env_handler_wait_times[slot_id] = (
                        self._env_handler_wait_times.get(slot_id, 0.0) + env_wait_elapsed
                    )

                if last_forward_inputs is not None:
                    last_forward_inputs = self._update_intervene_actions(env_output, last_forward_inputs)

                request_id = f"slot{slot_id}_ep{epoch_idx}_st{step_idx}_{request_counter}"
                request_counter += 1
                request = PerEnvInferenceRequest(
                    request_id=request_id,
                    env_id=slot_id,
                    stage_id=slot_id,
                    env_output=env_output,
                    step_idx=step_idx,
                    epoch_idx=epoch_idx,
                    mode="train",
                )
                future = self.batching_engine.submit_request(request)
                result: PerEnvInferenceResult = future.result()

                # logger.info(f"[DEBUG] Env {slot_id} epoch {epoch_idx} step {step_idx} env_output.dones = {env_output.get('dones').shape}")

                # if env_output.get('rewards') is not None:
                #     logger.info(f"[DEBUG] Env {slot_id} epoch {epoch_idx} step {step_idx} env_output.rewards.shape = {env_output.get('rewards').shape}")

                # if result.dones is not None:
                #     logger.info(f"[DEBUG] Env {slot_id} epoch {epoch_idx} step {step_idx} result.dones = {result.dones}")

                dones = env_output.get("dones")
                rewards = env_output.get("rewards")
                if "prev_logprobs" in result.result:
                    buffer.buffer.prev_logprobs.append(result.result["prev_logprobs"].cpu().contiguous())
                if "prev_values" in result.result:
                    buffer.buffer.prev_values.append(result.result["prev_values"].cpu().contiguous())
                if dones is not None:
                    buffer.buffer.dones.append(dones.cpu().contiguous())
                    buffer.buffer.truncations.append(env_output.get("truncations").cpu().contiguous())
                    buffer.buffer.terminations.append(env_output.get("terminations").cpu().contiguous())
                if rewards is not None:
                    buffer.buffer.rewards.append(rewards.cpu().contiguous())
                if last_forward_inputs is not None:
                    if self._mem_probe_should_log(
                        epoch_idx=epoch_idx, step_idx=step_idx, slot_id=slot_id
                    ):
                        if torch.cuda.is_available():
                            torch.cuda.synchronize()
                        cuda_before = self._count_cuda_tensors(last_forward_inputs)
                        offloaded = put_tensor_device(last_forward_inputs, "cpu")
                        cuda_after = self._count_cuda_tensors(offloaded)
                        alloc = torch.cuda.memory_allocated() if torch.cuda.is_available() else -1
                        reserved = torch.cuda.memory_reserved() if torch.cuda.is_available() else -1
                        peak = torch.cuda.max_memory_allocated() if torch.cuda.is_available() else -1
                        print(
                            f"[MEM_PROBE] rank={self._rank} slot={slot_id} "
                            f"epoch={epoch_idx} step={step_idx} "
                            f"cuda_tensors_before={cuda_before} after={cuda_after} "
                            f"alloc={alloc/2**30:.3f}GiB reserved={reserved/2**30:.3f}GiB "
                            f"peak_alloc={peak/2**30:.3f}GiB"
                        )
                        buffer.buffer.forward_inputs.append(offloaded)
                    else:
                        buffer.buffer.forward_inputs.append(
                            put_tensor_device(last_forward_inputs, "cpu")
                        )

                last_extracted_obs = result.extracted_obs
                last_forward_inputs = result.result.get("forward_inputs")
                # [DEBUG_TIMING] log when rollout sends actions back to env
                if self._debug_timing_enabled:
                    t_send = time.time()
                    self._write_debug_timing_log(
                        {
                            "event": "rollout_send",
                            "ts_rel": t_send - self._debug_t0,
                            "ts_abs": t_send,
                            "rollout_rank": self._rank,
                            "slot_id": slot_id,
                            "env_key": env_key,
                            "step_idx": step_idx,
                            "epoch_idx": epoch_idx,
                        }
                    )
                sender = self._output_sender
                if sender is not None:
                    sender.put(item=result.actions, key=env_key, async_op=True)
                else:
                    output_channel.put(item=result.actions, key=env_key, async_op=True)
                done_epoch = int(env_output.get("_debug_epoch_idx", epoch_idx))
                done_step = int(env_output.get("_debug_step_idx", step_idx))
                self._mark_slot_msg_done(slot_id, done_epoch, done_step)

            # [DEBUG_TIMING] second get in epoch (final step env output)
            env_wait_start_final = time.time()
            env_output = self._recv_env_output_for_slot(slot_id)
            env_wait_elapsed_final = time.time() - env_wait_start_final
            # [DEBUG dones/rewards] 最后一步应为 env 的 step n_chunk_steps-1，且 has_rewards=True
            msg_step_final = env_output.get("_debug_step_idx", "?")
            msg_epoch_final = env_output.get("_debug_epoch_idx", "?")
            from_global_env_id_final = env_output.get("_debug_global_env_id", "?")
            has_rewards_final = env_output.get("rewards") is not None
            # logger.info(
            #     "[DEBUG dones/rewards] slot_id=%s epoch_idx=%s final_recv: "
            #     "msg_step=%s msg_epoch=%s from_global_env_id=%s has_rewards=%s (expect msg_step=%s, has_rewards=True)",
            #     slot_id, epoch_idx, msg_step_final, msg_epoch_final, from_global_env_id_final, has_rewards_final,
            #     n_chunk_steps - 1,
            # )
            if self._debug_timing_enabled:
                t_recv_final = time.time()
                self._write_debug_timing_log(
                    {
                        "event": "rollout_recv",
                        "ts_rel": t_recv_final - self._debug_t0,
                        "ts_abs": t_recv_final,
                        "rollout_rank": self._rank,
                        "slot_id": slot_id,
                        "env_key": env_key,
                        "step_idx": n_chunk_steps,
                        "epoch_idx": epoch_idx,
                        "is_final_step": True,
                        "wait_time": env_wait_elapsed_final,
                        "from_env_rank": env_output.get("_debug_env_rank"),
                        "from_global_env_id": env_output.get("_debug_global_env_id"),
                    }
                )
            if last_forward_inputs is not None:
                last_forward_inputs = self._update_intervene_actions(env_output, last_forward_inputs)

            dones = env_output.get("dones")
            rewards = env_output.get("rewards")
            if dones is not None:
                buffer.buffer.dones.append(dones.cpu().contiguous())
                buffer.buffer.truncations.append(env_output.get("truncations").cpu().contiguous())
                buffer.buffer.terminations.append(env_output.get("terminations").cpu().contiguous())
            if rewards is not None:
                buffer.buffer.rewards.append(rewards.cpu().contiguous())
            if last_forward_inputs is not None:
                if (
                    self._mem_probe_enabled
                    and self._mem_probe_generate_cnt <= 1
                    and epoch_idx == 0
                    and slot_id == (self._mem_probe_slot_id if self._mem_probe_slot_id is not None else -1)
                ):
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                    cuda_before = self._count_cuda_tensors(last_forward_inputs)
                    offloaded = put_tensor_device(last_forward_inputs, "cpu")
                    cuda_after = self._count_cuda_tensors(offloaded)
                    alloc = torch.cuda.memory_allocated() if torch.cuda.is_available() else -1
                    reserved = torch.cuda.memory_reserved() if torch.cuda.is_available() else -1
                    peak = torch.cuda.max_memory_allocated() if torch.cuda.is_available() else -1
                    print(
                        f"[MEM_PROBE] rank={self._rank} slot={slot_id} "
                        f"epoch={epoch_idx} step=FINAL "
                        f"cuda_tensors_before={cuda_before} after={cuda_after} "
                        f"alloc={alloc/2**30:.3f}GiB reserved={reserved/2**30:.3f}GiB "
                        f"peak_alloc={peak/2**30:.3f}GiB"
                    )
                    buffer.buffer.forward_inputs.append(offloaded)
                else:
                    buffer.buffer.forward_inputs.append(
                        put_tensor_device(last_forward_inputs, "cpu")
                    )

            request_id = f"slot{slot_id}_ep{epoch_idx}_final_{request_counter}"
            request_counter += 1
            request = PerEnvInferenceRequest(
                request_id=request_id,
                env_id=slot_id,
                stage_id=slot_id,
                env_output=env_output,
                step_idx=n_chunk_steps,
                epoch_idx=epoch_idx,
                mode="train",
                is_final_step=True,
            )
            final_res: PerEnvInferenceResult = self.batching_engine.submit_request(request).result()
            if "prev_values" in final_res.result:
                buffer.buffer.prev_values.append(final_res.result["prev_values"].cpu().contiguous())
            if hasattr(self.hf_model, "q_head"):
                buffer.buffer.add_transition(last_extracted_obs, final_res.real_extracted_obs)
            buffer.completed_epochs += 1
            # Final step received - mark epoch as complete
            # The env_output from final_recv is the last step's output (step=n_chunk_steps-1)
            # Epoch is complete after processing this final step
            done_epoch = int(env_output.get("_debug_epoch_idx", epoch_idx))
            self._mark_slot_msg_done(slot_id, done_epoch, n_chunk_steps, is_final=True)

            if pipeline_mode == "async":
                self._send_rollout_batch(actor_channel, slot_id)
            # [DEBUG dones/rewards] 每 epoch 结束时检查：应满足 len(dones) == len(rewards) + 1
            n_dones = len(buffer.buffer.dones)
            n_rewards = len(buffer.buffer.rewards)
            expected = n_rewards + 1
            # if n_dones != expected:
            #     logger.warning(
            #         "[DEBUG dones/rewards] slot_id=%s epoch_idx=%s MISMATCH: "
            #         "len(dones)=%s len(rewards)=%s (expected dones=%s)",
            #         slot_id, epoch_idx, n_dones, n_rewards, expected,
            #     )
            # else:
            #     logger.info(
            #         "[DEBUG dones/rewards] slot_id=%s epoch_idx=%s ok: len(dones)=%s len(rewards)=%s",
            #         slot_id, epoch_idx, n_dones, n_rewards,
            #     )
            if epoch_idx < self.cfg.algorithm.rollout_epoch - 1:
                if self._epoch_prefetch_enabled:
                    if self._debug_timing_enabled:
                        t_ack = time.time()
                        self._write_debug_timing_log(
                            {
                                "event": "rollout_epoch_ack_skipped",
                                "ts_rel": t_ack - self._debug_t0,
                                "ts_abs": t_ack,
                                "rollout_rank": self._rank,
                                "slot_id": slot_id,
                                "env_key": env_key,
                                "epoch_idx": epoch_idx,
                                "next_epoch_idx": epoch_idx + 1,
                                "epoch_prefetch_enabled": True,
                            }
                        )
                else:
                    # [DEBUG_EPOCH] log before sending epoch_done
                    # logger.info(
                    #     "[DEBUG_EPOCH] slot_id=%s epoch_idx=%s: sending __epoch_done__ ack to env_key=%s",
                    #     slot_id, epoch_idx, env_key
                    # )
                    # [DEBUG_TIMING] log when sending epoch_done ack to env
                    if self._debug_timing_enabled:
                        t_ack = time.time()
                        self._write_debug_timing_log(
                            {
                                "event": "rollout_send",
                                "ts_rel": t_ack - self._debug_t0,
                                "ts_abs": t_ack,
                                "rollout_rank": self._rank,
                                "slot_id": slot_id,
                                "env_key": env_key,
                                "epoch_idx": epoch_idx,
                                "is_epoch_done_ack": True,
                            }
                        )
                    sender = self._output_sender
                    if sender is not None:
                        sender.put(item={"__epoch_done__": True}, key=env_key, async_op=True)
                    else:
                        output_channel.put(item={"__epoch_done__": True}, key=env_key, async_op=True)
                    # logger.info(
                    #     "[DEBUG_EPOCH] slot_id=%s epoch_idx=%s: __epoch_done__ ack sent successfully",
                    #     slot_id, epoch_idx
                    # )

    def _recv_env_output_for_slot(self, slot_id: int) -> Dict[str, Any]:
        q = self._slot_queue_map[slot_id]
        while True:
            try:
                msg = q.get(timeout=0.1)
                if self._debug_timing_enabled:
                    msg["_debug_slot_dequeue_ts"] = time.time()
                return msg
            except queue.Empty:
                if self._router_stop_event is not None and self._router_stop_event.is_set():
                    raise RuntimeError(
                        f"router stopped while waiting env output for slot_id={slot_id}"
                    )

    # Marker to indicate epoch is complete, ready for next epoch
    _EPOCH_COMPLETE_MARKER = 999999

    def _mark_slot_msg_done(self, slot_id: int, epoch_idx: int, done_step: int, is_final: bool = False) -> None:
        assert self._router_cv is not None
        with self._router_cv:
            self._slot_inflight[slot_id] = False
            self._slot_inflight_frontier[slot_id] = None
            if is_final:
                # Epoch is complete after final step, use special marker to indicate ready for next epoch
                self._slot_done_frontier[slot_id] = (epoch_idx, self._EPOCH_COMPLETE_MARKER)
            else:
                # Normal step (including init step=-1), record the actual step
                self._slot_done_frontier[slot_id] = (epoch_idx, done_step)
            self._router_cv.notify_all()

    def _run_multiplex_env_input_puller(
        self,
        env_keys: List[str],
        pending: Dict[int, Dict[int, List]],
    ) -> None:
        """Single-thread puller: read env outputs from Channel into ``pending``.

        Only this thread calls ``get`` for these keys, so recv stays thread-safe
        without SerializedChannelReceiver's global lock. Per-key FIFO and
        ``pending`` append semantics match the old per-key puller threads.
        """
        assert self._router_cv is not None
        assert self._router_stop_event is not None
        ch = self._input_channel
        assert ch is not None

        def _safe_int(value: Any, default: int) -> int:
            try:
                return int(value)
            except Exception:
                return default

        n = len(env_keys)
        if n == 0:
            return

        rr = 0
        first_pull_logged = False
        while not self._router_stop_event.is_set():
            ordered_keys = [env_keys[(rr + i) % n] for i in range(n)]
            try:
                env_key, msg = ch.get_any_timeout(ordered_keys, self._pull_timeout_s)
            except asyncio.QueueEmpty:
                rr = (rr + 1) % n
                if self._pull_timeout_s <= 0:
                    time.sleep(0.0001)
                continue
            pull_ts = time.time()
            if self._debug_timing_enabled and not first_pull_logged:
                first_pull_logged = True
                self._write_debug_timing_log(
                    {
                        "event": "rollout_first_pull",
                        "ts_rel": pull_ts - self._debug_t0,
                        "ts_abs": pull_ts,
                        "rollout_rank": self._rank,
                        "env_key": env_key,
                    }
                )

            sid_value = msg.get("_route_slot_id")
            sid: Optional[int] = None
            if sid_value is not None:
                try:
                    sid = int(sid_value)
                except Exception:
                    sid = None

            if sid is None:
                env_id_raw = msg.get("_debug_global_env_id")
                local_slot_raw = msg.get("_route_local_slot_index")
                if env_id_raw is not None and local_slot_raw is not None:
                    try:
                        sid = self._slot_id_by_env_local.get((int(env_id_raw), int(local_slot_raw)))
                    except Exception:
                        sid = None

            if sid is None or sid not in self._slot_spec_by_slot_id:
                logger.warning(
                    "[ROUTER PULLER] rollout_rank=%s key=%s got message without valid route slot id, skip msg keys=%s",
                    self._rank,
                    env_key,
                    list(msg.keys()),
                )
                rr = (ordered_keys.index(env_key) + rr + 1) % n
                continue

            eid = self._slot_spec_by_slot_id[sid].global_env_id
            epoch_idx = _safe_int(msg.get("_debug_epoch_idx"), 10**9)
            step_idx = _safe_int(msg.get("_debug_step_idx"), 10**9)
            pri = (epoch_idx, step_idx, eid)
            with self._router_cv:
                if self._debug_timing_enabled:
                    msg["_debug_pull_ts"] = pull_ts
                    msg["_debug_pending_ts"] = pull_ts
                    self._write_debug_timing_log(
                        {
                            "event": "rollout_pull",
                            "ts_rel": pull_ts - self._debug_t0,
                            "ts_abs": pull_ts,
                            "rollout_rank": self._rank,
                            "env_key": env_key,
                            "slot_id": sid,
                            "global_env_id": eid,
                            "msg_epoch_idx": epoch_idx,
                            "msg_step_idx": step_idx,
                            "env_ts_send": msg.get("_debug_ts_send"),
                            "env_put_start_ts": msg.get("_debug_put_start_ts"),
                            "env_put_submit_ts": msg.get("_debug_put_submit_ts"),
                            "transport_delay": (
                                pull_ts - msg.get("_debug_ts_send")
                                if msg.get("_debug_ts_send") is not None
                                else None
                            ),
                            "put_submit_to_pull_delay": (
                                pull_ts - msg.get("_debug_put_submit_ts")
                                if msg.get("_debug_put_submit_ts") is not None
                                else None
                            ),
                        }
                    )
                pending[eid][sid].append((pri, msg))
                self._router_cv.notify_all()
            rr = (ordered_keys.index(env_key) + rr + 1) % n

    def _run_env_output_router_loop(self) -> None:
        """
        Router: deterministic env-ordered dispatch with epoch boundary validation.

        All rollout workers process envs in the same fixed order (sorted by
        global_env_id) with step priority: earlier (epoch, step) is always
        dispatched first regardless of env_id.  This ensures every env's
        slots across different rollout workers finish at roughly the same
        time, minimising the wait on the env side.

        EPOCH BOUNDARY PROTECTION: The router validates that each slot only
        receives messages for its current epoch. Messages from future epochs
        are kept in pending queue until the slot advances to that epoch.

        NOTE: A single multiplex thread reads from the input Channel (no
        SerializedChannelReceiver lock), avoiding recv contention from N
        per-key puller threads while keeping the same pending semantics.
        """
        assert self._router_cv is not None
        assert self._router_stop_event is not None
        assert self._input_channel is not None

        if self._debug_timing_enabled:
            t_now = time.time()
            self._write_debug_timing_log(
                {
                    "event": "rollout_router_start",
                    "ts_rel": t_now - self._debug_t0,
                    "ts_abs": t_now,
                    "rollout_rank": self._rank,
                    "router_mode": "global_round_robin",
                    "slot_count": len(self.my_slot_specs),
                }
            )
        env_keys = [
            self._get_rollout_input_channel_key(self._rank, slot_spec.slot_id, "train")
            for slot_spec in self.my_slot_specs
        ]
        pending: Dict[int, Dict[int, List]] = {
            env_id: {sid: [] for sid in sids}
            for env_id, sids in self._env_slot_ids_local.items()
        }
        sorted_env_ids = sorted(self._env_slot_ids_local.keys())

        multiplex_puller = threading.Thread(
            target=self._run_multiplex_env_input_puller,
            args=(env_keys, pending),
            daemon=True,
            name=f"FlexMultiplexPull-r{self._rank}",
        )
        multiplex_puller.start()

        try:
            next_env_idx = 0
            router_loop_counter = 0
            last_router_wait_log = 0.0

            while not self._router_stop_event.is_set():
                with self._router_cv:
                    eid = sorted_env_ids[next_env_idx]
                    env_slots = self._env_slot_ids_local[eid]

                    # [EPOCH BARRIER] Check if all slots have messages and validate epoch boundaries
                    ready_slots = []
                    all_slots_valid = True
                    blocking_reason = None
                    slot_state_for_log = []  # (sid, pending_len, head_epoch_step, expected_epoch_step)
                    for sid in env_slots:
                        done_frontier = self._slot_done_frontier.get(sid)
                        if done_frontier is None:
                            expected_epoch, expected_step = 0, -1
                        elif done_frontier[1] == self._EPOCH_COMPLETE_MARKER:
                            expected_epoch, expected_step = done_frontier[0] + 1, -1
                        else:
                            expected_epoch, expected_step = done_frontier[0], done_frontier[1] + 1

                        if len(pending[eid][sid]) == 0:
                            # Missing message for this slot, wait
                            all_slots_valid = False
                            blocking_reason = f"slot_id={sid}: no pending message"
                            slot_state_for_log.append((sid, 0, None, (expected_epoch, expected_step)))
                            break
                        
                        # Check head message epoch and step
                        head_msg_epoch, head_msg_step = pending[eid][sid][0][0][0], pending[eid][sid][0][0][1]
                        slot_state_for_log.append(
                            (sid, len(pending[eid][sid]), (head_msg_epoch, head_msg_step), (expected_epoch, expected_step))
                        )
                        
                        # Validate: only accept messages matching expected epoch and step
                        if head_msg_epoch == expected_epoch and head_msg_step == expected_step:
                            ready_slots.append(sid)
                        elif head_msg_epoch < expected_epoch:
                            # Stale message from past epoch - this shouldn't happen
                            logger.warning(
                                f"[ROUTER EPOCH BARRIER] slot_id={sid}: received stale message "
                                f"epoch={head_msg_epoch} step={head_msg_step}, "
                                f"expected epoch={expected_epoch} step={expected_step}. "
                                f"Dropping message."
                            )
                            # Remove stale message
                            pending[eid][sid].pop(0)
                            all_slots_valid = False
                            blocking_reason = f"slot_id={sid}: stale message dropped"
                            break
                        else:
                            # Future message - either wrong step or wrong epoch
                            # Wait for slot handler to catch up
                            all_slots_valid = False
                            blocking_reason = (
                                f"slot_id={sid}: future msg (epoch={head_msg_epoch} step={head_msg_step}) "
                                f"vs expected (epoch={expected_epoch} step={expected_step}), "
                                f"done_frontier={done_frontier}"
                            )
                            break
                    
                    # Only dispatch if all slots for this env are ready
                    if not all_slots_valid or len(ready_slots) != len(env_slots):
                        router_loop_counter += 1
                        now = time.time()
                        if now - last_router_wait_log >= 2.0:
                            last_router_wait_log = now
                            logger.warning(
                                "[ROUTER WAIT] rollout_rank=%s eid=%s env_slots=%s ready=%s/%s reason=%s slot_state=%s",
                                self._rank,
                                eid,
                                env_slots,
                                len(ready_slots),
                                len(env_slots),
                                blocking_reason,
                                slot_state_for_log,
                            )
                        self._router_cv.wait(timeout=0.001)
                        continue
                    
                    router_loop_counter = 0  # Reset counter on successful dispatch

                    # All slots ready - dispatch messages
                    # Use epoch/step from first slot's head message
                    head = pending[eid][env_slots[0]][0][0]
                    frontier = (int(head[0]), int(head[1]))
                    for sid in env_slots:
                        _head_pri, msg = pending[eid][sid].pop(0)
                        dispatch_ts = time.time()
                        if self._debug_timing_enabled:
                            msg["_debug_dispatch_ts"] = dispatch_ts
                            self._write_debug_timing_log(
                                {
                                    "event": "rollout_router_dispatch",
                                    "ts_rel": dispatch_ts - self._debug_t0,
                                    "ts_abs": dispatch_ts,
                                    "rollout_rank": self._rank,
                                    "router_mode": "global_round_robin",
                                    "global_env_id": eid,
                                    "slot_id": sid,
                                    "msg_epoch_idx": int(head[0]),
                                    "msg_step_idx": int(head[1]),
                                    "pending_delay": (
                                        dispatch_ts - msg.get("_debug_pending_ts")
                                        if msg.get("_debug_pending_ts") is not None
                                        else None
                                    ),
                                }
                            )
                        self._slot_inflight_frontier[sid] = frontier
                        self._slot_inflight[sid] = True
                        self._slot_queue_map[sid].put(msg)

                    next_env_idx = (next_env_idx + 1) % len(sorted_env_ids)
                    # Relax env-level in-flight barrier:
                    # do not block router on current env's slot completion.
                    # Per-slot ordering is still guarded by done_frontier/inflight checks.
        finally:
            multiplex_puller.join(timeout=2.0)

    def _run_greedy_env_output_router_loop(self) -> None:
        """
        Greedy router (方案4): dispatch to any slot as soon as its next message is ready.
        No global env_id order - slots progress independently. Keeps epoch/step ordering per slot.

        Env input is read by a single multiplex thread (see
        ``_run_multiplex_env_input_puller``), not N per-key pullers with a
        global SerializedChannelReceiver lock.
        """
        assert self._router_cv is not None
        assert self._router_stop_event is not None
        assert self._input_channel is not None

        if self._debug_timing_enabled:
            t_now = time.time()
            self._write_debug_timing_log(
                {
                    "event": "rollout_router_start",
                    "ts_rel": t_now - self._debug_t0,
                    "ts_abs": t_now,
                    "rollout_rank": self._rank,
                    "router_mode": "greedy",
                    "slot_count": len(self.my_slot_specs),
                }
            )
        env_keys = [
            self._get_rollout_input_channel_key(self._rank, slot_spec.slot_id, "train")
            for slot_spec in self.my_slot_specs
        ]
        pending: Dict[int, Dict[int, List]] = {
            eid: {sid: [] for sid in sids}
            for eid, sids in self._env_slot_ids_local.items()
        }

        multiplex_puller = threading.Thread(
            target=self._run_multiplex_env_input_puller,
            args=(env_keys, pending),
            daemon=True,
            name=f"FlexGreedyMultiplexPull-r{self._rank}",
        )
        multiplex_puller.start()

        try:
            # Flat list (eid, sid) for all slots on this worker
            slot_pairs = [
                (eid, sid)
                for eid in sorted(self._env_slot_ids_local.keys())
                for sid in self._env_slot_ids_local[eid]
            ]

            while not self._router_stop_event.is_set():
                dispatched_any = False
                with self._router_cv:
                    for eid, sid in slot_pairs:
                        if self._slot_inflight.get(sid, False):
                            continue
                        if len(pending[eid][sid]) == 0:
                            continue

                        done_frontier = self._slot_done_frontier.get(sid)
                        if done_frontier is None:
                            expected_epoch, expected_step = 0, -1
                        elif done_frontier[1] == self._EPOCH_COMPLETE_MARKER:
                            expected_epoch, expected_step = done_frontier[0] + 1, -1
                        else:
                            expected_epoch, expected_step = done_frontier[0], done_frontier[1] + 1

                        head_pri, msg = pending[eid][sid][0][0], pending[eid][sid][0][1]
                        head_epoch, head_step = head_pri[0], head_pri[1]

                        if head_epoch == expected_epoch and head_step == expected_step:
                            pending[eid][sid].pop(0)
                            dispatch_ts = time.time()
                            if self._debug_timing_enabled:
                                msg["_debug_dispatch_ts"] = dispatch_ts
                                self._write_debug_timing_log(
                                    {
                                        "event": "rollout_router_dispatch",
                                        "ts_rel": dispatch_ts - self._debug_t0,
                                        "ts_abs": dispatch_ts,
                                        "rollout_rank": self._rank,
                                        "router_mode": "greedy",
                                        "global_env_id": eid,
                                        "slot_id": sid,
                                        "msg_epoch_idx": int(head_epoch),
                                        "msg_step_idx": int(head_step),
                                        "pending_delay": (
                                            dispatch_ts - msg.get("_debug_pending_ts")
                                            if msg.get("_debug_pending_ts") is not None
                                            else None
                                        ),
                                    }
                                )
                            self._slot_inflight_frontier[sid] = (int(head_epoch), int(head_step))
                            self._slot_inflight[sid] = True
                            self._slot_queue_map[sid].put(msg)
                            dispatched_any = True
                        elif head_epoch < expected_epoch:
                            logger.warning(
                                f"[GREEDY ROUTER] slot_id={sid}: dropping stale msg epoch={head_epoch} step={head_step}"
                            )
                            pending[eid][sid].pop(0)
                            dispatched_any = True

                if not dispatched_any:
                    with self._router_cv:
                        self._router_cv.wait(timeout=0.001)
        finally:
            multiplex_puller.join(timeout=2.0)

    def _run_env_group_greedy_router_loop(self) -> None:
        """
        Env-group greedy router:
        - Keep per-slot epoch/step ordering checks.
        - Dispatch all slots for one env together (env-group barrier).
        - No fixed global env order: whichever env becomes ready first gets dispatched.
        """
        assert self._router_cv is not None
        assert self._router_stop_event is not None
        assert self._input_channel is not None

        if self._debug_timing_enabled:
            t_now = time.time()
            self._write_debug_timing_log(
                {
                    "event": "rollout_router_start",
                    "ts_rel": t_now - self._debug_t0,
                    "ts_abs": t_now,
                    "rollout_rank": self._rank,
                    "router_mode": "env_group_greedy",
                    "slot_count": len(self.my_slot_specs),
                }
            )
        env_keys = [
            self._get_rollout_input_channel_key(self._rank, slot_spec.slot_id, "train")
            for slot_spec in self.my_slot_specs
        ]
        pending: Dict[int, Dict[int, List]] = {
            env_id: {sid: [] for sid in sids}
            for env_id, sids in self._env_slot_ids_local.items()
        }
        sorted_env_ids = sorted(self._env_slot_ids_local.keys())

        multiplex_puller = threading.Thread(
            target=self._run_multiplex_env_input_puller,
            args=(env_keys, pending),
            daemon=True,
            name=f"FlexEnvGroupGreedyPull-r{self._rank}",
        )
        multiplex_puller.start()

        def _expected_frontier(done_frontier: Optional[tuple]) -> tuple[int, int]:
            if done_frontier is None:
                return 0, -1
            if done_frontier[1] == self._EPOCH_COMPLETE_MARKER:
                return done_frontier[0] + 1, -1
            return done_frontier[0], done_frontier[1] + 1

        try:
            env_rr = 0
            while not self._router_stop_event.is_set():
                dispatched_any = False
                with self._router_cv:
                    n_env = len(sorted_env_ids)
                    for i in range(n_env):
                        eid = sorted_env_ids[(env_rr + i) % n_env]
                        env_slots = self._env_slot_ids_local[eid]

                        ready_payloads: List[tuple[int, int, Dict[str, Any]]] = []
                        env_ready = True
                        for sid in env_slots:
                            if self._slot_inflight.get(sid, False):
                                env_ready = False
                                break
                            if len(pending[eid][sid]) == 0:
                                env_ready = False
                                break

                            done_frontier = self._slot_done_frontier.get(sid)
                            expected_epoch, expected_step = _expected_frontier(done_frontier)
                            head_pri, msg = pending[eid][sid][0][0], pending[eid][sid][0][1]
                            head_epoch, head_step = int(head_pri[0]), int(head_pri[1])

                            if head_epoch == expected_epoch and head_step == expected_step:
                                ready_payloads.append((sid, head_epoch, msg))
                            elif head_epoch < expected_epoch:
                                logger.warning(
                                    "[ENV_GROUP_GREEDY] slot_id=%s dropping stale msg epoch=%s step=%s expected=(%s,%s)",
                                    sid,
                                    head_epoch,
                                    head_step,
                                    expected_epoch,
                                    expected_step,
                                )
                                pending[eid][sid].pop(0)
                                env_ready = False
                                break
                            else:
                                env_ready = False
                                break

                        if not env_ready or len(ready_payloads) != len(env_slots):
                            continue

                        # Dispatch all slots of this env together.
                        frontier_epoch = ready_payloads[0][1]
                        frontier_step = int(pending[eid][env_slots[0]][0][0][1])
                        frontier = (frontier_epoch, frontier_step)
                        for sid, _head_epoch, msg in ready_payloads:
                            pending[eid][sid].pop(0)
                            dispatch_ts = time.time()
                            if self._debug_timing_enabled:
                                msg["_debug_dispatch_ts"] = dispatch_ts
                                self._write_debug_timing_log(
                                    {
                                        "event": "rollout_router_dispatch",
                                        "ts_rel": dispatch_ts - self._debug_t0,
                                        "ts_abs": dispatch_ts,
                                        "rollout_rank": self._rank,
                                        "router_mode": "env_group_greedy",
                                        "global_env_id": eid,
                                        "slot_id": sid,
                                        "msg_epoch_idx": frontier[0],
                                        "msg_step_idx": frontier[1],
                                        "pending_delay": (
                                            dispatch_ts - msg.get("_debug_pending_ts")
                                            if msg.get("_debug_pending_ts") is not None
                                            else None
                                        ),
                                    }
                                )
                            self._slot_inflight_frontier[sid] = frontier
                            self._slot_inflight[sid] = True
                            self._slot_queue_map[sid].put(msg)

                        env_rr = (sorted_env_ids.index(eid) + 1) % n_env
                        dispatched_any = True
                        break

                if not dispatched_any:
                    with self._router_cv:
                        self._router_cv.wait(timeout=0.001)
        finally:
            multiplex_puller.join(timeout=2.0)

    async def evaluate(self, input_channel: Channel, output_channel: Channel):
        raise NotImplementedError(
            "FlexiblePerEnvAsyncRolloutWorker currently supports training generate only. "
            "Please set runner.val_check_interval=0 for this experimental mode."
        )

