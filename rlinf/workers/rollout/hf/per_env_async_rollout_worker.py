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
PerEnvAsyncRolloutWorker: Rollout worker for true per-env async pipeline.

Works with TruePerEnvAsyncEnvWorker to enable:
- env_id-based Channel routing (each env has its own message queue)
- Independent epoch loops per env (no epoch-level sync)
- Dynamic batching across all envs for efficient GPU utilization

Architecture:
    ┌─────────────────────────────────────────────────────────────────────────┐
    │                   PerEnvAsyncRolloutWorker                               │
    │                                                                          │
    │  Channel Keys:                                                           │
    │  - perenv_{rank}_0_train  ──→ EnvManager 0                              │
    │  - perenv_{rank}_1_train  ──→ EnvManager 1                              │
    │  - perenv_{rank}_N_train  ──→ EnvManager N                              │
    │                                                                          │
    │  ┌─────────────────────────────────────────────────────────────────┐    │
    │  │                  Per-Env Handler Threads                         │    │
    │  │                                                                  │    │
    │  │  Thread 0: recv(env_0) → submit → wait → send(env_0)            │    │
    │  │  Thread 1: recv(env_1) → submit → wait → send(env_1)            │    │
    │  │  Thread N: recv(env_N) → submit → wait → send(env_N)            │    │
    │  │                          │                                       │    │
    │  │                          ▼                                       │    │
    │  │               ┌──────────────────────┐                           │    │
    │  │               │ DynamicBatchingEngine│                           │    │
    │  │               │ - Batches requests   │                           │    │
    │  │               │ - Single GPU forward │                           │    │
    │  │               │ - Returns per-env    │                           │    │
    │  │               └──────────────────────┘                           │    │
    │  └─────────────────────────────────────────────────────────────────┘    │
    │                                                                          │
    │  Each env runs independently:                                            │
    │  - No waiting for other envs                                             │
    │  - Requests batched dynamically by the engine                            │
    │  - Results returned immediately when ready                               │
    └─────────────────────────────────────────────────────────────────────────┘

Key differences from DynamicBatchingRolloutWorker:
1. Uses per-env Channel keys (not stage-based)
2. Each env has its own handler thread for recv/send
3. No epoch-level synchronization
4. Supports env_ids finishing at different times
"""

import asyncio
import copy
import gc
import logging
import queue

import numpy as np
import threading
import time
from collections import defaultdict
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import torch
from omegaconf import DictConfig, OmegaConf, open_dict
from tqdm import tqdm

from rlinf.config import SupportedModel
from rlinf.data.io_struct import ChunkStepResult, EmbodiedRolloutResult
from rlinf.models import get_model, get_vla_model_config_and_processor
from rlinf.scheduler import Channel, Cluster, Worker
from rlinf.utils.metric_utils import compute_split_num
from rlinf.utils.nested_dict_process import put_tensor_device
from rlinf.utils.placement import HybridComponentPlacement
from rlinf.workers.rollout.hf.utils import init_real_obs

logger = logging.getLogger(__name__)


class SerializedChannelReceiver:
    """
    Wrapper around Channel that uses global lock to serialize recv() calls.

    Problem: Channel.get() internally calls _current_worker.recv() which has
    race conditions when called from multiple threads concurrently.

    Solution: Use global lock but with short hold time by checking queue
    status first. Only call blocking get() when we know message is available.
    """

    def __init__(self, input_channel: Channel):
        self.input_channel = input_channel
        self._lock = threading.Lock()

    def get(self, key: str) -> Dict[str, Any]:
        """Get from channel with global lock."""
        import time

        while True:
            # Check if message is available (non-blocking, no lock needed)
            if not self.input_channel.empty(key=key):
                # Message available, acquire lock and get it
                with self._lock:
                    # Double-check after acquiring lock
                    if not self.input_channel.empty(key=key):
                        return self.input_channel.get(key=key)

            # No message yet, sleep briefly and retry
            time.sleep(0.001)


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


@dataclass
class PerEnvInferenceRequest:
    """Inference request with env_id tracking."""
    request_id: str
    env_id: int  # Global env ID
    stage_id: int  # Local stage ID within this worker
    env_output: Dict[str, Any]
    step_idx: int
    epoch_idx: int
    mode: str = "train"
    is_final_step: bool = False


@dataclass
class PerEnvInferenceResult:
    """Result with env_id tracking."""
    request_id: str
    env_id: int
    actions: torch.Tensor
    result: Dict[str, Any]
    extracted_obs: Dict[str, Any]
    dones: Optional[torch.Tensor]
    rewards: Optional[torch.Tensor]
    real_extracted_obs: Optional[Dict[str, Any]]


@dataclass
class EnvRolloutBuffer:
    """Rollout buffer for a single environment."""
    env_id: int
    stage_id: int
    rollout_epoch: int
    buffer: EmbodiedRolloutResult = field(default=None)
    last_extracted_obs: Optional[Dict[str, Any]] = None
    last_forward_inputs: Optional[Dict[str, Any]] = None
    completed_epochs: int = 0

    def __post_init__(self):
        if self.buffer is None:
            self.buffer = EmbodiedRolloutResult(rollout_epoch=self.rollout_epoch)


class PerEnvDynamicBatchingEngine:
    """
    Dynamic batching engine optimized for per-env async pipeline.

    Similar to DynamicBatchingEngine but with:
    - env_id tracking for each request
    - Support for envs at different epochs/steps
    """

    def __init__(
        self,
        hf_model,
        cfg: DictConfig,
        max_batch_size: int = 32,
        batch_timeout_ms: float = 5.0,
        device: torch.device = None,
    ):
        self.hf_model = hf_model
        self.cfg = cfg
        self.max_batch_size = max_batch_size
        self.batch_timeout_ms = batch_timeout_ms
        self.device = device or torch.cuda.current_device()

        self.request_queue: queue.Queue[PerEnvInferenceRequest] = queue.Queue()
        self.result_futures: Dict[str, Future] = {}
        self._stop_sentinel = object()

        self.inference_thread: Optional[threading.Thread] = None
        self.should_stop = False

        self._setup_sample_params()

        # Metrics
        self.total_requests = 0
        self.total_batches = 0
        self.total_batch_size = 0
        self.total_generate_time = 0.0  # Total inference time in seconds

    def _setup_sample_params(self):
        """Setup sampling parameters."""
        length_params = OmegaConf.to_container(
            self.cfg.algorithm.length_params, resolve=True
        )
        sampling_params = OmegaConf.to_container(
            self.cfg.algorithm.sampling_params, resolve=True
        )

        self._train_sampling_params = {
            "do_sample": sampling_params["do_sample"],
            "temperature": sampling_params["temperature_train"],
            "top_k": sampling_params["top_k"],
            "top_p": sampling_params["top_p"],
            "max_new_tokens": length_params["max_new_token"],
            "use_cache": True,
        }

        self._eval_sampling_params = {
            "do_sample": sampling_params["do_sample"],
            "temperature": sampling_params["temperature_eval"],
            "top_k": sampling_params["top_k"],
            "top_p": sampling_params["top_p"],
            "max_new_tokens": length_params["max_new_token"],
            "use_cache": True,
        }

    def start(self):
        """Start background inference thread."""
        if self.inference_thread is not None and self.inference_thread.is_alive():
            raise RuntimeError("Engine already started")

        # A stopped engine should not carry stale requests/sentinels into the next
        # rollout generate() call. If stop() failed to join, the alive-thread guard
        # above prevents replacing the queue and silently leaking a thread.
        self.inference_thread = None
        self.request_queue = queue.Queue()
        self.result_futures.clear()
        self.should_stop = False
        self.inference_thread = threading.Thread(
            target=self._inference_loop,
            daemon=True,
            name="PerEnvDynamicBatching"
        )
        self.inference_thread.start()
        logger.info("PerEnvDynamicBatchingEngine started")

    def stop(self):
        """Stop background inference thread."""
        self.should_stop = True
        if self.inference_thread is not None:
            # Wake the inference loop if it is blocked in Queue.get(timeout=None).
            self.request_queue.put(self._stop_sentinel)
            self.inference_thread.join(timeout=5.0)
            if self.inference_thread.is_alive():
                logger.error(
                    "PerEnvDynamicBatchingEngine failed to stop inference thread; "
                    "keeping thread handle so the next start() fails instead of "
                    "leaking another CUDA inference thread."
                )
            else:
                self.inference_thread = None
        self._clear_pending_requests(RuntimeError("PerEnvDynamicBatchingEngine stopped"))
        logger.info(f"PerEnvDynamicBatchingEngine stopped. Stats: "
                   f"{self.total_requests} requests, {self.total_batches} batches, "
                   f"avg batch size: {self.total_batch_size / max(1, self.total_batches):.1f}")

    def _clear_pending_requests(self, exc: Exception):
        """Release queued requests and complete their futures during shutdown."""
        while True:
            try:
                request = self.request_queue.get_nowait()
            except queue.Empty:
                break
            if request is self._stop_sentinel:
                continue
            future = self.result_futures.pop(request.request_id, None)
            if future is not None and not future.done():
                future.set_exception(exc)

        for future in list(self.result_futures.values()):
            if not future.done():
                future.set_exception(exc)
        self.result_futures.clear()

    def submit_request(self, request: PerEnvInferenceRequest) -> Future:
        """Submit inference request asynchronously."""
        future = Future()
        self.result_futures[request.request_id] = future
        self.request_queue.put(request)
        self.total_requests += 1
        return future

    def _inference_loop(self):
        """Background loop for continuous batch processing."""
        while not self.should_stop:
            batch: List[PerEnvInferenceRequest] = []
            batch_start_time = time.time()

            while len(batch) < self.max_batch_size:
                remaining_time = self.batch_timeout_ms / 1000 - (time.time() - batch_start_time)
                if remaining_time <= 0 and len(batch) > 0:
                    break

                try:
                    timeout = max(0.001, remaining_time) if batch else None
                    request = self.request_queue.get(timeout=timeout)
                    if request is self._stop_sentinel:
                        self.should_stop = True
                        break
                    batch.append(request)
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
                results = self._run_batch_inference(batch)
                self.total_generate_time += time.time() - inference_start

                for request, result in zip(batch, results):
                    future = self.result_futures.pop(request.request_id, None)
                    if future is not None:
                        future.set_result(result)

                self.total_batches += 1
                self.total_batch_size += len(batch)

            except Exception as e:
                logger.error(f"Batch inference error: {e}")
                for request in batch:
                    future = self.result_futures.pop(request.request_id, None)
                    if future is not None:
                        future.set_exception(e)

    def _run_batch_inference(
        self, batch: List[PerEnvInferenceRequest]
    ) -> List[PerEnvInferenceResult]:
        """Run batched inference."""
        train_requests = [r for r in batch if r.mode == "train"]
        eval_requests = [r for r in batch if r.mode == "eval"]

        results = []
        if train_requests:
            results.extend(self._run_batch_inference_mode(train_requests, "train"))
        if eval_requests:
            results.extend(self._run_batch_inference_mode(eval_requests, "eval"))

        request_id_to_result = {r.request_id: r for r in results}
        return [request_id_to_result[req.request_id] for req in batch]

    def _run_batch_inference_mode(
        self, requests: List[PerEnvInferenceRequest], mode: str
    ) -> List[PerEnvInferenceResult]:
        """Run inference for a specific mode."""
        batched_env_output = self._collate_env_outputs([r.env_output for r in requests])

        extracted_obs = self.hf_model.preprocess_env_obs(batched_env_output["obs"])

        dones, rewards, real_extracted_obs = self._get_dones_and_rewards(
            batched_env_output, extracted_obs
        )

        kwargs = self._train_sampling_params if mode == "train" else self._eval_sampling_params

        if SupportedModel(self.cfg.actor.model.model_type) in [
            SupportedModel.OPENPI,
            SupportedModel.MLP_POLICY,
            SupportedModel.GR00T,
            SupportedModel.CNN_POLICY,
        ]:
            kwargs = {"mode": mode}

        kwargs["return_obs"] = not hasattr(self.hf_model, "q_head")

        actions, result = self._predict_action_batch_with_limit(extracted_obs, kwargs)

        return self._split_results(
            requests, actions, result, extracted_obs, dones, rewards, real_extracted_obs
        )

    def _get_batch_size(self, value: Any) -> int:
        if isinstance(value, torch.Tensor):
            return int(value.shape[0])
        if isinstance(value, np.ndarray):
            return int(value.shape[0])
        if isinstance(value, list):
            return len(value)
        if isinstance(value, dict):
            for child in value.values():
                size = self._get_batch_size(child)
                if size > 0:
                    return size
        return 0

    def _slice_batch_value(self, value: Any, start_idx: int, end_idx: int) -> Any:
        if isinstance(value, torch.Tensor):
            return value[start_idx:end_idx]
        if isinstance(value, np.ndarray):
            return value[start_idx:end_idx]
        if isinstance(value, list):
            return value[start_idx:end_idx]
        if isinstance(value, dict):
            return {
                key: self._slice_batch_value(child, start_idx, end_idx)
                for key, child in value.items()
            }
        return value

    def _merge_batch_values(self, values: List[Any]) -> Any:
        if not values:
            return None
        first = values[0]
        if first is None:
            return None
        if isinstance(first, torch.Tensor):
            return torch.cat(values, dim=0)
        if isinstance(first, np.ndarray):
            return np.concatenate(values, axis=0)
        if isinstance(first, dict):
            return {
                key: self._merge_batch_values([value[key] for value in values])
                for key in first.keys()
            }
        if isinstance(first, list):
            merged = []
            for value in values:
                merged.extend(value)
            return merged
        return first

    def _merge_result_dicts(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        if not results:
            return {}
        return {
            key: self._merge_batch_values([result[key] for result in results])
            for key in results[0].keys()
        }

    def _predict_action_batch_with_limit(
        self, env_obs: Dict[str, Any], kwargs: Dict[str, Any]
    ) -> tuple[Any, Dict[str, Any]]:
        batch_size = self._get_batch_size(env_obs)
        max_batch_size = self.max_batch_size
        if (
            max_batch_size is None
            or int(max_batch_size) <= 0
            or batch_size <= int(max_batch_size)
        ):
            with torch.no_grad():
                return self.hf_model.predict_action_batch(env_obs=env_obs, **kwargs)

        all_actions = []
        all_results = []
        for start_idx in range(0, batch_size, int(max_batch_size)):
            end_idx = min(start_idx + int(max_batch_size), batch_size)
            batch_obs = self._slice_batch_value(env_obs, start_idx, end_idx)
            with torch.no_grad():
                actions, result = self.hf_model.predict_action_batch(
                    env_obs=batch_obs,
                    **kwargs,
                )
            all_actions.append(actions)
            all_results.append(result)

        merged_actions = self._merge_batch_values(all_actions)
        merged_result = self._merge_result_dicts(all_results)
        return merged_actions, merged_result

    def _collate_env_outputs(
        self, env_outputs: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Collate multiple env outputs.

        Special handling for rewards: if some requests have None rewards (e.g., initial obs),
        we fill them with zero tensors to maintain batch dimension consistency.
        """
        if len(env_outputs) == 1:
            return env_outputs[0]

        batched = {}
        keys = env_outputs[0].keys()

        for key in keys:
            values = [e[key] for e in env_outputs if e]
            if all(v is None for v in values):
                batched[key] = None
                continue
            # Filter out None values for type checking and merging
            non_none_values = [v for v in values if v is not None]
            if not non_none_values:
                batched[key] = None
            elif isinstance(non_none_values[0], torch.Tensor):
                if len(non_none_values) < len(values):
                    # Mixed None and tensor - need special handling
                    if key == "rewards":
                        # For rewards, fill None with zero tensors to maintain dimensions
                        template = non_none_values[0]
                        filled_values = []
                        for v in values:
                            if v is None:
                                # Create zero tensor with same shape as template
                                filled_values.append(torch.zeros_like(template))
                            else:
                                filled_values.append(v)
                        batched[key] = torch.cat(filled_values, dim=0)
                    else:
                        # For other keys, just concatenate non-None values
                        batched[key] = torch.cat(non_none_values, dim=0)
                else:
                    batched[key] = torch.cat(non_none_values, dim=0)
            elif isinstance(non_none_values[0], np.ndarray):
                batched[key] = np.concatenate(non_none_values, axis=0)
            elif isinstance(non_none_values[0], dict):
                # Only merge non-None dicts
                batched[key] = self._collate_env_outputs(non_none_values)
            elif isinstance(non_none_values[0], list):
                # Concatenate lists (e.g., prompt lists)
                batched[key] = []
                for v in non_none_values:
                    batched[key].extend(v)
            else:
                # For other types (str, int, etc.), keep first value
                batched[key] = non_none_values[0]

        return batched

    def _split_results(
        self,
        requests: List[PerEnvInferenceRequest],
        actions: torch.Tensor,
        result: Dict[str, Any],
        extracted_obs: Dict[str, Any],
        dones: Optional[torch.Tensor],
        rewards: Optional[torch.Tensor],
        real_extracted_obs: Optional[Dict[str, Any]],
    ) -> List[PerEnvInferenceResult]:
        """Split batched results back to individual envs."""
        results = []

        batch_sizes = [
            self._get_batch_size(req.env_output.get("obs", {}))
            for req in requests
        ]

        start_idx = 0
        for req, bsz in zip(requests, batch_sizes):
            end_idx = start_idx + bsz

            req_actions = actions[start_idx:end_idx]

            req_result = {}
            for key, val in result.items():
                req_result[key] = self._slice_batch_value(val, start_idx, end_idx)

            req_extracted_obs = self._split_dict(extracted_obs, start_idx, end_idx)
            req_dones = dones[start_idx:end_idx] if dones is not None else None
            req_rewards = rewards[start_idx:end_idx] if rewards is not None else None

            # Convert filled zeros back to None if original request had None rewards
            # This happens when collate fills None with zeros for dimension consistency
            if req_rewards is not None and req.env_output.get("rewards") is None:
                req_rewards = None

            req_real_obs = None
            if real_extracted_obs is not None:
                req_real_obs = self._split_dict(real_extracted_obs, start_idx, end_idx)

            results.append(PerEnvInferenceResult(
                request_id=req.request_id,
                env_id=req.env_id,
                actions=req_actions,
                result=req_result,
                extracted_obs=req_extracted_obs,
                dones=req_dones,
                rewards=req_rewards,
                real_extracted_obs=req_real_obs,
            ))

            start_idx = end_idx

        return results

    def _split_dict(
        self, d: Dict[str, Any], start: int, end: int
    ) -> Dict[str, Any]:
        """Split nested dict of tensors."""
        result = {}
        for key, val in d.items():
            if isinstance(val, torch.Tensor):
                result[key] = val[start:end]
            elif isinstance(val, np.ndarray):
                result[key] = val[start:end]
            elif isinstance(val, list):
                result[key] = val[start:end]
            elif isinstance(val, dict):
                result[key] = self._split_dict(val, start, end)
            else:
                result[key] = val
        return result

    def _get_dones_and_rewards(
        self, env_output: Dict[str, Any], extracted_obs: Dict[str, Any]
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[Dict]]:
        """Get dones and rewards from env output."""
        real_extracted_obs = None

        if env_output.get("rewards") is None:
            if hasattr(self.hf_model, "q_head"):
                real_extracted_obs = init_real_obs(extracted_obs)
            dones = env_output.get("dones")
            if dones is not None:
                dones = dones.bool().cpu().contiguous()
            return dones, None, real_extracted_obs

        dones = env_output["dones"].bool().cpu().contiguous()
        rewards = env_output["rewards"].cpu().contiguous()

        if dones.any() and self.cfg.env.train.auto_reset:
            if hasattr(self.hf_model, "value_head") or hasattr(self.hf_model, "q_head"):
                final_obs = env_output.get("final_obs")
                if final_obs is not None:
                    final_extracted_obs = self.hf_model.preprocess_env_obs(final_obs)
                    if hasattr(self.hf_model, "q_head"):
                        real_extracted_obs = init_real_obs(final_extracted_obs)

                    kwargs = {"mode": "train", "return_obs": True}
                    actions, result = self._predict_action_batch_with_limit(
                        final_extracted_obs, kwargs
                    )

                    if "prev_values" in result:
                        _final_values = result["prev_values"]
                    else:
                        _final_values = torch.zeros(
                            (dones.shape[0], 1),
                            dtype=rewards.dtype,
                            device=rewards.device,
                        )

                    final_values = torch.zeros_like(_final_values[:, 0])
                    last_step_dones = dones[:, -1]
                    final_values[last_step_dones] = _final_values[:, 0][last_step_dones]
                    rewards[:, -1] += self.cfg.algorithm.gamma * final_values.cpu()

        if real_extracted_obs is None and hasattr(self.hf_model, "q_head"):
            real_extracted_obs = init_real_obs(extracted_obs)

        return dones, rewards, real_extracted_obs


class PerEnvAsyncRolloutWorker(Worker):
    """
    Rollout worker for true per-env async pipeline.

    Works with TruePerEnvAsyncEnvWorker using env_id-based Channel routing.

    Each env has:
    - Its own Channel key for communication
    - Its own handler thread for recv/send
    - Its own rollout buffer
    - No synchronization with other envs

    Configuration:
        rollout.per_env_async.enabled: true
        rollout.per_env_async.max_batch_size: 32
        rollout.per_env_async.batch_timeout_ms: 5.0
    """

    def __init__(self, cfg: DictConfig):
        Worker.__init__(self)
        print(f"[DEBUG] PerEnvAsyncRolloutWorker.__init__ called, class={self.__class__.__name__}")

        self.cfg = cfg
        self.should_stop = False

        self.actor_group_name = cfg.actor.group_name
        self.device = torch.cuda.current_device()

        self.num_pipeline_stages = cfg.rollout.pipeline_stage_num
        self.enable_offload = self.cfg.rollout.get("enable_offload", False)

        self.placement = HybridComponentPlacement(cfg, Cluster())

        actor_world_size = self.placement.get_world_size("actor")
        self.actor_weight_src_rank = self._rank % actor_world_size
        self.behavior_policy_version = 0

        # Flexible env-rollout mapping: compute which env_ids this worker handles
        self.env_world_size = self.placement.get_world_size("env")
        self.rollout_world_size = self.placement.get_world_size("rollout")
        self.total_env_ids = self.env_world_size * self.num_pipeline_stages

        # Per-env async config
        per_env_cfg = cfg.rollout.get("per_env_async", {})
        self.max_batch_size = per_env_cfg.get("max_batch_size", 32)
        self.batch_timeout_ms = per_env_cfg.get("batch_timeout_ms", 5.0)

        self.batching_engine: Optional[PerEnvDynamicBatchingEngine] = None
        self.env_buffers: Dict[int, EnvRolloutBuffer] = {}

        # Thread pool for per-env handlers
        self.handler_pool: Optional[ThreadPoolExecutor] = None

        # Message dispatcher for thread-safe message routing
        self.serialized_receiver: Optional[SerializedChannelReceiver] = None
        self.serialized_sender: Optional[SerializedChannelSender] = None
        self.serialized_actor_sender: Optional[SerializedChannelSender] = None

        # Timing metrics (wall-clock time, not cumulative across threads)
        self._generate_start_time = 0.0
        self._generate_end_time = 0.0
        self._env_handler_wait_times: Dict[int, float] = {}  # per-env wait time
        self._env_wait_lock = threading.Lock()

    def init_worker(self):
        """Initialize worker with model and batching engine."""
        print(f"[DEBUG] PerEnvAsyncRolloutWorker.init_worker() called")
        rollout_model_config = copy.deepcopy(self.cfg.actor.model)
        with open_dict(rollout_model_config):
            rollout_model_config.precision = self.cfg.rollout.model.precision
            rollout_model_config.path = self.cfg.rollout.model.model_path

        self.hf_model = get_model(rollout_model_config)

        if SupportedModel(self.cfg.actor.model.model_type) in [
            SupportedModel.OPENVLA,
            SupportedModel.OPENVLA_OFT,
        ]:
            model_config, input_processor = get_vla_model_config_and_processor(
                self.cfg.actor
            )
            self.hf_model.setup_config_and_processor(
                model_config, self.cfg, input_processor
            )

        self.hf_model.eval()

        # Initialize batching engine
        self.batching_engine = PerEnvDynamicBatchingEngine(
            hf_model=self.hf_model,
            cfg=self.cfg,
            max_batch_size=self.max_batch_size,
            batch_timeout_ms=self.batch_timeout_ms,
            device=self.device,
        )

        # Compute which env_ids this worker handles (flexible mapping)
        self.my_env_ids = self._compute_my_env_ids()

        # Initialize thread pool: one handler per env_id we handle.
        # max_workers must be >= 1 (ThreadPoolExecutor rejects 0). Some ranks may
        # have no env_ids under round-robin (e.g. rollout_world_size > num envs);
        # FlexiblePerEnvAsyncRolloutWorker then replaces this pool after flex_plan.
        self.handler_pool = ThreadPoolExecutor(
            max_workers=max(1, len(self.my_env_ids)),
            thread_name_prefix="PerEnvHandler"
        )

        # Initialize per-env buffers for all env_ids we handle
        for env_id in self.my_env_ids:
            self.env_buffers[env_id] = EnvRolloutBuffer(
                env_id=env_id,
                stage_id=env_id,  # Use env_id as identifier
                rollout_epoch=self.cfg.algorithm.rollout_epoch,
            )

        if self.enable_offload:
            self.offload_model()

        logger.info(f"PerEnvAsyncRolloutWorker initialized: rank={self._rank}, "
                   f"stages={self.num_pipeline_stages}, "
                   f"handling env_ids={self.my_env_ids}, num_handlers={len(self.my_env_ids)}")

    def _compute_my_env_ids(self) -> List[int]:
        """
        Compute which global env_ids this rollout worker handles.

        Uses round-robin distribution: env_id % rollout_world_size == my_rank
        This supports flexible env-rollout mapping where counts can differ.
        """
        return [env_id for env_id in range(self.total_env_ids)
                if env_id % self.rollout_world_size == self._rank]

    def _get_env_channel_key(self, global_env_id: int, mode: str = "train") -> str:
        """
        Get Channel key using global env_id.

        Key format: "perenv_{global_env_id}_{mode}"
        This supports flexible env-rollout worker mapping.
        """
        return f"perenv_{global_env_id}_{mode}"

    def load_checkpoint(self, load_path):
        """Load model checkpoint."""
        model_dict = torch.load(load_path)
        self.hf_model.load_state_dict(model_dict)

    async def sync_model_from_actor(self):
        """Sync model parameters from actor."""
        param_state_dict = await self.recv(
            self.actor_group_name, src_rank=self.actor_weight_src_rank, async_op=True
        ).async_wait()

        policy_version = param_state_dict.pop("__rlinf_policy_version__", None)
        if policy_version is not None:
            self.behavior_policy_version = int(policy_version.detach().cpu().item())
        self.hf_model.load_state_dict(param_state_dict)
        del param_state_dict
        gc.collect()
        torch.cuda.empty_cache()

    async def generate(
        self, input_channel: Channel, output_channel: Channel, actor_channel: Channel
    ):
        """
        Generate rollouts using true per-env async.

        Each env runs independently in its own handler thread.
        Requests are batched dynamically by the engine.

        IMPORTANT: Uses SerializedChannelReceiver to avoid race condition where
        multiple threads calling Channel.get() with different keys can receive
        each other's messages.
        """
        print(f"[DEBUG generate] ENTERED generate(), enable_offload={self.enable_offload}")

        if self.enable_offload:
            self.reload_model()

        print(f"[DEBUG generate] After reload_model, env_buffers keys={list(self.env_buffers.keys())}")

        # Reset timing metrics (wall-clock time)
        self._generate_start_time = time.time()
        self._env_handler_wait_times = {}
        self.batching_engine.total_generate_time = 0.0

        # Reset buffers
        for env_id, buffer in self.env_buffers.items():
            buffer.buffer = EmbodiedRolloutResult(
                rollout_epoch=self.cfg.algorithm.rollout_epoch
            )
            buffer.last_extracted_obs = None
            buffer.last_forward_inputs = None
            buffer.completed_epochs = 0

        # Start batching engine
        self.batching_engine.start()

        # Create serialized channel receiver/senders to avoid thread-unsafe recv/send
        self.serialized_receiver = SerializedChannelReceiver(input_channel)
        self.serialized_sender = SerializedChannelSender(output_channel)
        self.serialized_actor_sender = SerializedChannelSender(actor_channel)

        try:
            pipeline_mode = self.cfg.algorithm.get("pipeline_mode", "sync")

            # Launch independent handler for each env_id we handle
            loop = asyncio.get_event_loop()
            tasks = []
            for env_id in self.my_env_ids:
                task = loop.run_in_executor(
                    self.handler_pool,
                    self._run_env_handler,
                    env_id,
                    output_channel,
                    actor_channel,
                    pipeline_mode,
                )
                tasks.append(task)

            # Wait for all env handlers to complete
            await asyncio.gather(*tasks)

            # Send final batches for sync mode
            if pipeline_mode == "sync":
                for env_id in self.my_env_ids:
                    self._send_rollout_batch(actor_channel, env_id, use_key=False)
            elif pipeline_mode == "async":
                actor_channel.put(
                    item={"__done__": True}, key=f"rollout_{self._rank}", async_op=True
                )

        finally:
            self.batching_engine.stop()

        if self.enable_offload:
            self.offload_model()

        # Return timing metrics (wall-clock time, compatible with MultiStepRolloutWorker)
        # Use max of per-env wait times (parallel threads, so max = actual wall-clock wait)
        max_env_wait = max(self._env_handler_wait_times.values()) if self._env_handler_wait_times else 0.0
        timing_metrics = {
            "env_wait": max_env_wait,
            "generate": self.batching_engine.total_generate_time,
        }
        return timing_metrics

    def _run_env_handler(
        self,
        env_id: int,
        output_channel: Channel,
        actor_channel: Channel,
        pipeline_mode: str,
    ):
        """
        Run the rollout loop for a single environment.

        Args:
            env_id: Global env_id (not stage_id) - supports flexible env-rollout mapping

        与同步版本 (huggingface_worker.py) 对齐的数据流:
        1. 循环 n_chunk_steps 次: recv → infer → record → send
        2. Final Step: recv → record dones/rewards/forward_inputs → Bootstrap infer

        IMPORTANT: Uses self.serialized_receiver.get() instead of input_channel.get()
        to avoid race condition where multiple threads can receive each other's messages.
        """
        env_key = self._get_env_channel_key(env_id, "train")
        buffer = self.env_buffers[env_id]

        # # DEBUG: 检查 rank, env_id, env_key
        # logger.info(f"[DEBUG] RolloutWorker handler start: rank={self._rank}, "
        #            f"env_id={env_id}, env_key={env_key}")

        n_chunk_steps = (
            self.cfg.env.train.max_steps_per_rollout_epoch
            // self.cfg.actor.model.num_action_chunks
        )

        request_counter = 0

        for epoch_idx in range(self.cfg.algorithm.rollout_epoch):
            if pipeline_mode == "async":
                buffer.buffer = EmbodiedRolloutResult(rollout_epoch=1)
                buffer.last_extracted_obs = None
                buffer.last_forward_inputs = None

            last_extracted_obs = None
            last_forward_inputs = None

            # ==========================================================
            # 主循环: 与同步版本对齐 (recv → infer → record → send)
            # ==========================================================
            for step_idx in range(n_chunk_steps):
                # A. 接收 env_output (使用 serialized_receiver 避免竞争条件)
                env_wait_start = time.time()
                env_output = self.serialized_receiver.get(env_key)
                env_wait_elapsed = time.time() - env_wait_start
                with self._env_wait_lock:
                    self._env_handler_wait_times[env_id] = \
                        self._env_handler_wait_times.get(env_id, 0.0) + env_wait_elapsed

                # DEBUG: 检查每步的 rewards
                step_rewards = env_output.get("rewards")
                # if step_idx == 0 or step_idx == n_chunk_steps - 1:
                #     logger.info(f"[DEBUG] Env {env_id} step {step_idx}: rewards is None = {step_rewards is None}, "
                #                f"env_key={env_key}")

                # B. 更新 intervene actions
                if last_forward_inputs is not None:
                    last_forward_inputs = self._update_intervene_actions(
                        env_output, last_forward_inputs
                    )

                # C. 推理
                request_id = f"env{env_id}_ep{epoch_idx}_st{step_idx}_{request_counter}"
                request_counter += 1

                request = PerEnvInferenceRequest(
                    request_id=request_id, env_id=env_id, stage_id=env_id,
                    env_output=env_output, step_idx=step_idx, epoch_idx=epoch_idx, mode="train"
                )
                future = self.batching_engine.submit_request(request)
                result: PerEnvInferenceResult = future.result()

                # D. 从 env_output 提取 dones 和 rewards
                dones = env_output.get("dones")
                rewards = env_output.get("rewards")  # 第一步是 reset，rewards=None

                # E. 记录数据 (与同步版本的 ChunkStepResult 对齐)
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
                    buffer.buffer.forward_inputs.append(put_tensor_device(last_forward_inputs, "cpu"))

                # F. 更新状态
                last_extracted_obs = result.extracted_obs
                last_forward_inputs = result.result.get("forward_inputs")

                # G. 发送动作
                sender = self.serialized_sender
                if sender is not None:
                    sender.put(item=result.actions, key=env_key)
                else:
                    output_channel.put(item=result.actions, key=env_key)

            # ==========================================================
            # Final Step: 与同步版本对齐
            # 接收最后一个 env_output，记录 dones/rewards/forward_inputs
            # ==========================================================
            env_wait_start = time.time()
            env_output = self.serialized_receiver.get(env_key)
            env_wait_elapsed = time.time() - env_wait_start
            with self._env_wait_lock:
                self._env_handler_wait_times[env_id] = \
                    self._env_handler_wait_times.get(env_id, 0.0) + env_wait_elapsed

            # 更新 intervene actions
            if last_forward_inputs is not None:
                last_forward_inputs = self._update_intervene_actions(
                    env_output, last_forward_inputs
                )

            # 记录 Final Step 的数据
            dones = env_output.get("dones")
            rewards = env_output.get("rewards")

            # DEBUG: 检查 Final Step 的 rewards
            # logger.info(f"[DEBUG] Env {env_id} Final Step: rewards is None = {rewards is None}, "
            #            f"current rewards count = {len(buffer.buffer.rewards)}")

            if dones is not None:
                buffer.buffer.dones.append(dones.cpu().contiguous())
                buffer.buffer.truncations.append(env_output.get("truncations").cpu().contiguous())
                buffer.buffer.terminations.append(env_output.get("terminations").cpu().contiguous())
            if rewards is not None:
                buffer.buffer.rewards.append(rewards.cpu().contiguous())
            if last_forward_inputs is not None:
                buffer.buffer.forward_inputs.append(put_tensor_device(last_forward_inputs, "cpu"))

            # DEBUG: 检查记录后的状态
            # logger.info(f"[DEBUG] Env {env_id} after Final Step: "
            #            f"dones={len(buffer.buffer.dones)}, rewards={len(buffer.buffer.rewards)}, "
            #            f"forward_inputs={len(buffer.buffer.forward_inputs)}")

            # Bootstrap: 计算最后观测的 Value
            request_id = f"env{env_id}_ep{epoch_idx}_final_{request_counter}"
            request_counter += 1

            request = PerEnvInferenceRequest(
                request_id=request_id, env_id=env_id, stage_id=env_id,
                env_output=env_output, step_idx=n_chunk_steps,
                epoch_idx=epoch_idx, mode="train", is_final_step=True
            )
            future = self.batching_engine.submit_request(request)
            final_res: PerEnvInferenceResult = future.result()

            if "prev_values" in final_res.result:
                buffer.buffer.prev_values.append(final_res.result["prev_values"].cpu().contiguous())

            # q_head 转换
            if hasattr(self.hf_model, "q_head"):
                buffer.buffer.add_transition(last_extracted_obs, final_res.real_extracted_obs)

            buffer.completed_epochs += 1

            if pipeline_mode == "async":
                self._send_rollout_batch(actor_channel, env_id)

            # Send epoch completion acknowledgment to EnvWorker
            # This ensures EnvWorker waits for RolloutWorker to finish processing
            # the current epoch before starting the next one, preventing message misalignment
            # This is needed in BOTH sync and async modes to prevent message flow issues
            if epoch_idx < self.cfg.algorithm.rollout_epoch - 1:
                logger.info(f"[DEBUG RolloutWorker] Env {env_id} epoch {epoch_idx}: Completed, sending ack")
                sender = self.serialized_sender
                if sender is not None:
                    sender.put(item={"__epoch_done__": True}, key=env_key)
                else:
                    output_channel.put(item={"__epoch_done__": True}, key=env_key)
                logger.info(f"[DEBUG RolloutWorker] Env {env_id} epoch {epoch_idx}: Ack sent")

        # logger.debug(f"Env {env_id} rollout completed.")

    def _update_intervene_actions(self, env_output, forward_inputs):
        """Update forward inputs with intervene actions."""
        intervene_actions = env_output.get("intervene_actions")
        intervene_flags = env_output.get("intervene_flags")

        if intervene_actions is not None and forward_inputs is not None:
            if "action" in forward_inputs:
                policy_action = forward_inputs["action"].to(intervene_actions.device)
                policy_action = policy_action.reshape(
                    policy_action.shape[0], self.hf_model.num_action_chunks, -1
                )
                intervene_actions = intervene_actions.reshape(
                    intervene_actions.shape[0], self.hf_model.num_action_chunks, -1
                )
                action = intervene_actions * intervene_flags[..., None] + \
                         policy_action * (~intervene_flags[..., None])
                action = action.reshape(action.shape[0], -1)
                forward_inputs["action"] = action

        return forward_inputs

    def _send_rollout_batch(
        self, channel: Channel, env_id: int, use_key: bool = True
    ):
        """
        Send rollout batch to actor workers.

        This method matches the original MultiStepRolloutWorker.send_rollout_batch
        to ensure compatibility with actor's recv_rollout_batch.

        Args:
            channel: Channel to send data to actors
            env_id: Global env_id (supports flexible env-rollout mapping)
            use_key: Whether to use deterministic key routing (for async mode)
                     If False, send without key (for sync mode, faster)
        """
        buffer = self.env_buffers[env_id].buffer

        split_num = self.get_actor_split_num()
        splitted_rollout_result = buffer.to_splitted_dict(split_num)
        for item in splitted_rollout_result:
            item["__behavior_policy_version__"] = self.behavior_policy_version

        sender = self.serialized_actor_sender
        if use_key:
            # Async mode: use deterministic key routing based on env_id
            actor_world_size = self.placement.get_world_size("actor")
            for i in range(split_num):
                # Use env_id for deterministic routing
                global_msg_id = env_id * split_num + i
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
            for i in range(split_num):
                if sender is not None:
                    sender.put(item=splitted_rollout_result[i], key="default_queue", async_op=True)
                else:
                    channel.put(item=splitted_rollout_result[i], async_op=True)

    def get_actor_split_num(self):
        """Get the number of splits for sending to actors."""
        # Use total_env_ids for flexible env-rollout mapping
        send_num = self.total_env_ids
        recv_num = self.placement.get_world_size("actor")
        split_num = compute_split_num(recv_num, send_num)
        return split_num

    def offload_model(self):
        """Offload model to CPU."""
        self.hf_model.to("cpu")
        torch.cuda.empty_cache()

    def reload_model(self):
        """Reload model to GPU."""
        self.hf_model.to(self.device)

    def set_global_step(self, global_step):
        """Set global step for model (if supported)."""
        if hasattr(self.hf_model, "set_global_step"):
            self.hf_model.set_global_step(global_step)

    async def evaluate(self, input_channel: Channel, output_channel: Channel):
        """
        Evaluate using per-env async mode.

        This method matches the original MultiStepRolloutWorker.evaluate
        to ensure compatibility with the runner.
        """
        if self.enable_offload:
            self.reload_model()

        n_chunk_steps = (
            self.cfg.env.eval.max_steps_per_rollout_epoch
            // self.cfg.actor.model.num_action_chunks
        )

        # Start batching engine for eval
        self.batching_engine.start()

        try:
            loop = asyncio.get_event_loop()

            # Launch independent eval handlers for each env_id we handle
            tasks = []
            for env_id in self.my_env_ids:
                task = loop.run_in_executor(
                    self.handler_pool,
                    self._run_env_eval_handler,
                    env_id,
                    input_channel,
                    output_channel,
                    n_chunk_steps,
                )
                tasks.append(task)

            # Wait for all eval handlers to complete
            await asyncio.gather(*tasks)

        finally:
            self.batching_engine.stop()

        if self.enable_offload:
            self.offload_model()

    def _run_env_eval_handler(
        self,
        env_id: int,
        input_channel: Channel,
        output_channel: Channel,
        n_chunk_steps: int,
    ):
        """
        Handler for a single environment's evaluation.

        Args:
            env_id: Global env_id (supports flexible env-rollout mapping)

        Runs independently in its own thread.
        """
        env_key = self._get_env_channel_key(env_id, "eval")

        for epoch_idx in range(self.cfg.algorithm.eval_rollout_epoch):
            for step_idx in range(n_chunk_steps):
                # Receive env output
                env_output = input_channel.get(key=env_key)

                # Preprocess and predict
                extracted_obs = self.hf_model.preprocess_env_obs(env_output["obs"])

                # Create request for batching engine
                request_id = f"eval_env{env_id}_epoch{epoch_idx}_step{step_idx}"
                request = PerEnvInferenceRequest(
                    request_id=request_id,
                    env_id=env_id,
                    stage_id=env_id,
                    env_output=env_output,
                    step_idx=step_idx,
                    epoch_idx=epoch_idx,
                    mode="eval",
                    is_final_step=False,
                )

                future = self.batching_engine.submit_request(request)
                result: PerEnvInferenceResult = future.result()

                # Send actions back
                output_channel.put(item=result.actions, key=env_key)

    def __del__(self):
        """Cleanup resources."""
        if getattr(self, 'handler_pool', None):
            self.handler_pool.shutdown(wait=False)