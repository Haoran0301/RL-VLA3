import asyncio
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from typing import List, Optional

from omegaconf import DictConfig

from rlinf.data.io_struct import EmbodiedRolloutResult
from rlinf.scheduler import Channel
from rlinf.workers.rollout.hf.per_env_async_rollout_worker import (
    EnvRolloutBuffer,
    PerEnvAsyncRolloutWorker,
    PerEnvInferenceRequest,
    PerEnvInferenceResult,
    SerializedChannelReceiver,
)

logger = logging.getLogger(__name__)


class AggregatedPerEnvAsyncRolloutWorker(PerEnvAsyncRolloutWorker):
    """
    Experimental slot-based rollout worker for aggregated env stepping.

    It keeps DynamicBatchingEngine unchanged, but switches communication unit from
    env_id to slot_id where:
      slot_id = global_env_id * aggregate_slots_per_env + slot_idx
    """

    def __init__(self, cfg: DictConfig):
        super().__init__(cfg)
        per_env_cfg = cfg.rollout.get("per_env_async", {})
        self.aggregate_slots_per_env = int(per_env_cfg.get("aggregate_slots_per_env", 1))
        if self.aggregate_slots_per_env < 1:
            raise ValueError("aggregate_slots_per_env must be >= 1")
        self.total_slot_ids = self.total_env_ids * self.aggregate_slots_per_env
        self.my_slot_ids: List[int] = []

    def _get_slot_channel_key(self, slot_id: int, mode: str = "train") -> str:
        return f"perenv_slot_{slot_id}_{mode}"

    def _compute_my_slot_ids(self) -> List[int]:
        return [
            slot_id
            for slot_id in range(self.total_slot_ids)
            if slot_id % self.rollout_world_size == self._rank
        ]

    def init_worker(self):
        super().init_worker()
        if self.aggregate_slots_per_env == 1:
            self.my_slot_ids = self.my_env_ids
            logger.info("AggregatedPerEnvAsyncRolloutWorker runs in compatibility mode (slots=1).")
            return

        self.my_slot_ids = self._compute_my_slot_ids()

        if self.handler_pool is not None:
            self.handler_pool.shutdown(wait=False)
        self.handler_pool = ThreadPoolExecutor(
            max_workers=max(1, len(self.my_slot_ids)),
            thread_name_prefix="PerSlotHandler",
        )

        self.env_buffers = {}
        for slot_id in self.my_slot_ids:
            self.env_buffers[slot_id] = EnvRolloutBuffer(
                env_id=slot_id,
                stage_id=slot_id,
                rollout_epoch=self.cfg.algorithm.rollout_epoch,
            )

        logger.info(
            "AggregatedPerEnvAsyncRolloutWorker initialized: "
            f"rank={self._rank}, slots_per_env={self.aggregate_slots_per_env}, "
            f"handling slot_ids={self.my_slot_ids}"
        )

    def get_actor_split_num(self):
        send_num = self.total_slot_ids if self.aggregate_slots_per_env > 1 else self.total_env_ids
        recv_num = self.placement.get_world_size("actor")
        from rlinf.utils.metric_utils import compute_split_num

        return compute_split_num(recv_num, send_num)

    async def generate(
        self, input_channel: Channel, output_channel: Channel, actor_channel: Channel
    ):
        if self.aggregate_slots_per_env == 1:
            return await super().generate(input_channel, output_channel, actor_channel)

        if self.enable_offload:
            self.reload_model()

        self._generate_start_time = time.time()
        self._env_handler_wait_times = {}
        self.batching_engine.total_generate_time = 0.0

        for slot_id, buffer in self.env_buffers.items():
            buffer.buffer = EmbodiedRolloutResult(rollout_epoch=self.cfg.algorithm.rollout_epoch)
            buffer.last_extracted_obs = None
            buffer.last_forward_inputs = None
            buffer.completed_epochs = 0

        self.batching_engine.start()
        self.serialized_receiver = SerializedChannelReceiver(input_channel)

        try:
            pipeline_mode = self.cfg.algorithm.get("pipeline_mode", "sync")
            loop = asyncio.get_event_loop()
            tasks = []
            for slot_id in self.my_slot_ids:
                task = loop.run_in_executor(
                    self.handler_pool,
                    self._run_slot_handler,
                    slot_id,
                    output_channel,
                    actor_channel,
                    pipeline_mode,
                )
                tasks.append(task)

            await asyncio.gather(*tasks)

            if pipeline_mode == "sync":
                for slot_id in self.my_slot_ids:
                    self._send_rollout_batch(actor_channel, slot_id, use_key=False)
            elif pipeline_mode == "async":
                actor_channel.put(
                    item={"__done__": True}, key=f"rollout_{self._rank}", async_op=True
                )
        finally:
            self.batching_engine.stop()

        if self.enable_offload:
            self.offload_model()

        max_env_wait = max(self._env_handler_wait_times.values()) if self._env_handler_wait_times else 0.0
        return {
            "env_wait": max_env_wait,
            "generate": self.batching_engine.total_generate_time,
        }

    def _run_slot_handler(
        self,
        slot_id: int,
        output_channel: Channel,
        actor_channel: Channel,
        pipeline_mode: str,
    ):
        env_key = self._get_slot_channel_key(slot_id, "train")
        buffer = self.env_buffers[slot_id]
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

            for step_idx in range(n_chunk_steps):
                env_wait_start = time.time()
                env_output = self.serialized_receiver.get(env_key)
                env_wait_elapsed = time.time() - env_wait_start
                with self._env_wait_lock:
                    self._env_handler_wait_times[slot_id] = (
                        self._env_handler_wait_times.get(slot_id, 0.0) + env_wait_elapsed
                    )

                if last_forward_inputs is not None:
                    last_forward_inputs = self._update_intervene_actions(
                        env_output, last_forward_inputs
                    )

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
                    from rlinf.utils.nested_dict_process import put_tensor_device

                    buffer.buffer.forward_inputs.append(put_tensor_device(last_forward_inputs, "cpu"))

                last_extracted_obs = result.extracted_obs
                last_forward_inputs = result.result.get("forward_inputs")
                output_channel.put(item=result.actions, key=env_key)

            env_wait_start = time.time()
            env_output = self.serialized_receiver.get(env_key)
            env_wait_elapsed = time.time() - env_wait_start
            with self._env_wait_lock:
                self._env_handler_wait_times[slot_id] = (
                    self._env_handler_wait_times.get(slot_id, 0.0) + env_wait_elapsed
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
                from rlinf.utils.nested_dict_process import put_tensor_device

                buffer.buffer.forward_inputs.append(put_tensor_device(last_forward_inputs, "cpu"))

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
            future = self.batching_engine.submit_request(request)
            final_res: PerEnvInferenceResult = future.result()
            if "prev_values" in final_res.result:
                buffer.buffer.prev_values.append(final_res.result["prev_values"].cpu().contiguous())

            if hasattr(self.hf_model, "q_head"):
                buffer.buffer.add_transition(last_extracted_obs, final_res.real_extracted_obs)

            buffer.completed_epochs += 1

            if pipeline_mode == "async":
                self._send_rollout_batch(actor_channel, slot_id)

            if epoch_idx < self.cfg.algorithm.rollout_epoch - 1:
                output_channel.put(item={"__epoch_done__": True}, key=env_key)

    async def evaluate(self, input_channel: Channel, output_channel: Channel):
        if self.aggregate_slots_per_env > 1:
            raise NotImplementedError(
                "AggregatedPerEnvAsyncRolloutWorker currently supports training generate only. "
                "Please set runner.val_check_interval=0 for this experimental mode."
            )
        return await super().evaluate(input_channel, output_channel)
