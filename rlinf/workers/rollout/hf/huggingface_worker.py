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

import copy
import gc
from typing import Any

import numpy as np
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


class MultiStepRolloutWorker(Worker):
    def __init__(self, cfg: DictConfig):
        Worker.__init__(self)

        self.cfg = cfg
        self.should_stop = False

        self.actor_group_name = cfg.actor.group_name
        self.device = torch.cuda.current_device()

        self.num_pipeline_stages = cfg.rollout.pipeline_stage_num
        self.enable_offload = self.cfg.rollout.get("enable_offload", False)
        self.max_batch_size = self.cfg.rollout.get("max_batch_size", None)
        self.behavior_policy_version = 0

        self.placement = HybridComponentPlacement(cfg, Cluster())

        actor_world_size = self.placement.get_world_size("actor")
        self.actor_weight_src_rank = self._rank % actor_world_size

        # Calculate scatter_num for env-rollout communication
        rollout_world_size = self.placement.get_world_size("rollout")
        env_world_size = self.placement.get_world_size("env")
        if env_world_size > rollout_world_size:
            assert env_world_size % rollout_world_size == 0
            self.scatter_num = env_world_size // rollout_world_size
        else:
            self.scatter_num = 1

    def init_worker(self):
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

        self.setup_sample_params()
        if self.enable_offload:
            self.offload_model()

    def load_checkpoint(self, load_path):
        model_dict = torch.load(load_path)
        self.hf_model.load_state_dict(model_dict)

    def setup_sample_params(self):
        # length parameters for rollout
        self._length_params = OmegaConf.to_container(
            self.cfg.algorithm.length_params, resolve=True
        )
        # sampling parameters for rollout
        self._sampling_params = OmegaConf.to_container(
            self.cfg.algorithm.sampling_params, resolve=True
        )
        self._train_sampling_params = {
            "do_sample": self._sampling_params["do_sample"],
            "temperature": self._sampling_params["temperature_train"],
            "top_k": self._sampling_params["top_k"],
            "top_p": self._sampling_params["top_p"],
            "max_new_tokens": self._length_params["max_new_token"],
            "use_cache": True,
        }

        self._eval_sampling_params = {
            "do_sample": self._sampling_params["do_sample"],
            "temperature": self._sampling_params["temperature_eval"],
            "top_k": self._sampling_params["top_k"],
            "top_p": self._sampling_params["top_p"],
            "max_new_tokens": self._length_params["max_new_token"],
        }

    def predict(self, env_obs, mode="train"):
        kwargs = (
            self._train_sampling_params
            if mode == "train"
            else self._eval_sampling_params
        )

        if SupportedModel(self.cfg.actor.model.model_type) in [
            SupportedModel.OPENPI,
            SupportedModel.MLP_POLICY,
            SupportedModel.GR00T,
            SupportedModel.CNN_POLICY,
        ]:
            kwargs = {"mode": mode}

        kwargs["return_obs"] = not hasattr(self.hf_model, "q_head")

        # Get batch size from env_obs
        batch_size = self._get_batch_size(env_obs)

        # Check if we need to split into smaller batches
        if self.max_batch_size is not None and batch_size > self.max_batch_size:
            return self._predict_in_batches(env_obs, batch_size, kwargs)

        with torch.no_grad():
            actions, result = self.hf_model.predict_action_batch(
                env_obs=env_obs,
                **kwargs,
            )

        return actions, result

    def _get_batch_size(self, env_obs):
        """Get batch size from env_obs dict."""
        for key, value in env_obs.items():
            if isinstance(value, torch.Tensor):
                return value.shape[0]
            elif isinstance(value, list):
                return len(value)
            elif isinstance(value, dict):
                return self._get_batch_size(value)
        return 0

    def _split_env_obs(self, env_obs, start_idx, end_idx):
        """Split env_obs dict by batch indices."""
        split_obs = {}
        for key, value in env_obs.items():
            if isinstance(value, torch.Tensor):
                split_obs[key] = value[start_idx:end_idx]
            elif isinstance(value, list):
                split_obs[key] = value[start_idx:end_idx]
            elif isinstance(value, dict):
                split_obs[key] = self._split_env_obs(value, start_idx, end_idx)
            else:
                split_obs[key] = value
        return split_obs

    def _merge_results(self, results_list):
        """Merge list of result dicts into one."""
        if not results_list:
            return {}
        merged = {}
        for key in results_list[0].keys():
            values = [r[key] for r in results_list]
            if values[0] is None:
                merged[key] = None
            elif isinstance(values[0], torch.Tensor):
                merged[key] = torch.cat(values, dim=0)
            elif isinstance(values[0], np.ndarray):
                merged[key] = np.concatenate(values, axis=0)
            elif isinstance(values[0], dict):
                merged[key] = self._merge_results(values)
            else:
                merged[key] = values[0]
        return merged

    def _predict_in_batches(self, env_obs, batch_size, kwargs):
        """Split large batch and predict in smaller batches."""
        all_actions = []
        all_results = []

        for start_idx in range(0, batch_size, self.max_batch_size):
            end_idx = min(start_idx + self.max_batch_size, batch_size)
            batch_obs = self._split_env_obs(env_obs, start_idx, end_idx)

            with torch.no_grad():
                actions, result = self.hf_model.predict_action_batch(
                    env_obs=batch_obs,
                    **kwargs,
                )
            all_actions.append(actions)
            all_results.append(result)

        # Merge actions
        if isinstance(all_actions[0], np.ndarray):
            merged_actions = np.concatenate(all_actions, axis=0)
        else:
            merged_actions = torch.cat(all_actions, dim=0)

        # Merge results
        merged_result = self._merge_results(all_results)

        return merged_actions, merged_result

    def get_dones_and_rewards(
        self, env_output: dict[str, torch.Tensor], extracted_obs: dict[str, Any]
    ) -> tuple[torch.Tensor | None, torch.Tensor | None, dict[str, Any] | None]:
        """
        Get dones and rewards from environment batch, handling auto_reset if needed.

        Args:
            env_output: Environment batch containing dones, rewards, and optionally final_obs

        Returns:
            Tuple of (dones, rewards, real_extracted_obs). dones and rewards are tensors.
        """
        # First step: no rewards yet, only dones
        real_extracted_obs = None
        if env_output["rewards"] is None:
            if hasattr(self.hf_model, "q_head"):
                real_extracted_obs = init_real_obs(extracted_obs)
            return (
                env_output["dones"].bool().cpu().contiguous(),
                None,
                real_extracted_obs,
            )

        dones = env_output["dones"].bool().cpu().contiguous()
        rewards = env_output["rewards"].cpu().contiguous()

        # Handle auto_reset: add bootstrap value to rewards for done episodes
        # Note: currently this is not correct for chunk-size>1 with partial reset
        if dones.any() and self.cfg.env.train.auto_reset:
            if hasattr(self.hf_model, "value_head") or hasattr(self.hf_model, "q_head"):
                final_obs = env_output["final_obs"]
                with torch.no_grad():
                    final_extracted_obs = self.hf_model.preprocess_env_obs(final_obs)
                    if hasattr(self.hf_model, "q_head"):
                        real_extracted_obs = init_real_obs(final_extracted_obs)
                    actions, result = self.predict(final_extracted_obs)
                    if "prev_values" in result:
                        _final_values = result["prev_values"]
                    else:
                        _final_values = torch.zeros_like(actions[:, 0])
                final_values = torch.zeros_like(_final_values[:, 0])  # [bsz, ]
                last_step_dones = dones[:, -1]  # [bsz, ]

                final_values[last_step_dones] = _final_values[:, 0][last_step_dones]

                # Add bootstrap value to the last step of done episodes
                rewards[:, -1] += self.cfg.algorithm.gamma * final_values.cpu()

        if real_extracted_obs is None and hasattr(self.hf_model, "q_head"):
            real_extracted_obs = init_real_obs(extracted_obs)
        return dones, rewards, real_extracted_obs

    async def sync_model_from_actor(self):
        """Sync model parameters from the actor worker."""
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

    def update_intervene_actions(self, env_output, forward_inputs):
        intervene_actions = env_output["intervene_actions"]
        intervene_flags = env_output["intervene_flags"]
        if intervene_actions is not None:
            if "action" in forward_inputs:
                policy_action = forward_inputs["action"].to(intervene_actions.device)
                policy_action = policy_action.reshape(
                    policy_action.shape[0], self.hf_model.num_action_chunks, -1
                )
                intervene_actions = intervene_actions.reshape(
                    intervene_actions.shape[0], self.hf_model.num_action_chunks, -1
                )
                action = intervene_actions * intervene_flags[
                    ..., None
                ] + policy_action * (~intervene_flags[..., None])
                action = action.reshape(action.shape[0], -1)
                forward_inputs["action"] = action
            else:
                raise NotImplementedError(f"{forward_inputs.keys()=}")
        return forward_inputs

    async def generate(
        self, input_channel: Channel, output_channel: Channel, actor_channel: Channel
    ):
        if self.enable_offload:
            self.reload_model()
        pipeline_mode = self.cfg.algorithm.get("pipeline_mode", "sync")
        if pipeline_mode == "sync":
            self.buffer_list = [
            EmbodiedRolloutResult(rollout_epoch=self.cfg.algorithm.rollout_epoch)
            for _ in range(self.num_pipeline_stages)
        ]

        n_chunk_steps = (
            self.cfg.env.train.max_steps_per_rollout_epoch
            // self.cfg.actor.model.num_action_chunks
        )

        for _ in tqdm(
            range(self.cfg.algorithm.rollout_epoch),
            desc="Generating Rollout Epochs",
            disable=(self._rank != 0),
        ):
            if pipeline_mode == "async":
                self.buffer_list = [
                        EmbodiedRolloutResult(rollout_epoch=1)
                        for _ in range(self.num_pipeline_stages)
                        ]
            last_extracted_obs = [None for i in range(self.num_pipeline_stages)]
            last_forward_inputs = [
                None for i in range(self.num_pipeline_stages)
            ]  # save actions

            for _ in range(n_chunk_steps):
                for stage_id in range(self.num_pipeline_stages):
                    with self.worker_timer("env_wait"):
                        env_output = await self.recv_env_output(input_channel)

                    if last_forward_inputs[stage_id] is not None:
                        last_forward_inputs[stage_id] = self.update_intervene_actions(
                            env_output, last_forward_inputs[stage_id]
                        )

                    extracted_obs = self.hf_model.preprocess_env_obs(env_output["obs"])
                    dones, rewards, real_extracted_obs = self.get_dones_and_rewards(
                        env_output, extracted_obs
                    )
                    with self.worker_timer("generate"):
                        actions, result = self.predict(extracted_obs)
                    chunk_step_result = ChunkStepResult(
                        prev_logprobs=result["prev_logprobs"],
                        prev_values=result["prev_values"],
                        dones=dones,
                        truncations=env_output["truncations"],
                        terminations=env_output["terminations"],
                        rewards=rewards,  # the first step is reset step, reward is none, which will not be appended to the buffer
                        forward_inputs=last_forward_inputs[stage_id],
                    )
                    self.buffer_list[stage_id].append_result(chunk_step_result)
                    if last_extracted_obs[stage_id] is not None and hasattr(
                        self.hf_model, "q_head"
                    ):
                        self.buffer_list[stage_id].add_transition(
                            last_extracted_obs[stage_id], real_extracted_obs
                        )
                    last_extracted_obs[stage_id] = extracted_obs
                    last_forward_inputs[stage_id] = result["forward_inputs"]

                    self.send_chunk_actions(output_channel, actions)

            for stage_id in range(self.num_pipeline_stages):
                with self.worker_timer("env_wait"):
                    env_output = await self.recv_env_output(input_channel)
                last_forward_inputs[stage_id] = self.update_intervene_actions(
                    env_output, last_forward_inputs[stage_id]
                )

                extracted_obs = self.hf_model.preprocess_env_obs(env_output["obs"])
                # Get dones and rewards from environment batch (final step of epoch)
                dones, rewards, real_extracted_obs = self.get_dones_and_rewards(
                    env_output, extracted_obs
                )
                self.buffer_list[stage_id].dones.append(dones)
                self.buffer_list[stage_id].truncations.append(env_output["truncations"])
                self.buffer_list[stage_id].terminations.append(
                    env_output["terminations"]
                )
                self.buffer_list[stage_id].rewards.append(rewards)
                self.buffer_list[stage_id].forward_inputs.append(
                    put_tensor_device(last_forward_inputs[stage_id], "cpu")
                )

                with self.worker_timer("generate"):
                    actions, result = self.predict(extracted_obs)
                # For the final step, we only need prev_values for bootstrapping
                # This is a special case that doesn't create a full ChunkStepResult
                if "prev_values" in result:
                    self.buffer_list[stage_id].prev_values.append(
                        result["prev_values"].cpu().contiguous()
                    )
                if hasattr(self.hf_model, "q_head"):
                    self.buffer_list[stage_id].add_transition(
                        last_extracted_obs[stage_id], real_extracted_obs
                    )
            if pipeline_mode == "async":
                for i in range(self.num_pipeline_stages):
                    self.send_rollout_batch(actor_channel, i)
        # Send rollout batch based on pipeline mode
        if pipeline_mode == "sync":
            # Sync mode: send all data at once after all epochs complete
            for i in range(self.num_pipeline_stages):
                self.send_rollout_batch(actor_channel, i, use_key=False)
        elif pipeline_mode == "async":
            actor_channel.put(item={"__done__": True}, key=f"rollout_{self._rank}", async_op=True)
        if self.enable_offload:
            self.offload_model()

        # Return timing metrics
        timing_metrics = {}
        if "env_wait" in self._timer_metrics:
            timing_metrics["env_wait"] = self._timer_metrics.pop("env_wait")
        if "generate" in self._timer_metrics:
            timing_metrics["generate"] = self._timer_metrics.pop("generate")
        return timing_metrics
    async def evaluate(self, input_channel: Channel, output_channel: Channel):
        if self.enable_offload:
            self.reload_model()

        n_chunk_steps = (
            self.cfg.env.eval.max_steps_per_rollout_epoch
            // self.cfg.actor.model.num_action_chunks
        )
        for _ in tqdm(
            range(self.cfg.algorithm.eval_rollout_epoch),
            desc="Evaluating Rollout Epochs",
            disable=(self._rank != 0),
        ):
            for _ in range(n_chunk_steps):
                for _ in range(self.num_pipeline_stages):
                    env_output = await self.recv_env_output(input_channel, mode="eval")
                    extracted_obs = self.hf_model.preprocess_env_obs(env_output["obs"])
                    actions, _ = self.predict(extracted_obs, mode="eval")
                    self.send_chunk_actions(output_channel, actions, mode="eval")

        if self.enable_offload:
            self.offload_model()

    def offload_model(self):
        self.hf_model = self.hf_model.to("cpu")
        gc.collect()
        torch.cuda.empty_cache()

    def reload_model(self):
        self.hf_model = self.hf_model.to(self.device)

    async def recv_env_output(
        self, input_channel: Channel, mode="train"
    ) -> dict[str, torch.Tensor]:
        assert mode in ["train", "eval"], f"{mode=} is not supported"
        if self.scatter_num == 1:
            # Original case: one env per rollout
            env_output = await input_channel.get(
                key=f"{self._rank}_{mode}", async_op=True
            ).async_wait()
        else:
            # New case: multiple envs per rollout, need to gather and merge
            env_outputs = []
            for env_local_id in range(self.scatter_num):
                env_out = await input_channel.get(
                    key=f"{self._rank}_{env_local_id}_{mode}", async_op=True
                ).async_wait()
                env_outputs.append(env_out)
            env_output = self._merge_env_outputs(env_outputs)
        return env_output

    def _merge_env_outputs(self, env_outputs: list[dict]) -> dict:
        """Merge multiple env outputs into one by concatenating tensors."""
        if not env_outputs:
            return {}
        merged = {}
        for key in env_outputs[0].keys():
            values = [out[key] for out in env_outputs]
            # Skip if all values are None
            if all(v is None for v in values):
                merged[key] = None
                continue
            # Filter out None values for merging
            non_none_values = [v for v in values if v is not None]
            if not non_none_values:
                merged[key] = None
            elif isinstance(non_none_values[0], torch.Tensor):
                merged[key] = torch.cat(non_none_values, dim=0)
            elif isinstance(non_none_values[0], list):
                merged[key] = sum(non_none_values, [])
            elif isinstance(non_none_values[0], dict):
                # Only merge non-None dicts
                merged[key] = self._merge_env_outputs(non_none_values)
            else:
                merged[key] = non_none_values[0]
        return merged

    def send_chunk_actions(self, output_channel: Channel, chunk_actions, mode="train"):
        assert mode in ["train", "eval"], f"{mode=} is not supported"
        if self.scatter_num == 1:
            # Original case: one env per rollout
            output_channel.put(
                item=chunk_actions, key=f"{self._rank}_{mode}", async_op=True
            )
        else:
            # New case: multiple envs per rollout, split and send to each env
            import numpy as np
            action_splits = np.array_split(chunk_actions, self.scatter_num)
            for env_local_id in range(self.scatter_num):
                output_channel.put(
                    item=action_splits[env_local_id],
                    key=f"{self._rank}_{env_local_id}_{mode}",
                    async_op=True,
                )

    def send_rollout_batch(
        self, actor_channel: Channel, stage_id: int, use_key: bool = True
    ):
        """
        Send rollout batch to actor workers.

        Args:
            actor_channel: Channel to send data to actors
            stage_id: Pipeline stage ID
            use_key: Whether to use deterministic key routing (for async mode)
                     If False, send without key (for sync mode, faster)
        """
        split_num = self.get_actor_split_num()
        splitted_rollout_result = self.buffer_list[stage_id].to_splitted_dict(split_num)
        for item in splitted_rollout_result:
            item["__behavior_policy_version__"] = self.behavior_policy_version

        if use_key:
            # Async mode: use deterministic key routing to prevent message competition
            actor_world_size = self.placement.get_world_size("actor")
            for i in range(split_num):
                global_msg_id = self._rank * split_num + i
                target_actor = global_msg_id % actor_world_size
                actor_channel.put(
                    item=splitted_rollout_result[i],
                    key=f"actor_{target_actor}",
                    async_op=True,
                )
        else:
            # Sync mode: send without key (faster, no message competition in sync mode)
            for i in range(split_num):
                actor_channel.put(item=splitted_rollout_result[i], async_op=True)
    def get_actor_split_num(self):
        send_num = self.placement.get_world_size("rollout") * self.num_pipeline_stages
        recv_num = self.placement.get_world_size("actor")
        split_num = compute_split_num(recv_num, send_num)
        return split_num

    def set_global_step(self, global_step):
        if hasattr(self.hf_model, "set_global_step"):
            self.hf_model.set_global_step(global_step)
