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

import asyncio
import logging
import os
from functools import partial

import numpy as np
import torch
from omegaconf import DictConfig
from torch import nn
from torch.distributed.tensor import DTensor
from torch.multiprocessing.reductions import reduce_tensor

import rlinf.algorithms  # noqa: F401
from rlinf.algorithms.registry import calculate_adv_and_returns, policy_loss
from rlinf.algorithms.utils import (
    kl_penalty,
)
from rlinf.config import SupportedModel
from rlinf.data.io_struct import BatchResizingIterator, RolloutResult
from rlinf.hybrid_engines.fsdp.fsdp_model_manager import (
    FSDPModelManager,
)
from rlinf.models import get_model
from rlinf.scheduler import Channel, Cluster, Worker
from rlinf.utils.data_iter_utils import get_iterator_k_split
from rlinf.utils.distributed import all_reduce_dict, masked_normalization
from rlinf.utils.distributed import (
    compute_rollout_metrics as compute_math_rollout_metrics,
)
from rlinf.utils.metric_utils import (
    append_to_dict,
    compute_loss_mask,
    compute_rollout_metrics,
    compute_split_num,
)
from rlinf.utils.nested_dict_process import (
    cat_list_of_dict_tensor,
    get_num_micro_batches,
    iter_dict_micro_batches,
    put_tensor_device,
    split_dict_to_chunk,
)
from rlinf.utils.placement import (
    HybridComponentPlacement,
    ModelParallelComponentPlacement,
)
from rlinf.utils.utils import (
    clear_memory,
    compute_entropy_from_logits,
    compute_logprobs_from_logits,
    cpu_weight_swap,
    get_loss_agg_func,
    masked_mean,
    reshape_entropy,
    retrieve_model_state_dict_in_cpu,
)
from rlinf.workers.rollout.utils import RankMapper
from rlinf.workers.per_env_flex_plan import build_flex_plan

logger = logging.getLogger(__name__)


def compare_rollout_actor_logprobs(
    rollout_logprobs: torch.Tensor,
    actor_logprobs: torch.Tensor,
    loss_mask: torch.Tensor = None,
    prefix: str = "",
    clip_ratio: float | None = None,
) -> dict:
    """
    Compare logprobs from rollout and actor to verify consistency.

    Args:
        rollout_logprobs: logprobs from rollout (prev_logprobs)
        actor_logprobs: logprobs computed by actor
        loss_mask: optional mask for valid positions
        prefix: prefix for metric names

    Returns:
        dict with comparison metrics
    """
    metrics = {}

    # Ensure same shape
    if rollout_logprobs.shape != actor_logprobs.shape:
        metrics[f"{prefix}shape_mismatch"] = 1.0
        metrics[f"{prefix}rollout_shape"] = str(rollout_logprobs.shape)
        metrics[f"{prefix}actor_shape"] = str(actor_logprobs.shape)
        return metrics

    # Compute difference
    diff = actor_logprobs - rollout_logprobs
    abs_diff = torch.abs(diff)

    if loss_mask is not None:
        # Expand mask to match logprobs shape if needed
        while loss_mask.dim() < abs_diff.dim():
            loss_mask = loss_mask.unsqueeze(-1)
        mask = loss_mask.expand_as(abs_diff)
        valid_count = mask.sum().item()
        if valid_count > 0:
            masked_abs_diff = (abs_diff * mask).sum() / valid_count
            masked_diff = (diff * mask).sum() / valid_count
        else:
            masked_abs_diff = abs_diff.mean()
            masked_diff = diff.mean()
    else:
        masked_abs_diff = abs_diff.mean()
        masked_diff = diff.mean()
        valid_count = abs_diff.numel()

    def _masked_mean(value: torch.Tensor) -> torch.Tensor:
        if loss_mask is not None and valid_count > 0:
            return (value * mask).sum() / valid_count
        return value.mean()

    # Basic statistics. Here diff is the sampled-action log-ratio:
    # log pi_current(a|s) - log pi_behavior(a|s).
    metrics[f"{prefix}mean_abs_diff"] = masked_abs_diff.item()
    metrics[f"{prefix}mean_diff"] = masked_diff.item()
    metrics[f"{prefix}max_abs_diff"] = abs_diff.max().item()
    metrics[f"{prefix}min_abs_diff"] = abs_diff.min().item()
    metrics[f"{prefix}std_diff"] = diff.std().item()
    metrics[f"{prefix}mean_log_ratio"] = masked_diff.item()
    metrics[f"{prefix}mean_abs_log_ratio"] = masked_abs_diff.item()
    metrics[f"{prefix}max_abs_log_ratio"] = abs_diff.max().item()

    # Sampled-action approximation of KL(behavior || current):
    # E_a~behavior[log pi_behavior(a|s) - log pi_current(a|s)].
    approx_kl_behavior_to_current = _masked_mean(-diff)
    metrics[f"{prefix}approx_kl_behavior_to_current"] = approx_kl_behavior_to_current.item()

    # Keep the historical name for compatibility with existing compare_logprobs logs.
    kl_approx = masked_diff
    metrics[f"{prefix}kl_approx"] = kl_approx.item()

    if clip_ratio is not None:
        ratio = torch.exp(torch.clamp(diff, min=-20.0, max=20.0))
        clip_frac = _masked_mean((torch.abs(ratio - 1.0) > clip_ratio).float())
        metrics[f"{prefix}clip_frac"] = clip_frac.item()

    # Relative error
    rollout_abs = torch.abs(rollout_logprobs)
    rel_error = abs_diff / (rollout_abs + 1e-8)
    if loss_mask is not None and valid_count > 0:
        rel_error_mean = (rel_error * mask).sum() / valid_count
    else:
        rel_error_mean = rel_error.mean()
    metrics[f"{prefix}mean_rel_error"] = rel_error_mean.item()

    # Correlation coefficient
    rollout_flat = rollout_logprobs.flatten()
    actor_flat = actor_logprobs.flatten()
    if rollout_flat.std() > 1e-8 and actor_flat.std() > 1e-8:
        corr = torch.corrcoef(torch.stack([rollout_flat, actor_flat]))[0, 1]
        metrics[f"{prefix}correlation"] = corr.item()
    else:
        metrics[f"{prefix}correlation"] = 1.0

    # Percentage of values within tolerance
    for tol in [1e-6, 1e-4, 1e-2, 1e-1]:
        within_tol = (abs_diff < tol).float()
        if loss_mask is not None and valid_count > 0:
            pct = (within_tol * mask).sum() / valid_count
        else:
            pct = within_tol.mean()
        metrics[f"{prefix}pct_within_{tol}"] = pct.item()

    return metrics


def process_nested_dict_for_adv(nested_dict, rollout_epoch):
    """
    original shape: [rollout_epoch x n_chunk_steps, bsz, num_action_chunks, ...]
    target shape: [n_chunk_steps, rollout_epoch x bsz, num_action_chunks, ...]
    """
    ret_dict = {}
    for key, value in nested_dict.items():
        if isinstance(value, torch.Tensor):
            new_value = value.reshape(
                rollout_epoch, -1, *value.shape[1:]
            )  # [rollout_epoch, n_chunk_step, bsz, ...]
            new_value = new_value.transpose(
                0, 1
            )  # [n_chunk_step, rollout_epoch, bsz, ...]
            new_value = new_value.reshape(new_value.shape[0], -1, *new_value.shape[3:])
            ret_dict[key] = new_value
        elif isinstance(value, dict):
            ret_dict[key] = process_nested_dict_for_adv(value, rollout_epoch)
    return ret_dict


def process_nested_dict_for_train(nested_dict, shuffle_id):
    ret_dict = {}
    for key, value in nested_dict.items():
        if key in ["dones", "terminations", "truncations", "prev_values"]:
            value = value[:-1]
        if "env_info" in key:
            raise NotImplementedError
        if value is None:
            ret_dict[key] = None
        if isinstance(value, torch.Tensor):
            ret_dict[key] = value.reshape(-1, *value.shape[2:])[shuffle_id]
        elif isinstance(value, dict):
            ret_dict[key] = process_nested_dict_for_train(value, shuffle_id)
    return ret_dict


def reshape_nested_dict_for_recompute(nested_dict):
    """
    Reshape nested dict from [n_chunk_steps, batch_size, ...] to [n_chunk_steps * batch_size, ...].
    Similar to process_nested_dict_for_train but without shuffle and without special key handling.
    Used for _recompute_old_logprobs and _compare_logprobs_before_training.
    """
    ret_dict = {}
    for key, value in nested_dict.items():
        if value is None:
            ret_dict[key] = None
        elif isinstance(value, torch.Tensor):
            if value.dim() >= 2:
                ret_dict[key] = value.reshape(-1, *value.shape[2:])
            else:
                ret_dict[key] = value
        elif isinstance(value, dict):
            ret_dict[key] = reshape_nested_dict_for_recompute(value)
        else:
            ret_dict[key] = value
    return ret_dict


def slice_nested_dict(nested_dict, start_idx, end_idx):
    """
    Slice nested dict along the first dimension (batch dimension).
    Handles both tensor and nested dict types recursively.
    """
    ret_dict = {}
    for key, value in nested_dict.items():
        if value is None:
            ret_dict[key] = None
        elif isinstance(value, torch.Tensor):
            ret_dict[key] = value[start_idx:end_idx]
        elif isinstance(value, dict):
            ret_dict[key] = slice_nested_dict(value, start_idx, end_idx)
        else:
            ret_dict[key] = value
    return ret_dict


def get_nested_k_split_for_specific_keys(nested_dict, num_splits, key_list):
    """
    Get k-split iterator for some keys in nested_dict.
    """
    extra_dict = {}
    for key in key_list:
        if key not in nested_dict.keys():
            continue
        value = nested_dict[key]
        if isinstance(value, dict):
            extra_dict[key] = split_dict_to_chunk(value, num_splits)
        elif isinstance(value, torch.Tensor):
            continue
        else:
            raise NotImplementedError(
                f"Only support dict and tensor type, but got {type(value)}"
            )
    # {key1: [d1, d2, ...], key2: [d1, d2, ...]} -> [{key1: d1, key2: d1}, {key1: d2, key2: d2}, ...]
    extra_list = [
        {k: extra_dict[k][i] for k in extra_dict.keys()} for i in range(num_splits)
    ]
    return extra_list


class FSDPActor(FSDPModelManager, Worker):
    def __init__(
        self, cfg: DictConfig, placement: ModelParallelComponentPlacement
    ) -> None:
        """
        FSDPActor worker used to train the model with data from rollout workers.

        Args:
            cfg (DictConfig): The global yaml configuration.
            placement (ModelParallelComponentPlacement): The accelerator placement for actor worker.
        """
        Worker.__init__(self)
        super().__init__(cfg.actor, self._world_size, self._rank)

        self.cfg = cfg

        self.response_len = (
            self.cfg.actor.model.encoder_seq_length - self.cfg.data.max_prompt_length
        )
        self.calculate_entropy = self.cfg.algorithm.calculate_entropy
        self.calculate_entropy_loss = (
            self.cfg.algorithm.entropy_bonus > 0 and self.calculate_entropy
        )
        self.kl_beta = self.cfg.algorithm.kl_beta
        self.kl_penalty_type = self.cfg.algorithm.kl_penalty_type

        self.total_batch_size_per_dp = (
            self.cfg.data.rollout_batch_size
            * self.cfg.algorithm.group_size
            // self._world_size
        )

        self._rollout_group_name = cfg.rollout.group_name
        self._component_placement = placement
        self.is_pipeline = self._component_placement.is_disaggregated
        self.ref_policy_state_dict = None
        if self.is_pipeline:
            self._inference_group_name = cfg.inference.group_name
            self._inference_world_size = self._component_placement.get_world_size(
                "inference"
            )
            self._inference_dst_map: dict[int, list[str]] = {}
        else:
            self._inference_group_name = None
            self._inference_world_size = 0
            self._inference_dst_map = None
        self.loss_agg_func = get_loss_agg_func(self.cfg.algorithm.loss_agg_func)
        self.enable_offload = (
            self.cfg.actor.get("enable_offload", False) and not self.is_pipeline
        )
        self.micro_batch_size = self.cfg.actor.micro_batch_size
        self.n_mini_batches = self.cfg.algorithm.n_minibatches
        self.task_type = self.cfg.runner.task_type

    def init_worker(self) -> None:
        """
        Initialize the actor worker. build the model and use corresponding training backend
        (FSDP/FSDP2) to wrap it. If needed, offload model parameters and optimizer states to CPU.
        If kl_beta > 0, retrieve the reference policy model state dict to CPU.
        If mode is disaggregated, setup which inference ranks it needs to sync weights to by
        doing a handshake with inference workers.
        """
        self.setup_model_and_optimizer()
        if self.cfg.algorithm.kl_beta > 0 and self.cfg.actor.get(
            "combine_reference_model", True
        ):
            self.ref_policy_state_dict = retrieve_model_state_dict_in_cpu(self.model)

        if self.enable_offload and not self.is_pipeline:
            self.offload_param_and_grad()
            self.offload_optimizer()
        self._setup_rollout_weight_dst_ranks()

    def _setup_rollout_weight_dst_ranks(self) -> None:
        """Setup destination ranks for token and weight communication."""
        rank_map = RankMapper.get_actor_rank_to_rollout_rank_map(
            self._component_placement
        )
        self._weight_dst_rank_in_rollout = rank_map[self._rank]
        self.log_info(
            f"Actor rank {self._rank} will send weights to {self._weight_dst_rank_in_rollout}"
        )

    def del_reshard_state_dict(self) -> None:
        """Just for interface compatibility with MegatronActor."""
        if hasattr(self, "rollout_state_dict"):
            del self.rollout_state_dict
        clear_memory(sync=False)

    def sync_model_to_inference(self) -> None:
        """
        Sync the model's full state dict to the inference worker.
        The model state_dict is the reference of actor's model
        parameters(by setting cpu_offload=False).
        """
        if not self._inference_dst_map:
            self._strategy.setup_actor_sync_inference_ranks(self)

        if self.is_optimizer_offloaded:
            self.offload_optimizer()

        if self.is_weight_offloaded:
            self.load_param_and_grad(self.device, False)

        inference_state_dict = self.get_model_state_dict(
            cpu_offload=False, full_state_dict=False
        )
        # NOTE: we have already know which inference rank needs which params
        # by calling _strategy.setup_actor_sync_inference_ranks() to do handshake
        # with each inference rank. just send them accordingly.
        for rank, needed_params in self._inference_dst_map.items():
            sended_params = {}
            for name in needed_params:
                if name in inference_state_dict:
                    # mentioned again, no ShardedTensor here.
                    sended_params[name] = (
                        inference_state_dict[name].to_local()
                        if isinstance(inference_state_dict[name], DTensor)
                        else inference_state_dict[name]
                    )
            self.send(
                object=sended_params,
                dst_group_name=self._inference_group_name,
                dst_rank=rank,
                async_op=True,
            )

        if self.enable_offload and not self.is_weight_offloaded:
            self.offload_param_and_grad()

        torch.distributed.barrier()

    def sync_model_to_rollout(self) -> None:
        """
        Sync the model's full state dict to the rollout worker.
        """
        if self.enable_offload and not self.is_optimizer_offloaded:
            self.offload_optimizer()

        if self.enable_offload and self.is_weight_offloaded:
            self.load_param_and_grad(self.device, True)

        self.rollout_state_dict = self.get_model_state_dict(
            cpu_offload=False, full_state_dict=True
        )

        has_visual = any("visual." in k for k in self.rollout_state_dict.keys())

        state_dict = {}

        if self._weight_dst_rank_in_rollout is not None:
            for k, v in self.rollout_state_dict.items():
                name = k
                if has_visual:
                    if name.startswith("model.language_model."):
                        name = "model." + name[21:]
                    # NOTE:
                    # if transformers version is 4.56.1 or older(not tested),
                    # the following line should be uncommented

                    # elif name.startswith("model."):
                    #     name = name[6:]
                state_dict[name] = reduce_tensor(v) if not self.is_pipeline else v
            if not self.is_pipeline:
                self.send(
                    state_dict,
                    self._rollout_group_name,
                    self._weight_dst_rank_in_rollout,
                )
            else:
                for weight_dst_rank in self._weight_dst_rank_in_rollout:
                    self.send(
                        state_dict,
                        self._rollout_group_name,
                        weight_dst_rank,
                    )

        state_dict.clear()
        if self.enable_offload and not self.is_weight_offloaded:
            self.offload_param_and_grad()

    def get_batch(
        self, channel: Channel
    ) -> tuple[dict[str, torch.Tensor], RolloutResult]:
        result: RolloutResult = channel.get()

        batch = result.to_actor_batch(
            self.cfg.data.max_prompt_length,
            self.cfg.actor.model.encoder_seq_length,
            self.tokenizer.eos_token_id,
        )
        return batch, result

    def _load_weight_and_optimizer(self) -> None:
        # Acquire the GPUs to ensure that no one is using them before loading models
        # Otherwise, it may lead to OOM
        with self.device_lock:
            if not self.enable_offload:
                return
            if self.is_weight_offloaded:
                self.load_param_and_grad(self.device)
            if self.is_optimizer_offloaded:
                self.load_optimizer(self.device)

    @torch.no_grad()
    def inference_step(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        self.model.eval()
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        position_ids = batch["position_ids"]

        multi_modal_inputs = {}
        if "multi_modal_inputs" in batch.keys():
            for key in batch["multi_modal_inputs"][0].keys():
                multi_modal_inputs[key] = torch.cat(
                    [inputs[key] for inputs in batch["multi_modal_inputs"]],
                    dim=0,
                ).cuda()

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            use_cache=False,
            **multi_modal_inputs,
        )

        logits = outputs.logits
        logits = logits[:, -self.response_len - 1 : -1, :]
        logits = logits / self.cfg.algorithm.sampling_params.temperature

        responses = input_ids[:, -self.response_len :]
        logprobs = compute_logprobs_from_logits(logits, responses)
        return logprobs

    def run_inference(
        self,
        input_channel: Channel,
        output_channel: Channel,
        compute_ref_logprobs: bool,
    ) -> None:
        """
        Compute prev/ref logprobs using the actor Model's forward.

        Args:
            input_channel: The input channel to read from.
            output_channel: The output channel to send results to.
            compute_ref_logprobs: Whether to compute reference logprobs.
        """
        recv_batch_size = 0
        while recv_batch_size < self.total_batch_size_per_dp:
            batch, rollout_result = self.get_batch(input_channel)
            recv_batch_size += rollout_result.num_sequence
            self._load_weight_and_optimizer()

            num_splits = (
                rollout_result.num_sequence
                // self.cfg.algorithm.logprob_forward_micro_batch_size
            )
            micro_batches_iter = get_iterator_k_split(
                batch,
                num_splits=num_splits,
            )
            micro_batches = list(micro_batches_iter)

            prev_logprobs = []
            with self.worker_timer():
                for micro_batch in micro_batches:
                    prev_logprobs.append(self.inference_step(micro_batch).cpu())

                if rollout_result.rollout_logprobs is not None:
                    # Rollout has returned logprobs, store the recomputed logprobs in recompute_prev_logprobs
                    rollout_result.recompute_prev_logprobs = torch.cat(prev_logprobs)
                else:
                    # Otherwise, directly store the logprobs in prev_logprobs (the final logprobs used for training)
                    rollout_result.prev_logprobs = torch.cat(prev_logprobs)

            if compute_ref_logprobs:
                assert self.ref_policy_state_dict is not None, (
                    "Reference policy state dict is None but compute_ref_logprobs is True"
                )
                ref_logprobs = []
                with cpu_weight_swap(self.model, self.ref_policy_state_dict):
                    for micro_batch in micro_batches:
                        ref_logprobs.append(self.inference_step(micro_batch).cpu())
                    rollout_result.ref_logprobs = torch.cat(ref_logprobs)

            output_channel.put(rollout_result)

        assert recv_batch_size == self.total_batch_size_per_dp, (
            f"Expected {self.total_batch_size_per_dp} sequences from channel, but got {recv_batch_size}"
        )

    def training_step(
        self, batch: dict[str, torch.Tensor] | BatchResizingIterator
    ) -> tuple[dict[str, torch.Tensor], float, list[float]]:
        if isinstance(batch, dict):
            global_batch_size = batch["input_ids"].shape[0]
            assert global_batch_size % self.micro_batch_size == 0, (
                f"global batch size {global_batch_size} can not divide micro_batch_size {self.micro_batch_size}"
            )
            micro_batch_cnt = global_batch_size // self.micro_batch_size
            self.gradient_accumulation = micro_batch_cnt
            micro_batches = get_iterator_k_split(batch, micro_batch_cnt)
            micro_batches_iter = iter(micro_batches)
        else:
            global_batch_size = self.total_batch_size_per_dp // self.n_mini_batches
            micro_batch_cnt = global_batch_size // self.micro_batch_size
            self.gradient_accumulation = micro_batch_cnt

            def iterator_wrapper():
                for _ in range(micro_batch_cnt):
                    yield next(batch)

            micro_batches_iter = iterator_wrapper()
        self.optimizer.zero_grad()
        mbs_metrics_list = {}
        for idx, m_batch in enumerate(micro_batches_iter):
            backward_ctx = self.before_micro_batch(
                self.model,
                is_last_micro_batch=(idx + 1) == self.gradient_accumulation,
            )
            for k, v in m_batch.items():
                m_batch[k] = v.cuda() if isinstance(v, torch.Tensor) else v

            multi_modal_inputs = {}
            if "multi_modal_inputs" in m_batch.keys():
                for key in m_batch["multi_modal_inputs"][0].keys():
                    multi_modal_inputs[key] = torch.cat(
                        [inputs[key] for inputs in m_batch["multi_modal_inputs"]],
                        dim=0,
                    ).cuda()

            input_ids = m_batch["input_ids"]
            attention_mask = m_batch["attention_mask"]
            position_ids = m_batch["position_ids"]
            prev_logprobs = m_batch["prev_logprobs"]
            advantages = m_batch["advantages"]
            ref_logprobs = None
            if "ref_logprobs" in m_batch:
                ref_logprobs = m_batch["ref_logprobs"]

            loss_mask = m_batch["attention_mask"][:, -self.response_len :]

            clip_ratio = self.cfg.algorithm.ratio_clip_eps
            clip_ratio_low = self.cfg.algorithm.get("clip_ratio_low", None)
            clip_ratio_high = self.cfg.algorithm.get("clip_ratio_high", None)
            clip_ratio_low = (
                clip_ratio_low if clip_ratio_low is not None else clip_ratio
            )
            clip_ratio_high = (
                clip_ratio_high if clip_ratio_high is not None else clip_ratio
            )
            clip_ratio_c = self.cfg.algorithm.get("clip_ratio_c", 3.0)

            with self.amp_context:
                output = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    **multi_modal_inputs,
                    use_cache=False,
                )

                logits: torch.Tensor = output.logits

                logits.div_(self.cfg.algorithm.sampling_params.temperature)

                responses = input_ids[:, -self.response_len :]
                logits = logits[
                    :, -self.response_len - 1 : -1, :
                ]  # (bsz, response_length, vocab_size)
                logprobs = compute_logprobs_from_logits(logits, responses)

                if self.cfg.algorithm.get("importance_sampling_fix", False):
                    rollout_prev_logprobs = prev_logprobs
                    recompute_prev_logprobs = m_batch["recompute_prev_logprobs"]
                    advantages = advantages * torch.clamp(
                        (recompute_prev_logprobs - rollout_prev_logprobs).exp(),
                        min=self.cfg.algorithm.importance_sampling_clip,
                    )

                loss, mbs_metrics_data = policy_loss(
                    loss_type=self.cfg.algorithm.loss_type,
                    loss_agg_func=self.loss_agg_func,
                    logprobs=logprobs,
                    old_logprobs=prev_logprobs,
                    advantages=advantages,
                    clip_ratio_low=clip_ratio_low,
                    clip_ratio_high=clip_ratio_high,
                    clip_ratio_c=clip_ratio_c,
                    loss_mask=loss_mask,
                    task_type=self.task_type,
                )

                entropy_loss = torch.tensor(0.0, device=torch.cuda.current_device())
                if self.calculate_entropy:
                    entropy = compute_entropy_from_logits(
                        logits,
                    )

                    entropy_loss = self.loss_agg_func(entropy, mask=loss_mask)
                    if self.calculate_entropy_loss:
                        loss = loss - self.cfg.algorithm.entropy_bonus * entropy_loss

                kl_loss = torch.tensor(0.0, device=torch.cuda.current_device())
                if self.kl_beta > 0 and ref_logprobs is not None:
                    kld = kl_penalty(ref_logprobs, logprobs, self.kl_penalty_type)
                    kl_loss = self.loss_agg_func(kld, loss_mask)
                    loss = loss + kl_loss * self.kl_beta

                # add to log
                # scale loss for gradient accumulation and backprop
                loss = loss / self.gradient_accumulation
                with backward_ctx:
                    self.grad_scaler.scale(loss).backward()

            mbs_metrics_data.update(
                {
                    "final_loss": loss.detach(),
                    "entropy_loss": entropy_loss.detach(),
                    "kl_loss": kl_loss.detach(),
                }
            )

            append_to_dict(mbs_metrics_list, mbs_metrics_data)

        grad_norm, lr_list = self.optimizer_step()
        return mbs_metrics_list, grad_norm, lr_list

    def run_training_pipeline(self, input_channel: Channel) -> tuple[dict, list]:
        self.model.train()
        train_batch_iterator = BatchResizingIterator(
            cfg=self.cfg,
            get_batch_fn=partial(self.get_batch, input_channel),
            micro_batch_size=self.micro_batch_size,
            total_batch_size=self.total_batch_size_per_dp,
            num_global_batches=self.n_mini_batches,
            forward_only=False,
        )
        train_batch_iterator.register_get_batch_handler(
            self.compute_advantages_and_returns
        )

        if self.cfg.algorithm.normalize_advantages:

            def normalize_advantages(batch: dict[str, torch.Tensor]):
                mask = batch["attention_mask"][:, -self.response_len :]
                batch["advantages"] = masked_normalization(batch["advantages"], mask)
                return batch

            train_batch_iterator.register_global_batch_handler(normalize_advantages)

        self._load_weight_and_optimizer()
        training_metrics_list = []
        with self.worker_timer():
            for _ in range(self.n_mini_batches):
                metrics, grad_norm, lr_list = self.training_step(
                    batch=train_batch_iterator
                )

                # aggregate metrics across micro-batches
                mean_metric_dict = {
                    key: torch.mean(torch.stack(value))
                    for key, value in metrics.items()
                }
                mean_metric_dict = all_reduce_dict(
                    mean_metric_dict, op=torch.distributed.ReduceOp.AVG
                )

                mean_metric_dict["actor/grad_norm"] = float(grad_norm)
                mean_metric_dict["actor/lr"] = lr_list[0]
                training_metrics_list.append(mean_metric_dict)

        # put lr scheduler step here
        self.lr_scheduler.step()

        # Rollout metrics
        batch = train_batch_iterator.get_all_batches()
        rollout_metrics, _, _ = compute_math_rollout_metrics(
            batch, self.cfg.data.max_prompt_length, self.response_len
        )

        return rollout_metrics, training_metrics_list

    def run_training(self, input_channel: Channel) -> tuple[dict, list]:
        # Get all batches for this DP
        if self.is_pipeline:
            with self.worker_timer():
                return self.run_training_pipeline(input_channel)

        batches = []
        recv_batch_size = 0
        while recv_batch_size < self.total_batch_size_per_dp:
            batch, rollout_result = self.get_batch(input_channel)
            batches.append(batch)
            recv_batch_size += rollout_result.num_sequence
        assert recv_batch_size == self.total_batch_size_per_dp, (
            f"Expected {self.total_batch_size_per_dp} sequences from channel, but got {recv_batch_size}"
        )
        global_batch = RolloutResult.merge_batches(batches)

        # Compute advantages and returns
        global_batch = self.compute_advantages_and_returns(global_batch)

        if self.cfg.algorithm.normalize_advantages:
            mask = global_batch["attention_mask"][:, -self.response_len :]
            global_batch["advantages"] = masked_normalization(
                global_batch["advantages"], mask
            )

        # Must be called after batch is retrieved, which is when rollout has stopped
        # Otherwise, loading model might cause OOM
        self._load_weight_and_optimizer()

        mini_batches = get_iterator_k_split(
            global_batch,
            num_splits=self.cfg.algorithm.n_minibatches,
            shuffle=self.cfg.algorithm.get("shuffle_rollout", True),
            shuffle_seed=self.cfg.actor.seed,
        )

        self.model.train()
        assert (
            self.cfg.actor.global_batch_size
            % (self.cfg.actor.micro_batch_size * self._world_size)
            == 0
        )

        training_metrics_list = []
        # Global batch iterations
        with self.worker_timer():
            for mini_batch in mini_batches:
                metrics, grad_norm, lr_list = self.training_step(batch=mini_batch)

                # aggregate metrics across micro-batches
                mean_metric_dict = {
                    key: torch.mean(torch.stack(value))
                    for key, value in metrics.items()
                }
                mean_metric_dict = all_reduce_dict(
                    mean_metric_dict, op=torch.distributed.ReduceOp.AVG
                )

                mean_metric_dict["actor/grad_norm"] = float(grad_norm)
                mean_metric_dict["actor/lr"] = lr_list[0]
                training_metrics_list.append(mean_metric_dict)

        # put lr scheduler step here
        self.lr_scheduler.step()

        # Rollout metrics
        rollout_metrics, _, _ = compute_math_rollout_metrics(
            global_batch, self.cfg.data.max_prompt_length, self.response_len
        )

        return rollout_metrics, training_metrics_list

    # Advantages and returns
    def compute_advantages_and_returns(self, batch: dict[str, torch.Tensor]):
        """Compute the advantages and returns.

        Args:
            batch (Dict[str, torch.Tensor]): The rollout batch.
        """
        with self.worker_timer():
            if batch.get("advantages", None) is None:
                mask = batch["attention_mask"][:, -self.response_len :]
                advantages, _ = calculate_adv_and_returns(
                    task_type=self.task_type,
                    adv_type=self.cfg.algorithm.adv_type,
                    rewards=batch["rewards"].cuda(),
                    loss_mask=mask.cuda(),
                    group_size=self.cfg.algorithm.group_size,
                    kl_beta=self.cfg.algorithm.get("reinpp_kl_beta", 0.0),
                    kl_penalty_type=self.kl_penalty_type,
                    logprob=batch["prev_logprobs"].cuda()
                    if "prev_logprobs" in batch
                    else None,
                    ref_logprob=batch["ref_logprobs"].cuda()
                    if "ref_logprobs" in batch
                    else None,
                    use_reinpp_baseline=self.cfg.algorithm.get(
                        "use_reinpp_baseline", False
                    ),
                )
                batch["advantages"] = advantages

        return batch


class EmbodiedFSDPActor(FSDPModelManager, Worker):
    def __init__(self, cfg: DictConfig):
        Worker.__init__(self)
        super().__init__(cfg.actor, self._world_size, self._rank)
        self.cfg = cfg
        self._env_group_name = cfg.env.group_name
        self._rollout_group_name = cfg.rollout.group_name
        self._component_placement = HybridComponentPlacement(cfg, Cluster())

        # stage_num: default to 2, use for pipeline rollout process
        self.stage_num = cfg.rollout.pipeline_stage_num

        self.enable_offload = self.cfg.actor.get("enable_offload", False)

    def _setup_rollout_weight_dst_ranks(self) -> None:
        """
        Setup destination ranks for weight communication.
        It can support any topology between actor and rollout workers.
        Assuming there are M actor ranks and N rollout ranks, each actor rank
        will send weights to most ceil(N/M) rollout ranks according to the modulo rule.
        """
        rollout_world_size = self._component_placement.get_world_size("rollout")
        actor_world_size = self._world_size
        rank = self._rank
        self._weight_dst_rank_in_rollout = []
        rollout_ranks_per_actor = (
            rollout_world_size + actor_world_size - 1
        ) // actor_world_size
        for i in range(rollout_ranks_per_actor):
            if i * actor_world_size + rank < rollout_world_size:
                self._weight_dst_rank_in_rollout.append(i * actor_world_size + rank)

    def init_worker(self) -> None:
        """
        Initialize the actor worker. build the model and use corresponding training backend,
        if needed, offload model parameters and optimizer states to CPU.
        """
        self.setup_model_and_optimizer()

        if self.enable_offload:
            self.offload_param_and_grad()
            self.offload_optimizer()
        self._setup_rollout_weight_dst_ranks()

    def model_provider_func(self) -> nn.Module:
        model = get_model(self.cfg.actor.model)
        if model is not None:
            return model
        return super().model_provider_func()

    def sync_model_to_rollout(self) -> None:
        """
        Sync the model's full state dict to the rollout worker.
        """
        if self.enable_offload and not self.is_optimizer_offloaded:
            self.offload_optimizer()

        if self.enable_offload and self.is_weight_offloaded:
            self.load_param_and_grad(self.device)

        state_dict = self.get_model_state_dict(cpu_offload=False, full_state_dict=True)
        state_dict["__rlinf_policy_version__"] = torch.tensor(
            int(self.optimizer_steps),
            dtype=torch.long,
            device=torch.cuda.current_device(),
        )
        for rank in self._weight_dst_rank_in_rollout:
            self.send(
                state_dict,
                self._rollout_group_name,
                rank,
                async_op=True,
            )
        if self.enable_offload and not self.is_weight_offloaded:
            self.offload_param_and_grad()

    def recv_rollout_batch(self, input_channel: Channel) -> None:
        """
        Receive rollout batch from rollout workers.

        Args:
            input_channel: The input channel to read from.
        """
        # Check if per_env_async mode is enabled
        per_env_async_cfg = self.cfg.rollout.get("per_env_async", {})
        per_env_async_enabled = per_env_async_cfg.get("enabled", False)
        aggregate_slots_per_env = int(per_env_async_cfg.get("aggregate_slots_per_env", 1))

        if per_env_async_enabled:
            # Per-Env Async mode: each slot sends independently
            env_world_size = self._component_placement.get_world_size("env")
            rollout_world_size = self._component_placement.get_world_size("rollout")
            default_batch_size = (
                self.cfg.env.train.total_num_envs
                // max(1, env_world_size)
                // max(1, self.stage_num)
            )
            flex_cfg = per_env_async_cfg.get("flex", None)
            if flex_cfg and flex_cfg.get("enabled", False):
                plan = build_flex_plan(
                    cfg=self.cfg,
                    env_world_size=env_world_size,
                    rollout_world_size=rollout_world_size,
                    default_stage_num=self.stage_num,
                    default_batch_size=default_batch_size,
                )
                send_num = plan.total_slot_count
            else:
                send_num = env_world_size * self.stage_num * aggregate_slots_per_env
        else:
            # Normal mode: each rollout worker sends per stage
            send_num = self._component_placement.get_world_size("rollout") * self.stage_num

        recv_num = self._component_placement.get_world_size("actor")
        split_num = compute_split_num(send_num, recv_num)

        self.rollout_batch = {}
        recv_list = []
        for _ in range(split_num):
            recv_list.append(input_channel.get())

        # [DEBUG] 检查接收到的每个 split 的数据大小
        logger.info(f"[DEBUG Actor] Received {len(recv_list)} splits")
        for i, recv_data in enumerate(recv_list):
            if "rewards" in recv_data and recv_data["rewards"] is not None:
                logger.info(f"[DEBUG Actor]   split {i}: rewards shape={recv_data['rewards'].shape}")
            if "dones" in recv_data and recv_data["dones"] is not None:
                logger.info(f"[DEBUG Actor]   split {i}: dones shape={recv_data['dones'].shape}")

        # shape [num_chunk, bsz, chunk_size], cat dim 1
        self.rollout_batch = self._cat_rollout_batches(recv_list, dim=1)

        # [DEBUG] 检查合并后的数据大小
        logger.info(f"[DEBUG Actor] After cat_list_of_dict_tensor:")
        for key in ["prev_logprobs", "prev_values", "dones", "rewards"]:
            if key in self.rollout_batch and self.rollout_batch[key] is not None:
                logger.info(f"[DEBUG Actor]   {key}: shape={self.rollout_batch[key].shape}")

        self.rollout_batch = self._process_received_rollout_batch(self.rollout_batch)

    def _process_received_rollout_batch(
        self, rollout_batch: dict[str, torch.Tensor], is_single_epoch: bool = False
    ) -> dict[str, torch.Tensor]:
        """
        Process received rollout batch data.

        Args:
            rollout_batch: Raw rollout batch data
            is_single_epoch: If True, skip cross-epoch reshape (for async mode)
                            If False, reshape data across epochs (for sync mode)

        For sync mode (is_single_epoch=False):
            original shape: [rollout_epoch x n_chunk_steps, bsz, num_action_chunks, ...]
            target shape: [n_chunk_steps, rollout_epoch x bsz, num_action_chunks, ...]

        For async mode (is_single_epoch=True):
            shape remains: [n_chunk_steps, bsz, num_action_chunks, ...]
        """
        if not is_single_epoch:
            # Sync mode: reshape data across epochs
            rollout_epoch = self.cfg.algorithm.rollout_epoch
            rollout_batch = process_nested_dict_for_adv(rollout_batch, rollout_epoch)

        # Compute loss_mask if needed
        if (
            not self.cfg.env.train.auto_reset
            and not self.cfg.env.train.ignore_terminations
        ):
            dones = rollout_batch["dones"]
            loss_mask, loss_mask_sum = compute_loss_mask(dones)

            if self.cfg.algorithm.reward_type == "chunk_level":
                loss_mask = loss_mask.any(dim=-1, keepdim=True)
                loss_mask_sum = loss_mask_sum[..., -1:]

            rollout_batch["loss_mask"] = loss_mask
            rollout_batch["loss_mask_sum"] = loss_mask_sum

        # Filter data by rewards if enabled
        if self.cfg.algorithm.get("filter_rewards", False):
            rollout_batch = self._apply_reward_filter(rollout_batch)

        return rollout_batch

    def _apply_reward_filter(
        self, rollout_batch: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        """Apply reward-based filtering to rollout batch."""
        rewards = rollout_batch["rewards"]
        if rollout_batch.get("loss_mask", None) is not None:
            rewards = rewards * rollout_batch["loss_mask"]
        n_chunk_step, batch_size, num_action_chunks = rewards.shape

        group_size = self.cfg.algorithm.group_size
        if batch_size % group_size != 0:
            logger.warning(
                "Skipping reward filter: batch_size=%s not divisible by group_size=%s "
                "(e.g. streaming micro-batches).",
                batch_size,
                group_size,
            )
            return rollout_batch
        n_prompts = batch_size // group_size

        # Calculate rewards by prompt
        rewards = rewards.transpose(0, 1)
        rewards = rewards.reshape(rewards.shape[0], -1)
        reward_matrix = rewards.reshape(n_prompts, group_size, rewards.shape[-1])
        reward_matrix = reward_matrix.sum(dim=-1)
        mean_reward_in_group = reward_matrix.mean(dim=1)

        # Create mask based on reward bounds
        reward_filter_mask = (
            mean_reward_in_group >= self.cfg.algorithm.rewards_lower_bound
        ) & (mean_reward_in_group <= self.cfg.algorithm.rewards_upper_bound)

        # Extend mask dimension
        reward_filter_mask = reward_filter_mask.repeat_interleave(group_size)
        reward_filter_mask = (
            reward_filter_mask.unsqueeze(0).expand(n_chunk_step, -1).unsqueeze(-1)
        )

        # Update loss_mask
        if rollout_batch.get("loss_mask", None) is not None:
            rollout_batch["loss_mask"] = reward_filter_mask & rollout_batch["loss_mask"]
        else:
            rollout_batch["loss_mask"] = reward_filter_mask

        return rollout_batch

    def recompute_old_logprobs(self) -> None:
        """
        Public method to recompute old_logprobs using actor model.
        Called by Runner when algorithm.recompute_old_logprobs is True.
        """
        if self.is_weight_offloaded:
            self.load_param_and_grad(self.device)
        self._recompute_old_logprobs()
        if self.enable_offload:
            self.offload_param_and_grad()

    def compute_advantages_and_returns(self) -> dict[str, torch.Tensor]:
        """
        Compute the advantages and returns for rollout batch.
        Works for both sync mode (multi-epoch aggregated data) and async mode (single epoch data).
        """
        kwargs = {
            "task_type": self.cfg.runner.task_type,
            "adv_type": self.cfg.algorithm.adv_type,
            "rewards": self.rollout_batch["rewards"],
            "dones": self.rollout_batch["dones"],
            "values": self.rollout_batch.get("prev_values", None),
            "gamma": self.cfg.algorithm.get("gamma", 1),
            "gae_lambda": self.cfg.algorithm.get("gae_lambda", 1),
            "group_size": self.cfg.algorithm.get("group_size", 8),
            "reward_type": self.cfg.algorithm.reward_type,
            "loss_mask": self.rollout_batch.get("loss_mask", None),
            "loss_mask_sum": self.rollout_batch.get("loss_mask_sum", None),
        }

        advantages_and_returns = calculate_adv_and_returns(**kwargs)

        self.rollout_batch.update(advantages_and_returns)
        if kwargs["loss_mask"] is not None:
            self.rollout_batch.update({"loss_mask": kwargs["loss_mask"]})
        if kwargs["loss_mask_sum"] is not None:
            self.rollout_batch.update({"loss_mask_sum": kwargs["loss_mask_sum"]})

        rollout_metrics = compute_rollout_metrics(self.rollout_batch)
        if self.cfg.algorithm.get("policy_staleness_metrics", False):
            rollout_metrics.update(self._compute_policy_staleness_metrics())
        return rollout_metrics

    @staticmethod
    def _to_policy_version(value) -> int | None:
        if value is None:
            return None
        if torch.is_tensor(value):
            return int(value.detach().cpu().item())
        return int(value)

    _BEHAVIOR_POLICY_VERSION_KEYS = (
        "__behavior_policy_version__",
        "__behavior_policy_version_min__",
        "__behavior_policy_version_max__",
    )

    def _pop_behavior_policy_versions(self, recv_list: list[dict]) -> list[int]:
        versions = []
        for data in recv_list:
            version = data.pop("__behavior_policy_version__", None)
            for key in self._BEHAVIOR_POLICY_VERSION_KEYS[1:]:
                data.pop(key, None)
            version = self._to_policy_version(version)
            if version is not None:
                versions.append(version)
        return versions

    def _attach_behavior_policy_versions(
        self, rollout_batch: dict, versions: list[int]
    ) -> None:
        if not versions:
            return
        rollout_batch["__behavior_policy_version__"] = float(np.mean(versions))
        rollout_batch["__behavior_policy_version_min__"] = float(min(versions))
        rollout_batch["__behavior_policy_version_max__"] = float(max(versions))

    def _cat_rollout_batches(self, list_of_dict: list[dict], dim: int = 1) -> dict:
        """Concat rollout batches after extracting non-tensor metadata fields."""
        if not list_of_dict:
            return {}
        versions = self._pop_behavior_policy_versions(list_of_dict)
        merged = cat_list_of_dict_tensor(list_of_dict, dim=dim)
        self._attach_behavior_policy_versions(merged, versions)
        return merged

    @torch.no_grad()
    def _compute_policy_staleness_metrics(self) -> dict:
        metrics = {}
        behavior_version = self.rollout_batch.get("__behavior_policy_version__", None)
        if behavior_version is not None:
            train_version = float(self.optimizer_steps)
            metrics["staleness/policy_version_lag"] = train_version - float(behavior_version)
            metrics["staleness/behavior_policy_version"] = float(behavior_version)
            metrics["staleness/behavior_policy_version_min"] = float(
                self.rollout_batch.get("__behavior_policy_version_min__", behavior_version)
            )
            metrics["staleness/behavior_policy_version_max"] = float(
                self.rollout_batch.get("__behavior_policy_version_max__", behavior_version)
            )
            metrics["staleness/train_policy_version"] = train_version

        self.model.eval()
        n_chunk_steps = self.rollout_batch["prev_logprobs"].shape[0]
        batch_size = self.rollout_batch["prev_logprobs"].shape[1]
        total_samples = n_chunk_steps * batch_size
        reshaped_batch = reshape_nested_dict_for_recompute(self.rollout_batch)
        micro_batch_size = self.cfg.actor.micro_batch_size
        actor_logprobs_list = []
        prev_logprobs_list = []
        loss_mask_list = []
        compute_values = self.cfg.algorithm.adv_type == "gae"

        for start_idx in range(0, total_samples, micro_batch_size):
            end_idx = min(start_idx + micro_batch_size, total_samples)
            micro_data = slice_nested_dict(reshaped_batch, start_idx, end_idx)
            micro_data.pop("__behavior_policy_version__", None)
            micro_data.pop("__behavior_policy_version_min__", None)
            micro_data.pop("__behavior_policy_version_max__", None)
            micro_data = put_tensor_device(
                micro_data, f"cuda:{int(os.environ['LOCAL_RANK'])}"
            )
            prev_logprobs = micro_data["prev_logprobs"]
            loss_mask = micro_data.get("loss_mask", None)

            if SupportedModel(self.cfg.actor.model.model_type) in [
                SupportedModel.OPENVLA,
                SupportedModel.OPENVLA_OFT,
            ]:
                micro_data["temperature"] = (
                    self.cfg.algorithm.sampling_params.temperature_train
                )
                micro_data["top_k"] = self.cfg.algorithm.sampling_params.top_k

            with self.amp_context:
                output_dict = self.model(
                    data=micro_data,
                    compute_logprobs=True,
                    compute_entropy=False,
                    compute_values=compute_values,
                    use_cache=False,
                )

            actor_logprobs = output_dict["logprobs"]
            if SupportedModel(self.cfg.actor.model.model_type) in [SupportedModel.GR00T]:
                prev_logprobs = output_dict["prev_logprobs"]

            actor_logprobs_list.append(actor_logprobs.detach().cpu())
            prev_logprobs_list.append(prev_logprobs.detach().cpu())
            if loss_mask is not None:
                loss_mask_list.append(loss_mask.detach().cpu())

        actor_logprobs = torch.cat(actor_logprobs_list, dim=0)
        prev_logprobs = torch.cat(prev_logprobs_list, dim=0)
        loss_mask = torch.cat(loss_mask_list, dim=0) if loss_mask_list else None
        metrics.update(
            compare_rollout_actor_logprobs(
                rollout_logprobs=prev_logprobs,
                actor_logprobs=actor_logprobs,
                loss_mask=loss_mask,
                prefix="staleness/",
                clip_ratio=self.cfg.algorithm.get("clip_ratio_high", None),
            )
        )
        metrics["staleness/num_samples"] = float(total_samples)
        self.model.train()
        numeric_metrics = {
            key: value for key, value in metrics.items() if isinstance(value, (int, float))
        }
        return all_reduce_dict(
            numeric_metrics, op=torch.distributed.ReduceOp.AVG
        )

    @torch.no_grad()
    def _compare_logprobs_before_training(self) -> dict:
        """
        Compare rollout logprobs vs actor logprobs BEFORE any training updates.
        This ensures we compare with the same model version.
        Processes ALL samples in micro-batches to avoid OOM.
        """
        self.model.eval()

        rank = int(os.environ.get("RANK", 0))
        if rank == 0:
            print("[Logprob Compare] Starting comparison on all samples...", flush=True)

        # Get total samples (data is already in [total_samples, ...] format after process_nested_dict_for_train)
        total_samples = self.rollout_batch["prev_logprobs"].shape[0]
        micro_batch_size = self.cfg.actor.micro_batch_size

        # Collect logprobs from all micro-batches
        all_actor_logprobs = []
        all_prev_logprobs = []
        all_loss_masks = []

        compute_values = True if self.cfg.algorithm.adv_type == "gae" else False

        for start_idx in range(0, total_samples, micro_batch_size):
            end_idx = min(start_idx + micro_batch_size, total_samples)

            # Slice data for this micro-batch (handles nested dicts recursively)
            micro_data = slice_nested_dict(self.rollout_batch, start_idx, end_idx)
            micro_data = put_tensor_device(
                micro_data, f"cuda:{int(os.environ['LOCAL_RANK'])}"
            )

            prev_logprobs = micro_data["prev_logprobs"]
            loss_mask = micro_data.get("loss_mask", None)

            # Set model-specific params
            if SupportedModel(self.cfg.actor.model.model_type) in [
                SupportedModel.OPENVLA,
                SupportedModel.OPENVLA_OFT,
            ]:
                micro_data["temperature"] = (
                    self.cfg.algorithm.sampling_params.temperature_train
                )
                micro_data["top_k"] = self.cfg.algorithm.sampling_params.top_k

            # Forward pass to get actor logprobs
            with self.amp_context:
                output_dict = self.model(
                    data=micro_data,
                    compute_logprobs=True,
                    compute_entropy=False,
                    compute_values=compute_values,
                    use_cache=False,
                )

            actor_logprobs = output_dict["logprobs"]

            # For GR00T, use the recomputed prev_logprobs
            if SupportedModel(self.cfg.actor.model.model_type) in [SupportedModel.GR00T]:
                prev_logprobs = output_dict["prev_logprobs"]

            all_actor_logprobs.append(actor_logprobs.cpu())
            all_prev_logprobs.append(prev_logprobs.cpu())
            if loss_mask is not None:
                all_loss_masks.append(loss_mask.cpu())

        # Concatenate all micro-batch results
        all_actor_logprobs = torch.cat(all_actor_logprobs, dim=0)
        all_prev_logprobs = torch.cat(all_prev_logprobs, dim=0)
        all_loss_masks = torch.cat(all_loss_masks, dim=0) if all_loss_masks else None

        # Compare
        compare_metrics = compare_rollout_actor_logprobs(
            rollout_logprobs=all_prev_logprobs,
            actor_logprobs=all_actor_logprobs,
            loss_mask=all_loss_masks,
            prefix="logprob_compare/",
        )

        # Log results on rank 0
        if rank == 0:
            print("\n" + "=" * 60, flush=True)
            print(f"[Logprob Compare] All metrics (total_samples={total_samples}):", flush=True)
            for key, value in compare_metrics.items():
                if isinstance(value, float):
                    print(f"  {key}: {value:.6f}", flush=True)
                else:
                    print(f"  {key}: {value}", flush=True)
            print("=" * 60 + "\n", flush=True)

        self.model.train()
        return compare_metrics

    @torch.no_grad()
    def _recompute_old_logprobs(self) -> None:
        """
        Recompute old_logprobs using actor model BEFORE any training updates.
        This replaces the prev_logprobs from rollout with freshly computed logprobs.

        This is useful when:
        1. Rollout worker uses a different model version than actor
        2. Want to ensure logprobs consistency for PPO training

        Note: Data is reshaped from [n_chunk_steps, batch_size, ...] to
        [n_chunk_steps * batch_size, ...] before processing, then reshaped back.
        """
        self.model.eval()

        rank = int(os.environ.get("RANK", 0))
        if rank == 0:
            print("[Recompute Old Logprobs] Starting...", flush=True)

        # Get original batch dimensions for reshaping back
        n_chunk_steps = self.rollout_batch["prev_logprobs"].shape[0]
        batch_size = self.rollout_batch["prev_logprobs"].shape[1]
        total_samples = n_chunk_steps * batch_size

        # Reshape data from [n_chunk_steps, batch_size, ...] to [total_samples, ...]
        reshaped_batch = reshape_nested_dict_for_recompute(self.rollout_batch)

        # Process in micro-batches to avoid OOM
        micro_batch_size = self.cfg.actor.micro_batch_size
        recomputed_logprobs_list = []

        for start_idx in range(0, total_samples, micro_batch_size):
            end_idx = min(start_idx + micro_batch_size, total_samples)

            # Slice data for this micro-batch (handles nested dicts recursively)
            micro_data = slice_nested_dict(reshaped_batch, start_idx, end_idx)
            micro_data = put_tensor_device(
                micro_data, f"cuda:{int(os.environ['LOCAL_RANK'])}"
            )

            # Set model-specific params
            if SupportedModel(self.cfg.actor.model.model_type) in [
                SupportedModel.OPENVLA,
                SupportedModel.OPENVLA_OFT,
            ]:
                micro_data["temperature"] = (
                    self.cfg.algorithm.sampling_params.temperature_train
                )
                micro_data["top_k"] = self.cfg.algorithm.sampling_params.top_k

            compute_values = self.cfg.algorithm.adv_type == "gae"

            # Forward pass to get logprobs
            with self.amp_context:
                output_dict = self.model(
                    data=micro_data,
                    compute_logprobs=True,
                    compute_entropy=False,
                    compute_values=compute_values,
                    use_cache=False,
                )

            # Get the recomputed logprobs
            if SupportedModel(self.cfg.actor.model.model_type) in [SupportedModel.GR00T]:
                logprobs = output_dict["prev_logprobs"]
            else:
                logprobs = output_dict["logprobs"]

            recomputed_logprobs_list.append(logprobs.cpu())

        # Concatenate all micro-batch results: [total_samples, ...]
        recomputed_logprobs = torch.cat(recomputed_logprobs_list, dim=0)

        # Reshape back to [n_chunk_steps, batch_size, ...]
        recomputed_logprobs = recomputed_logprobs.reshape(
            n_chunk_steps, batch_size, *recomputed_logprobs.shape[1:]
        )

        # Store as actor_old_logprobs (keep original prev_logprobs from rollout)
        self.rollout_batch["actor_old_logprobs"] = recomputed_logprobs

        if rank == 0:
            print("[Recompute Old Logprobs] Done. Stored as 'actor_old_logprobs'.", flush=True)

        self.model.train()

    @torch.no_grad()
    def _recompute_old_logprobs_with_snapshot(
        self, initial_state_dict: dict[str, torch.Tensor]
    ) -> None:
        """
        Recompute old_logprobs using a saved weights snapshot (Solution A for async mode).

        This method temporarily swaps the model weights with the initial snapshot,
        computes old_logprobs, then restores the current weights. This ensures all
        epochs use the same model version (v0) for old_logprobs computation.

        Note: Data is reshaped from [n_chunk_steps, batch_size, ...] to
        [n_chunk_steps * batch_size, ...] before processing, then reshaped back.

        Args:
            initial_state_dict: The saved initial model weights (v0 version)
        """
        self.model.eval()

        rank = int(os.environ.get("RANK", 0))
        if rank == 0:
            print("[Recompute Old Logprobs with Snapshot] Starting...", flush=True)

        # Get original batch dimensions for reshaping back
        n_chunk_steps = self.rollout_batch["prev_logprobs"].shape[0]
        batch_size = self.rollout_batch["prev_logprobs"].shape[1]
        total_samples = n_chunk_steps * batch_size

        # Reshape data from [n_chunk_steps, batch_size, ...] to [total_samples, ...]
        reshaped_batch = reshape_nested_dict_for_recompute(self.rollout_batch)

        # Process in micro-batches to avoid OOM
        micro_batch_size = self.cfg.actor.micro_batch_size
        recomputed_logprobs_list = []

        # Use cpu_weight_swap to temporarily load initial weights
        with cpu_weight_swap(self.model, initial_state_dict):
            for start_idx in range(0, total_samples, micro_batch_size):
                end_idx = min(start_idx + micro_batch_size, total_samples)

                # Slice data for this micro-batch (handles nested dicts recursively)
                micro_data = slice_nested_dict(reshaped_batch, start_idx, end_idx)
                micro_data = put_tensor_device(
                    micro_data, f"cuda:{int(os.environ['LOCAL_RANK'])}"
                )

                # Set model-specific params
                if SupportedModel(self.cfg.actor.model.model_type) in [
                    SupportedModel.OPENVLA,
                    SupportedModel.OPENVLA_OFT,
                ]:
                    micro_data["temperature"] = (
                        self.cfg.algorithm.sampling_params.temperature_train
                    )
                    micro_data["top_k"] = self.cfg.algorithm.sampling_params.top_k

                compute_values = self.cfg.algorithm.adv_type == "gae"

                # Forward pass to get logprobs using initial weights
                with self.amp_context:
                    output_dict = self.model(
                        data=micro_data,
                        compute_logprobs=True,
                        compute_entropy=False,
                        compute_values=compute_values,
                        use_cache=False,
                    )

                # Get the recomputed logprobs
                if SupportedModel(self.cfg.actor.model.model_type) in [SupportedModel.GR00T]:
                    logprobs = output_dict["prev_logprobs"]
                else:
                    logprobs = output_dict["logprobs"]

                recomputed_logprobs_list.append(logprobs.cpu())

        # Concatenate all micro-batch results: [total_samples, ...]
        recomputed_logprobs = torch.cat(recomputed_logprobs_list, dim=0)

        # Reshape back to [n_chunk_steps, batch_size, ...]
        recomputed_logprobs = recomputed_logprobs.reshape(
            n_chunk_steps, batch_size, *recomputed_logprobs.shape[1:]
        )

        # Store as actor_old_logprobs (keep original prev_logprobs from rollout)
        self.rollout_batch["actor_old_logprobs"] = recomputed_logprobs

        if rank == 0:
            print(
                "[Recompute Old Logprobs with Snapshot] Done. "
                "Used initial weights (v0) for all epochs.",
                flush=True,
            )

        self.model.train()

    def run_training(self) -> None:
        """
        Run the training process using the received rollout batch.
        """
        if self.is_weight_offloaded:
            self.load_param_and_grad(self.device)
        if self.is_optimizer_offloaded:
            self.load_optimizer(self.device)

        self.model.train()
        rollout_size = (
            self.rollout_batch["prev_logprobs"].shape[0]
            * self.rollout_batch["prev_logprobs"].shape[1]
        )

        # [DEBUG] 检查各字段的大小，用于调试 rewards 长度问题
        logger.info(f"[DEBUG Actor] rollout_batch shapes before process_nested_dict_for_train:")
        for key in ["prev_logprobs", "prev_values", "dones", "rewards", "terminations", "truncations"]:
            if key in self.rollout_batch and self.rollout_batch[key] is not None:
                shape = self.rollout_batch[key].shape
                flat_size = shape[0] * shape[1] if len(shape) >= 2 else shape[0]
                logger.info(f"[DEBUG Actor]   {key}: shape={shape}, flat_size={flat_size}")
        logger.info(f"[DEBUG Actor] rollout_size (from prev_logprobs): {rollout_size}")

        g = torch.Generator()
        g.manual_seed(self.cfg.actor.seed + self._rank)
        shuffle_id = torch.randperm(rollout_size, generator=g)

        with torch.no_grad():
            self.rollout_batch = process_nested_dict_for_train(
                self.rollout_batch, shuffle_id
            )

        assert self.cfg.actor.global_batch_size % self._world_size == 0, (
            "global_batch_size is not divisible by actor world_size"
        )

        batch_size_per_rank = self.cfg.actor.global_batch_size // self._world_size
        self.gradient_accumulation = get_num_micro_batches(
            batch_size_per_rank, self.cfg.actor.micro_batch_size
        )

        # Split to make minibatch iterator for updating the actor
        # See PPO paper for details. https://arxiv.org/abs/1707.06347
        rollout_size = self.rollout_batch["prev_logprobs"].size(0)
        assert rollout_size % batch_size_per_rank == 0, (
            f"{rollout_size} is not divisible by {batch_size_per_rank}"
        )
        metrics = {}

        # Compare rollout vs actor logprobs BEFORE any training updates
        if self.cfg.algorithm.get("compare_logprobs", False):
            metrics.update(self._compare_logprobs_before_training())

        update_epoch = self.cfg.algorithm.get("update_epoch", 1)
        for _ in range(update_epoch):
            rollout_dataloader_iter = get_iterator_k_split(
                self.rollout_batch,
                rollout_size // batch_size_per_rank,
            )
            for train_global_batch in rollout_dataloader_iter:
                # split batch into micro_batches
                train_global_batch_size = train_global_batch["prev_logprobs"].shape[0]
                assert (
                    train_global_batch_size
                    == self.cfg.actor.global_batch_size
                    // torch.distributed.get_world_size()
                )
                num_micro_batches = get_num_micro_batches(
                    train_global_batch_size, self.cfg.actor.micro_batch_size
                )
                train_micro_batch = iter_dict_micro_batches(
                    train_global_batch,
                    batch_size=train_global_batch_size,
                    micro_batch_size=self.cfg.actor.micro_batch_size,
                )

                self.optimizer.zero_grad()
                for idx, data in enumerate(train_micro_batch):
                    actual_micro_batch_size = data["prev_logprobs"].shape[0]
                    data = put_tensor_device(
                        data, f"cuda:{int(os.environ['LOCAL_RANK'])}"
                    )
                    backward_ctx = self.before_micro_batch(
                        self.model,
                        is_last_micro_batch=(idx + 1) == num_micro_batches,
                    )
                    advantages = data["advantages"]
                    # Use actor_old_logprobs if available, otherwise use prev_logprobs
                    old_logprobs = data.get("actor_old_logprobs", data["prev_logprobs"])
                    returns = data.get("returns", None)
                    prev_values = data.get("prev_values", None)
                    loss_mask = data.get("loss_mask", None)
                    loss_mask_sum = data.get("loss_mask_sum", None)

                    if SupportedModel(self.cfg.actor.model.model_type) in [
                        SupportedModel.OPENVLA,
                        SupportedModel.OPENVLA_OFT,
                    ]:
                        data["temperature"] = (
                            self.cfg.algorithm.sampling_params.temperature_train
                        )
                        data["top_k"] = self.cfg.algorithm.sampling_params.top_k

                    compute_values = (
                        True if self.cfg.algorithm.adv_type == "gae" else False
                    )

                    with self.amp_context:
                        output_dict = self.model(
                            data=data,
                            compute_logprobs=True,
                            compute_entropy=self.cfg.algorithm.entropy_bonus > 0,
                            compute_values=compute_values,
                            use_cache=False,
                        )

                    if SupportedModel(self.cfg.actor.model.model_type) in [
                        SupportedModel.GR00T
                    ]:
                        old_logprobs = output_dict["prev_logprobs"]

                    kwargs = {
                        "loss_type": self.cfg.algorithm.loss_type,
                        "logprob_type": self.cfg.algorithm.logprob_type,
                        "reward_type": self.cfg.algorithm.reward_type,
                        "single_action_dim": self.cfg.actor.model.get("action_dim", 7),
                        "logprobs": output_dict["logprobs"],
                        "values": output_dict.get("values", None),
                        "old_logprobs": old_logprobs,
                        "advantages": advantages,
                        "returns": returns,
                        "prev_values": prev_values,
                        "clip_ratio_high": self.cfg.algorithm.clip_ratio_high,
                        "clip_ratio_low": self.cfg.algorithm.clip_ratio_low,
                        "value_clip": self.cfg.algorithm.get("value_clip", None),
                        "huber_delta": self.cfg.algorithm.get("huber_delta", None),
                        "loss_mask": loss_mask,
                        "loss_mask_sum": loss_mask_sum,
                        "max_episode_steps": self.cfg.env.train.max_episode_steps,
                        "task_type": self.cfg.runner.task_type,
                        "critic_warmup": self.optimizer_steps
                        < self.critic_warmup_steps,
                    }
                    loss, metrics_data = policy_loss(**kwargs)

                    entropy_loss = torch.tensor(0.0, device=torch.cuda.current_device())
                    if (
                        self.cfg.algorithm.entropy_bonus > 0
                        and not kwargs["critic_warmup"]
                    ):
                        entropy = output_dict["entropy"]
                        entropy = reshape_entropy(
                            entropy,
                            entropy_type=self.cfg.algorithm.entropy_type,
                            action_dim=self.cfg.actor.model.get("action_dim", 7),
                            batch_size=output_dict["logprobs"].shape[0],
                        )
                        entropy_loss = masked_mean(entropy, mask=loss_mask)
                        loss -= self.cfg.algorithm.entropy_bonus * entropy_loss
                    metrics_data["entropy_loss"] = entropy_loss.detach().item()

                    loss = loss * (actual_micro_batch_size / train_global_batch_size)
                    with backward_ctx:
                        self.grad_scaler.scale(loss).backward()

                    metrics_data["loss"] = loss.detach().item()
                    append_to_dict(metrics, metrics_data)

                torch.cuda.empty_cache()

                grad_norm, lr_list = self.optimizer_step()
                data = {
                    "actor/grad_norm": grad_norm,
                    "actor/lr": lr_list[0],
                }
                if len(lr_list) > 1:
                    data["critic/lr"] = lr_list[1]
                append_to_dict(metrics, data)
        # put LR scheduler step here
        self.lr_scheduler.step()
        self.optimizer.zero_grad()
        clear_memory()
        mean_metric_dict = {key: np.mean(value) for key, value in metrics.items()}
        mean_metric_dict = all_reduce_dict(
            mean_metric_dict, op=torch.distributed.ReduceOp.AVG
        )

        return mean_metric_dict

    def set_global_step(self, global_step) -> None:
        """
        Set the global step for the model, if needed.
        """
        if hasattr(self.model, "set_global_step"):
            self.model.set_global_step(global_step)

    def run_training_single_epoch(self) -> dict:
        """
        Run training for a single epoch's data.

        与 run_training() 类似，但针对单个 epoch 的数据进行训练。
        单个 epoch 的数据形状为 [n_chunk_steps, bsz, num_action_chunks, ...]。
        """
        if self.is_weight_offloaded:
            self.load_param_and_grad(self.device)
        if self.is_optimizer_offloaded:
            self.load_optimizer(self.device)

        self.model.train()
        rollout_size = (
            self.rollout_batch["prev_logprobs"].shape[0]
            * self.rollout_batch["prev_logprobs"].shape[1]
        )
        g = torch.Generator()
        g.manual_seed(self.cfg.actor.seed + self._rank)
        shuffle_id = torch.randperm(rollout_size, generator=g)

        with torch.no_grad():
            self.rollout_batch = process_nested_dict_for_train(
                self.rollout_batch, shuffle_id
            )

        assert self.cfg.actor.global_batch_size % self._world_size == 0, (
            "global_batch_size is not divisible by actor world_size"
        )

        batch_size_per_rank = self.cfg.actor.global_batch_size // self._world_size
        self.gradient_accumulation = get_num_micro_batches(
            batch_size_per_rank, self.cfg.actor.micro_batch_size
        )

        # Split to make minibatch iterator for updating the actor
        # For single epoch: rollout_size = n_chunk_steps * bsz (already flattened by process_nested_dict_for_train)
        rollout_size = self.rollout_batch["prev_logprobs"].size(0)
        # For async pipeline with single epoch, we need to handle the case where
        # rollout_size might not be divisible by batch_size_per_rank
        # In this case, we find a suitable effective batch size
        if rollout_size % batch_size_per_rank == 0:
            # Data size is divisible by configured batch size
            effective_batch_size = batch_size_per_rank
        else:
            # Data size is not divisible by configured batch size
            # Use rollout_size as the effective batch size (single batch for this epoch)
            effective_batch_size = rollout_size
        metrics = {}
        update_epoch = self.cfg.algorithm.get("update_epoch", 1)
        for _ in range(update_epoch):
            rollout_dataloader_iter = get_iterator_k_split(
                self.rollout_batch,
                rollout_size // effective_batch_size,
            )
            for train_global_batch in rollout_dataloader_iter:
                # split batch into micro_batches
                train_global_batch_size = train_global_batch["prev_logprobs"].shape[0]
                # For async pipeline, train_global_batch_size might be different from configured global_batch_size
                # This is acceptable - we handle flexible batch sizes for async pipeline

                num_micro_batches = get_num_micro_batches(
                    train_global_batch_size, self.cfg.actor.micro_batch_size
                )
                train_micro_batch = iter_dict_micro_batches(
                    train_global_batch,
                    batch_size=train_global_batch_size,
                    micro_batch_size=self.cfg.actor.micro_batch_size,
                )

                self.optimizer.zero_grad()
                for idx, data in enumerate(train_micro_batch):
                    actual_micro_batch_size = data["prev_logprobs"].shape[0]
                    data = put_tensor_device(
                        data, f"cuda:{int(os.environ['LOCAL_RANK'])}"
                    )
                    backward_ctx = self.before_micro_batch(
                        self.model,
                        is_last_micro_batch=(idx + 1) == num_micro_batches,
                    )
                    advantages = data["advantages"]
                    # Use actor_old_logprobs if available, otherwise use prev_logprobs
                    old_logprobs = data.get("actor_old_logprobs", data["prev_logprobs"])
                    returns = data.get("returns", None)
                    prev_values = data.get("prev_values", None)
                    loss_mask = data.get("loss_mask", None)
                    loss_mask_sum = data.get("loss_mask_sum", None)

                    if SupportedModel(self.cfg.actor.model.model_type) in [
                        SupportedModel.OPENVLA,
                        SupportedModel.OPENVLA_OFT,
                    ]:
                        data["temperature"] = (
                            self.cfg.algorithm.sampling_params.temperature_train
                        )
                        data["top_k"] = self.cfg.algorithm.sampling_params.top_k

                    compute_values = (
                        True if self.cfg.algorithm.adv_type == "gae" else False
                    )

                    with self.amp_context:
                        output_dict = self.model(
                            data=data,
                            compute_logprobs=True,
                            compute_entropy=self.cfg.algorithm.entropy_bonus > 0,
                            compute_values=compute_values,
                            use_cache=False,
                        )

                    if SupportedModel(self.cfg.actor.model.model_type) in [
                        SupportedModel.GR00T
                    ]:
                        old_logprobs = output_dict["prev_logprobs"]

                    kwargs = {
                        "loss_type": self.cfg.algorithm.loss_type,
                        "logprob_type": self.cfg.algorithm.logprob_type,
                        "reward_type": self.cfg.algorithm.reward_type,
                        "single_action_dim": self.cfg.actor.model.get("action_dim", 7),
                        "logprobs": output_dict["logprobs"],
                        "values": output_dict.get("values", None),
                        "old_logprobs": old_logprobs,
                        "advantages": advantages,
                        "returns": returns,
                        "prev_values": prev_values,
                        "clip_ratio_high": self.cfg.algorithm.clip_ratio_high,
                        "clip_ratio_low": self.cfg.algorithm.clip_ratio_low,
                        "value_clip": self.cfg.algorithm.get("value_clip", None),
                        "huber_delta": self.cfg.algorithm.get("huber_delta", None),
                        "loss_mask": loss_mask,
                        "loss_mask_sum": loss_mask_sum,
                        "max_episode_steps": self.cfg.env.train.max_episode_steps,
                        "task_type": self.cfg.runner.task_type,
                        "critic_warmup": self.optimizer_steps
                        < self.critic_warmup_steps,
                    }
                    loss, metrics_data = policy_loss(**kwargs)

                    entropy_loss = torch.tensor(0.0, device=torch.cuda.current_device())
                    if (
                        self.cfg.algorithm.entropy_bonus > 0
                        and not kwargs["critic_warmup"]
                    ):
                        entropy = output_dict["entropy"]
                        entropy = reshape_entropy(
                            entropy,
                            entropy_type=self.cfg.algorithm.entropy_type,
                            action_dim=self.cfg.actor.model.get("action_dim", 7),
                            batch_size=output_dict["logprobs"].shape[0],
                        )
                        entropy_loss = masked_mean(entropy, mask=loss_mask)
                        loss -= self.cfg.algorithm.entropy_bonus * entropy_loss
                    metrics_data["entropy_loss"] = entropy_loss.detach().item()

                    loss = loss * (actual_micro_batch_size / train_global_batch_size)
                    with backward_ctx:
                        self.grad_scaler.scale(loss).backward()

                    metrics_data["loss"] = loss.detach().item()
                    append_to_dict(metrics, metrics_data)

                torch.cuda.empty_cache()

                grad_norm, lr_list = self.optimizer_step()
                data = {
                    "actor/grad_norm": grad_norm,
                    "actor/lr": lr_list[0],
                }
                if len(lr_list) > 1:
                    data["critic/lr"] = lr_list[1]
                append_to_dict(metrics, data)

        # put LR scheduler step here
        self.lr_scheduler.step()
        self.optimizer.zero_grad()
        clear_memory()
        mean_metric_dict = {key: np.mean(value) for key, value in metrics.items()}
        # With barrier synchronization in async_train_loop, all actors are now synchronized
        # so all_reduce_dict should work correctly
        mean_metric_dict = all_reduce_dict(
            mean_metric_dict, op=torch.distributed.ReduceOp.AVG
        )

        return mean_metric_dict

    async def async_train_loop(self, input_channel: Channel, num_epochs: int):
        """
        异步训练循环：持续从 channel 获取数据并训练

        根据配置参数决定使用哪种训练模式：
        - streaming_mode: True  -> 流式模式（每收到1条消息就训练）
        - streaming_mode: False -> 批量模式（收齐N条消息后训练）

        Args:
            input_channel: 数据输入 channel
            num_epochs: 预期接收的 epoch 数量

        Configuration:
            actor.streaming_mode: bool
                当设为 True 时，启用流式训练模式（方案 B1）。
                每收到 1 条消息，所有 Actor 同步训练。
            actor.gradient_accumulation_across_epochs: bool
                当设为 True 时，启用跨 epoch 梯度累积。
        """
        # 检查是否启用流式模式
        use_streaming = self.cfg.actor.get("streaming_mode", False)

        if use_streaming:
            # 流式模式（方案 B1）：每收到 1 条消息就训练
            return await self.async_train_loop_streaming(
                input_channel, num_epochs
            )

        # 检查是否启用跨 epoch 梯度累积
        use_gradient_accumulation = self.cfg.actor.get(
            "gradient_accumulation_across_epochs", False
        )

        if use_gradient_accumulation:
            # 批量模式 + 梯度累积（文档 13.9）
            return await self.async_train_loop_with_gradient_accumulation(
                input_channel, num_epochs
            )

        # 原有的 per-epoch 训练逻辑
        epoch_count = 0
        all_metrics = []

        # 记录总训练时间
        import time as time_module
        total_training_time = 0.0

        # 方案 A：保存初始权重快照用于计算 old_logprobs
        initial_state_dict = None
        if self.cfg.algorithm.get("recompute_old_logprobs", False):
            rank = int(os.environ.get("RANK", 0))
            if rank == 0:
                print("[Async Train] Saving initial weights snapshot for old_logprobs...", flush=True)
            initial_state_dict = retrieve_model_state_dict_in_cpu(self.model)
            if rank == 0:
                print("[Async Train] Initial weights snapshot saved.", flush=True)

        while epoch_count < num_epochs:
            # 从 channel 获取一个 epoch 的数据
            self.recv_single_epoch_batch(input_channel)

            # 检查是否收到结束信号
            if self.rollout_batch.get("__done__", False):
                break

            # CRITICAL: Barrier to synchronize all actors before training
            # FSDP requires all workers to be synchronized during forward/backward passes
            # Without this barrier, different actors may be at different epochs,
            # causing FSDP's all-gather/reduce-scatter operations to deadlock
            torch.distributed.barrier()

            # Recompute old_logprobs using initial weights snapshot
            if self.cfg.algorithm.get("recompute_old_logprobs", False):
                if self.is_weight_offloaded:
                    self.load_param_and_grad(self.device)
                self._recompute_old_logprobs_with_snapshot(initial_state_dict)

            # 计算 advantage
            rollout_metrics = self.compute_advantages_and_returns()

            # 执行训练并记录时间
            training_start = time_module.time()
            training_metrics = self.run_training_single_epoch()
            training_time = time_module.time() - training_start
            total_training_time += training_time

            epoch_count += 1
            all_metrics.append({
                "rollout": rollout_metrics,
                "training": training_metrics,
                "epoch": epoch_count,
                "training_time": training_time  # 添加训练时间
            })

        # 清理快照
        if initial_state_dict is not None:
            del initial_state_dict
        # Ensure no leftover rollout messages are carried into the next global step.
        self._drain_actor_rollout_messages(input_channel)
        clear_memory(sync=False)

        return all_metrics

    def recv_single_message(self, input_channel: Channel) -> bool:
        """
        流式接收：每次只接收 1 条消息，立即返回用于训练。

        用于方案 B1：所有 Actor 同步接收 1 条消息后一起训练。

        Returns:
            bool: True 表示收到有效数据，False 表示收到结束信号
        """
        recv_key = f"actor_{self._rank}"
        data = input_channel.get(key=recv_key)

        # 检查结束信号
        if isinstance(data, dict) and data.get("__done__", False):
            self.rollout_batch = {"__done__": True}
            return False

        # 处理单条消息
        self.rollout_batch = self._process_received_rollout_batch(
            data, is_single_epoch=True
        )
        return True

    def _drain_actor_rollout_messages(self, input_channel: Channel) -> int:
        """
        Drain (discard) any leftover rollout messages in this actor's channel queue.

        In async pipeline, if per-epoch message counting is slightly off or if some
        messages arrive late, leftover messages can be consumed in the next global
        step, causing epoch/step misalignment. We intentionally drop them to avoid
        cross-step computation.
        """
        recv_key = f"actor_{self._rank}"
        drained = 0
        while True:
            try:
                _ = input_channel.get_nowait(key=recv_key)
            except asyncio.QueueEmpty:
                break
            drained += 1
        if drained > 0:
            logger.warning(
                "[Async Train] Drained %s leftover messages from key=%s to avoid cross-step "
                "misalignment.",
                drained,
                recv_key,
            )
        return drained

    def get_messages_per_epoch(self) -> int:
        """计算每个 epoch 每个 actor 应接收的消息总数"""
        rollout_world_size = self._component_placement.get_world_size("rollout")
        env_world_size = self._component_placement.get_world_size("env")
        actor_world_size = self._component_placement.get_world_size("actor")
        per_env_async_cfg = self.cfg.rollout.get("per_env_async", {})
        per_env_async_enabled = per_env_async_cfg.get("enabled", False)
        aggregate_slots_per_env = int(per_env_async_cfg.get("aggregate_slots_per_env", 1))

        if per_env_async_enabled:
            default_batch_size = (
                self.cfg.env.train.total_num_envs
                // max(1, env_world_size)
                // max(1, self.stage_num)
            )
            flex_cfg = per_env_async_cfg.get("flex", None)
            if flex_cfg and flex_cfg.get("enabled", False):
                plan = build_flex_plan(
                    cfg=self.cfg,
                    env_world_size=env_world_size,
                    rollout_world_size=rollout_world_size,
                    default_stage_num=self.stage_num,
                    default_batch_size=default_batch_size,
                )
                send_num = plan.total_slot_count
            else:
                send_num = env_world_size * self.stage_num * aggregate_slots_per_env
        else:
            send_num = rollout_world_size * self.stage_num
        recv_num = actor_world_size
        split_num = compute_split_num(recv_num, send_num)

        # In async streaming mode, rollout workers send:
        #   - `send_num` logical units (per-env flex slots / or per-stage slots)
        #   - each logical unit is split into `split_num` channel messages
        # so total messages per epoch across all actor ranks is:
        #   send_num * split_num
        #
        # NOTE:
        #   When flex mode is enabled, `self.stage_num` is not the real producer dimension
        #   (it is derived from `env.per_env_async.flex.manager_batch_sizes_by_env_rank`).
        #   Therefore total_messages must be computed from `send_num`, not `rollout_world_size * self.stage_num`.
        total_messages = send_num * split_num
        my_message_count = total_messages // actor_world_size
        if self._rank < total_messages % actor_world_size:
            my_message_count += 1
        return my_message_count

    def recv_single_epoch_batch(self, input_channel: Channel) -> None:
        """
        接收单个 epoch 的 rollout 数据，使用确定性路由（批量模式）

        Per-Env Async 模式下的消息流:
        - 每个 rollout worker 有 stage_num 个 stage
        - 每个 stage 每个 epoch 发送 split_num 条消息
        - 总消息数 = rollout_world_size * stage_num * split_num
        """
        rollout_world_size = self._component_placement.get_world_size("rollout")
        env_world_size = self._component_placement.get_world_size("env")
        actor_world_size = self._component_placement.get_world_size("actor")
        per_env_async_cfg = self.cfg.rollout.get("per_env_async", {})
        per_env_async_enabled = per_env_async_cfg.get("enabled", False)
        aggregate_slots_per_env = int(per_env_async_cfg.get("aggregate_slots_per_env", 1))

        # Calculate split_num (messages per stage per rollout worker)
        # Must match rollout's get_actor_split_num(): compute_split_num(recv_num, send_num)
        if per_env_async_enabled:
            default_batch_size = (
                self.cfg.env.train.total_num_envs
                // max(1, env_world_size)
                // max(1, self.stage_num)
            )
            flex_cfg = per_env_async_cfg.get("flex", None)
            if flex_cfg and flex_cfg.get("enabled", False):
                plan = build_flex_plan(
                    cfg=self.cfg,
                    env_world_size=env_world_size,
                    rollout_world_size=rollout_world_size,
                    default_stage_num=self.stage_num,
                    default_batch_size=default_batch_size,
                )
                send_num = plan.total_slot_count
            else:
                send_num = env_world_size * self.stage_num * aggregate_slots_per_env
        else:
            send_num = rollout_world_size * self.stage_num
        recv_num = actor_world_size
        split_num = compute_split_num(recv_num, send_num)

        # Calculate how many messages this actor should receive per epoch
        # Total messages = rollout_world_size * stage_num * split_num
        # (每个 rollout worker 的每个 stage 都发送 split_num 条消息)
        total_messages = send_num * split_num
        my_message_count = total_messages // actor_world_size
        if self._rank < total_messages % actor_world_size:
            my_message_count += 1

        self.rollout_batch = {}
        recv_list = []

        # Use actor_{rank} as key for deterministic routing
        recv_key = f"actor_{self._rank}"
        for _ in range(my_message_count):
            data = input_channel.get(key=recv_key)
            # 检查结束信号
            if isinstance(data, dict) and data.get("__done__", False):
                self.rollout_batch = {"__done__": True}
                return
            recv_list.append(data)

        self.rollout_batch = self._cat_rollout_batches(recv_list, dim=1)

        # Process single epoch data (skip cross-epoch reshape)
        self.rollout_batch = self._process_received_rollout_batch(
            self.rollout_batch, is_single_epoch=True
        )

    def _flatten_batch_for_gradient_accumulation(
        self, batch: dict[str, torch.Tensor]
    ) -> tuple[dict[str, torch.Tensor], int]:
        """
        Flatten batch from [n_chunk_steps, batch_size, ...] to [total_samples, ...]
        for gradient accumulation training.

        This method processes the batch similar to process_nested_dict_for_train
        but without shuffling (shuffling happens at slice level).

        Args:
            batch: Rollout batch with shape [n_chunk_steps, batch_size, ...]

        Returns:
            Tuple of (flattened_batch, total_samples)
        """
        rollout_size = batch["prev_logprobs"].shape[0] * batch["prev_logprobs"].shape[1]

        # Create identity permutation (no shuffle at this stage)
        identity_id = torch.arange(rollout_size)

        with torch.no_grad():
            flattened_batch = process_nested_dict_for_train(batch, identity_id)

        return flattened_batch, rollout_size

    def _train_batch_slice_accumulate(
        self,
        batch_slice: dict[str, torch.Tensor],
        target_batch_size: int,
        is_last_slice_before_update: bool = False,
    ) -> dict:
        """
        Train on a batch slice with gradient accumulation (no optimizer.step()).

        This method performs forward and backward passes on the given batch slice,
        accumulating gradients without updating parameters.

        Loss scaling: Each sample's contribution to the gradient is `1/target_batch_size`,
        ensuring consistent gradient magnitude regardless of slice size.

        Args:
            batch_slice: Sliced batch data with shape [slice_size, ...]
            target_batch_size: Target batch size for one optimizer.step()
                              Used for correct loss scaling.
            is_last_slice_before_update: If True, this is the last slice before
                              optimizer.step() will be called. Only in this case
                              should FSDP perform Reduce-Scatter on the final
                              micro-batch. This is critical for FSDP full_shard
                              mode with cross-epoch gradient accumulation.

        Returns:
            Training metrics for this slice
        """
        self.model.train()
        slice_size = batch_slice["prev_logprobs"].shape[0]

        # Shuffle the slice
        g = torch.Generator()
        g.manual_seed(self.cfg.actor.seed + self._rank + slice_size)  # Vary seed by slice
        shuffle_id = torch.randperm(slice_size, generator=g)

        with torch.no_grad():
            shuffled_slice = {}
            for key, value in batch_slice.items():
                if value is None:
                    shuffled_slice[key] = None
                elif isinstance(value, torch.Tensor):
                    shuffled_slice[key] = value[shuffle_id]
                elif isinstance(value, dict):
                    shuffled_slice[key] = {
                        k: v[shuffle_id] if isinstance(v, torch.Tensor) else v
                        for k, v in value.items()
                    }
                else:
                    shuffled_slice[key] = value
            batch_slice = shuffled_slice

        micro_batch_size = self.cfg.actor.micro_batch_size
        actual_num_micro_batches = get_num_micro_batches(slice_size, micro_batch_size)
        micro_batch_iter = iter_dict_micro_batches(
            batch_slice,
            batch_size=slice_size,
            micro_batch_size=micro_batch_size,
        )

        metrics = {}

        # Do NOT zero_grad here - we're accumulating gradients across epochs

        # Calculate the loss scale factor:
        # Each sample should contribute 1/target_batch_size to the gradient.
        # Since loss is averaged over micro-batch, we scale by:
        # (micro_batch_actual_size / target_batch_size)
        # This ensures consistent per-sample gradient contribution.

        for idx, data in enumerate(micro_batch_iter):
            data = put_tensor_device(data, f"cuda:{int(os.environ['LOCAL_RANK'])}")

            # Get actual micro-batch size (may vary for last batch with remainder)
            actual_micro_batch_size = data["prev_logprobs"].shape[0]

            # Only set is_last_micro_batch=True when:
            # 1. This is the last micro-batch in this slice, AND
            # 2. This slice is the last one before optimizer.step()
            # This ensures FSDP only performs Reduce-Scatter right before optimizer.step()
            is_last = (idx + 1) == actual_num_micro_batches and is_last_slice_before_update
            backward_ctx = self.before_micro_batch(
                self.model,
                is_last_micro_batch=is_last,
            )

            advantages = data["advantages"]
            old_logprobs = data.get("actor_old_logprobs", data["prev_logprobs"])
            returns = data.get("returns", None)
            prev_values = data.get("prev_values", None)
            loss_mask = data.get("loss_mask", None)
            loss_mask_sum = data.get("loss_mask_sum", None)

            if SupportedModel(self.cfg.actor.model.model_type) in [
                SupportedModel.OPENVLA,
                SupportedModel.OPENVLA_OFT,
            ]:
                data["temperature"] = (
                    self.cfg.algorithm.sampling_params.temperature_train
                )
                data["top_k"] = self.cfg.algorithm.sampling_params.top_k

            compute_values = self.cfg.algorithm.adv_type == "gae"

            with self.amp_context:
                output_dict = self.model(
                    data=data,
                    compute_logprobs=True,
                    compute_entropy=self.cfg.algorithm.entropy_bonus > 0,
                    compute_values=compute_values,
                    use_cache=False,
                )

            if SupportedModel(self.cfg.actor.model.model_type) in [SupportedModel.GR00T]:
                old_logprobs = output_dict["prev_logprobs"]

            kwargs = {
                "loss_type": self.cfg.algorithm.loss_type,
                "logprob_type": self.cfg.algorithm.logprob_type,
                "reward_type": self.cfg.algorithm.reward_type,
                "single_action_dim": self.cfg.actor.model.get("action_dim", 7),
                "logprobs": output_dict["logprobs"],
                "values": output_dict.get("values", None),
                "old_logprobs": old_logprobs,
                "advantages": advantages,
                "returns": returns,
                "prev_values": prev_values,
                "clip_ratio_high": self.cfg.algorithm.clip_ratio_high,
                "clip_ratio_low": self.cfg.algorithm.clip_ratio_low,
                "value_clip": self.cfg.algorithm.get("value_clip", None),
                "huber_delta": self.cfg.algorithm.get("huber_delta", None),
                "loss_mask": loss_mask,
                "loss_mask_sum": loss_mask_sum,
                "max_episode_steps": self.cfg.env.train.max_episode_steps,
                "task_type": self.cfg.runner.task_type,
                "critic_warmup": self.optimizer_steps < self.critic_warmup_steps,
            }
            loss, metrics_data = policy_loss(**kwargs)

            entropy_loss = torch.tensor(0.0, device=torch.cuda.current_device())
            if self.cfg.algorithm.entropy_bonus > 0 and not kwargs["critic_warmup"]:
                entropy = output_dict["entropy"]
                entropy = reshape_entropy(
                    entropy,
                    entropy_type=self.cfg.algorithm.entropy_type,
                    action_dim=self.cfg.actor.model.get("action_dim", 7),
                    batch_size=output_dict["logprobs"].shape[0],
                )
                entropy_loss = masked_mean(entropy, mask=loss_mask)
                loss -= self.cfg.algorithm.entropy_bonus * entropy_loss
            metrics_data["entropy_loss"] = entropy_loss.detach().item()

            # Correct loss scaling for gradient accumulation:
            # The loss from policy_loss is already mean-reduced over the micro-batch.
            # We need to scale it so that each sample contributes 1/target_batch_size.
            # Since loss is mean over actual_micro_batch_size samples:
            #   loss = sum(sample_losses) / actual_micro_batch_size
            # We want total gradient = sum(sample_gradients) / target_batch_size
            # So we scale: loss *= (actual_micro_batch_size / target_batch_size)
            loss = loss * (actual_micro_batch_size / target_batch_size)

            with backward_ctx:
                self.grad_scaler.scale(loss).backward()

            metrics_data["loss"] = loss.detach().item()
            append_to_dict(metrics, metrics_data)

        return metrics

    async def async_train_loop_streaming(
        self, input_channel: Channel, num_epochs: int
    ):
        """
        流式异步训练循环（方案 B1）：每收到 1 条消息，所有 Actor 同步训练。

        与 async_train_loop_with_gradient_accumulation 的区别：
        - 批量模式：收齐 N 条消息后训练
        - 流式模式：每收到 1 条消息就训练（所有 Actor 同步）

        Args:
            input_channel: 数据输入 channel
            num_epochs: 预期接收的 epoch 数量
        """
        import time as time_module

        rank = int(os.environ.get("RANK", 0))
        all_metrics = []
        total_training_time = 0.0

        # 配置
        target_batch_size = self.cfg.actor.global_batch_size // self._world_size
        update_epoch = self.cfg.algorithm.get("update_epoch", 1)
        messages_per_epoch = self.get_messages_per_epoch()
        total_messages = num_epochs * messages_per_epoch

        if rank == 0:
            print(
                f"[Streaming Train] Starting: target_batch_size={target_batch_size}, "
                f"update_epoch={update_epoch}, messages_per_epoch={messages_per_epoch}, "
                f"total_messages={total_messages}",
                flush=True,
            )

        # 初始化
        if self.is_weight_offloaded:
            self.load_param_and_grad(self.device)
        if self.is_optimizer_offloaded:
            self.load_optimizer(self.device)
        self.optimizer.zero_grad()

        accumulated_samples = 0
        message_count = 0
        leftover_batch = None
        leftover_size = 0

        adv_type = self.cfg.algorithm.get("adv_type", "gae")
        grpo_group_size = int(self.cfg.algorithm.get("group_size", 1))
        # GRPO / reinpp 等需要在 batch 维上按 group_size 分组；流式单条消息常为 bsz=1。
        needs_grouped_adv = adv_type != "gae" and grpo_group_size > 1
        streaming_group_buffer: list = []

        while message_count < total_messages:
            # 所有 Actor 同步接收 1 条消息
            has_data = self.recv_single_message(input_channel)

            if not has_data:
                break

            message_count += 1

            # Barrier: 确保所有 Actor 都收到消息后再训练
            torch.distributed.barrier()

            if needs_grouped_adv:
                streaming_group_buffer.append(self.rollout_batch)
                buf_bsz = sum(int(b["rewards"].shape[1]) for b in streaming_group_buffer)
                if buf_bsz % grpo_group_size != 0:
                    continue
                self.rollout_batch = self._cat_rollout_batches(
                    streaming_group_buffer, dim=1
                )
                streaming_group_buffer.clear()
                if self.cfg.algorithm.get("filter_rewards", False):
                    self.rollout_batch = self._apply_reward_filter(self.rollout_batch)

            # 计算 advantage
            rollout_metrics = self.compute_advantages_and_returns()

            training_start = time_module.time()

            # 展平数据
            current_batch, current_size = self._flatten_batch_for_gradient_accumulation(
                self.rollout_batch
            )

            # 合并 leftover
            if leftover_batch is not None and leftover_size > 0:
                combined_batch = cat_list_of_dict_tensor(
                    [leftover_batch, current_batch], dim=0
                )
                combined_size = leftover_size + current_size
            else:
                combined_batch = current_batch
                combined_size = current_size

            # 训练循环（支持 update_epoch）
            epoch_metrics_list = []
            for update_idx in range(update_epoch):
                offset = 0
                while offset < combined_size:
                    remaining = target_batch_size - accumulated_samples
                    samples_to_use = min(remaining, combined_size - offset)

                    batch_slice = slice_nested_dict(
                        combined_batch, offset, offset + samples_to_use
                    )

                    # 判断这个 slice 训练完后是否会触发 optimizer.step()
                    will_update_after_this_slice = (
                        accumulated_samples + samples_to_use
                    ) >= target_batch_size

                    # CRITICAL FIX for FSDP full_shard mode:
                    # Check if this is the LAST slice of the LAST message.
                    # If so, we MUST trigger Reduce-Scatter even if we won't
                    # reach target_batch_size, because we need to sync gradients
                    # before the final optimizer.step().
                    is_last_slice_of_last_message = (
                        message_count >= total_messages
                        and update_idx == update_epoch - 1
                        and (offset + samples_to_use) >= combined_size
                    )
                    # Force reduce-scatter on the last slice of the last message
                    force_reduce_scatter = (
                        is_last_slice_of_last_message
                        and not will_update_after_this_slice
                    )

                    slice_metrics = self._train_batch_slice_accumulate(
                        batch_slice,
                        target_batch_size=target_batch_size,
                        is_last_slice_before_update=(
                            will_update_after_this_slice or force_reduce_scatter
                        ),
                    )

                    accumulated_samples += samples_to_use
                    offset += samples_to_use

                    if accumulated_samples >= target_batch_size:
                        grad_norm, lr_list = self.optimizer_step()
                        self.optimizer.zero_grad()

                        update_metrics = {
                            "actor/grad_norm": grad_norm,
                            "actor/lr": lr_list[0],
                        }
                        for k, v in slice_metrics.items():
                            update_metrics[k] = np.mean(v) if isinstance(v, list) else v

                        update_metrics = all_reduce_dict(
                            update_metrics, op=torch.distributed.ReduceOp.AVG
                        )
                        epoch_metrics_list.append(update_metrics)
                        accumulated_samples = 0

            # 保存 leftover
            leftover_batch = None
            leftover_size = 0

            training_time = time_module.time() - training_start
            total_training_time += training_time

            # 聚合 metrics
            if epoch_metrics_list:
                agg_metrics = {}
                for k in epoch_metrics_list[0]:
                    agg_metrics[k] = np.mean([m[k] for m in epoch_metrics_list])
            else:
                agg_metrics = {}

            all_metrics.append({
                "rollout": rollout_metrics,
                "training": agg_metrics,
                "message": message_count,
                "training_time": training_time,
            })

        if streaming_group_buffer:
            dropped = sum(int(b["rewards"].shape[1]) for b in streaming_group_buffer)
            if rank == 0:
                logger.warning(
                    "[Streaming Train] Dropping %s samples left in buffer (incomplete "
                    "group for adv_type=%s, group_size=%s).",
                    dropped,
                    adv_type,
                    grpo_group_size,
                )

        # 处理剩余梯度
        # NOTE: With the force_reduce_scatter fix above, this should rarely happen.
        # The last slice of the last message will have is_last_slice_before_update=True,
        # which triggers Reduce-Scatter. This code is kept as a safety net.
        if accumulated_samples > 0:
            if rank == 0:
                print(
                    f"[Streaming Train] Final update with {accumulated_samples} "
                    f"accumulated samples (< target {target_batch_size})",
                    flush=True,
                )
            # Gradients should already be reduced (sharded) due to force_reduce_scatter
            self.optimizer_step()
            self.optimizer.zero_grad()

        self.lr_scheduler.step()
        # Ensure no leftover rollout messages are carried into the next global step.
        self._drain_actor_rollout_messages(input_channel)
        clear_memory(sync=False)

        return all_metrics

    async def async_train_loop_with_gradient_accumulation(
        self, input_channel: Channel, num_epochs: int
    ):
        """
        异步训练循环：严格控制每次更新恰好使用 target_batch_size 个样本

        这是文档 13.9 中描述的"严格 Batch Size 梯度累积方案"的实现。
        核心思想是通过拆分数据确保每次更新恰好使用 target_batch_size 个样本。

        Args:
            input_channel: 数据输入 channel
            num_epochs: 预期接收的 epoch 数量

        Example:
            假设 target_batch_size = 512，每次接收 384 samples：

            接收 Epoch 0 (384 samples):
              - 前向 + 反向传播 384 samples（累积梯度）
              - accumulated_samples = 384
              - 384 < 512，不更新参数

            接收 Epoch 1 (384 samples):
              - 需要 512 - 384 = 128 samples 才能更新
              - 从 384 samples 中取出前 128 samples，训练
              - accumulated_samples = 512
              - 更新参数！optimizer.step()
              - 剩余 256 samples 继续累积
              - 训练这 256 samples（累积梯度）
              - accumulated_samples = 256
              - 不更新参数
        """
        epoch_count = 0
        all_metrics = []
        rank = int(os.environ.get("RANK", 0))

        # 记录总训练时间
        import time as time_module
        total_training_time = 0.0

        # 严格的 batch size 控制
        target_batch_size = self.cfg.actor.global_batch_size // self._world_size
        accumulated_samples = 0

        # 获取 update_epoch 配置（每个 epoch 的数据训练多少次）
        update_epoch = self.cfg.algorithm.get("update_epoch", 1)

        # 方案 A：保存初始权重快照用于计算 old_logprobs
        initial_state_dict = None
        if self.cfg.algorithm.get("recompute_old_logprobs", False):
            if rank == 0:
                print(
                    "[Async Train w/ Grad Accum] Saving initial weights snapshot...",
                    flush=True,
                )
            initial_state_dict = retrieve_model_state_dict_in_cpu(self.model)

        # 初始化梯度
        if self.is_weight_offloaded:
            self.load_param_and_grad(self.device)
        if self.is_optimizer_offloaded:
            self.load_optimizer(self.device)
        self.optimizer.zero_grad()

        # 用于存储跨 epoch 的剩余数据
        leftover_batch = None
        leftover_size = 0

        if rank == 0:
            print(
                f"[Async Train w/ Grad Accum] Starting with target_batch_size={target_batch_size}, "
                f"update_epoch={update_epoch}",
                flush=True,
            )

        while epoch_count < num_epochs:
            # 从 channel 获取一个 epoch 的数据
            self.recv_single_epoch_batch(input_channel)

            # 检查是否收到结束信号
            if self.rollout_batch.get("__done__", False):
                break

            # CRITICAL: Barrier to synchronize all actors
            torch.distributed.barrier()

            # Recompute old_logprobs if needed
            if self.cfg.algorithm.get("recompute_old_logprobs", False):
                self._recompute_old_logprobs_with_snapshot(initial_state_dict)

            # 计算 advantage
            rollout_metrics = self.compute_advantages_and_returns()

            # 记录训练开始时间
            epoch_training_start = time_module.time()

            # 将数据展平
            current_batch, current_size = self._flatten_batch_for_gradient_accumulation(
                self.rollout_batch
            )

            if rank == 0:
                print(
                    f"[Async Train w/ Grad Accum] Epoch {epoch_count}: received {current_size} samples, "
                    f"accumulated={accumulated_samples}, leftover={leftover_size}",
                    flush=True,
                )

            # 合并 leftover 数据（如果有）
            if leftover_batch is not None and leftover_size > 0:
                combined_batch = cat_list_of_dict_tensor(
                    [leftover_batch, current_batch], dim=0
                )
                combined_size = leftover_size + current_size
            else:
                combined_batch = current_batch
                combined_size = current_size

            # 处理合并后的数据
            # update_epoch: 对同一批数据训练多次（PPO 风格）
            epoch_training_metrics_list = []  # 收集这个 epoch 内所有更新的 metrics

            for update_idx in range(update_epoch):
                offset = 0
                while offset < combined_size:
                    # 计算还需要多少样本才能达到 target_batch_size
                    remaining_needed = target_batch_size - accumulated_samples

                    # 计算这次可以使用多少样本
                    samples_to_use = min(remaining_needed, combined_size - offset)

                    # 切片数据
                    batch_slice = slice_nested_dict(combined_batch, offset, offset + samples_to_use)

                    # 判断这个 slice 训练完后是否会触发 optimizer.step()
                    # 只有在这种情况下，才让 FSDP 执行 Reduce-Scatter
                    will_update_after_this_slice = (accumulated_samples + samples_to_use) >= target_batch_size

                    # 训练这个切片（累积梯度）
                    # Loss 缩放由 target_batch_size 控制，确保每个样本贡献 1/target_batch_size
                    slice_metrics = self._train_batch_slice_accumulate(
                        batch_slice,
                        target_batch_size=target_batch_size,
                        is_last_slice_before_update=will_update_after_this_slice,
                    )

                    accumulated_samples += samples_to_use
                    offset += samples_to_use

                    # 如果累积的样本数达到目标，更新参数
                    if accumulated_samples >= target_batch_size:
                        grad_norm, lr_list = self.optimizer_step()
                        self.optimizer.zero_grad()

                        update_metrics = {
                            "actor/grad_norm": grad_norm,
                            "actor/lr": lr_list[0],
                            "accumulated_samples": accumulated_samples,
                        }
                        if len(lr_list) > 1:
                            update_metrics["critic/lr"] = lr_list[1]

                        # Merge slice metrics with update metrics
                        for key, value in slice_metrics.items():
                            if isinstance(value, list):
                                update_metrics[key] = np.mean(value)
                            else:
                                update_metrics[key] = value
                        update_metrics = all_reduce_dict(
                            update_metrics, op=torch.distributed.ReduceOp.AVG
                        )
                        epoch_training_metrics_list.append(update_metrics)

                        if rank == 0:
                            print(
                                f"[Async Train w/ Grad Accum] Parameter update! "
                                f"update_idx={update_idx}, accumulated_samples={accumulated_samples}, "
                                f"grad_norm={grad_norm:.4f}",
                                flush=True,
                            )

                        accumulated_samples = 0

            # 保存剩余的数据（如果 offset < combined_size，理论上不会发生）
            if offset < combined_size:
                leftover_batch = slice_nested_dict(combined_batch, offset, combined_size)
                leftover_size = combined_size - offset
            else:
                leftover_batch = None
                leftover_size = 0

            # 合并这个 epoch 内所有更新的 metrics 为单个 dict
            # 如果有多次更新，取平均值；如果没有更新，使用 slice_metrics
            if epoch_training_metrics_list:
                aggregated_training_metrics = {}
                for key in epoch_training_metrics_list[0].keys():
                    values = [m[key] for m in epoch_training_metrics_list if key in m]
                    if values:
                        aggregated_training_metrics[key] = sum(values) / len(values)
            else:
                # 这个 epoch 没有触发参数更新，使用最后一个 slice 的 metrics
                aggregated_training_metrics = {}
                for key, value in slice_metrics.items():
                    if isinstance(value, list):
                        aggregated_training_metrics[key] = np.mean(value)
                    else:
                        aggregated_training_metrics[key] = value

            # 记录训练结束时间
            epoch_training_time = time_module.time() - epoch_training_start
            total_training_time += epoch_training_time

            epoch_count += 1
            all_metrics.append({
                "rollout": rollout_metrics,
                "training": aggregated_training_metrics,  # dict, not list
                "epoch": epoch_count,
                "training_time": epoch_training_time  # 添加训练时间
            })

        # 处理剩余的梯度（如果有累积的样本但未更新）
        if accumulated_samples > 0:
            if rank == 0:
                print(
                    f"[Async Train w/ Grad Accum] Final update with {accumulated_samples} "
                    f"accumulated samples (< target {target_batch_size})",
                    flush=True,
                )
            grad_norm, lr_list = self.optimizer_step()
            self.optimizer.zero_grad()

        # LR scheduler step
        self.lr_scheduler.step()

        # 清理
        if initial_state_dict is not None:
            del initial_state_dict
        # Ensure no leftover rollout messages are carried into the next global step.
        self._drain_actor_rollout_messages(input_channel)
        clear_memory(sync=False)

        return all_metrics