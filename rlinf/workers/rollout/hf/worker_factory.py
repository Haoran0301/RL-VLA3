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
Worker factory for creating appropriate RolloutWorker based on configuration.

This module provides a unified interface for creating rollout workers,
supporting per-env async mode:

1. MultiStepRolloutWorker - Default serial processing
2. PerEnvAsyncRolloutWorker - Per-env async with independent env handlers

Usage:
    # In config yaml:
    rollout:
      # Per-env async (must pair with TruePerEnvAsyncEnvWorker)
      per_env_async:
        enabled: true
        max_batch_size: 32
        batch_timeout_ms: 5.0

    # In training script:
    from rlinf.workers.rollout.hf.worker_factory import get_rollout_worker_cls

    rollout_worker_cls = get_rollout_worker_cls(cfg)
    rollout_group = rollout_worker_cls.create_group(cfg).launch(...)
"""

import logging
from typing import Type

from omegaconf import DictConfig

logger = logging.getLogger(__name__)


def get_rollout_worker_cls(cfg: DictConfig) -> Type:
    """
    Get the appropriate RolloutWorker class based on configuration.

    Configuration options:
        rollout.per_env_async.enabled: true  -> PerEnvAsyncRolloutWorker
        Otherwise                            -> MultiStepRolloutWorker

    Note: PerEnvAsyncRolloutWorker must be paired with TruePerEnvAsyncEnvWorker.

    Args:
        cfg: Configuration dict

    Returns:
        RolloutWorker class (not instance)
    """
    # Check for per-env async
    per_env_async_cfg = cfg.rollout.get("per_env_async", None)
    print(f"[DEBUG worker_factory] per_env_async_cfg = {per_env_async_cfg}")
    print(
        f"[DEBUG worker_factory] enabled = "
        f"{per_env_async_cfg.get('enabled', False) if per_env_async_cfg else None}"
    )
    if per_env_async_cfg and per_env_async_cfg.get("enabled", False):
        flex_cfg = per_env_async_cfg.get("flex", None)
        if flex_cfg and flex_cfg.get("enabled", False):
            logger.info("Using FlexiblePerEnvAsyncRolloutWorker (per-env async flex mode)")
            from rlinf.workers.rollout.hf.flexible_per_env_async_rollout_worker import (
                FlexiblePerEnvAsyncRolloutWorker,
            )

            return FlexiblePerEnvAsyncRolloutWorker

        aggregate_slots = int(per_env_async_cfg.get("aggregate_slots_per_env", 1))
        aggregate_enabled = per_env_async_cfg.get("aggregate_slots_enabled", False) or aggregate_slots > 1
        if aggregate_enabled:
            logger.info(
                "Using AggregatedPerEnvAsyncRolloutWorker "
                f"(per-env async aggregated mode, slots={aggregate_slots})"
            )
            from rlinf.workers.rollout.hf.aggregated_per_env_async_rollout_worker import (
                AggregatedPerEnvAsyncRolloutWorker,
            )

            return AggregatedPerEnvAsyncRolloutWorker

        logger.info("Using PerEnvAsyncRolloutWorker (per-env async mode)")
        from rlinf.workers.rollout.hf.per_env_async_rollout_worker import (
            PerEnvAsyncRolloutWorker,
        )

        return PerEnvAsyncRolloutWorker

    # Default
    logger.info("Using MultiStepRolloutWorker (default mode)")
    from rlinf.workers.rollout.hf.huggingface_worker import MultiStepRolloutWorker
    return MultiStepRolloutWorker


def create_rollout_worker(cfg: DictConfig):
    """
    Create a RolloutWorker instance based on configuration.

    This is a convenience function that gets the appropriate class
    and instantiates it.

    Args:
        cfg: Configuration dict

    Returns:
        RolloutWorker instance
    """
    worker_cls = get_rollout_worker_cls(cfg)
    return worker_cls(cfg)
