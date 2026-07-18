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
Worker factory for creating appropriate EnvWorker based on configuration.

This module provides a unified interface for creating env workers,
supporting per-env async mode:

1. Default EnvWorker - Serial stage processing
2. TruePerEnvAsyncEnvWorker - True per-env async (each env runs independently)

Configuration:
    # True per-env async (each env runs its own epoch loop, no sync)
    env.per_env_async.enabled: true
"""

import logging
from typing import Type

from omegaconf import DictConfig

logger = logging.getLogger(__name__)


def get_env_worker_cls(cfg: DictConfig) -> Type:
    """
    Get the appropriate EnvWorker class based on configuration.

    Configuration options:
        env.per_env_async.enabled: true  -> TruePerEnvAsyncEnvWorker
        Otherwise                        -> EnvWorker

    Args:
        cfg: Configuration dict

    Returns:
        EnvWorker class (not instance)
    """
    # Check for per-env async
    per_env_async_cfg = cfg.env.get("per_env_async", None)
    if per_env_async_cfg and per_env_async_cfg.get("enabled", False):
        flex_cfg = per_env_async_cfg.get("flex", None)
        if flex_cfg and flex_cfg.get("enabled", False):
            logger.info("Using FlexiblePerEnvAsyncEnvWorker (per-env async flex mode)")
            from rlinf.workers.env.per_env_async.flexible_per_env_async_worker import (
                FlexiblePerEnvAsyncEnvWorker,
            )

            return FlexiblePerEnvAsyncEnvWorker

        aggregate_slots = int(per_env_async_cfg.get("aggregate_slots_per_env", 1))
        aggregate_enabled = per_env_async_cfg.get("aggregate_slots_enabled", False) or aggregate_slots > 1
        if aggregate_enabled:
            logger.info(
                "Using AggregatedPerEnvAsyncEnvWorker "
                f"(per-env async aggregated mode, slots={aggregate_slots})"
            )
            from rlinf.workers.env.per_env_async.aggregated_per_env_async_worker import (
                AggregatedPerEnvAsyncEnvWorker,
            )

            return AggregatedPerEnvAsyncEnvWorker

        logger.info("Using TruePerEnvAsyncEnvWorker (per-env async mode)")
        from rlinf.workers.env.per_env_async.true_per_env_async_worker import (
            TruePerEnvAsyncEnvWorker,
        )

        return TruePerEnvAsyncEnvWorker

    # Default
    logger.info("Using EnvWorker (default mode)")
    from rlinf.workers.env.env_worker import EnvWorker
    return EnvWorker


def create_env_worker(cfg: DictConfig):
    """
    Create an EnvWorker instance based on configuration.

    This is a convenience function that gets the appropriate class
    and instantiates it.

    Args:
        cfg: Configuration dict

    Returns:
        EnvWorker instance
    """
    worker_cls = get_env_worker_cls(cfg)
    return worker_cls(cfg)
