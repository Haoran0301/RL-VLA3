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
Per-Env Async Pipeline for RLinf.

This module implements per-env level async pipeline where each EnvManager
runs its OWN independent epoch loop with no synchronization between environments.

Usage:
    # In config yaml:
    env:
      per_env_async:
        enabled: true
        thread_pool_size: 16

    rollout:
      per_env_async:
        enabled: true
        max_batch_size: 32
        batch_timeout_ms: 5.0

    # In training script:
    from rlinf.workers.env.per_env_async import get_env_worker_cls

    env_worker_cls = get_env_worker_cls(cfg)
    env_group = env_worker_cls.create_group(cfg).launch(...)

Key components:
- TruePerEnvAsyncEnvWorker: Each env runs independently (no epoch sync)
- get_env_worker_cls: Factory function to get appropriate worker class
"""

from rlinf.workers.env.per_env_async.true_per_env_async_worker import TruePerEnvAsyncEnvWorker
from rlinf.workers.env.per_env_async.aggregated_per_env_async_worker import (
    AggregatedPerEnvAsyncEnvWorker,
)
from rlinf.workers.env.per_env_async.flexible_per_env_async_worker import (
    FlexiblePerEnvAsyncEnvWorker,
)
from rlinf.workers.env.per_env_async.worker_factory import (
    get_env_worker_cls,
    create_env_worker,
)

__all__ = [
    "TruePerEnvAsyncEnvWorker",
    "AggregatedPerEnvAsyncEnvWorker",
    "FlexiblePerEnvAsyncEnvWorker",
    "get_env_worker_cls",
    "create_env_worker",
]
