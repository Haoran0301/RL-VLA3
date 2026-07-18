#!/usr/bin/env python3
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
简单脚本：测试 LiberoEnv.step() 的单步耗时，测试不同并行环境数。

用法:
  python -m rlinf.envs.libero.bench_step_time [--num-envs-list N1,N2,...] [--num-steps N] [--warmup K]
  cd /path/to/rlinf_maniskill && python -m rlinf.envs.libero.bench_step_time
"""

import argparse
import time

import numpy as np
from omegaconf import OmegaConf

from rlinf.envs.libero.libero_env import LiberoEnv


def _make_env_cfg(total_num_envs: int = 8):
    """构建与 LiberoEnv 兼容的最小 env 配置。"""
    cfg_dict = {
        "env_type": "libero",
        "task_suite_name": "libero_spatial",
        "total_num_envs": total_num_envs,
        "auto_reset": True,
        "ignore_terminations": True,
        "max_steps_per_rollout_epoch": 240,
        "max_episode_steps": 240,
        "use_rel_reward": True,
        "reward_coef": 1.0,
        "reset_gripper_open": True,
        "is_eval": False,
        "seed": 0,
        "group_size": 1,
        "use_fixed_reset_state_ids": True,
        "use_ordered_reset_state_ids": False,
        "specific_reset_id": None,
        "video_cfg": {
            "save_video": False,
            "info_on_video": False,
            "video_base_dir": "/tmp/test_video",
        },
        "init_params": {
            "camera_heights": 256,
            "camera_widths": 256,
        },
    }
    return OmegaConf.create(cfg_dict)


def _sample_action(env, num_envs: int):
    """采样随机动作。Libero 使用 7 维动作。"""
    try:
        action_space = env.env.get_env_attr("action_space", id=0)[0]
        if action_space is not None:
            return np.array([action_space.sample() for _ in range(num_envs)])
    except Exception:
        pass
    return np.random.uniform(-1, 1, (num_envs, 7)).astype(np.float32)


def main():
    parser = argparse.ArgumentParser(description="Benchmark LiberoEnv step time")
    parser.add_argument(
        "--num-envs-list",
        type=str,
        default="4,8,16,32",
        help="逗号分隔的并行环境数量列表，如 4,8,16,32",
    )
    parser.add_argument("--num-steps", type=int, default=10, help="计时步数")
    parser.add_argument("--warmup", type=int, default=2, help="预热步数（不参与计时）")
    args = parser.parse_args()

    num_envs_list = [int(x.strip()) for x in args.num_envs_list.split(",")]

    for num_envs in num_envs_list:
        cfg = _make_env_cfg(total_num_envs=num_envs)
        env = LiberoEnv(
            cfg=cfg,
            num_envs=num_envs,
            seed_offset=0,
            total_num_processes=1,
            worker_info=None,
        )

        obs, info = env.reset()
        action = _sample_action(env, num_envs)

        # 预热
        for _ in range(args.warmup):
            obs, reward, term, trunc, info = env.step(action, auto_reset=False)
            action = _sample_action(env, num_envs)

        # 计时
        step_times_ms = []
        for _ in range(args.num_steps):
            action = _sample_action(env, num_envs)
            t0 = time.perf_counter()
            obs, reward, term, trunc, info = env.step(action, auto_reset=False)
            t1 = time.perf_counter()
            step_times_ms.append((t1 - t0) * 1000)

        mean_ms = sum(step_times_ms) / len(step_times_ms)
        variance = sum((t - mean_ms) ** 2 for t in step_times_ms) / len(step_times_ms)
        std_ms = variance**0.5
        min_ms = min(step_times_ms)
        max_ms = max(step_times_ms)
        fps = 1000.0 / mean_ms * num_envs

        print(f"LiberoEnv step 耗时 (num_envs={num_envs}, {args.num_steps} steps):")
        print(f"  平均: {mean_ms:.2f} ms")
        print(f"  标准差: {std_ms:.2f} ms")
        print(f"  最小: {min_ms:.2f} ms")
        print(f"  最大: {max_ms:.2f} ms")
        print(f"  约 FPS (每 env): {fps:.1f}")
        print()

        if hasattr(env, "close"):
            env.close()


if __name__ == "__main__":
    main()
