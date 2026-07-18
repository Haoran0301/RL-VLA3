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
简单脚本：测试 MetaWorldEnv.step() 的单步耗时，测试不同并行环境数。

无头/无 DISPLAY 环境（如 Docker）需使用 EGL 渲染，脚本会在创建环境前设置 MUJOCO_GL=egl。

用法:
  python -m rlinf.envs.metaworld.bench_step_time [--num-envs-list N1,N2,...] [--num-steps N] [--warmup K]
  cd /path/to/rlinf_maniskill && python -m rlinf.envs.metaworld.bench_step_time
"""

import argparse
import os
import time
import warnings

import numpy as np
from omegaconf import OmegaConf

from rlinf.envs.metaworld.metaworld_env import MetaWorldEnv


def _make_env_cfg(total_num_envs: int = 8):
    """构建与 MetaWorldEnv 兼容的最小 env 配置。"""
    cfg_dict = {
        "env_type": "metaworld",
        "task_suite_name": "metaworld_50",
        "total_num_envs": total_num_envs,
        "auto_reset": True,
        "ignore_terminations": True,
        "max_steps_per_rollout_epoch": 100,
        "max_episode_steps": 100,
        "use_rel_reward": True,
        "reward_coef": 1.0,
        "is_eval": False,
        "seed": 0,
        "group_size": 1,
        "use_fixed_reset_state_ids": False,
        "use_ordered_reset_state_ids": False,
        "video_cfg": {
            "save_video": False,
            "info_on_video": False,
            "video_base_dir": "/tmp/test_video",
        },
        "init_params": {
            "camera_heights": 480,
            "camera_widths": 480,
        },
    }
    return OmegaConf.create(cfg_dict)


def _sample_action(env, num_envs: int):
    """采样随机动作。MetaWorld 使用 4 维动作。"""
    try:
        action_space = env.env.get_env_attr("action_space", id=0)[0]
        if action_space is not None:
            return np.array([action_space.sample() for _ in range(num_envs)])
    except Exception:
        pass
    return np.random.uniform(-1, 1, (num_envs, 4)).astype(np.float32)


def main():
    # 无头环境（无 DISPLAY）下必须用 EGL，否则 MuJoCo 会报 OpenGL context 错误
    os.environ.setdefault("MUJOCO_GL", "egl")
    # 抑制 bench 时无关的终端输出
    warnings.filterwarnings("ignore", message=".*Gym.*gymnasium.*")
    warnings.filterwarnings("ignore", module="glfw")

    parser = argparse.ArgumentParser(description="Benchmark MetaWorldEnv step time")
    parser.add_argument(
        "--num-envs-list",
        type=str,
        default="1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16",
        help="逗号分隔的并行环境数量列表，如 4,8,16,32",
    )
    parser.add_argument("--num-steps", type=int, default=10, help="计时步数")
    parser.add_argument("--warmup", type=int, default=2, help="预热步数（不参与计时）")
    args = parser.parse_args()

    num_envs_list = [int(x.strip()) for x in args.num_envs_list.split(",")]

    for num_envs in num_envs_list:
        cfg = _make_env_cfg(total_num_envs=num_envs)
        env = MetaWorldEnv(
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

        print(f"MetaWorldEnv step 耗时 (num_envs={num_envs}, {args.num_steps} steps):")
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
