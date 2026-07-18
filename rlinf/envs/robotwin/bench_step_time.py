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
简单脚本：测试 RoboTwinEnv.step() 的单步耗时，测试不同并行环境数。

依赖：robotwin_env 使用 robotwin.envs.vector_env，需设置 ROBOTWIN_PATH 并将该路径加入 sys.path。
用法:
  python -m rlinf.envs.robotwin.bench_step_time [--num-envs-list N1,N2,...] [--num-steps N] [--warmup K]
  cd /path/to/rlinf_maniskill && python -m rlinf.envs.robotwin.bench_step_time
"""

import argparse
import os
import sys
import time

# RoboTwinEnv / VectorEnv 依赖 ASSETS_PATH 与 envs 包，需在导入前设置
os.environ.setdefault("ROBOT_PLATFORM", "ALOHA")
robotwin_path = os.environ.setdefault("ROBOTWIN_PATH", "/ufs/shr/RoboTwin-RLinf_support")
if robotwin_path not in sys.path:
    sys.path.insert(0, robotwin_path)

import numpy as np
from omegaconf import OmegaConf

from rlinf.envs.robotwin.robotwin_env import RoboTwinEnv


def _make_env_cfg(total_num_envs: int = 2, horizon: int = 1):
    """构建与 RoboTwinEnv 兼容的 env 配置，与 robotwin_env 及 env/train/robotwin_place_empty_cup 对齐。"""
    cfg_dict = {
        "env_type": "robotwin",
        "assets_path": robotwin_path,
        "total_num_envs": total_num_envs,
        "auto_reset": False,
        "ignore_terminations": False,
        "use_rel_reward": True,
        "use_custom_reward": True,
        "reward_coef": 1.0,
        "center_crop": False,
        "is_eval": False,
        "seed": 0,
        "group_size": 1,
        "use_fixed_reset_state_ids": True,
        "max_steps_per_rollout_epoch": 200,
        "max_episode_steps": 200,
        "horizon": horizon,
        "video_cfg": {
            "save_video": False,
            "info_on_video": False,
            "video_base_dir": "/tmp/test_video",
        },
        "task_config": {
            "task_name": "place_empty_cup",
            "step_lim": 200,
            "save_path": "/tmp/robotwin_bench",
            "embodiment": ["piper", "piper", 0.6],
            "camera": {
                "head_camera_type": "D435",
                "wrist_camera_type": "D435",
                "collect_head_camera": True,
                "collect_wrist_camera": False,
            },
        },
    }
    return OmegaConf.create(cfg_dict)


def _sample_action(num_envs: int, horizon: int = 1, action_dim: int = 14):
    """采样随机动作。RoboTwinEnv.step 接受 (n_envs, horizon, action_dim) 或 (n_envs, action_dim)。"""
    return np.clip(
        np.random.uniform(0, 1, (num_envs, horizon, action_dim)).astype(np.float32),
        0,
        1,
    )


def main():
    parser = argparse.ArgumentParser(description="Benchmark RoboTwinEnv step time")
    parser.add_argument(
        "--num-envs-list",
        type=str,
        default="2,4,8,16,32,64,128,256",
        help="逗号分隔的并行环境数量列表（RoboTwin 通常使用较少环境），如 2,4",
    )
    parser.add_argument("--num-steps", type=int, default=10, help="计时步数")
    parser.add_argument("--warmup", type=int, default=2, help="预热步数（不参与计时）")
    parser.add_argument("--horizon", type=int, default=1, help="动作 horizon")
    args = parser.parse_args()

    num_envs_list = [int(x.strip()) for x in args.num_envs_list.split(",")]

    for num_envs in num_envs_list:
        cfg = _make_env_cfg(total_num_envs=num_envs, horizon=args.horizon)
        env = RoboTwinEnv(
            cfg=cfg,
            num_envs=num_envs,
            seed_offset=0,
            total_num_processes=1,
            worker_info=None,
        )

        obs, info = env.reset()
        action = _sample_action(num_envs, args.horizon)

        print('hello')
        print('hello')
        print('hello')
        # 预热
        for _ in range(args.warmup):
            obs, reward, term, trunc, info = env.step(action, auto_reset=False)
            action = _sample_action(num_envs, args.horizon)

        # 计时
        step_times_ms = []
        for _ in range(args.num_steps):
            action = _sample_action(num_envs, args.horizon)
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

        print(f"RoboTwinEnv step 耗时 (num_envs={num_envs}, {args.num_steps} steps):")
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
