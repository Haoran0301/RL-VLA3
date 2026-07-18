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
简单脚本：测试 CalvinEnv.step() 的单步耗时，测试不同并行环境数。

用法:
  python -m rlinf.envs.calvin.bench_step_time [--num-envs-list N1,N2,...] [--num-steps N] [--warmup K]
  cd /path/to/rlinf_maniskill && python -m rlinf.envs.calvin.bench_step_time
"""

import argparse
import time

import numpy as np
from omegaconf import OmegaConf

from rlinf.envs.calvin.calvin_gym_env import CalvinEnv


def _make_env_cfg(total_num_envs: int = 8):
    """构建与 CalvinEnv 兼容的最小 env 配置。"""
    cfg_dict = {
        "env_type": "calvin",
        "task_suite_name": "calvin_d",
        "total_num_envs": total_num_envs,
        "auto_reset": True,
        "ignore_terminations": True,
        "max_steps_per_rollout_epoch": 480,
        "max_episode_steps": 480,
        "use_rel_reward": True,
        "reward_coef": 1.0,
        "seed": 0,
        "group_size": 1,
        "use_fixed_reset_state_ids": False,
        "use_ordered_reset_state_ids": False,
        "is_eval": False,
        "video_cfg": {
            "save_video": False,
            "info_on_video": False,
            "video_base_dir": "/tmp/test_video",
        },
        "init_params": {
            "camera_heights": 200,
            "camera_widths": 200,
        },
    }
    return OmegaConf.create(cfg_dict)


def _sample_action(env, num_envs: int):
    """采样随机动作。Calvin 使用 7 维动作，最后一维为夹爪且必须为 -1 或 1（离散）。"""
    try:
        action_space = env.env.get_env_attr("action_space", id=0)[0]
        if action_space is not None:
            actions = np.array([action_space.sample() for _ in range(num_envs)]).astype(
                np.float32
            )
            # 确保夹爪维为 ±1，避免 robot.apply_action 中 assert self.gripper_action in (-1, 1) 报错
            if actions.shape[-1] >= 1:
                actions[:, -1] = np.sign(actions[:, -1])
                actions[:, -1] = np.where(actions[:, -1] == 0, 1.0, actions[:, -1])
            return actions
    except Exception:
        pass
    # fallback：前 6 维连续，最后一维夹爪为离散 ±1
    action = np.random.uniform(-1, 1, (num_envs, 7)).astype(np.float32)
    action[:, -1] = np.random.choice([-1.0, 1.0], size=num_envs)
    return action


def main():
    parser = argparse.ArgumentParser(description="Benchmark CalvinEnv step time")
    parser.add_argument(
        "--num-envs-list",
        type=str,
        default="4,8,16,32",
        help="逗号分隔的并行环境数量列表，如 4,8,16,32",
    )
    parser.add_argument("--num-steps", type=int, default=20, help="计时步数")
    parser.add_argument("--warmup", type=int, default=5, help="预热步数（不参与计时）")
    args = parser.parse_args()

    num_envs_list = [int(x.strip()) for x in args.num_envs_list.split(",")]

    for num_envs in num_envs_list:
        cfg = _make_env_cfg(total_num_envs=num_envs)
        env = CalvinEnv(
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

        print(f"CalvinEnv step 耗时 (num_envs={num_envs}, {args.num_steps} steps):")
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
