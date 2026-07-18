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
简单脚本：测试 BehaviorEnv.step() 的单步耗时，测试不同并行环境数。

用法:
  python -m rlinf.envs.behavior.bench_step_time [--num-envs-list N1,N2,...] [--num-steps N] [--warmup K]
  cd /path/to/rlinf_maniskill && python -m rlinf.envs.behavior.bench_step_time
"""

import argparse
import os
import time

import torch
import yaml
from omegaconf import OmegaConf

from rlinf.envs.behavior.behavior_env import BehaviorEnv


def _make_env_cfg(total_num_envs: int = 4, task_idx: int = 0):
    """构建与 BehaviorEnv 兼容的 env 配置。"""
    import omnigibson as og

    config_filename = os.path.join(og.example_config_path, "r1pro_behavior.yaml")
    omnigibson_cfg = yaml.load(
        open(config_filename, "r"), Loader=yaml.FullLoader
    )
    omnigibson_cfg = OmegaConf.create(omnigibson_cfg)

    cfg_dict = {
        "env_type": "behavior",
        "total_num_envs": total_num_envs,
        "auto_reset": False,
        "ignore_terminations": True,
        "max_steps_per_rollout_epoch": 2000,
        "max_episode_steps": 2000,
        "use_rel_reward": True,
        "seed": 0,
        "group_size": 1,
        "base_config_name": "r1pro_behavior",
        "task_idx": task_idx,
        "video_cfg": {
            "save_video": False,
            "info_on_video": False,
            "video_base_dir": "/tmp/test_video",
        },
        "omnigibson_cfg": omnigibson_cfg,
    }
    return OmegaConf.create(cfg_dict)


def _sample_action(env, num_envs: int, action_dim: int = 23):
    """采样随机动作。Behavior 使用 23 维动作。"""
    try:
        if hasattr(env.env, "action_space"):
            action_space = env.env.action_space
            if action_space is not None:
                return torch.stack(
                    [torch.from_numpy(action_space.sample()) for _ in range(num_envs)]
                ).to(env.device)
    except Exception:
        pass
    return torch.randn(num_envs, action_dim, device=env.device, dtype=torch.float32)


def main():
    parser = argparse.ArgumentParser(description="Benchmark BehaviorEnv step time")
    parser.add_argument(
        "--num-envs-list",
        type=str,
        default="2,4",
        help="逗号分隔的并行环境数量列表，如 2,4",
    )
    parser.add_argument("--num-steps", type=int, default=10, help="计时步数")
    parser.add_argument("--warmup", type=int, default=2, help="预热步数（不参与计时）")
    parser.add_argument("--task-idx", type=int, default=0, help="任务索引 (0-49)")
    args = parser.parse_args()

    num_envs_list = [int(x.strip()) for x in args.num_envs_list.split(",")]

    for num_envs in num_envs_list:
        cfg = _make_env_cfg(total_num_envs=num_envs, task_idx=args.task_idx)
        env = BehaviorEnv(
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
            obs, reward, term, trunc, info = env.step(action)
            action = _sample_action(env, num_envs)

        # 排空 warmup 残留的 GPU 操作
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        # 计时
        step_times_ms = []
        for _ in range(args.num_steps):
            action = _sample_action(env, num_envs)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            obs, reward, term, trunc, info = env.step(action)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t1 = time.perf_counter()
            step_times_ms.append((t1 - t0) * 1000)

        mean_ms = sum(step_times_ms) / len(step_times_ms)
        variance = sum((t - mean_ms) ** 2 for t in step_times_ms) / len(step_times_ms)
        std_ms = variance**0.5
        min_ms = min(step_times_ms)
        max_ms = max(step_times_ms)
        fps = 1000.0 / mean_ms * num_envs

        print(f"BehaviorEnv step 耗时 (num_envs={num_envs}, {args.num_steps} steps):")
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
