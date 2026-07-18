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
简单脚本：测试 RobocasaEnv.step() 的单步耗时，测试不同并行环境数。

用法:
  python -m rlinf.envs.robocasa.bench_step_time [--num-envs-list N1,N2,...] [--num-steps N] [--warmup K]
  cd /path/to/rlinf_maniskill && python -m rlinf.envs.robocasa.bench_step_time
"""

import argparse
import time

import numpy as np
from omegaconf import OmegaConf

from rlinf.envs.robocasa.robocasa_env import RobocasaEnv

try:
    import torch
    _TORCH_CUDA_AVAILABLE = torch.cuda.is_available()
except Exception:
    _TORCH_CUDA_AVAILABLE = False


def _get_gpu_memory_stats():
    """获取当前进程的 GPU 显存峰值（MB）与显存利用率峰值（%）。无 CUDA 时返回 None, None。"""
    if not _TORCH_CUDA_AVAILABLE:
        return None, None
    try:
        peak_allocated = torch.cuda.max_memory_allocated()
        total_memory = torch.cuda.get_device_properties(0).total_memory
        peak_mb = peak_allocated / (1024**2)
        peak_util_pct = (peak_allocated / total_memory) * 100.0 if total_memory else 0.0
        return peak_mb, peak_util_pct
    except Exception:
        return None, None


def _make_env_cfg(total_num_envs: int = 8):
    """构建与 RobocasaEnv 兼容的最小 env 配置。"""
    cfg_dict = {
        "env_type": "robocasa",
        "robot_name": "PandaOmron",
        "camera_names": ["robot0_agentview_left", "robot0_eye_in_hand"],
        "task_names": ["CloseDrawer"],
        "total_num_envs": total_num_envs,
        "auto_reset": True,
        "ignore_terminations": True,
        "max_steps_per_rollout_epoch": 300,
        "max_episode_steps": 300,
        "use_fixed_reset_state_ids": False,
        "use_ordered_reset_state_ids": False,
        "use_rel_reward": True,
        "reward_coef": 1.0,
        "is_eval": False,
        "seed": 0,
        "group_size": 1,
        "video_cfg": {
            "save_video": False,
            "info_on_video": False,
            "video_base_dir": "/tmp/test_video",
        },
        "init_params": {
            "camera_heights": 224,
            "camera_widths": 224,
        },
    }
    return OmegaConf.create(cfg_dict)


def _sample_action(env, num_envs: int):
    """采样随机动作。Robocasa/Robosuite 使用 7 维动作（单臂）。"""
    try:
        action_space = env.env.get_env_attr("action_space", id=0)[0]
        if action_space is not None:
            return np.array([action_space.sample() for _ in range(num_envs)])
    except Exception:
        pass
    return np.random.uniform(-1, 1, (num_envs, 12)).astype(np.float32)


def main():
    parser = argparse.ArgumentParser(description="Benchmark RobocasaEnv step time")
    parser.add_argument(
        "--num-envs-list",
        type=str,
        default="4,8,16,32",
        help="逗号分隔的并行环境数量列表，如 4,8,16,32",
    )
    parser.add_argument("--num-steps", type=int, default=20, help="计时步数")
    parser.add_argument("--warmup", type=int, default=2, help="预热步数（不参与计时）")
    args = parser.parse_args()

    num_envs_list = [int(x.strip()) for x in args.num_envs_list.split(",")]

    for num_envs in num_envs_list:
        cfg = _make_env_cfg(total_num_envs=num_envs)
        env = RobocasaEnv(
            cfg=cfg,
            num_envs=num_envs,
            seed_offset=0,
            total_num_processes=1,
            worker_info=None,
        )

        obs, info = env.reset()
        action = _sample_action(env, num_envs)

        # print(f"obs: {obs}")
        # print(f"action: {action.shape}")
        # print(f"info: {info}")
        # 预热
        for _ in range(args.warmup):
            obs, reward, term, trunc, info = env.step(action, auto_reset=False)
            action = _sample_action(env, num_envs)

        # 清零显存峰值统计，仅统计下面计时区间的峰值
        if _TORCH_CUDA_AVAILABLE:
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()

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

        print(f"RobocasaEnv step 耗时 (num_envs={num_envs}, {args.num_steps} steps):")
        print(f"  平均: {mean_ms:.2f} ms")
        print(f"  标准差: {std_ms:.2f} ms")
        print(f"  最小: {min_ms:.2f} ms")
        print(f"  最大: {max_ms:.2f} ms")
        print(f"  约 FPS (每 env): {fps:.1f}")
        peak_mb, peak_util_pct = _get_gpu_memory_stats()
        if peak_mb is not None and peak_util_pct is not None:
            print(f"  GPU 显存峰值: {peak_mb:.2f} MB")
            print(f"  GPU 显存利用率峰值: {peak_util_pct:.2f}%")
        else:
            print(f"  GPU 显存: 未检测到 CUDA，跳过")
        print()

        if hasattr(env, "close"):
            env.close()


if __name__ == "__main__":
    main()
