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
测试多个 ManiskillEnv 轮流 onload/offload 并 step 的平均耗时。

与 bench_step_time.py 使用完全相同的环境配置，但场景为：
- 开多个（如 2 个）ManiskillEnv，每个具有相同数量的 num_envs
- 轮流执行：offload 当前不在用的 → onload 要用的 → step 一步
- 统计该模式下的平均 step 时间（含 onload/offload 开销）

用法:
  python -m rlinf.envs.maniskill.bench_step_time_alternating [--num-envs N] [--num-multi-envs K] [--num-steps N] [--warmup K]
"""

import argparse
import time

import torch
from omegaconf import OmegaConf

from rlinf.envs import get_env_cls
from rlinf.envs.env_manager import EnvManager


def _make_env_cfg(total_num_envs: int):
    """与 bench_step_time.py 完全相同的 env 配置。"""
    cfg_dict = {
        "env": {
            "train": {
                "env_type": "maniskill",
                "total_num_envs": total_num_envs,
                "auto_reset": True,
                "ignore_terminations": True,
                "use_rel_reward": True,
                "seed": 0,
                "group_size": 8,
                "use_fixed_reset_state_ids": False,
                "max_episode_steps": 80,
                "video_cfg": {
                    "save_video": False,
                    "info_on_video": False,
                    "video_base_dir": "/tmp/test_video",
                },
                "init_params": {
                    "id": "PutOnPlateInScene25Main-v3",
                    "obs_mode": "rgb+segmentation",
                    "control_mode": "arm_pd_ee_target_delta_pose_align2_gripper_pd_joint_pos",
                    "sim_backend": "gpu",
                    "sim_config": {
                        "sim_freq": 500,
                        "control_freq": 5,
                    },
                    "max_episode_steps": 80,
                    "sensor_configs": {
                        "shader_pack": "default",
                    },
                    "render_mode": "all",
                    "obj_set": "train",
                    "use_multiple_plates": False,
                },
            }
        }
    }
    cfg = OmegaConf.create(cfg_dict)
    return cfg.env.train


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark alternating onload/offload + step for multiple ManiskillEnv"
    )
    parser.add_argument(
        "--num-envs",
        type=int,
        default=320,
        help="每个 ManiskillEnv 的并行环境数量",
    )
    parser.add_argument(
        "--num-multi-envs",
        type=int,
        default=1,
        help="ManiskillEnv 实例数量（轮流 onload/offload）",
    )
    parser.add_argument("--num-steps", type=int, default=10, help="计时循环数（每循环每个 env 各 step 一次）")
    parser.add_argument("--warmup", type=int, default=2, help="预热循环数（不参与计时）")
    args = parser.parse_args()

    num_envs = args.num_envs
    num_multi = args.num_multi_envs
    cfg = _make_env_cfg(total_num_envs=num_envs)
    env_cls = get_env_cls("maniskill", cfg)

    # 使用 EnvManager + enable_offload 实现 onload(start_env) / offload(stop_env)
    managers = []
    for i in range(num_multi):
        mgr = EnvManager(
            cfg=cfg,
            rank=i,
            num_envs=num_envs,
            seed_offset=i,
            total_num_processes=num_multi,
            env_cls=env_cls,
            worker_info=None,
            enable_offload=True,
        )
        managers.append(mgr)

    # 初始化：逐个 onload → reset → warmup → offload
    for i, mgr in enumerate(managers):
        mgr.start_env()
        mgr.reset()
        action = mgr.sample_action_space()
        for _ in range(args.warmup):
            mgr.step(action, auto_reset=False)
            action = mgr.sample_action_space()
        # mgr.stop_env()

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    # 计时：轮流 onload → step → offload（任意时刻仅一个 env 在 GPU 上）
    cycle_times_ms = []  # 完整周期：onload + step + offload
    step_only_times_ms = []  # 仅 step 耗时

    for cycle in range(args.num_steps):
        for i in range(num_multi):
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t0 = time.perf_counter()

            # managers[i].start_env()
            action = managers[i].sample_action_space()

            t_after_onload = time.perf_counter()
            obs, reward, term, trunc, info = managers[i].step(action, auto_reset=False)
            t_after_step = time.perf_counter()

            # managers[i].stop_env()

            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t1 = time.perf_counter()

            cycle_times_ms.append((t1 - t0) * 1000)
            step_only_times_ms.append((t_after_step - t_after_onload) * 1000)


    for i in range(num_multi):            
        managers[i].stop_env()


    mean_cycle = sum(cycle_times_ms) / len(cycle_times_ms)
    std_cycle = (sum((t - mean_cycle) ** 2 for t in cycle_times_ms) / len(cycle_times_ms)) ** 0.5
    mean_step = sum(step_only_times_ms) / len(step_only_times_ms)
    std_step = (sum((t - mean_step) ** 2 for t in step_only_times_ms) / len(step_only_times_ms)) ** 0.5

    print(f"Alternating onload/offload + step (num_multi_envs={num_multi}, num_envs={num_envs}, {args.num_steps} cycles):")
    print(f"  完整周期 (onload+step+offload) 平均: {mean_cycle:.2f} ms (std: {std_cycle:.2f})")
    print(f"  仅 step 平均: {mean_step:.2f} ms (std: {std_step:.2f})")
    print(f"  约 FPS (每 env, 按完整周期): {1000.0 / mean_cycle * num_envs:.1f}")
    print(f"  约 FPS (每 env, 按 step 计): {1000.0 / mean_step * num_envs:.1f}")


if __name__ == "__main__":
    main()
