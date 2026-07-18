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
简单脚本：测试 ManiskillEnv.step() 的单步耗时。

用法:
  python -m rlinf.envs.maniskill.bench_step_time [--num-envs N] [--num-steps N] [--warmup K]
  # 或从项目根目录:
  cd /path/to/rlinf_maniskill && python -m rlinf.envs.maniskill.bench_step_time
"""

import argparse
import time

import torch
from omegaconf import OmegaConf

from rlinf.envs.maniskill.maniskill_env import ManiskillEnv


def _make_env_cfg(total_num_envs: int = 32):
    """构建与 ManiskillEnv 兼容的最小 env 配置。"""
    cfg_dict = {
        "env": {
            "train": {
                "env_type": "maniskill",
                "total_num_envs": total_num_envs,
                "auto_reset": True,           # 与实际配置一致
                "ignore_terminations": True,
                "use_rel_reward": True,
                "seed": 0,
                "group_size": 8,              # 4 envs / group_size=2 → 2 groups
                "use_fixed_reset_state_ids": False,
                "max_episode_steps": 80,      # 与实际配置一致
                "video_cfg": {
                    "save_video": False,      # 关闭视频节省时间
                    "info_on_video": False,
                    "video_base_dir": "/tmp/test_video"
                },
                "init_params": {
                    "id": "PutOnPlateInScene25Main-v3",  # 与实际环境一致
                    "obs_mode": "rgb+segmentation",      # 与实际配置一致
                    "control_mode": "arm_pd_ee_target_delta_pose_align2_gripper_pd_joint_pos",
                    "sim_backend": "gpu",
                    "sim_config": {
                        "sim_freq": 500,
                        "control_freq": 5
                    },
                    "max_episode_steps": 80,
                    "sensor_configs": {
                        "shader_pack": "default"         # 必需的渲染参数
                    },
                    "render_mode": "all",
                    "obj_set": "train",
                    "use_multiple_plates": False
                }
            }
        }
    }
    cfg = OmegaConf.create(cfg_dict)
    return cfg.env.train

def main():
    parser = argparse.ArgumentParser(description="Benchmark ManiskillEnv step time")
    parser.add_argument("--num-envs", type=int, default=640, help="并行环境数量")
    parser.add_argument("--num-steps", type=int, default=10, help="计时步数")
    parser.add_argument("--warmup", type=int, default=2, help="预热步数（不参与计时）")
    args = parser.parse_args()
    
    mean_ms_list = []
    std_ms_list = []
    min_ms_list = []
    max_ms_list = []
    fps_list = []
    for num_envs in [800]:
        cfg = _make_env_cfg(total_num_envs=num_envs)
        env = ManiskillEnv(
            cfg=cfg,
            num_envs=num_envs,
            seed_offset=0,
            total_num_processes=1,
            worker_info=None,
            # record_metrics=True,
        )

        obs, info = env.reset()


        action = env.sample_action_space()

        # 预热
        for _ in range(args.warmup):
            obs, reward, term, trunc, info = env.step(action, auto_reset=False)
            action = env.sample_action_space()

        # 排空 warmup 残留的 GPU 操作
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        # 计时
        step_times_ms = []
        for _ in range(args.num_steps):
            action = env.sample_action_space()
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            obs, reward, term, trunc, info = env.step(action, auto_reset=False)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t1 = time.perf_counter()
            step_times_ms.append((t1 - t0) * 1000)

        mean_ms = sum(step_times_ms) / len(step_times_ms)
        variance = sum((t - mean_ms) ** 2 for t in step_times_ms) / len(step_times_ms)
        std_ms = variance ** 0.5
        min_ms = min(step_times_ms)
        max_ms = max(step_times_ms)
        mean_ms_list.append(mean_ms)
        std_ms_list.append(std_ms)
        min_ms_list.append(min_ms)
        max_ms_list.append(max_ms)
        fps_list.append(1000.0 / mean_ms * num_envs)
        print(f"ManiskillEnv step 耗时 (num_envs={num_envs}, {args.num_steps} steps):")
        print(f"  平均: {mean_ms_list[-1]:.2f} ms")
        print(f"  标准差: {std_ms_list[-1]:.2f} ms")
        print(f"  最小: {min_ms_list[-1]:.2f} ms")
        print(f"  最大: {max_ms_list[-1]:.2f} ms")
        print(f"  约 FPS (每 env): {fps_list[-1]:.1f}")

    # print(f"Mean ms list: {mean_ms_list}")
    # print(f"Std ms list: {std_ms_list}")
    # print(f"Min ms list: {min_ms_list}")
    # print(f"Max ms list: {max_ms_list}")
    # print(f"Fps list: {fps_list}")



if __name__ == "__main__":
    main()
