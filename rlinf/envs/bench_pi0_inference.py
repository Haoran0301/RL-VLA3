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
Benchmark Pi0 / Pi0.5 / GR00t inference speed under different batch sizes，支持多环境输入格式。

支持模型类型:
- pi0:  OpenPi Pi0 (config_name: pi0_maniskill, pi0_libero, ...)
- pi05: OpenPi Pi0.5 (config_name: pi05_maniskill, pi05_libero, ...)
- gr00t: GR00T-N1.5 (仅支持 maniskill, libero)

各环境传入模型的数据格式（基于 dataconfig 与 policy 调研）:

| 环境 | main_images | wrist_images | state_dim | Pi0 config | GR00t |
|------|-------------|--------------|-----------|-------------|-------|
| maniskill | [B,256,256,3] | None | 8 | pi0/pi05_maniskill | maniskill_widowx |
| libero | [B,256,256,3] | [B,256,256,3] | 8 | pi0/pi05_libero | libero_franka |
| calvin | [B,200,200,3] | [B,84,84,3] | 7 | pi0/pi05_calvin | - |
| metaworld | [B,480,480,3] | None | 4 | pi0/pi05_metaworld | - |
| robocasa | [B,224,224,3] | [B,224,224,3] | 8 | pi0_robocasa | - |

用法:
  python -m rlinf.envs.bench_pi0_inference --model-type pi0 --env robocasa --model-path /ufs/models/RLinf/RLinf-Pi0-RoboCasa
  python -m rlinf.envs.bench_pi0_inference --model-type pi05 --env libero --model-path /ufs/models/RLinf/RLinf-Pi05-ManiSkill-25Main-SFT/
  python -m rlinf.envs.bench_pi0_inference --model-type gr00t --env libero --model-path /path/to/gr00t_ckpt
"""

import argparse
import copy
import statistics
import time
from typing import Any, Dict, List

import torch
from omegaconf import OmegaConf, open_dict

from rlinf.models import get_model

# 各环境的 Pi0/Pi0.5 输入格式配置（基于 policies/*_policy.py 与 env obs 调研）
# pi05_config_name: None 表示该环境无 Pi0.5 配置（如 robocasa）
ENV_PI0_FORMATS = {
    "maniskill": {
        "main_h": 256,
        "main_w": 256,
        "wrist_h": None,
        "wrist_w": None,
        "has_wrist": False,
        "state_dim": 8,
        "config_name": "pi0_maniskill",
        "pi05_config_name": "pi05_maniskill",
        "action_dim": 7,
        "num_images_in_input": 1,
        "action_horizon": 8,
    },
    "calvin": {
        "main_h": 200,
        "main_w": 200,
        "wrist_h": 84,
        "wrist_w": 84,
        "has_wrist": True,
        "state_dim": 7,  # ee_pos(3) + ee_rot(3) + gripper(1)
        "config_name": "pi0_calvin",
        "pi05_config_name": "pi05_calvin",
        "action_dim": 7,
        "num_images_in_input": 2,
        "action_horizon": 5,
    },
    "libero": {
        "main_h": 256,
        "main_w": 256,
        "wrist_h": 256,
        "wrist_w": 256,
        "has_wrist": True,
        "state_dim": 8,
        "config_name": "pi0_libero",
        "pi05_config_name": "pi05_libero",
        "action_dim": 7,
        "num_images_in_input": 2,
        "action_horizon": 8,
    },
    "metaworld": {
        "main_h": 480,
        "main_w": 480,
        "wrist_h": None,
        "wrist_w": None,
        "has_wrist": False,
        "state_dim": 4,
        "config_name": "pi0_metaworld",
        "pi05_config_name": "pi05_metaworld",
        "action_dim": 4,
        "num_images_in_input": 1,
        "action_horizon": 5,
    },
    "robocasa": {
        "main_h": 224,
        "main_w": 224,
        "wrist_h": 224,
        "wrist_w": 224,
        "has_wrist": True,
        "state_dim": 8,  # eef_pos(3)+eef_quat(4)+gripper(1), policy make_robocasa_example
        "config_name": "pi0_robocasa",
        "pi05_config_name": None,  # 无 pi05_robocasa
        "action_dim": 12,
        "num_images_in_input": 2,
        "action_horizon": 10,
    },
}

# GR00t 仅支持 maniskill、libero
ENV_GR00T_FORMATS = {
    "maniskill": {
        "embodiment_tag": "maniskill_widowx",
        "obs_converter_type": "maniskill",
        "action_dim": 7,
    },
    "libero": {
        "embodiment_tag": "libero_franka",
        "obs_converter_type": "libero",
        "action_dim": 7,
    },
}


def _parse_batch_sizes(s: str) -> List[int]:
    vals = [int(x.strip()) for x in s.split(",") if x.strip()]
    if not vals:
        raise ValueError("batch sizes cannot be empty")
    if any(v <= 0 for v in vals):
        raise ValueError(f"all batch sizes must be > 0, got {vals}")
    return vals


def _get_openpi_config_name(env_fmt: Dict[str, Any], model_type: str) -> str:
    if model_type == "pi05":
        cfg = env_fmt.get("pi05_config_name")
        if cfg is None:
            raise ValueError(
                f"env {env_fmt.get('config_name', '?')} 无 Pi0.5 配置，请使用 --model-type pi0 或换环境"
            )
        return cfg
    return env_fmt["config_name"]


def _build_openpi_model_cfg(args: argparse.Namespace, env_fmt: Dict[str, Any]) -> object:
    config_name = _get_openpi_config_name(env_fmt, args.model_type)
    env_fmt = {**env_fmt, "config_name": config_name}

    if args.train_config is not None:
        cfg = OmegaConf.load(args.train_config)
        if not hasattr(cfg, "actor") or not hasattr(cfg.actor, "model"):
            raise ValueError(
                f"train config {args.train_config} does not contain actor.model"
            )
        model_cfg = copy.deepcopy(cfg.actor.model)
        if hasattr(cfg, "rollout") and hasattr(cfg.rollout, "model"):
            if hasattr(cfg.rollout.model, "precision"):
                with open_dict(model_cfg):
                    model_cfg.precision = cfg.rollout.model.precision
    else:
        action_dim = (
            args.action_dim if args.action_dim is not None else env_fmt["action_dim"]
        )
        model_cfg = OmegaConf.create(
            {
                "model_type": "openpi",
                "model_path": args.model_path,
                "precision": args.precision,
                "trust_remote_code": True,
                "is_lora": False,
                "add_value_head": args.add_value_head,
                "num_action_chunks": args.num_action_chunks,
                "action_dim": action_dim,
                "num_steps": args.model_num_steps,
                "openpi": {
                    "config_name": env_fmt["config_name"],
                    "num_images_in_input": env_fmt["num_images_in_input"],
                    "noise_method": args.noise_method,
                    "action_horizon": env_fmt["action_horizon"],
                    "noise_params": [0.16, 0.12, 200],
                    "joint_logprob": args.joint_logprob,
                    "action_chunk": args.num_action_chunks,
                    "num_steps": args.model_num_steps,
                    "action_env_dim": action_dim,
                    "add_value_head": args.add_value_head,
                },
            }
        )

    if not hasattr(model_cfg, "model_type") or model_cfg.model_type != "openpi":
        raise ValueError(
            f"this benchmark requires model_type=openpi, got {getattr(model_cfg, 'model_type', None)}"
        )

    with open_dict(model_cfg):
        if args.model_path is not None:
            model_cfg.model_path = args.model_path
        if not hasattr(model_cfg, "num_action_chunks"):
            model_cfg.num_action_chunks = args.num_action_chunks
        if not hasattr(model_cfg, "action_dim"):
            model_cfg.action_dim = env_fmt["action_dim"]
        if not hasattr(model_cfg, "num_steps"):
            model_cfg.num_steps = args.model_num_steps
        if not hasattr(model_cfg, "precision"):
            model_cfg.precision = args.precision
        if not hasattr(model_cfg, "add_value_head"):
            model_cfg.add_value_head = args.add_value_head
        if not hasattr(model_cfg, "openpi"):
            model_cfg.openpi = OmegaConf.create({})
        model_cfg.openpi.config_name = env_fmt["config_name"]
        if not hasattr(model_cfg.openpi, "num_images_in_input"):
            model_cfg.openpi.num_images_in_input = env_fmt["num_images_in_input"]
        if not hasattr(model_cfg.openpi, "noise_method"):
            model_cfg.openpi.noise_method = args.noise_method
        if not hasattr(model_cfg.openpi, "action_horizon"):
            model_cfg.openpi.action_horizon = env_fmt["action_horizon"]
        if not hasattr(model_cfg.openpi, "noise_params"):
            model_cfg.openpi.noise_params = [0.16, 0.12, 200]
        if not hasattr(model_cfg.openpi, "joint_logprob"):
            model_cfg.openpi.joint_logprob = args.joint_logprob
        if not hasattr(model_cfg.openpi, "action_chunk"):
            model_cfg.openpi.action_chunk = model_cfg.num_action_chunks
        if not hasattr(model_cfg.openpi, "num_steps"):
            model_cfg.openpi.num_steps = model_cfg.num_steps
        if not hasattr(model_cfg.openpi, "action_env_dim"):
            model_cfg.openpi.action_env_dim = model_cfg.action_dim
        if not hasattr(model_cfg.openpi, "add_value_head"):
            model_cfg.openpi.add_value_head = model_cfg.add_value_head

    if not hasattr(model_cfg, "model_path") or model_cfg.model_path in (None, ""):
        raise ValueError("model_path is required (via --model-path or --train-config)")

    return model_cfg


def _build_gr00t_model_cfg(args: argparse.Namespace, gr00t_fmt: Dict[str, Any]) -> object:
    """构建 GR00t 模型配置，仅支持 maniskill、libero。"""
    model_cfg = OmegaConf.create(
        {
            "model_type": "gr00t",
            "model_path": args.model_path,
            "precision": args.precision,
            "trust_remote_code": True,
            "is_lora": False,
            "embodiment_tag": gr00t_fmt["embodiment_tag"],
            "obs_converter_type": gr00t_fmt["obs_converter_type"],
            "action_dim": gr00t_fmt["action_dim"],
            "num_action_chunks": args.num_action_chunks or 5,
            "denoising_steps": args.denoising_steps,
            "rl_head_config": OmegaConf.create(
                {
                    "joint_logprob": False,
                    "noise_method": "flow_sde",
                    "ignore_last": False,
                    "safe_get_logprob": False,
                    "noise_anneal": False,
                    "noise_params": [0.7, 0.3, 400],
                    "noise_level": 0.5,
                    "add_value_head": args.add_value_head,
                    "chunk_critic_input": False,
                    "detach_critic_input": True,
                    "valid_action_dim": gr00t_fmt["action_dim"],
                    "disable_dropout": True,
                }
            ),
        }
    )
    if not model_cfg.model_path or model_cfg.model_path in (None, ""):
        raise ValueError("model_path is required (via --model-path)")
    return model_cfg


def _make_fake_env_obs(
    max_batch: int,
    env_fmt: Dict[str, Any],
    device: torch.device,
    main_h: int,
    main_w: int,
    state_dim: int,
) -> Dict:
    """构造符合各环境 Pi0 输入格式的假观测。"""
    main_images = torch.randint(
        low=0,
        high=256,
        size=(max_batch, main_h, main_w, 3),
        dtype=torch.uint8,
        device=device,
    )
    states = torch.zeros((max_batch, state_dim), dtype=torch.float32, device=device)
    task_descriptions = ["do something"] * max_batch

    obs = {
        "main_images": main_images,
        "wrist_images": None,
        "states": states,
        "task_descriptions": task_descriptions,
    }

    if env_fmt["has_wrist"]:
        wh, ww = env_fmt["wrist_h"], env_fmt["wrist_w"]
        obs["wrist_images"] = torch.randint(
            low=0,
            high=256,
            size=(max_batch, wh, ww, 3),
            dtype=torch.uint8,
            device=device,
        )

    return obs


def _slice_obs(obs: Dict, batch_size: int) -> Dict:
    out = {
        "main_images": obs["main_images"][:batch_size],
        "states": obs["states"][:batch_size],
        "task_descriptions": obs["task_descriptions"][:batch_size],
    }
    if obs["wrist_images"] is not None:
        out["wrist_images"] = obs["wrist_images"][:batch_size]
    else:
        out["wrist_images"] = None
    return out


def _copy_obs_for_gr00t(obs: Dict) -> Dict:
    """GR00t predict_action_batch 会原地修改 states 和 main_images，需传副本。"""
    out = {
        "main_images": obs["main_images"].clone(),
        "states": obs["states"].clone(),
        "task_descriptions": obs["task_descriptions"],
    }
    out["wrist_images"] = obs["wrist_images"].clone() if obs["wrist_images"] is not None else None
    return out


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark Pi0 / Pi0.5 / GR00t inference latency under different batch sizes"
    )
    parser.add_argument(
        "--model-type",
        type=str,
        default="pi0",
        choices=["pi0", "pi05", "gr00t"],
        help="模型类型: pi0, pi05, gr00t",
    )
    parser.add_argument(
        "--env",
        type=str,
        default="maniskill",
        choices=list(ENV_PI0_FORMATS.keys()),
        help="环境类型，决定输入格式；gr00t 仅支持 maniskill/libero",
    )
    parser.add_argument(
        "--batch-sizes",
        type=str,
        default="1,2,4,8,16,32,64,80,90,100",
        help="comma-separated batch sizes",
    )
    parser.add_argument("--num-steps", type=int, default=20, help="timed steps per batch size")
    parser.add_argument("--warmup", type=int, default=5, help="warmup steps per batch size")
    parser.add_argument(
        "--train-config",
        type=str,
        default=None,
        help="optional merged training config path",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=None,
        help="override model path; required if train config does not provide one",
    )
    parser.add_argument(
        "--precision",
        type=str,
        default="bf16",
        choices=["bf16", "fp16", "fp32"],
    )
    parser.add_argument("--add-value-head", action="store_true")
    parser.add_argument("--num-action-chunks", type=int, default=None, help="default from env")
    parser.add_argument("--action-dim", type=int, default=None, help="default from env")
    parser.add_argument("--model-num-steps", type=int, default=4)
    parser.add_argument("--denoising-steps", type=int, default=4, help="GR00t denoising steps")
    parser.add_argument(
        "--noise-method",
        type=str,
        default="flow_noise",
        choices=["flow_sde", "flow_noise", "flow_cps"],
    )
    parser.add_argument("--joint-logprob", action="store_true")
    parser.add_argument("--image-size", type=int, default=None, help="override main image H/W")
    parser.add_argument("--state-dim", type=int, default=None, help="override state dim")
    args = parser.parse_args()

    if args.model_type == "gr00t" and args.env not in ENV_GR00T_FORMATS:
        raise ValueError("GR00t only supports env in %s, got env=%s" % (list(ENV_GR00T_FORMATS.keys()), args.env))

    env_fmt = ENV_PI0_FORMATS[args.env].copy()
    if args.num_action_chunks is None:
        args.num_action_chunks = 5

    main_h = args.image_size if args.image_size is not None else env_fmt["main_h"]
    main_w = args.image_size if args.image_size is not None else env_fmt["main_w"]
    state_dim = args.state_dim if args.state_dim is not None else env_fmt["state_dim"]

    batch_sizes = _parse_batch_sizes(args.batch_sizes)
    max_batch = max(batch_sizes)
    if args.num_steps <= 0 or args.warmup < 0:
        raise ValueError("num_steps must be > 0 and warmup must be >= 0")

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this benchmark")
    device = torch.device("cuda")

    if args.model_type == "gr00t":
        gr00t_fmt = ENV_GR00T_FORMATS[args.env]
        model_cfg = _build_gr00t_model_cfg(args, gr00t_fmt)
    else:
        model_cfg = _build_openpi_model_cfg(args, env_fmt)

    print("=== Model Init (aligned with rollout worker) ===")
    print(f"model_type: {args.model_type}")
    print(f"env:        {args.env}")
    print(f"model_path: {model_cfg.model_path}")
    print(f"precision:  {model_cfg.precision}")
    if args.model_type == "gr00t":
        print(f"embodiment: {model_cfg.embodiment_tag}")
        print(f"obs_conv:   {model_cfg.obs_converter_type}")
    else:
        print(f"openpi.cfg: {model_cfg.openpi.config_name}")
        print(f"noise:      {model_cfg.openpi.noise_method}")
    print(f"num_chunks: {model_cfg.num_action_chunks}")
    print(f"action_dim: {model_cfg.action_dim}")
    print(f"device:     {device}")

    model = get_model(model_cfg)
    model.eval()

    fake_obs = _make_fake_env_obs(
        max_batch=max_batch,
        env_fmt=env_fmt,
        device=device,
        main_h=main_h,
        main_w=main_w,
        state_dim=state_dim,
    )

    print("\n=== Benchmark ===")
    print(
        f"warmup={args.warmup}, num_steps={args.num_steps}, "
        f"batch_sizes={batch_sizes}, image={main_h}x{main_w}, state_dim={state_dim}"
    )
    if env_fmt["has_wrist"]:
        print(f"wrist_image: {env_fmt['wrist_h']}x{env_fmt['wrist_w']}")
    else:
        print("wrist_image: None")
    print("-" * 96)
    print(f"{'batch':>8} | {'mean_ms':>10} | {'std_ms':>10} | {'p50_ms':>10} | {'p95_ms':>10} | {'samples/s':>12}")
    print("-" * 96)

    use_gr00t = args.model_type == "gr00t"
    has_preprocess = not use_gr00t and hasattr(model, "preprocess_env_obs") and model.preprocess_env_obs is not None

    with torch.no_grad():
        for bs in batch_sizes:
            obs_bs = _slice_obs(fake_obs, bs)
            if has_preprocess:
                obs_bs = model.preprocess_env_obs(obs_bs)

            for _ in range(args.warmup):
                inp = _copy_obs_for_gr00t(obs_bs) if use_gr00t else obs_bs
                _ = model.predict_action_batch(env_obs=inp, mode="train")
            torch.cuda.synchronize()

            times_ms = []
            for _ in range(args.num_steps):
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                inp = _copy_obs_for_gr00t(obs_bs) if use_gr00t else obs_bs
                _ = model.predict_action_batch(env_obs=inp, mode="train")
                torch.cuda.synchronize()
                t1 = time.perf_counter()
                times_ms.append((t1 - t0) * 1000.0)

            mean_ms = statistics.fmean(times_ms)
            std_ms = statistics.pstdev(times_ms) if len(times_ms) > 1 else 0.0
            sorted_t = sorted(times_ms)
            p50_ms = sorted_t[len(sorted_t) // 2]
            p95_ms = sorted_t[min(len(sorted_t) - 1, int(len(sorted_t) * 0.95))]
            samples_per_s = bs * 1000.0 / mean_ms

            print(
                f"{bs:8d} | {mean_ms:10.2f} | {std_ms:10.2f} | {p50_ms:10.2f} | {p95_ms:10.2f} | {samples_per_s:12.1f}"
            )

    print("-" * 96)
    print("Done.")


if __name__ == "__main__":
    main()
# _action_batch(env_obs=obs_bs, mode="train")
#                 torch.cuda.synchronize()
#                 t1 = time.perf_counter()
#                 times_ms.append((t1 - t0) * 1000.0)

#             mean_ms = statistics.fmean(times_ms)
#             std_ms = statistics.pstdev(times_ms) if len(times_ms) > 1 else 0.0
#             sorted_t = sorted(times_ms)
#             p50_ms = sorted_t[len(sorted_t) // 2]
#             p95_ms = sorted_t[min(len(sorted_t) - 1, int(len(sorted_t) * 0.95))]
#             samples_per_s = bs * 1000.0 / mean_ms

#             print(
#                 f"{bs:8d} | {mean_ms:10.2f} | {std_ms:10.2f} | {p50_ms:10.2f} | {p95_ms:10.2f} | {samples_per_s:12.1f}"
#             )

#     print("-" * 96)
#     print("Done.")


# if __name__ == "__main__":
#     main()
