# RL-VLA³ (COLM 2026)

Official code for **RL-VLA³**, published at **COLM 2026**.

**RL-VLA³** is a fully asynchronous distributed reinforcement learning framework for post-training Vision-Language-Action (VLA) models.

Physical simulators introduce highly variable, resource-intensive latencies that mismatch synchronous LLM-style RL pipelines. RL-VLA³ decouples **Simulator**, **Generator**, and **Trainer** into independently progressing resource groups, and improves throughput via:

- **Asynchronous rollout** between parallel Simulators and Generators
- **Dynamic batching** for Generator inference
- **Fine-grained environment sharding** across Generators
- **Asynchronous training** that streams finished trajectories to the Trainer without global barriers

Across LIBERO, ManiSkill, Meta-World, and RoboCasa, with backbones such as π₀ / π₀.₅, GR00T N1.5, and OpenVLA-OFT, RL-VLA³ achieves up to **85.2%** higher throughput than synchronous baselines while preserving sample efficiency, with scaling validated from **8 to 256 GPUs**.

## Acknowledgement

This codebase is modified from [RLinf](https://github.com/RLinf/RLinf).  
We thank the RLinf authors for the open-source infrastructure that this project builds upon.

## What's New vs. RLinf

Relative to upstream RLinf, the main system changes live under [`rlinf/workers/`](rlinf/workers/): **async** and **flexible** workers for fine-grained Simulator–Generator interaction and async training.

| Component | Key modules |
|-----------|-------------|
| Env (Simulator) | `rlinf/workers/env/per_env_async/` — e.g. `flexible_per_env_async_worker.py`, `aggregated_per_env_async_worker.py` |
| Rollout (Generator) | `rlinf/workers/rollout/hf/` — e.g. `flexible_per_env_async_rollout_worker.py`, `per_env_async_rollout_worker.py` |
| Flex routing / sharding | `rlinf/workers/per_env_flex_plan.py` |
| Actor (Trainer async) | async / streaming paths in `rlinf/workers/actor/` (used with `algorithm.pipeline_mode: async`) |

Worker selection is driven by YAML (`env.per_env_async.*`, `rollout.per_env_async.*`, `algorithm.pipeline_mode`) via the factories in those packages. Paper experiment configs are under `examples/embodiment/config/`.

## Environment & Dependencies

Our runtime environment and dependency stack are the same as [RLinf](https://github.com/RLinf/RLinf).  
Please follow the official RLinf documentation for installation, Docker images, and embodied simulator setup:

- **[RLinf Docs](https://rlinf.readthedocs.io/en/latest/)** (installation, VLA / embodied quick start, examples)

Optional local helpers under `requirements/` mirror the upstream embodied install flow; when in doubt, prefer the RLinf docs above.

## Quick Start

After the RLinf-compatible environment is ready:

```bash
bash scripts/run_libero.sh      [config_name]
bash scripts/run_maniskill.sh   [config_name]
bash scripts/run_metaworld.sh   [config_name]
bash scripts/run_robocasa.sh    [config_name]
```

Example:

```bash
bash scripts/run_libero.sh libero_col_aa_env256_pipe4
bash scripts/run_maniskill.sh maniskill_hyb62_aa_env3264_pipe2
```

Entrypoint: `examples/embodiment/train_embodied_agent.py`  
Configs: `examples/embodiment/config/{libero,maniskill,metaworld,robocasa}/`  
Naming convention: see [`examples/embodiment/config/README.md`](examples/embodiment/config/README.md).

## Config Modes

| Tag | Meaning |
|-----|---------|
| `ss` | Sync rollout + sync training (baseline) |
| `as` | Async rollout + sync training |
| `aa` | Async rollout + async training (full RL-VLA³) |
| `col` | Colocated placement |
| `hyb62` | Hybrid placement (Simulator GPUs 0–5, Generator GPUs 6–7) |

Key YAML switches:

```yaml
algorithm.pipeline_mode: async
env.per_env_async.enabled: true
rollout.per_env_async.enabled: true
rollout.per_env_async.max_batch_size: ...
rollout.per_env_async.batch_timeout_ms: ...
actor.gradient_accumulation_across_epochs: true
```

## Citation

```bibtex
@article{sun2026rl,
  title={RL-VLA$^{3}$: A Flexible and Asynchronous Reinforcement Learning Framework for VLA Training},
  author={Sun, Haoran and Guo, Yongjian and Guan, Zhong and Di, Shuai and Bai, Xiaodong and Long, Jing and Zhao, Tianyun and Luo, Mingxi and Zhao, Hongke and Wu, Likang and others},
  journal={arXiv e-prints},
  pages={arXiv--2602},
  year={2026}
}
```

Please also cite [RLinf](https://github.com/RLinf/RLinf) when appropriate.

## License

Apache License 2.0 (same as the upstream RLinf release).
