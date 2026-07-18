# Experiment Configs (Paper-Aligned)

Configs reproduce the main settings in the RL-VLA³ paper (throughput tables & ablations).

## Naming

```text
{env}_{placement}_{mode}_env{N}_pipe{P}[_b{B}][_t{T}ms][_train]
```

| Token | Meaning |
|-------|---------|
| `env` | `libero` / `maniskill` / `metaworld` / `robocasa` |
| `placement` | `col` = colocated; `hyb62` = hybrid (sim 0–5, gen 6–7 on 8 GPUs) |
| `mode` | `ss` = sync rollout + sync train; `as` = async rollout only; `aa` = full async |
| `env{N}` | Total parallel environments |
| `pipe{P}` | Environment batches (pipeline stages) per Simulator GPU |
| `b{B}` | Dynamic-batching max batch number (optional) |
| `t{T}ms` | Dynamic-batching latency threshold in ms (optional) |
| `_train` | Longer run used for success-rate curves |

## Layout

```text
config/
├── libero/                       # Appendix Table: LIBERO + GR00T (colocated)
├── maniskill/                    # Appendix Table: ManiSkill (col + hyb62)
├── metaworld/                    # Appendix Table: Meta-World (col + hyb62)
├── robocasa/                     # Appendix Table: RoboCasa (col + hyb62)
│   └── ablation_dynamic_batching/  # RoboCasa dynamic-batching sweep
├── env/                          # Shared env defaults
├── model/                        # Shared model defaults
└── training_backend/             # FSDP etc.
```

## Paper Mapping (8-GPU)

### LIBERO (colocated, 256 envs)

| Config | Rollout Async | Train Async | Batches/GPU |
|--------|---------------|-------------|-------------|
| `libero_col_ss_env256_pipe1` | off | off | 1 |
| `libero_col_ss_env256_pipe2` | off | off | 2 |
| `libero_col_as_env256_pipe2` | on | off | 2 |
| `libero_col_aa_env256_pipe2` | on | on | 2 |
| `libero_col_ss_env256_pipe4` | off | off | 4 |
| `libero_col_as_env256_pipe4` | on | off | 4 |
| `libero_col_aa_env256_pipe4` | on | on | 4 |

### ManiSkill

| Config | Placement | Mode |
|--------|-----------|------|
| `maniskill_col_ss_env2048_pipe{1,2,4}` | col | ss |
| `maniskill_col_as_env2048_pipe{2,4}` | col | as |
| `maniskill_hyb62_{ss,as,aa}_env3264_pipe2` | hyb62 | ss/as/aa |
| `*_train` | longer runs for success curves | |

### Meta-World

| Config | Placement | Mode |
|--------|-----------|------|
| `metaworld_col_{ss,as,aa}_env512_pipe{1,2,4}` | col | … |
| `metaworld_hyb62_{ss,as,aa}_env768_pipe2` | hyb62 | … |
| `*_train` | success-rate curves | |

### RoboCasa

| Config | Notes |
|--------|-------|
| `robocasa_col_{ss,as,aa}_env160_pipe{1,2}` | colocated main table |
| `robocasa_hyb62_{ss,as,aa}_env168_pipe2_*` | hybrid main table |
| `ablation_dynamic_batching/*` | max-batch × latency sweep |

## Launch

From repo root:

```bash
bash scripts/run_libero.sh libero_col_aa_env256_pipe4
bash scripts/run_maniskill.sh maniskill_hyb62_aa_env3264_pipe2
bash scripts/run_metaworld.sh metaworld_col_aa_env512_pipe4
bash scripts/run_robocasa.sh robocasa_col_aa_env160_pipe2
```
