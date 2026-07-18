#!/usr/bin/env bash
set -euo pipefail

export NCCL_P2P_DISABLE=1
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"

CONFIG_NAME="${1:-maniskill_col_ss_env2048_pipe1}"
echo "[ManiSkill] Running config: ${CONFIG_NAME}"
bash examples/embodiment/run_maniskill.sh "${CONFIG_NAME}"
