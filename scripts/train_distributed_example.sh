#!/bin/bash
set -euo pipefail

# export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export TOKENIZERS_PARALLELISM="false"

if [ -f scripts/train_local.sh ]; then
  source scripts/train_local.sh
fi

PYTHON_BIN="${PYTHON_BIN:-python}"
CONFIG_PATH="${CONFIG_PATH:-configs/train/train_wan22_ti2v_5b_action_adaln.yaml}"
ACCELERATE_CONFIG="${ACCELERATE_CONFIG:-configs/train/accelerate_multi_gpu.yaml}"
MACHINE_RANK="${MACHINE_RANK:-}"
DATASET_NUM_WORKERS="${DATASET_NUM_WORKERS:-8}"
MAX_TRAIN_STEPS="${MAX_TRAIN_STEPS:-20000}"
SAVE_STEPS="${SAVE_STEPS:-2000}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-16}"
DETERMINISTIC="${DETERMINISTIC:-0}"
USE_WANDB="${USE_WANDB:-0}"
USE_SWANLAB="${USE_SWANLAB:-0}"
RUN_NAME="${RUN_NAME:-}"
RESUME_FROM="${RESUME_FROM:-}"
OUTPUT_PATH="${OUTPUT_PATH:-}"

LAUNCH_CMD=(
  "${PYTHON_BIN}" -m accelerate.commands.launch
  --config_file "${ACCELERATE_CONFIG}"
)

if [ -n "${MACHINE_RANK}" ]; then
  LAUNCH_CMD+=(--machine_rank "${MACHINE_RANK}")
fi

TRAIN_CMD=(
  scripts/train.py
  --config "${CONFIG_PATH}"
  --model_paths "${MODEL_DIR}"
  --dataset_base_path "${DATASET_DIR}"
  --dataset_metadata_path "${DATASET_METADATA_PATH}"
  --action_stat_path "${ACTION_STAT_PATH}"
  --dataset_num_workers "${DATASET_NUM_WORKERS}"
  --gradient_accumulation_steps "${GRADIENT_ACCUMULATION_STEPS}"
  --max_train_steps "${MAX_TRAIN_STEPS}"
  --save_steps "${SAVE_STEPS}"
  --use_gradient_checkpointing
)

if [ "${DETERMINISTIC}" = "1" ]; then
  TRAIN_CMD+=(--deterministic)
fi

if [ "${USE_WANDB}" = "1" ]; then
  TRAIN_CMD+=(--use_wandb)
fi

if [ "${USE_SWANLAB}" = "1" ]; then
  TRAIN_CMD+=(--use_swanlab)
fi

if [ -n "${RUN_NAME}" ]; then
  TRAIN_CMD+=(--run_name "${RUN_NAME}")
fi

if [ -n "${OUTPUT_PATH}" ]; then
  TRAIN_CMD+=(--output_path "${OUTPUT_PATH}")
fi

if [ -n "${RESUME_FROM}" ]; then
  TRAIN_CMD+=(--resume_from "${RESUME_FROM}")
fi

"${LAUNCH_CMD[@]}" "${TRAIN_CMD[@]}"
