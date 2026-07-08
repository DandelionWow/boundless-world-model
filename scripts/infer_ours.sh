#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

cd "${REPO_ROOT}"

# export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

# CONFIG_PATH="${CONFIG_PATH:-configs/infer/robotwin_ti2v_720p.yaml}"
# MODEL_PATHS="${MODEL_PATHS:-models/Wan2.2-TI2V-5B}"
# CKPT_PATH="${CKPT_PATH:-checkpoints/action_model.safetensors}"
# DATASET_BASE_PATH="${DATASET_BASE_PATH:-data/RoboTwin2.0_lerobot}"
# DATASET_METADATA_PATH="${DATASET_METADATA_PATH:-data/RoboTwin2.0_lerobot/metadata/episodes_val.jsonl}"
# ACTION_STAT_PATH="${ACTION_STAT_PATH:-data/RoboTwin2.0_lerobot/metadata/stat.json}"
# OUTPUT_PATH="${OUTPUT_PATH:-outputs/infer}"
# MAX_SAMPLES="${MAX_SAMPLES:-1}"
# PYTHON_BIN="${PYTHON_BIN:-python}"

# ===========================================
# Environment Configuration
# ===========================================

# export CUDA_VISIBLE_DEVICES='0, 1, 2, 3'
# export CUDA_VISIBLE_DEVICES='3'

# ===========================================
# Path Configuration
# ===========================================

CONFIG_PATH='configs/infer/infer.yaml'
MODEL_PATHS='/data/wangbowen/models/Wan-AI/Wan2.2-TI2V-5B/'
CKPT_PATH='/data/wangbowen/models/BLM/step-12000.safetensors'
DATASET_BASE_PATH='data/RoboTwin2.0_boundless'
DATASET_METADATA_PATH='data/RoboTwin2.0_boundless/RoboTwin2.0_boundless.jsonl'
ACTION_STAT_PATH='data/RoboTwin2.0_boundless/stat.json'
OUTPUT_PATH='outputs/inference'

# ===========================================
# Inference Configuration
# ===========================================

# START_INDEX='1'
# MAX_SAMPLES='1'

CUDA_VISIBLE_DEVICES='2' python scripts/infer_ours.py \
  --config "${CONFIG_PATH}" \
  --model_paths "${MODEL_PATHS}" \
  --ckpt_path "${CKPT_PATH}" \
  --dataset_base_path "${DATASET_BASE_PATH}" \
  --dataset_metadata_path "${DATASET_METADATA_PATH}" \
  --action_stat_path "${ACTION_STAT_PATH}" \
  --output_path "${OUTPUT_PATH}" \
  --start_index 30 \
  --max_samples 10

# CUDA_VISIBLE_DEVICES='3' python scripts/infer_ours.py \
#   --config "${CONFIG_PATH}" \
#   --model_paths "${MODEL_PATHS}" \
#   --ckpt_path "${CKPT_PATH}" \
#   --dataset_base_path "${DATASET_BASE_PATH}" \
#   --dataset_metadata_path "${DATASET_METADATA_PATH}" \
#   --action_stat_path "${ACTION_STAT_PATH}" \
#   --output_path "${OUTPUT_PATH}" \
#   --start_index 10 \
#   --max_samples 10