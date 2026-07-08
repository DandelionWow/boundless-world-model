#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

cd "${REPO_ROOT}"

# ===========================================
# Environment Configuration
# ===========================================

CUDA_VISIBLE_DEVICES='2,3'

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

START_INDEX='50'
MAX_SAMPLES='0'
OVERWRITE=''
# OVERWRITE='--overwrite'

python scripts/infer_ours_distributed.py \
  --config "${CONFIG_PATH}" \
  --model_paths "${MODEL_PATHS}" \
  --ckpt_path "${CKPT_PATH}" \
  --dataset_base_path "${DATASET_BASE_PATH}" \
  --dataset_metadata_path "${DATASET_METADATA_PATH}" \
  --action_stat_path "${ACTION_STAT_PATH}" \
  --output_path "${OUTPUT_PATH}" \
  --start_index "${START_INDEX}" \
  --max_samples "${MAX_SAMPLES}" \
  --gpus "${CUDA_VISIBLE_DEVICES}" \
  ${OVERWRITE}
