#!/bin/sh
set -e
# Activate the conda environment
CONDA_BASE=$(conda info --base)
source "${CONDA_BASE}/etc/profile.d/conda.sh"

conda activate global-transport

DATA_DIR="PATH/TO/IMAGENET256/"
COND_FILE="PATH/TO/CONDITIONS/train_text_emb_dfn5b.npy"
OUTPUT_DIR="PATH/TO/OUTPUT/"
WANDB_DIR="PATH/TO/WANDB_LOG/"
WANDB_PROJ="WANDB_PROJECT_NAME"
EXP_NAME="sit-cont-b-2-256-ot"

accelerate launch \
  --num_machines 1 \
  --multi_gpu \
  --num_processes 8 \
  train_fm.py \
  --torch-compile \
  --mixed-precision "bf16" \
  --project-name "${WANDB_PROJ}" \
  --data-dir "${DATA_DIR}" \
  --cond-file "${COND_FILE}" \
  --cond-dim 1024 \
  --output-dir "${OUTPUT_DIR}" \
  --wandb-dir "${WANDB_DIR}" \
  --exp-name "${EXP_NAME}" \
  --model "SiT-B/2" \
  --attn-func "fa3" \
  --resolution 256 \
  --seed 0 \
  --P-mean 0.0 \
  --max-train-steps 800000 \
  --use-ot \
  --proj-coeff 0  # proj-coeff > 0 to enable REPA, otherwise set proj-coeff = 0 for standard training
