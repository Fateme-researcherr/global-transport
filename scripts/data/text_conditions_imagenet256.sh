#!/bin/sh
set -e
# Activate the conda environment
CONDA_BASE=$(conda info --base)
source "${CONDA_BASE}/etc/profile.d/conda.sh"

conda activate global-transport

DATA_DIR="PATH/TO/IMAGENET256/"
COND_DIR="PATH/TO/CONDITIONS/"

# Continuous conditions (caption setup of C2OT): captions from visual-layer/imagenet-1k-vl-enriched
# encoded with the DFN5B-CLIP-ViT-H-14 text encoder into 1024-d unit vectors.
# Needs two extra packages: pip install open_clip_torch pyarrow
mkdir -p "${COND_DIR}"

python preprocessing/text_conditions.py captions \
  --split train \
  --dest "${COND_DIR}/train_captions.json"

python preprocessing/text_conditions.py captions \
  --split validation \
  --dest "${COND_DIR}/val_captions.json"

# row i of the training conditions belongs to the i-th latent in DATA_DIR/vae-sd
python preprocessing/text_conditions.py encode \
  --captions "${COND_DIR}/train_captions.json" \
  --dest "${COND_DIR}/train_text_emb_dfn5b.npy" \
  --data-dir "${DATA_DIR}"

python preprocessing/text_conditions.py encode \
  --captions "${COND_DIR}/val_captions.json" \
  --dest "${COND_DIR}/val_text_emb_dfn5b.npy"

python preprocessing/text_conditions.py verify \
  --conds "${COND_DIR}/train_text_emb_dfn5b.npy" \
  --data-dir "${DATA_DIR}"
