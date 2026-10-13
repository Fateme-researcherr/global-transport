#!/bin/sh
set -e
# Activate the conda environment
CONDA_BASE=$(conda info --base)
source "${CONDA_BASE}/etc/profile.d/conda.sh"

conda activate global-transport

DATA_DIR="PATH/TO/IMAGENET256/"

# Pre-processed ImageNet-256 (edm2 layout: sd-vae-ft-mse latents + dataset.json labels and the
# 256x256 images) from https://huggingface.co/datasets/Yuehao/imagenet-1k-256, ~295 GB of chunked
# zip archives. Needs the HuggingFace CLI: pip install -U "huggingface_hub[cli]"
hf download Yuehao/imagenet-1k-256 --repo-type dataset --local-dir "${DATA_DIR}"

cat "${DATA_DIR}"/vae-sd.zip.chunk_* > "${DATA_DIR}/vae-sd.zip"
unzip -qn "${DATA_DIR}/vae-sd.zip" -d "${DATA_DIR}/vae-sd"

# The images are only needed for REPA (--proj-coeff > 0) and for the caption checks
cat "${DATA_DIR}"/images.zip.chunk_* > "${DATA_DIR}/images.zip"
unzip -qn "${DATA_DIR}/images.zip" -d "${DATA_DIR}/images"
