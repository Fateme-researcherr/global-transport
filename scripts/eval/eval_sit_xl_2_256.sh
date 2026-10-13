#!/bin/sh
set -e
# Activate the conda environment
CONDA_BASE=$(conda info --base)
source "${CONDA_BASE}/etc/profile.d/conda.sh"

conda activate global-transport

# Resolve project root from the script's own location so nothing depends on cwd.
PROJECT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
cd "${PROJECT_DIR}"

# Path to the SiT-XL/2 checkpoint you trained (final step).
CKPT_DIR="PATH/TO/sit-xl-2-256/checkpoints/0800000.pt"
SAMPLE_DIR="${PROJECT_DIR}/samples"
# ImageNet-256 reference stats. Download once into ckpt/:
#   hf download kyungmnlee/DMF imgnet256.pkl --local-dir ckpt
REF="${PROJECT_DIR}/ckpt/imgnet256.pkl"

# Sweep over sampling steps (16/32/64) x CFG guidance scale (1.0 ... 4.0).
for ns in 16 32 64; do
  for cfg in 1.0 1.5 2.0 2.5 3.0 3.5 4.0; do
    torchrun \
      --nnodes 1 \
      --nproc_per_node 8 \
      --master-port 29502 \
      generate.py \
      --ckpt "${CKPT_DIR}" \
      --sample-dir "${SAMPLE_DIR}" \
      --num-fid-samples 50000 \
      --per-proc-batch-size 32 \
      --model "SiT-XL/2" \
      --resolution 256 \
      --mode "euler" \
      --num-steps "${ns}" \
      --cfg-scale "${cfg}" \
      --global-seed 3
  done
done

cd preprocessing/
for ns in 16 32 64; do
  for cfg in 1.0 1.5 2.0 2.5 3.0 3.5 4.0; do
    python calculate_metrics.py calc \
    --images="${SAMPLE_DIR}/SiT-XL-2-seed-3-mode-euler-steps-${ns}-shift-1.0-cfg-${cfg}-gmin-0.0-gmax-1.0" \
    --ref="${REF}"
  done
done
