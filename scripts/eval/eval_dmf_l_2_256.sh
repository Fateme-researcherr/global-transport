#!/bin/sh
set -e
# Activate the conda environment
CONDA_BASE=$(conda info --base)
source "${CONDA_BASE}/etc/profile.d/conda.sh"

conda activate global-transport

# Resolve project root from the script's own location so nothing depends on cwd.
PROJECT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
cd "${PROJECT_DIR}"

# Path to the DMFT-L/2 checkpoint you trained (final step).
CKPT_DIR="PATH/TO/dmf-l-2-256/checkpoints/0400000.pt"
SAMPLE_DIR="${PROJECT_DIR}/samples"
# ImageNet-256 reference stats. Download once into ckpt/:
#   hf download kyungmnlee/DMF imgnet256.pkl --local-dir ckpt
REF="${PROJECT_DIR}/ckpt/imgnet256.pkl"

# Sweep over sampling steps (1/2/4); flow maps are sampled without CFG (the guidance is trained in).
for ns in 1 2 4; do
  torchrun \
    --nnodes 1 \
    --nproc_per_node 8 \
    --master-port 29502 \
    generate.py \
    --ckpt "${CKPT_DIR}" \
    --sample-dir "${SAMPLE_DIR}" \
    --num-fid-samples 50000 \
    --per-proc-batch-size 32 \
    --model "DMFT-L/2" \
    --dmf-depth 18 \
    --resolution 256 \
    --mode "euler" \
    --num-steps "${ns}" \
    --global-seed 3
done

cd preprocessing/
for ns in 1 2 4; do
  python calculate_metrics.py calc \
  --images="${SAMPLE_DIR}/DMFT-L-2-seed-3-mode-euler-steps-${ns}-shift-1.0-cfg-1.0-gmin-0.0-gmax-1.0" \
  --ref="${REF}"
done
