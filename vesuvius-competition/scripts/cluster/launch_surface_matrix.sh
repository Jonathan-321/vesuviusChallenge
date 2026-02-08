#!/bin/bash
# Launch 4-run 48h matrix.
# Run from repo root on OSCER:
#   bash scripts/cluster/launch_surface_matrix.sh

set -euo pipefail

CONFIGS=(
  "configs/experiments/surface_unet3d_q1.yaml"
  "configs/experiments/surface_unet3d_q2.yaml"
  "configs/experiments/surface_unet3d_q3.yaml"
  "configs/experiments/surface_unet3d_q4.yaml"
)

for cfg in "${CONFIGS[@]}"; do
  echo "Submitting ${cfg}"
  sbatch --export=CFG="${cfg}" scripts/cluster/train_surface_matrix.sbatch
done
