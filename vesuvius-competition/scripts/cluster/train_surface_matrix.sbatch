#!/bin/bash
# Usage:
#   sbatch --export=CFG=configs/experiments/surface_unet3d_q1.yaml scripts/cluster/train_surface_matrix.sbatch

#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --job-name=vesu3d
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

if [ -z "${CFG:-}" ]; then
  echo "CFG is required. Example:"
  echo "  sbatch --export=CFG=configs/experiments/surface_unet3d_q1.yaml scripts/cluster/train_surface_matrix.sbatch"
  exit 1
fi

module purge
module load Python/3.10.8-GCCcore-12.2.0

REPO_DIR=${REPO_DIR:-/scratch/jowny/vesuvius/vesuviusChallenge/vesuvius-competition/vesuvius-competition}
cd "${REPO_DIR}"

source venv/bin/activate
export PYTHONPATH="${PWD}"
mkdir -p logs

python train.py --config "${CFG}" --device cuda
