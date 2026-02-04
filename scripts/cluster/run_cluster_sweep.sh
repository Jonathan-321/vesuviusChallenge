#!/bin/bash
# Run validation sweep on OSCER cluster

# SSH into cluster and run commands
ssh jowny@schooner.oscer.ou.edu << 'EOF'
set -e

echo "=== Setting up environment ==="
cd /scratch/jowny/vesuvius/vesuviusChallenge/vesuvius-competition

# Export PYTHONPATH
export PYTHONPATH=$PWD:$PWD/src

# Activate venv
source venv/bin/activate

echo "=== Verifying environment ==="
pwd
echo "PYTHONPATH: $PYTHONPATH"
python -c "import sys; print('Python:', sys.executable)"
python -c "import train, src.models; print('Imports OK')"

echo "=== Running validation sweep ==="
# Run sweep with extended threshold ranges
python scripts/local_val_sweep.py \
    --model-path models/from_modal/surface_unet3d_tuned_epoch15.pth \
    --config-path configs/experiments/surface_unet3d_tuned.yaml \
    --max-volumes 2 \
    --t-low-values "0.28,0.30,0.32,0.34,0.36,0.38" \
    --t-high-values "0.82,0.84,0.86,0.88" \
    --dust-values "50,100" \
    --no-tta

echo "=== Sweep completed ==="
EOF