#!/bin/bash
# Inspect predictions for failing volume 419698042

ssh jowny@schooner.oscer.ou.edu << 'EOF'
set -e

cd /scratch/jowny/vesuvius/vesuviusChallenge/vesuvius-competition
export PYTHONPATH=$PWD:$PWD/src
source venv/bin/activate

python << 'PYTHON'
import numpy as np
import torch
from pathlib import Path
import sys
sys.path.append('.')

from train import load_config
from src.inference.predict import VesuviusPredictor3D
from src.inference.create_submission import topo_postprocess
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Setup
model_path = "models/from_modal/surface_unet3d_tuned_epoch15.pth"
config_path = "configs/experiments/surface_unet3d_tuned.yaml"
vid = "419698042"

cfg = load_config(config_path)
predictor = VesuviusPredictor3D(
    model_path=model_path,
    device="cuda" if torch.cuda.is_available() else "cpu",
    roi_size=tuple(cfg["data"]["patch_size"]),
    overlap=0.5,
    class_index=1,
    config_path=config_path,
)

# Load volume
vol_path = Path("data/processed_3d/images/volume_" + vid + ".npz")
mask_path = Path("data/processed_3d/masks/mask_" + vid + ".npz")
volume = np.load(vol_path)["data"]
mask = np.load(mask_path)["data"]
gt = (mask == 1).astype(np.uint8)

print(f"\nVolume {vid} shape: {volume.shape}")
print(f"Volume stats - min: {volume.min():.3f}, max: {volume.max():.3f}, mean: {volume.mean():.3f}")

# Predict
print("\nRunning prediction...")
probs = predictor.predict_volume(volume)
print(f"Prob stats - min: {probs.min():.3f}, max: {probs.max():.3f}, mean: {probs.mean():.3f}")

# Apply post-processing
pred = topo_postprocess(probs, t_low=0.35, t_high=0.85, z_radius=1, xy_radius=0, dust_min_size=50)
print(f"Pred binary - sum: {pred.sum()}, frac: {pred.mean():.4f}")
print(f"GT binary - sum: {gt.sum()}, frac: {gt.mean():.4f}")

# Save a middle slice visualization
mid_z = volume.shape[0] // 2
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
axes[0].imshow(probs[mid_z], cmap='viridis', vmin=0, vmax=1)
axes[0].set_title(f'Probabilities (z={mid_z})')
axes[1].imshow(pred[mid_z], cmap='gray')
axes[1].set_title('Predictions')
axes[2].imshow(gt[mid_z], cmap='gray') 
axes[2].set_title('Ground Truth')
plt.tight_layout()
plt.savefig(f'volume_{vid}_slice.png')
print(f"Saved visualization to volume_{vid}_slice.png")
PYTHON
EOF