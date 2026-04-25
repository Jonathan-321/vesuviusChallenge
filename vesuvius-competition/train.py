#!/usr/bin/env python3
"""
Main training script for Vesuvius Challenge
Supports both local and Modal training

Usage:
    python train.py --config configs/experiments/baseline.yaml
    python train.py --config configs/experiments/attention_unet.yaml --resume checkpoint.pth
"""

import sys
from pathlib import Path

# Ensure repo root on path so `src` package is importable when run directly.
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.training.train import main, load_config, get_volume_ids  # noqa: E402

__all__ = ["main", "load_config", "get_volume_ids"]


if __name__ == '__main__':
    main()
