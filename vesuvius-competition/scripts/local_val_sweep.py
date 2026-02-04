#!/usr/bin/env python
"""
Lightweight local validation sweep (no Modal).

Requirements:
- processed volumes in `data/processed_3d/images/volume_*.npz`
  and masks in `data/processed_3d/masks/mask_*.npz`
- checkpoint path (e.g., models/from_modal/surface_unet3d_tuned_epoch15.pth)
- config yaml (e.g., configs/experiments/surface_unet3d_tuned.yaml)
"""
import argparse
import sys
from pathlib import Path
from typing import List, Sequence

import numpy as np
import torch

# Ensure project root on path
ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))

from src.training.train import load_config, get_volume_ids  # noqa: E402
from src.inference.predict import VesuviusPredictor3D  # noqa: E402
from src.inference.create_submission import topo_postprocess  # noqa: E402


def parse_floats(values: str) -> List[float]:
    return [float(x) for x in values.split(",") if x]


def parse_ints(values: str) -> List[int]:
    return [int(x) for x in values.split(",") if x]


def dice_score(pred: np.ndarray, target: np.ndarray, eps: float = 1e-6) -> float:
    """Simple binary Dice."""
    pred = pred.astype(np.bool_)
    target = target.astype(np.bool_)
    inter = np.logical_and(pred, target).sum()
    return (2 * inter + eps) / (pred.sum() + target.sum() + eps)


def choose_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def sweep(
    model_path: Path,
    config_path: Path,
    max_volumes: int,
    t_low_values: Sequence[float],
    t_high_values: Sequence[float],
    dust_values: Sequence[int],
    use_tta: bool,
):
    cfg = load_config(config_path)

    # Resolve processed dir relative to repo
    proc_dir = Path(cfg["data"]["processed_dir"])
    if not proc_dir.is_absolute():
        # Resolve relative to repo root (where configs/data/processed_3d lives)
        proc_dir = (ROOT / proc_dir).resolve()
    cfg["data"]["processed_dir"] = str(proc_dir)

    train_ids, val_ids = get_volume_ids(
        cfg["data"]["processed_dir"],
        cfg["validation"]["split_ratio"],
        cfg["validation"]["seed"],
    )
    val_ids = val_ids[:max_volumes]

    device = choose_device()
    predictor = VesuviusPredictor3D(
        model_path=str(model_path),
        device=device,
        roi_size=tuple(cfg["data"]["patch_size"]),
        overlap=0.5,
        class_index=1,
        config_path=str(config_path),
    )

    results = []
    for vid in val_ids:
        vol_path = proc_dir / "images" / f"volume_{vid}.npz"
        mask_path = proc_dir / "masks" / f"mask_{vid}.npz"
        volume = np.load(vol_path)["data"]
        mask = np.load(mask_path)["data"]
        gt = (mask == 1).astype(np.uint8)

        probs = (
            predictor.predict_volume_tta(volume)
            if use_tta
            else predictor.predict_volume(volume)
        )

        for tl in t_low_values:
            for th in t_high_values:
                for dust in dust_values:
                    pred = topo_postprocess(
                        probs,
                        t_low=tl,
                        t_high=th,
                        z_radius=1,
                        xy_radius=0,
                        dust_min_size=dust,
                    )
                    dice = dice_score(pred, gt)
                    results.append(
                        {
                            "vid": vid,
                            "t_low": tl,
                            "t_high": th,
                            "dust": dust,
                            "dice": dice,
                        }
                    )
                    print(
                        f"vid={vid} tl={tl:.2f} th={th:.2f} dust={dust:3d} "
                        f"dice={dice:.4f}"
                    )
    # Summaries
    best = sorted(results, key=lambda x: x["dice"], reverse=True)[:10]
    print("\nTop results:")
    for r in best:
        print(
            f"vid={r['vid']} tl={r['t_low']:.2f} th={r['t_high']:.2f} "
            f"dust={r['dust']:3d} dice={r['dice']:.4f}"
        )


def main():
    parser = argparse.ArgumentParser(description="Local val sweep (no Modal)")
    parser.add_argument(
        "--model-path",
        type=Path,
        default=Path("models/from_modal/surface_unet3d_tuned_epoch15.pth"),
    )
    parser.add_argument(
        "--config-path",
        type=Path,
        default=Path("configs/experiments/surface_unet3d_tuned.yaml"),
    )
    parser.add_argument("--max-volumes", type=int, default=2)
    parser.add_argument(
        "--t-low-values", type=str, default="0.3,0.35,0.4",
        help="Comma-separated list",
    )
    parser.add_argument(
        "--t-high-values", type=str, default="0.85,0.9",
        help="Comma-separated list",
    )
    parser.add_argument(
        "--dust-values", type=str, default="50,100",
        help="Comma-separated list",
    )
    parser.add_argument("--no-tta", action="store_true", help="Disable TTA")
    args = parser.parse_args()

    sweep(
        model_path=args.model_path,
        config_path=args.config_path,
        max_volumes=args.max_volumes,
        t_low_values=parse_floats(args.t_low_values),
        t_high_values=parse_floats(args.t_high_values),
        dust_values=parse_ints(args.dust_values),
        use_tta=not args.no_tta,
    )


if __name__ == "__main__":
    main()
