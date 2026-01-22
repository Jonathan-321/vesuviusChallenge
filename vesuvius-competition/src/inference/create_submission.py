"""
Create competition submission for Vesuvius Challenge
"""
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import zipfile
import logging
from typing import List, Dict, Optional, Tuple
import json
import tifffile
import scipy.ndimage as ndi
from skimage.morphology import remove_small_objects

from .predict import VesuviusPredictor, VesuviusPredictor3D

logger = logging.getLogger(__name__)


def validate_coverage(mask: np.ndarray, vol_id: str, min_cov: float = 0.01, max_cov: float = 0.30) -> float:
    """
    Quick sanity check on mask coverage to catch inverted predictions.
    """
    coverage = float(mask.mean())
    if coverage < min_cov or coverage > max_cov:
        logger.warning(
            f"[{vol_id}] coverage {coverage:.2%} outside expected range "
            f"({min_cov:.0%}-{max_cov:.0%})."
        )
    else:
        logger.info(f"[{vol_id}] coverage {coverage:.2%}")
    return coverage


def build_anisotropic_struct(z_radius: int, xy_radius: int):
    """Build anisotropic structuring element for 3D morphology."""
    z, r = z_radius, xy_radius

    if z == 0 and r == 0:
        return None

    if z == 0 and r > 0:
        size = 2 * r + 1
        struct = np.zeros((1, size, size), dtype=bool)
        cy, cx = r, r
        for dy in range(-r, r + 1):
            for dx in range(-r, r + 1):
                if dy * dy + dx * dx <= r * r:
                    struct[0, cy + dy, cx + dx] = True
        return struct

    if z > 0 and r == 0:
        struct = np.zeros((2 * z + 1, 1, 1), dtype=bool)
        struct[:, 0, 0] = True
        return struct

    depth = 2 * z + 1
    size = 2 * r + 1
    struct = np.zeros((depth, size, size), dtype=bool)
    cz, cy, cx = z, r, r
    for dz in range(-z, z + 1):
        for dy in range(-r, r + 1):
            for dx in range(-r, r + 1):
                if dy * dy + dx * dx <= r * r:
                    struct[cz + dz, cy + dy, cx + dx] = True
    return struct


def topo_postprocess(
    probs: np.ndarray,
    t_low: float = 0.5,
    t_high: float = 0.9,
    z_radius: int = 1,
    xy_radius: int = 0,
    dust_min_size: int = 100,
) -> np.ndarray:
    """3D postprocessing: hysteresis + anisotropic closing + dust removal."""
    strong = probs >= t_high
    weak = probs >= t_low

    if not strong.any():
        return np.zeros_like(probs, dtype=np.uint8)

    struct_hyst = ndi.generate_binary_structure(3, 3)
    mask = ndi.binary_propagation(strong, mask=weak, structure=struct_hyst)

    if not mask.any():
        return np.zeros_like(probs, dtype=np.uint8)

    if z_radius > 0 or xy_radius > 0:
        struct_close = build_anisotropic_struct(z_radius, xy_radius)
        if struct_close is not None:
            mask = ndi.binary_closing(mask, structure=struct_close)

    if dust_min_size > 0:
        mask = remove_small_objects(mask.astype(bool), min_size=dust_min_size)

    return mask.astype(np.uint8)


def create_submission_3d(
    model_path: str,
    test_dir: str,
    output_zip: str,
    test_csv: Optional[str] = None,
    config_path: Optional[str] = None,
    device: str = 'cuda',
    roi_size: Tuple[int, int, int] = (160, 160, 160),
    overlap: float = 0.5,
    class_index: int = 1,
    tta: bool = True,
    t_low: float = 0.5,
    t_high: float = 0.9,
    z_radius: int = 1,
    xy_radius: int = 0,
    dust_min_size: int = 100,
):
    """
    Create 3D submission ZIP with per-id .tif masks.
    """
    predictor = VesuviusPredictor3D(
        model_path=model_path,
        device=device,
        roi_size=roi_size,
        overlap=overlap,
        class_index=class_index,
        config_path=config_path,
    )

    test_path = Path(test_dir)
    if test_csv is None:
        test_csv = str((test_path.parent / "test.csv").resolve())

    test_df = pd.read_csv(test_csv)
    output_dir = Path(output_zip).parent / "submission_masks"
    output_dir.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(output_zip, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for image_id in tqdm(test_df["id"], desc="Predicting 3D volumes"):
            tif_path = test_path / f"{image_id}.tif"
            volume = tifffile.imread(str(tif_path))

            if tta:
                probs = predictor.predict_volume_tta(volume)
            else:
                probs = predictor.predict_volume(volume)

            mask = topo_postprocess(
                probs,
                t_low=t_low,
                t_high=t_high,
                z_radius=z_radius,
                xy_radius=xy_radius,
                dust_min_size=dust_min_size,
            )

            out_path = output_dir / f"{image_id}.tif"
            tifffile.imwrite(str(out_path), mask.astype(np.uint8))
            zf.write(out_path, arcname=f"{image_id}.tif")
            out_path.unlink()

    logger.info(f"3D submission saved to {output_zip}")


def create_submission(
    model_path: str,
    test_dir: str,
    output_path: str,
    threshold: float = 0.5,
    device: str = 'cuda',
    use_tta: bool = False,
    submission_format: str = 'kaggle'
):
    """
    Create a competition submission
    
    Args:
        model_path: Path to trained model
        test_dir: Directory with test volumes
        output_path: Path to save submission
        threshold: Prediction threshold
        device: Device for inference
        use_tta: Whether to use test-time augmentation
        submission_format: Format for submission ('kaggle' or 'custom')
    """
    # Initialize predictor
    predictor = VesuviusPredictor(model_path, device, use_tta)
    
    # Get test volumes
    test_path = Path(test_dir)
    volume_files = sorted(list(test_path.glob('volume_*.npz')))
    
    logger.info(f"Found {len(volume_files)} test volumes")
    
    # Store predictions
    predictions = {}
    
    # Process each volume
    for vol_file in tqdm(volume_files, desc="Processing volumes"):
        vol_id = vol_file.stem.replace('volume_', '')
        
        # Load volume
        with np.load(vol_file) as data:
            volume = data['data'] if 'data' in data else data['volume'] if 'volume' in data else data[list(data.keys())[0]]
        
        # Predict
        if use_tta:
            pred_prob = predictor.predict_with_tta(volume)
        else:
            pred_prob = predictor.predict_volume(volume)
        
        # Threshold
        pred_binary = (pred_prob > threshold).astype(np.uint8)
        validate_coverage(pred_binary, vol_id)
        
        predictions[vol_id] = {
            'binary': pred_binary,
            'probability': pred_prob
        }
        
        logger.info(f"Volume {vol_id}: {pred_binary.sum():,} ink pixels predicted")
    
    # Create submission based on format
    if submission_format == 'kaggle':
        create_kaggle_submission(predictions, output_path)
    else:
        create_custom_submission(predictions, output_path)
    
    logger.info(f"Submission saved to {output_path}")


def create_kaggle_submission(predictions: Dict, output_path: str):
    """
    Create Kaggle-style submission CSV
    
    Format: id,value
    Where id is "volumeId_y_x" and value is 0/1
    """
    submission_data = []
    
    for vol_id, pred_data in predictions.items():
        binary_pred = pred_data['binary']
        h, w = binary_pred.shape
        
        # Get coordinates of all predicted ink pixels
        ink_coords = np.argwhere(binary_pred == 1)
        
        for y, x in ink_coords:
            submission_data.append({
                'id': f"{vol_id}_{y}_{x}",
                'value': 1
            })
    
    # Create DataFrame
    submission_df = pd.DataFrame(submission_data)
    
    # If no ink predictions, create minimal submission
    if len(submission_df) == 0:
        submission_df = pd.DataFrame({
            'id': ['dummy_0_0'],
            'value': [0]
        })
    
    # Save CSV
    submission_df.to_csv(output_path, index=False)
    logger.info(f"Kaggle submission: {len(submission_df)} ink pixels")


def create_custom_submission(predictions: Dict, output_path: str):
    """
    Create custom submission format with full masks
    """
    output_dir = Path(output_path).parent / "submission_masks"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save each prediction
    metadata = {
        'format': 'custom',
        'volumes': {}
    }
    
    for vol_id, pred_data in predictions.items():
        # Save binary mask
        mask_path = output_dir / f"pred_{vol_id}.npz"
        np.savez_compressed(
            mask_path,
            binary=pred_data['binary'],
            probability=pred_data['probability']
        )
        
        # Update metadata
        metadata['volumes'][vol_id] = {
            'mask_file': str(mask_path.name),
            'ink_pixels': int(pred_data['binary'].sum()),
            'shape': pred_data['binary'].shape
        }
    
    # Save metadata
    metadata_path = output_dir / "metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    # Create zip file
    zip_path = Path(output_path).with_suffix('.zip')
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
        for file_path in output_dir.glob('*'):
            zf.write(file_path, file_path.name)
    
    logger.info(f"Custom submission saved to {zip_path}")


def ensemble_predictions(
    model_paths: List[str],
    test_dir: str,
    output_path: str,
    weights: Optional[List[float]] = None,
    threshold: float = 0.5,
    device: str = 'cuda',
    use_tta: bool = False
):
    """
    Create ensemble submission from multiple models
    
    Args:
        model_paths: List of model checkpoints
        test_dir: Test data directory
        output_path: Output path
        weights: Model weights for averaging (default: equal)
        threshold: Prediction threshold
        device: Device for inference
        use_tta: Use test-time augmentation
    """
    if weights is None:
        weights = [1.0 / len(model_paths)] * len(model_paths)
    
    assert len(weights) == len(model_paths), "Number of weights must match models"
    assert abs(sum(weights) - 1.0) < 1e-6, "Weights must sum to 1"
    
    # Get test volumes
    test_path = Path(test_dir)
    volume_files = sorted(list(test_path.glob('volume_*.npz')))
    
    logger.info(f"Ensemble of {len(model_paths)} models on {len(volume_files)} volumes")
    
    # Store ensemble predictions
    ensemble_preds = {}
    
    # Process each volume
    for vol_file in tqdm(volume_files, desc="Processing volumes"):
        vol_id = vol_file.stem.replace('volume_', '')
        
        # Load volume
        with np.load(vol_file) as data:
            volume = data['data'] if 'data' in data else data['volume'] if 'volume' in data else data[list(data.keys())[0]]
        
        # Get predictions from each model
        prob_sum = np.zeros((volume.shape[1], volume.shape[2]), dtype=np.float32)
        
        for model_path, weight in zip(model_paths, weights):
            predictor = VesuviusPredictor(model_path, device, use_tta)
            
            if use_tta:
                pred_prob = predictor.predict_with_tta(volume)
            else:
                pred_prob = predictor.predict_volume(volume)
            
            prob_sum += pred_prob * weight
        
        # Threshold ensemble prediction
        pred_binary = (prob_sum > threshold).astype(np.uint8)
        validate_coverage(pred_binary, vol_id)
        
        ensemble_preds[vol_id] = {
            'binary': pred_binary,
            'probability': prob_sum
        }
        
        logger.info(f"Volume {vol_id}: {pred_binary.sum():,} ink pixels (ensemble)")
    
    # Create submission
    create_kaggle_submission(ensemble_preds, output_path)
    
    logger.info(f"Ensemble submission saved to {output_path}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Create competition submission")
    parser.add_argument("--model", required=True, help="Model checkpoint path")
    parser.add_argument("--test", required=True, help="Test data directory")
    parser.add_argument("--output", required=True, help="Output submission path")
    parser.add_argument("--threshold", type=float, default=0.5, help="Prediction threshold")
    parser.add_argument("--device", default="cuda", help="Device")
    parser.add_argument("--tta", action="store_true", help="Use test-time augmentation")
    parser.add_argument(
        "--format",
        choices=['kaggle', 'custom', 'kaggle_3d'],
        default='kaggle',
        help="Submission format",
    )
    parser.add_argument("--config", default=None, help="Config path for 3D models")
    parser.add_argument("--test-csv", default=None, help="Path to test.csv for 3D submission")
    parser.add_argument("--roi", type=int, nargs=3, default=[160, 160, 160], help="ROI size for 3D sliding window")
    parser.add_argument("--overlap", type=float, default=0.5, help="Overlap for 3D sliding window")
    parser.add_argument("--class-index", type=int, default=1, help="Foreground class index for 3D")
    parser.add_argument("--t-low", type=float, default=0.5, help="Hysteresis low threshold")
    parser.add_argument("--t-high", type=float, default=0.9, help="Hysteresis high threshold")
    parser.add_argument("--z-radius", type=int, default=1, help="Anisotropic closing z-radius")
    parser.add_argument("--xy-radius", type=int, default=0, help="Anisotropic closing xy-radius")
    parser.add_argument("--dust-min", type=int, default=100, help="Remove small objects below this size")
    
    args = parser.parse_args()
    
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
    
    if args.format == 'kaggle_3d':
        create_submission_3d(
            model_path=args.model,
            test_dir=args.test,
            output_zip=args.output,
            test_csv=args.test_csv,
            config_path=args.config,
            device=args.device,
            roi_size=tuple(args.roi),
            overlap=args.overlap,
            class_index=args.class_index,
            tta=args.tta,
            t_low=args.t_low,
            t_high=args.t_high,
            z_radius=args.z_radius,
            xy_radius=args.xy_radius,
            dust_min_size=args.dust_min,
        )
    else:
        create_submission(
            args.model,
            args.test,
            args.output,
            args.threshold,
            args.device,
            args.tta,
            args.format
        )
