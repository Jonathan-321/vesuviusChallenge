"""
Create competition submission for Vesuvius Challenge
"""
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import zipfile
import logging
from typing import List, Dict, Optional
import json

from .predict import VesuviusPredictor

logger = logging.getLogger(__name__)


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
    parser.add_argument("--format", choices=['kaggle', 'custom'], default='kaggle', help="Submission format")
    
    args = parser.parse_args()
    
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
    
    create_submission(
        args.model,
        args.test,
        args.output,
        args.threshold,
        args.device,
        args.tta,
        args.format
    )