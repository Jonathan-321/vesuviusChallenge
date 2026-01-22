"""
Evaluate trained model on validation data
"""
import sys
sys.path.append('src')

import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
import logging
from typing import List, Dict, Tuple

from inference.predict import VesuviusPredictor
from evaluation.metrics import (
    compute_metrics, compute_per_class_metrics, 
    find_optimal_threshold, print_metrics
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)


def evaluate_on_validation(
    model_path: str,
    data_dir: str,
    val_split: float = 0.2,
    device: str = 'cuda',
    num_samples: int = 10
) -> Dict[str, float]:
    """
    Evaluate model on validation data
    
    Args:
        model_path: Path to model checkpoint
        data_dir: Directory with processed data
        val_split: Validation split ratio
        device: Device for inference
        num_samples: Number of samples to evaluate per volume
        
    Returns:
        Aggregated metrics
    """
    # Initialize predictor
    predictor = VesuviusPredictor(model_path, device, use_tta=False, batch_size=4)
    
    # Get validation volumes
    data_path = Path(data_dir)
    volume_root = data_path / "images" if (data_path / "images").exists() else data_path
    mask_root = data_path / "masks" if (data_path / "masks").exists() else data_path
    volume_files = sorted(list(volume_root.glob('volume_*.npz')))
    
    # Use same split as training
    np.random.seed(42)
    shuffled_files = np.random.permutation(volume_files)
    n_val = max(1, int(len(shuffled_files) * val_split))
    val_files = shuffled_files[:n_val]
    
    logger.info(f"Evaluating on {len(val_files)} validation volumes")
    
    all_metrics = []
    all_y_true = []
    all_y_pred = []
    all_y_prob = []
    
    # Process each validation volume
    for vol_file in tqdm(val_files, desc="Evaluating"):
        vol_id = vol_file.stem.replace('volume_', '')
        mask_file = mask_root / f"mask_{vol_id}.npz"
        
        if not mask_file.exists():
            logger.warning(f"Mask not found for {vol_id}")
            continue
        
        # Load data
        with np.load(vol_file) as vdata:
            volume = vdata['data'] if 'data' in vdata else vdata['volume'] if 'volume' in vdata else vdata[list(vdata.keys())[0]]
        
        with np.load(mask_file) as mdata:
            mask = mdata['data'] if 'data' in mdata else mdata['mask'] if 'mask' in mdata else mdata[list(mdata.keys())[0]]
        
        # Ensure correct shapes
        if volume.ndim != 3 or mask.ndim != 2 or mask.shape != volume.shape[1:]:
            logger.warning(f"Skipping {vol_id} due to shape mismatch")
            continue
        
        # Sample random patches for faster evaluation
        d, h, w = volume.shape
        patch_size = (predictor.in_channels, 128, 128)
        pd, ph, pw = patch_size
        if d < pd:
            logger.warning(f"Skipping {vol_id} due to insufficient depth ({d} < {pd})")
            continue
        
        for _ in range(num_samples):
            # Random position
            z = np.random.randint(0, max(1, d - pd + 1))
            y = np.random.randint(0, max(1, h - ph + 1))
            x = np.random.randint(0, max(1, w - pw + 1))
            
            # Extract patch
            vol_patch = volume[z:z+pd, y:y+ph, x:x+pw]
            mask_patch = mask[y:y+ph, x:x+pw]
            
            # Normalize
            vol_patch = vol_patch.astype(np.float32) / 65535.0
            mask_patch = (mask_patch > 127).astype(np.uint8)
            
            # Predict
            with torch.no_grad():
                vol_tensor = torch.from_numpy(vol_patch).unsqueeze(0).unsqueeze(0).to(device)
                # Reshape for 2D model
                vol_2d = vol_tensor.squeeze(1).permute(0, 1, 2, 3)
                
                pred = predictor.model(vol_2d)
                pred_prob = torch.sigmoid(pred).cpu().numpy()[0, 0]
            
            # Threshold
            pred_binary = (pred_prob > 0.5).astype(np.uint8)
            
            # Store for aggregation
            all_y_true.append(mask_patch.flatten())
            all_y_pred.append(pred_binary.flatten())
            all_y_prob.append(pred_prob.flatten())
    
    # Aggregate all predictions
    y_true_all = np.concatenate(all_y_true)
    y_pred_all = np.concatenate(all_y_pred)
    y_prob_all = np.concatenate(all_y_prob)
    
    # Compute metrics at default threshold
    metrics = compute_metrics(y_true_all, y_pred_all, y_prob_all, threshold=0.5)
    
    # Find optimal threshold
    logger.info("Finding optimal threshold...")
    opt_thresh_f1, best_f1 = find_optimal_threshold(y_true_all, y_prob_all, metric='f1_score')
    opt_thresh_iou, best_iou = find_optimal_threshold(y_true_all, y_prob_all, metric='iou')
    
    # Compute metrics at optimal threshold
    y_pred_opt = (y_prob_all > opt_thresh_f1).astype(np.uint8)
    metrics_opt = compute_metrics(y_true_all, y_pred_opt, y_prob_all, threshold=opt_thresh_f1)
    
    # Per-class metrics
    class_metrics = compute_per_class_metrics(y_true_all, y_pred_all)
    
    # Print results
    print_metrics(metrics, "Validation Metrics (Threshold=0.5)")
    print_metrics(metrics_opt, f"Validation Metrics (Optimal Threshold={opt_thresh_f1:.2f})")
    
    print(f"\n{'='*60}")
    print("Per-Class Analysis")
    print(f"{'='*60}")
    print(f"\nInk Regions:")
    print(f"  Pixel Count:     {class_metrics['ink']['pixel_count']:,}")
    print(f"  Pixel %:         {class_metrics['ink']['pixel_percentage']:.2f}%")
    print(f"  Precision:       {class_metrics['ink']['precision']:.4f}")
    print(f"  Recall:          {class_metrics['ink']['recall']:.4f}")
    print(f"  F1 Score:        {class_metrics['ink']['f1_score']:.4f}")
    
    print(f"\nNon-Ink Regions:")
    print(f"  Pixel Count:     {class_metrics['non_ink']['pixel_count']:,}")
    print(f"  Pixel %:         {class_metrics['non_ink']['pixel_percentage']:.2f}%")
    print(f"  Precision:       {class_metrics['non_ink']['precision']:.4f}")
    print(f"  Recall:          {class_metrics['non_ink']['recall']:.4f}")
    print(f"  F1 Score:        {class_metrics['non_ink']['f1_score']:.4f}")
    
    print(f"\n{'='*60}")
    print("Optimal Thresholds")
    print(f"{'='*60}")
    print(f"  Best F1 Score:   {best_f1:.4f} @ threshold {opt_thresh_f1:.2f}")
    print(f"  Best IoU:        {best_iou:.4f} @ threshold {opt_thresh_iou:.2f}")
    
    return metrics_opt


def main():
    """Run evaluation"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate Vesuvius model")
    parser.add_argument("--model", default="./models/from_modal/best_model.pth", help="Model path")
    parser.add_argument("--data", default="./data/processed", help="Data directory")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu", help="Device")
    parser.add_argument("--samples", type=int, default=20, help="Samples per volume")
    
    args = parser.parse_args()
    
    logger.info(f"Evaluating model: {args.model}")
    logger.info(f"Using device: {args.device}")
    
    metrics = evaluate_on_validation(
        args.model,
        args.data,
        device=args.device,
        num_samples=args.samples
    )
    
    # Save metrics
    import json
    metrics_path = Path(args.model).parent / "validation_metrics.json"
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"Saved metrics to {metrics_path}")


if __name__ == "__main__":
    main()
