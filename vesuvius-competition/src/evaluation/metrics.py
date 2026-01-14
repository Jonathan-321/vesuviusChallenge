"""
Evaluation metrics for Vesuvius Challenge
"""
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    jaccard_score, matthews_corrcoef, confusion_matrix,
    roc_auc_score, average_precision_score
)
from typing import Dict, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: Optional[np.ndarray] = None,
    threshold: float = 0.5
) -> Dict[str, float]:
    """
    Compute comprehensive metrics for binary segmentation
    
    Args:
        y_true: Ground truth binary mask
        y_pred: Predicted binary mask
        y_prob: Prediction probabilities (if available)
        threshold: Threshold used for binarization
        
    Returns:
        Dictionary of metrics
    """
    # Flatten arrays
    y_true_flat = y_true.flatten()
    y_pred_flat = y_pred.flatten()
    
    # Basic metrics
    metrics = {
        'accuracy': accuracy_score(y_true_flat, y_pred_flat),
        'precision': precision_score(y_true_flat, y_pred_flat, zero_division=0),
        'recall': recall_score(y_true_flat, y_pred_flat, zero_division=0),
        'f1_score': f1_score(y_true_flat, y_pred_flat, zero_division=0),
        'iou': jaccard_score(y_true_flat, y_pred_flat, zero_division=0),
        'threshold': threshold
    }
    
    # MCC with error handling
    try:
        metrics['mcc'] = matthews_corrcoef(y_true_flat, y_pred_flat)
    except:
        metrics['mcc'] = 0.0
    
    # Confusion matrix with handling for single class
    cm = confusion_matrix(y_true_flat, y_pred_flat, labels=[0, 1])
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
    else:
        # Handle case where only one class is present
        if y_true_flat.sum() == 0 and y_pred_flat.sum() == 0:
            # All negatives
            tn = len(y_true_flat)
            tp = fp = fn = 0
        elif y_true_flat.sum() == len(y_true_flat) and y_pred_flat.sum() == len(y_pred_flat):
            # All positives
            tp = len(y_true_flat)
            tn = fp = fn = 0
        else:
            # Mixed case, use what we have
            tn = ((1 - y_true_flat) * (1 - y_pred_flat)).sum()
            tp = (y_true_flat * y_pred_flat).sum()
            fp = ((1 - y_true_flat) * y_pred_flat).sum()
            fn = (y_true_flat * (1 - y_pred_flat)).sum()
    
    metrics.update({
        'true_positives': int(tp),
        'true_negatives': int(tn),
        'false_positives': int(fp),
        'false_negatives': int(fn),
        'specificity': tn / (tn + fp) if (tn + fp) > 0 else 0
    })
    
    # Probabilistic metrics if available
    if y_prob is not None:
        y_prob_flat = y_prob.flatten()
        # Only compute if we have both classes
        if len(np.unique(y_true_flat)) > 1:
            try:
                metrics['auc_roc'] = roc_auc_score(y_true_flat, y_prob_flat)
                metrics['avg_precision'] = average_precision_score(y_true_flat, y_prob_flat)
            except:
                metrics['auc_roc'] = 0.0
                metrics['avg_precision'] = 0.0
        else:
            metrics['auc_roc'] = 0.0
            metrics['avg_precision'] = 0.0
    
    # Dice coefficient (important for segmentation)
    dice = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0
    metrics['dice'] = dice
    
    return metrics


def compute_per_class_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray
) -> Dict[str, Dict[str, float]]:
    """
    Compute metrics for ink and non-ink regions separately
    """
    # Metrics for ink regions (positive class)
    ink_pixels = y_true == 1
    if ink_pixels.sum() > 0:
        ink_metrics = {
            'precision': precision_score(y_true[ink_pixels], y_pred[ink_pixels], zero_division=0),
            'recall': recall_score(y_true[ink_pixels], y_pred[ink_pixels], zero_division=0),
            'f1_score': f1_score(y_true[ink_pixels], y_pred[ink_pixels], zero_division=0),
            'pixel_count': int(ink_pixels.sum()),
            'pixel_percentage': float(ink_pixels.sum() / y_true.size * 100)
        }
    else:
        ink_metrics = {
            'precision': 0.0,
            'recall': 0.0,
            'f1_score': 0.0,
            'pixel_count': 0,
            'pixel_percentage': 0.0
        }
    
    # Metrics for non-ink regions (negative class)
    non_ink_pixels = y_true == 0
    if non_ink_pixels.sum() > 0:
        non_ink_metrics = {
            'precision': precision_score(1 - y_true[non_ink_pixels], 1 - y_pred[non_ink_pixels], zero_division=0),
            'recall': recall_score(1 - y_true[non_ink_pixels], 1 - y_pred[non_ink_pixels], zero_division=0),
            'f1_score': f1_score(1 - y_true[non_ink_pixels], 1 - y_pred[non_ink_pixels], zero_division=0),
            'pixel_count': int(non_ink_pixels.sum()),
            'pixel_percentage': float(non_ink_pixels.sum() / y_true.size * 100)
        }
    else:
        non_ink_metrics = {
            'precision': 0.0,
            'recall': 0.0,
            'f1_score': 0.0,
            'pixel_count': 0,
            'pixel_percentage': 0.0
        }
    
    return {
        'ink': ink_metrics,
        'non_ink': non_ink_metrics
    }


def find_optimal_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    metric: str = 'f1_score',
    thresholds: Optional[np.ndarray] = None
) -> Tuple[float, float]:
    """
    Find optimal threshold for binarization
    
    Args:
        y_true: Ground truth
        y_prob: Prediction probabilities
        metric: Metric to optimize ('f1_score', 'iou', 'mcc')
        thresholds: Thresholds to test
        
    Returns:
        Tuple of (optimal_threshold, best_metric_value)
    """
    if thresholds is None:
        thresholds = np.linspace(0.1, 0.9, 17)
    
    best_threshold = 0.5
    best_value = 0
    
    for thresh in thresholds:
        y_pred = (y_prob > thresh).astype(np.uint8)
        
        if metric == 'f1_score':
            value = f1_score(y_true.flatten(), y_pred.flatten(), zero_division=0)
        elif metric == 'iou':
            value = jaccard_score(y_true.flatten(), y_pred.flatten(), zero_division=0)
        elif metric == 'mcc':
            try:
                value = matthews_corrcoef(y_true.flatten(), y_pred.flatten())
            except:
                value = 0
        else:
            raise ValueError(f"Unknown metric: {metric}")
        
        if value > best_value:
            best_value = value
            best_threshold = thresh
    
    return best_threshold, best_value


def print_metrics(metrics: Dict[str, float], title: str = "Evaluation Metrics"):
    """Pretty print metrics"""
    print(f"\n{'='*60}")
    print(f"{title:^60}")
    print(f"{'='*60}")
    
    # Main metrics
    print(f"\nMain Metrics:")
    print(f"  Dice Score:     {metrics.get('dice', 0):.4f}")
    print(f"  IoU (Jaccard):  {metrics.get('iou', 0):.4f}")
    print(f"  F1 Score:       {metrics.get('f1_score', 0):.4f}")
    print(f"  MCC:            {metrics.get('mcc', 0):.4f}")
    
    # Accuracy metrics
    print(f"\nAccuracy Metrics:")
    print(f"  Accuracy:       {metrics.get('accuracy', 0):.4f}")
    print(f"  Precision:      {metrics.get('precision', 0):.4f}")
    print(f"  Recall:         {metrics.get('recall', 0):.4f}")
    print(f"  Specificity:    {metrics.get('specificity', 0):.4f}")
    
    # Probabilistic metrics
    if 'auc_roc' in metrics:
        print(f"\nProbabilistic Metrics:")
        print(f"  AUC-ROC:        {metrics['auc_roc']:.4f}")
        print(f"  Avg Precision:  {metrics['avg_precision']:.4f}")
    
    # Confusion matrix
    print(f"\nConfusion Matrix:")
    print(f"  True Positives:  {metrics.get('true_positives', 0):,}")
    print(f"  True Negatives:  {metrics.get('true_negatives', 0):,}")
    print(f"  False Positives: {metrics.get('false_positives', 0):,}")
    print(f"  False Negatives: {metrics.get('false_negatives', 0):,}")
    
    print(f"\nThreshold: {metrics.get('threshold', 0.5):.2f}")
    print(f"{'='*60}\n")