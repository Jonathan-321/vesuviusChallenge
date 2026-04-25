"""
Advanced loss functions for Vesuvius Challenge
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from monai.losses import DiceLoss, FocalLoss, DiceFocalLoss


class TopologyAwareLoss(nn.Module):
    """
    Topology-aware loss that emphasizes boundary accuracy
    """
    
    def __init__(self, alpha=0.5, beta=0.3, gamma=0.2):
        super().__init__()
        self.alpha = alpha  # Dice weight
        self.beta = beta    # Focal weight
        self.gamma = gamma  # Boundary weight
        
        self.dice_loss = DiceLoss(sigmoid=True, squared_pred=True)
        self.focal_loss = FocalLoss(gamma=2.0)
        self.bce_loss = nn.BCEWithLogitsLoss()
    
    def forward(self, pred, target):
        # Dice loss for overall segmentation
        dice = self.dice_loss(pred, target)
        
        # Focal loss for hard examples
        focal = self.focal_loss(pred, target)
        
        # Boundary-aware component
        boundary = self.compute_boundary_loss(pred, target)
        
        return self.alpha * dice + self.beta * focal + self.gamma * boundary
    
    def compute_boundary_loss(self, pred, target):
        """
        Compute boundary-weighted loss
        Higher weight on pixels near boundaries
        """
        # Compute boundaries using sobel
        target_boundaries = self.compute_boundaries(target)
        
        # Weight map (higher near boundaries)
        weights = 1.0 + 10.0 * target_boundaries
        
        # Weighted BCE
        bce = F.binary_cross_entropy_with_logits(pred, target, reduction='none')
        weighted_bce = (bce * weights).mean()
        
        return weighted_bce
    
    def compute_boundaries(self, mask):
        """Compute boundary map using gradients"""
        # Sobel filters
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], 
                               dtype=mask.dtype, device=mask.device).view(1, 1, 3, 3)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], 
                               dtype=mask.dtype, device=mask.device).view(1, 1, 3, 3)
        
        # Pad to maintain size
        mask_padded = F.pad(mask, (1, 1, 1, 1), mode='replicate')
        
        # Compute gradients
        grad_x = F.conv2d(mask_padded, sobel_x, padding=0)
        grad_y = F.conv2d(mask_padded, sobel_y, padding=0)
        
        # Magnitude
        boundaries = torch.sqrt(grad_x**2 + grad_y**2 + 1e-8)
        
        return boundaries


class CombinedLoss(nn.Module):
    """
    Combined Dice + Focal loss with class weighting
    """
    
    def __init__(self, dice_weight=0.5, focal_weight=0.5, pos_weight=30.0):
        super().__init__()
        self.dice_weight = dice_weight
        self.focal_weight = focal_weight
        
        self.dice_loss = DiceLoss(sigmoid=True, squared_pred=True)
        self.focal_loss = FocalLoss(gamma=2.0)
        self.pos_weight = torch.tensor([pos_weight])
    
    def forward(self, pred, target):
        dice = self.dice_loss(pred, target)
        focal = self.focal_loss(pred, target)
        
        return self.dice_weight * dice + self.focal_weight * focal


class DiceBCELoss(nn.Module):
    """
    Combination of Dice loss and BCE loss
    """
    
    def __init__(self, dice_weight=0.5, bce_weight=0.5, pos_weight=30.0):
        super().__init__()
        self.dice_weight = dice_weight
        self.bce_weight = bce_weight
        
        self.dice_loss = DiceLoss(sigmoid=True)
        self.bce_loss = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor([pos_weight])
        )
    
    def forward(self, pred, target):
        dice = self.dice_loss(pred, target)
        bce = self.bce_loss(pred, target)
        
        return self.dice_weight * dice + self.bce_weight * bce


class DiceCELoss(nn.Module):
    """
    Combination of Dice loss (softmax) and Cross-Entropy for multi-class
    """

    def __init__(self, dice_weight=0.5, ce_weight=0.5):
        super().__init__()
        self.dice_weight = dice_weight
        self.ce_weight = ce_weight
        self.dice_loss = DiceLoss(softmax=True, to_onehot_y=True)
        self.ce_loss = nn.CrossEntropyLoss()

    def forward(self, pred, target):
        if target.ndim == pred.ndim - 1:
            dice_target = target.unsqueeze(1)
            ce_target = target
        else:
            dice_target = target
            ce_target = target.squeeze(1) if target.shape[1] == 1 else target

        dice = self.dice_loss(pred, dice_target)
        ce = self.ce_loss(pred, ce_target.long())
        return self.dice_weight * dice + self.ce_weight * ce


class TverskyLoss(nn.Module):
    """
    Tversky loss - good for handling class imbalance
    alpha=0.7, beta=0.3 emphasizes false negatives (finding all ink)
    """
    
    def __init__(self, alpha=0.7, beta=0.3, smooth=1e-6):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth
    
    def forward(self, pred, target):
        pred = torch.sigmoid(pred)
        
        # True Positives, False Positives, False Negatives
        tp = (pred * target).sum(dim=(2, 3))
        fp = ((1 - target) * pred).sum(dim=(2, 3))
        fn = (target * (1 - pred)).sum(dim=(2, 3))
        
        tversky = (tp + self.smooth) / (tp + self.alpha * fp + self.beta * fn + self.smooth)
        
        return 1 - tversky.mean()


class FocalTverskyLoss(nn.Module):
    """
    Focal Tversky Loss - very effective for segmentation
    """
    
    def __init__(self, alpha=0.7, beta=0.3, gamma=1.5):
        super().__init__()
        self.tversky = TverskyLoss(alpha=alpha, beta=beta)
        self.gamma = gamma
    
    def forward(self, pred, target):
        tversky = self.tversky(pred, target)
        focal_tversky = torch.pow(tversky, self.gamma)
        return focal_tversky


def get_loss_function(config):
    """
    Factory function to create loss from config
    """
    loss_config = config.get('loss', {})
    loss_type = loss_config.get('type', 'combined').lower()
    
    if loss_type == 'dice':
        return DiceLoss(sigmoid=True, squared_pred=True)
    
    elif loss_type == 'focal':
        return FocalLoss(gamma=2.0)
    
    elif loss_type == 'combined':
        return CombinedLoss(
            dice_weight=loss_config.get('dice_weight', 0.5),
            focal_weight=loss_config.get('focal_weight', 0.5),
            pos_weight=loss_config.get('pos_weight', 30.0)
        )
    
    elif loss_type == 'dice_bce':
        return DiceBCELoss(
            dice_weight=loss_config.get('dice_weight', 0.5),
            bce_weight=loss_config.get('bce_weight', 0.5),
            pos_weight=loss_config.get('pos_weight', 30.0)
        )

    elif loss_type in ('dice_ce', 'dice_cross_entropy'):
        return DiceCELoss(
            dice_weight=loss_config.get('dice_weight', 0.5),
            ce_weight=loss_config.get('ce_weight', 0.5)
        )

    elif loss_type in ('cross_entropy', 'ce'):
        return nn.CrossEntropyLoss()
    
    elif loss_type == 'topology_aware':
        return TopologyAwareLoss(
            alpha=loss_config.get('dice_weight', 0.5),
            beta=loss_config.get('focal_weight', 0.3),
            gamma=loss_config.get('boundary_weight', 0.2)
        )
    
    elif loss_type == 'tversky':
        return TverskyLoss(
            alpha=loss_config.get('alpha', 0.7),
            beta=loss_config.get('beta', 0.3)
        )
    
    elif loss_type == 'focal_tversky':
        return FocalTverskyLoss(
            alpha=loss_config.get('alpha', 0.7),
            beta=loss_config.get('beta', 0.3),
            gamma=loss_config.get('gamma', 1.5)
        )
    
    elif loss_type == 'dice_focal':
        return DiceFocalLoss(
            sigmoid=True,
            lambda_dice=loss_config.get('dice_weight', 0.5),
            lambda_focal=loss_config.get('focal_weight', 0.5)
        )
    
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")
