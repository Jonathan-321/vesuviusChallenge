"""
Inference prediction module for Vesuvius Challenge
"""
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from tqdm import tqdm
import segmentation_models_pytorch as smp
from src.models import get_model
from monai.inferers import SlidingWindowInferer
from typing import Dict, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


def _normalize_volume(volume: np.ndarray) -> np.ndarray:
    """Normalize volume to [0, 1] based on dtype."""
    if np.issubdtype(volume.dtype, np.floating):
        return volume.astype(np.float32)
    max_value = np.iinfo(volume.dtype).max
    return volume.astype(np.float32) / float(max_value)


class VesuviusPredictor:
    """Predictor for Vesuvius ink detection"""
    
    def __init__(
        self,
        model_path: str,
        device: str = 'cuda',
        use_tta: bool = False,
        batch_size: int = 1
    ):
        """
        Initialize predictor
        
        Args:
            model_path: Path to trained model checkpoint
            device: Device to use for inference
            use_tta: Whether to use test-time augmentation
            batch_size: Batch size for inference
        """
        # Resolve device with MPS fallback
        if device == 'cuda' and torch.cuda.is_available():
            self.device = 'cuda'
        elif device in ('mps', 'cuda') and torch.backends.mps.is_available():
            self.device = 'mps'
        else:
            self.device = 'cpu'
        self.use_tta = use_tta
        self.batch_size = batch_size
        self.model_config = None
        self.data_config = None
        self.patch_size = (32, 128, 128)
        self.in_channels = 32
        
        # Load model
        self.model = self._load_model(model_path)
        self.model.eval()
        
    def _load_model(self, model_path: str) -> nn.Module:
        """Load trained model from checkpoint and reconstruct the architecture"""
        checkpoint = torch.load(model_path, map_location=self.device)

        state_dict = checkpoint
        if isinstance(checkpoint, dict):
            self.model_config = checkpoint.get('config', {}).get('model', None)
            self.data_config = checkpoint.get('config', {}).get('data', None)
            state_dict = checkpoint.get('model_state_dict') or checkpoint.get('state_dict') or checkpoint

        if self.model_config:
            # Build model from stored config for correct encoder/in_channels
            model = get_model(checkpoint['config'])
            self.in_channels = self.model_config.get('in_channels', self.in_channels)
        else:
            # Fallback to baseline UNet if config is missing
            model = smp.Unet(
                encoder_name="resnet34",
                encoder_weights=None,
                in_channels=self.in_channels,
                classes=1,
                activation=None
            )

        # Derive default patch size from config if available
        if self.data_config and self.data_config.get('patch_size'):
            try:
                depth, height, width = self.data_config['patch_size']
                self.patch_size = (int(depth), int(height), int(width))
                self.in_channels = self.patch_size[0]
            except Exception:
                logger.warning("Invalid patch_size in checkpoint config; using fallback.")

        # Load weights with a bit of tolerance to avoid minor key mismatches
        load_result = model.load_state_dict(state_dict, strict=False)
        if hasattr(load_result, 'missing_keys') and load_result.missing_keys:
            logger.warning(f"Missing keys when loading checkpoint: {load_result.missing_keys}")
        if hasattr(load_result, 'unexpected_keys') and load_result.unexpected_keys:
            logger.warning(f"Unexpected keys when loading checkpoint: {load_result.unexpected_keys}")
        model = model.to(self.device)
        
        logger.info(f"Loaded model from {model_path}")
        logger.info(f"Model trained for {checkpoint.get('epoch', 'unknown')} epochs")
        logger.info(f"Best val loss: {checkpoint.get('best_val_loss', 'unknown')}")
        
        return model
    
    def predict_volume(
        self,
        volume: np.ndarray,
        stride: int = 64,
        patch_size: Optional[Tuple[int, int, int]] = None
    ) -> np.ndarray:
        """
        Predict on a full volume using sliding window
        
        Args:
            volume: Input volume of shape (D, H, W)
            stride: Stride for sliding window
            patch_size: Size of patches (depth, height, width)
            
        Returns:
            Predicted mask of shape (H, W)
        """
        patch_size = patch_size or self.patch_size
        # Ensure depth matches model channels
        if patch_size[0] != self.in_channels:
            logger.warning(
                f"Patch depth {patch_size[0]} != model in_channels {self.in_channels}; "
                f"using model depth."
            )
            patch_size = (self.in_channels, patch_size[1], patch_size[2])

        d, h, w = volume.shape
        pd, ph, pw = patch_size
        
        # Normalize volume
        volume = _normalize_volume(volume)
        
        # Initialize output
        output = np.zeros((h, w), dtype=np.float32)
        count = np.zeros((h, w), dtype=np.float32)
        
        # Pad volume if needed
        pad_d = max(0, pd - d)
        pad_h = max(0, stride - (h - ph) % stride) if h > ph else ph - h
        pad_w = max(0, stride - (w - pw) % stride) if w > pw else pw - w
        
        if pad_d > 0 or pad_h > 0 or pad_w > 0:
            volume = np.pad(volume, ((0, pad_d), (0, pad_h), (0, pad_w)), mode='constant')
            d, h_pad, w_pad = volume.shape
        else:
            h_pad, w_pad = h, w
        
        # Sliding window prediction
        positions = []
        for z in range(0, d - pd + 1, stride):
            for y in range(0, h_pad - ph + 1, stride):
                for x in range(0, w_pad - pw + 1, stride):
                    positions.append((z, y, x))
        
        # Process in batches
        with torch.no_grad():
            for i in tqdm(range(0, len(positions), self.batch_size), desc="Predicting"):
                batch_positions = positions[i:i + self.batch_size]
                batch_patches = []
                
                for z, y, x in batch_positions:
                    patch = volume[z:z+pd, y:y+ph, x:x+pw]
                    batch_patches.append(patch)
                
                # Convert to tensor
                batch_tensor = torch.from_numpy(np.array(batch_patches)).unsqueeze(1).to(self.device)
                
                # Reshape for 2D model (use depth as channels)
                b, c, d, h_p, w_p = batch_tensor.shape
                batch_2d = batch_tensor.squeeze(1).permute(0, 1, 2, 3)  # (b, d, h, w)
                
                # Predict
                predictions = self.model(batch_2d)
                predictions = torch.sigmoid(predictions)
                predictions = predictions.cpu().numpy()
                
                # Apply predictions to output
                for j, (z, y, x) in enumerate(batch_positions):
                    pred = predictions[j, 0]  # Remove channel dimension
                    # Handle edge cases where patch extends beyond padded volume
                    y_end = min(y + ph, output.shape[0])
                    x_end = min(x + pw, output.shape[1])
                    pred_h = y_end - y
                    pred_w = x_end - x
                    output[y:y_end, x:x_end] += pred[:pred_h, :pred_w]
                    count[y:y_end, x:x_end] += 1
        
        # Average overlapping predictions
        output = output / np.maximum(count, 1)
        
        # Crop to original size
        output = output[:h, :w]
        
        return output
    
    def predict_with_tta(
        self,
        volume: np.ndarray,
        stride: int = 64,
        patch_size: Optional[Tuple[int, int, int]] = None
    ) -> np.ndarray:
        """
        Predict with test-time augmentation
        
        Args:
            volume: Input volume
            stride: Stride for sliding window
            patch_size: Patch size
            
        Returns:
            Averaged predictions
        """
        predictions = []
        
        # Original
        pred = self.predict_volume(volume, stride, patch_size)
        predictions.append(pred)
        
        # Horizontal flip
        volume_flip = volume[:, :, ::-1]
        pred_flip = self.predict_volume(volume_flip, stride, patch_size)
        pred_flip = pred_flip[:, ::-1]
        predictions.append(pred_flip)
        
        # Vertical flip
        volume_flip = volume[:, ::-1, :]
        pred_flip = self.predict_volume(volume_flip, stride, patch_size)
        pred_flip = pred_flip[::-1, :]
        predictions.append(pred_flip)
        
        # Average predictions
        return np.mean(predictions, axis=0)
    
    def predict_file(
        self,
        volume_path: str,
        output_path: Optional[str] = None,
        threshold: float = 0.5
    ) -> np.ndarray:
        """
        Predict on a volume file
        
        Args:
            volume_path: Path to volume .npz file
            output_path: Optional path to save predictions
            threshold: Threshold for binary prediction
            
        Returns:
            Binary predictions
        """
        # Load volume
        with np.load(volume_path) as data:
            volume = data['data'] if 'data' in data else data['volume'] if 'volume' in data else data[list(data.keys())[0]]
        
        logger.info(f"Loaded volume from {volume_path}, shape: {volume.shape}")
        
        # Predict
        if self.use_tta:
            predictions = self.predict_with_tta(volume)
        else:
            predictions = self.predict_volume(volume)
        
        # Threshold
        binary_pred = (predictions > threshold).astype(np.uint8)
        
        # Save if requested
        if output_path:
            np.savez_compressed(output_path, prediction=binary_pred, probability=predictions)
            logger.info(f"Saved predictions to {output_path}")
        
        return binary_pred


class VesuviusPredictor3D:
    """Predictor for 3D surface detection"""

    def __init__(
        self,
        model_path: str,
        device: str = 'cuda',
        roi_size: Tuple[int, int, int] = (160, 160, 160),
        overlap: float = 0.5,
        sw_batch_size: int = 1,
        class_index: int = 1,
        config_path: Optional[str] = None,
    ):
        if device == 'cuda' and torch.cuda.is_available():
            self.device = 'cuda'
        elif device in ('mps', 'cuda') and torch.backends.mps.is_available():
            self.device = 'mps'
        else:
            self.device = 'cpu'

        self.roi_size = roi_size
        self.overlap = overlap
        self.sw_batch_size = sw_batch_size
        self.class_index = class_index
        self.model_config = None
        self.num_classes = None

        self.model = self._load_model(model_path, config_path)
        self.model.eval()

    def _load_model(self, model_path: str, config_path: Optional[str]) -> nn.Module:
        """Load 3D model from checkpoint/config."""
        checkpoint = torch.load(model_path, map_location=self.device)
        config = None
        state_dict = checkpoint
        if isinstance(checkpoint, dict):
            config = checkpoint.get('config')
            state_dict = checkpoint.get('model_state_dict') or checkpoint.get('state_dict') or checkpoint

        if config is None and config_path:
            from train import load_config
            config = load_config(config_path)

        if config is None:
            raise ValueError("3D predictor requires config (checkpoint config or --config).")

        self.model_config = config.get('model', {})
        self.num_classes = int(self.model_config.get('out_channels', 1))
        if self.num_classes <= 1:
            self.class_index = 0

        model = get_model(config)
        model.load_state_dict(state_dict, strict=False)
        model = model.to(self.device)
        return model

    def _predict_logits(self, volume: np.ndarray) -> torch.Tensor:
        """Run sliding window inference and return logits."""
        volume = _normalize_volume(volume)
        volume_tensor = torch.from_numpy(volume).unsqueeze(0).unsqueeze(0).to(self.device)

        inferer = SlidingWindowInferer(
            roi_size=self.roi_size,
            overlap=self.overlap,
            mode="gaussian",
            sw_batch_size=self.sw_batch_size,
        )
        with torch.no_grad():
            logits = inferer(volume_tensor, self.model)
        return logits

    def predict_volume(self, volume: np.ndarray) -> np.ndarray:
        """Predict probability map for class_index (D, H, W)."""
        logits = self._predict_logits(volume)
        if self.num_classes and self.num_classes > 1:
            probs = torch.softmax(logits, dim=1)
            prob_map = probs[0, self.class_index]
        else:
            probs = torch.sigmoid(logits)
            prob_map = probs[0, 0]
        return prob_map.cpu().numpy()

    def predict_volume_tta(self, volume: np.ndarray) -> np.ndarray:
        """Predict with 4x rotation TTA in HW plane."""
        probs = []
        for k in range(4):
            vol_rot = np.rot90(volume, k=-k, axes=(1, 2))
            prob_rot = self.predict_volume(vol_rot)
            prob_unrot = np.rot90(prob_rot, k=k, axes=(1, 2))
            probs.append(prob_unrot)
        return np.mean(probs, axis=0)


def predict_directory(
    model_path: str,
    input_dir: str,
    output_dir: str,
    device: str = 'cuda',
    use_tta: bool = False,
    threshold: float = 0.5
):
    """
    Predict on all volumes in a directory
    
    Args:
        model_path: Path to model checkpoint
        input_dir: Directory with volume files
        output_dir: Directory to save predictions
        device: Device for inference
        use_tta: Whether to use TTA
        threshold: Prediction threshold
    """
    predictor = VesuviusPredictor(model_path, device, use_tta)
    
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Find all volume files
    volume_files = sorted(list(input_path.glob('volume_*.npz')))
    logger.info(f"Found {len(volume_files)} volumes to process")
    
    for vol_file in volume_files:
        vol_id = vol_file.stem.replace('volume_', '')
        output_file = output_path / f"pred_{vol_id}.npz"
        
        logger.info(f"Processing {vol_file.name}...")
        predictor.predict_file(str(vol_file), str(output_file), threshold)
    
    logger.info("Prediction complete!")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Vesuvius ink detection inference")
    parser.add_argument("--model", required=True, help="Path to model checkpoint")
    parser.add_argument("--input", required=True, help="Input directory or file")
    parser.add_argument("--output", required=True, help="Output directory")
    parser.add_argument("--device", default="cuda", help="Device (cuda/cpu)")
    parser.add_argument("--tta", action="store_true", help="Use test-time augmentation")
    parser.add_argument("--threshold", type=float, default=0.5, help="Prediction threshold")
    
    args = parser.parse_args()
    
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
    
    if Path(args.input).is_file():
        predictor = VesuviusPredictor(args.model, args.device, args.tta)
        predictor.predict_file(args.input, args.output, args.threshold)
    else:
        predict_directory(args.model, args.input, args.output, args.device, args.tta, args.threshold)
