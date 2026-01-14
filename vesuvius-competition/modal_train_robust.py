"""
Robust Modal training with resumption capability
"""
import modal
import signal
import json
from datetime import datetime
from pathlib import Path

app = modal.App("vesuvius-training-robust")
volume = modal.Volume.from_name("vesuvius-data", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install([
        "torch>=2.1.0",
        "torchvision>=0.16.0",
        "numpy>=1.24.0",
        "pillow>=10.0.0",
        "tqdm>=4.65.0",
        "pyyaml>=6.0",
        "albumentations>=1.3.0",
        "segmentation-models-pytorch>=0.3.3",
        "pandas>=2.0.0",
        "scikit-learn>=1.3.0",
        "matplotlib>=3.7.0",
    ])
)


class TrainingLogger:
    """Logger that writes to both console and volume"""
    def __init__(self, log_path):
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        
    def log(self, message):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_line = f"[{timestamp}] {message}"
        print(log_line)
        
        # Append to log file
        with open(self.log_path, 'a') as f:
            f.write(log_line + '\n')


@app.function(
    image=image,
    gpu="A100-80GB",
    volumes={"/mnt": volume},
    timeout=60 * 60 * 24,  # 24 hours
)
def train_with_resume(
    config_name: str = "baseline",
    max_epochs: int = 20,
    resume: bool = True
):
    """
    Train with automatic resumption and robust logging
    """
    import os
    import sys
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, Dataset
    import numpy as np
    from tqdm import tqdm
    import segmentation_models_pytorch as smp
    
    # Setup logger
    logger = TrainingLogger("/mnt/logs/training.log")
    logger.log(f"Starting training - Config: {config_name}, Max epochs: {max_epochs}")
    
    # Setup signal handler for graceful shutdown
    def signal_handler(signum, frame):
        logger.log(f"Received signal {signum} - Saving checkpoint and exiting gracefully")
        # The checkpoint saving will happen in the except block
        raise KeyboardInterrupt("Training interrupted by signal")
    
    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)
    
    try:
        logger.log(f"GPU: {torch.cuda.get_device_name(0)}")
        logger.log(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB")
        
        # Simple dataset (same as before)
        class VesuviusDataset(Dataset):
            def __init__(self, volume_ids, data_dir, samples_per_volume=100):
                self.data_dir = Path(data_dir)
                self.samples_per_volume = samples_per_volume
                self.volumes = []
                self.masks = []
                
                for vid in volume_ids:
                    vol_path = self.data_dir / f"volume_{vid}.npz"
                    mask_path = self.data_dir / f"mask_{vid}.npz"
                    if vol_path.exists() and mask_path.exists():
                        try:
                            with np.load(vol_path) as vol_data:
                                vol = vol_data['data'] if 'data' in vol_data else vol_data['volume'] if 'volume' in vol_data else vol_data[list(vol_data.keys())[0]]
                                vol = vol.copy()
                            
                            with np.load(mask_path) as mask_data:
                                mask = mask_data['data'] if 'data' in mask_data else mask_data['mask'] if 'mask' in mask_data else mask_data[list(mask_data.keys())[0]]
                                mask = mask.copy()
                                
                            self.volumes.append(vol)
                            self.masks.append(mask)
                        except Exception as e:
                            logger.log(f"Error loading {vid}: {e}")
                
                logger.log(f"Loaded {len(self.volumes)} volumes")
                
            def __len__(self):
                return len(self.volumes) * self.samples_per_volume
            
            def __getitem__(self, idx):
                vol_idx = idx % len(self.volumes)
                volume = self.volumes[vol_idx]
                mask = self.masks[vol_idx]
                
                # Random crop
                d, h, w = volume.shape
                pd, ph, pw = 32, 128, 128
                
                # Ensure we have enough dimensions
                if d < pd or h < ph or w < pw:
                    pad_d = max(0, pd - d)
                    pad_h = max(0, ph - h) 
                    pad_w = max(0, pw - w)
                    volume = np.pad(volume, ((0, pad_d), (0, pad_h), (0, pad_w)), mode='constant')
                    mask = np.pad(mask, ((0, pad_h), (0, pad_w)), mode='constant')
                    d, h, w = volume.shape
                    
                z = np.random.randint(0, max(1, d - pd + 1))
                y = np.random.randint(0, max(1, h - ph + 1))
                x = np.random.randint(0, max(1, w - pw + 1))
                
                vol_patch = volume[z:z+pd, y:y+ph, x:x+pw].astype(np.float32) / 65535.0
                mask_patch = mask[y:y+ph, x:x+pw].astype(np.float32) / 255.0
                
                # Convert to tensors
                vol_tensor = torch.from_numpy(vol_patch).unsqueeze(0)
                mask_tensor = torch.from_numpy(mask_patch).unsqueeze(0).unsqueeze(0)
                
                return vol_tensor, mask_tensor
        
        # Get volume IDs
        data_path = Path("/mnt/processed")
        volume_files = sorted(list(data_path.glob('volume_*.npz')))
        
        # Only use volumes with correct shapes
        valid_volume_ids = []
        for vol_file in volume_files:
            vol_id = vol_file.stem.replace('volume_', '')
            mask_file = data_path / f"mask_{vol_id}.npz"
            
            if mask_file.exists():
                try:
                    with np.load(vol_file) as vdata:
                        vshape = vdata[list(vdata.keys())[0]].shape
                    if vshape == (65, 320, 320):
                        valid_volume_ids.append(vol_id)
                except:
                    pass
        
        volume_ids = valid_volume_ids
        logger.log(f"Found {len(volume_ids)} valid volumes")
        
        # Split data
        np.random.seed(42)
        shuffled_ids = np.random.permutation(volume_ids)
        n_val = max(1, int(len(shuffled_ids) * 0.2))
        val_ids = shuffled_ids[:n_val].tolist()
        train_ids = shuffled_ids[n_val:].tolist()
        
        logger.log(f"Train: {len(train_ids)}, Val: {len(val_ids)}")
        
        # Create datasets and loaders
        train_dataset = VesuviusDataset(train_ids, data_path, samples_per_volume=200)
        val_dataset = VesuviusDataset(val_ids, data_path, samples_per_volume=50)
        
        train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True, num_workers=0)
        val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False, num_workers=0)
        
        # Create model
        model = smp.Unet(
            encoder_name="resnet34",
            encoder_weights=None,
            in_channels=32,
            classes=1,
            activation=None
        ).cuda()
        
        total_params = sum(p.numel() for p in model.parameters())
        logger.log(f"Model: UNet-ResNet34, Parameters: {total_params:,}")
        
        # Loss and optimizer
        criterion = smp.losses.DiceLoss(mode='binary', from_logits=True)
        optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs)
        
        # Check for existing checkpoint
        start_epoch = 0
        best_val_loss = float('inf')
        checkpoint_path = Path("/mnt/models/baseline_best.pth")
        
        if resume and checkpoint_path.exists():
            logger.log(f"Loading checkpoint from {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint.get('epoch', 0) + 1
            best_val_loss = checkpoint.get('best_val_loss', float('inf'))
            logger.log(f"Resuming from epoch {start_epoch} with best val loss {best_val_loss:.6f}")
        
        # Training loop with progress tracking
        for epoch in range(start_epoch, max_epochs):
            epoch_start_time = datetime.now()
            
            # Training
            model.train()
            train_loss = 0
            progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{max_epochs}")
            
            for batch_idx, (inputs, targets) in enumerate(progress_bar):
                inputs, targets = inputs.cuda(), targets.cuda()
                
                # Reshape for 2D model
                b, c, d, h, w = inputs.shape
                inputs_2d = inputs.squeeze(1).permute(0, 1, 2, 3)
                
                outputs = model(inputs_2d)
                targets = targets.squeeze(2)
                
                loss = criterion(outputs, targets)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                
                # Update progress bar
                progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})
                
                # Log periodically
                if batch_idx % 100 == 0:
                    logger.log(f"Epoch {epoch+1}, Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.4f}")
            
            avg_train_loss = train_loss / len(train_loader)
            
            # Validation
            model.eval()
            val_loss = 0
            with torch.no_grad():
                for inputs, targets in val_loader:
                    inputs, targets = inputs.cuda(), targets.cuda()
                    
                    b, c, d, h, w = inputs.shape
                    inputs_2d = inputs.squeeze(1).permute(0, 1, 2, 3)
                    
                    outputs = model(inputs_2d)
                    targets = targets.squeeze(2)
                    
                    loss = criterion(outputs, targets)
                    val_loss += loss.item()
            
            avg_val_loss = val_loss / len(val_loader)
            
            # Log epoch results
            epoch_time = (datetime.now() - epoch_start_time).total_seconds()
            logger.log(f"Epoch {epoch+1} complete in {epoch_time:.1f}s - Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
            
            # Save checkpoint every epoch
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss,
                'best_val_loss': min(best_val_loss, avg_val_loss),
            }
            
            # Save best model
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                checkpoint['best_val_loss'] = best_val_loss
                checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(checkpoint, checkpoint_path)
                logger.log(f"✅ Saved best model (val_loss: {best_val_loss:.4f})")
            
            # Also save latest checkpoint
            latest_path = Path("/mnt/models/latest_checkpoint.pth")
            torch.save(checkpoint, latest_path)
            
            # Update learning rate
            scheduler.step()
            
            # Commit volume periodically
            if epoch % 5 == 0:
                volume.commit()
                logger.log("Volume committed")
        
        logger.log(f"✅ Training completed! Best val loss: {best_val_loss:.4f}")
        
    except Exception as e:
        logger.log(f"❌ Training interrupted: {type(e).__name__}: {str(e)}")
        
        # Save emergency checkpoint
        try:
            if 'model' in locals() and 'optimizer' in locals() and 'epoch' in locals():
                emergency_checkpoint = {
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'interrupted': True,
                    'error': str(e)
                }
                emergency_path = Path("/mnt/models/emergency_checkpoint.pth")
                emergency_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(emergency_checkpoint, emergency_path)
                logger.log(f"💾 Saved emergency checkpoint at epoch {epoch}")
        except:
            logger.log("Failed to save emergency checkpoint")
        
        raise
    
    finally:
        # Always commit volume before exiting
        volume.commit()
        logger.log("Final volume commit completed")
    
    return f"Training complete! Best validation loss: {best_val_loss:.4f}"


@app.local_entrypoint()
def main(resume: bool = True, epochs: int = 20):
    """
    Entry point with resumption support
    
    Usage:
        modal run modal_train_robust.py                    # Resume from checkpoint
        modal run modal_train_robust.py --resume=false    # Start fresh
        modal run modal_train_robust.py --epochs=30       # Train for 30 epochs
    """
    print(f"🚀 Launching robust training (resume={resume}, epochs={epochs})")
    result = train_with_resume.remote(
        config_name="baseline",
        max_epochs=epochs,
        resume=resume
    )
    print(f"✅ {result}")


if __name__ == "__main__":
    main()