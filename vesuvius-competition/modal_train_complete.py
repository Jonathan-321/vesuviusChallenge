"""
Complete training to 20 epochs with automatic continuation
"""
import modal
from datetime import datetime

app = modal.App("vesuvius-complete-training")
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


@app.function(
    image=image,
    gpu="A100-80GB",
    volumes={"/mnt": volume},
    timeout=60 * 60 * 24,  # 24 hours
)
def complete_training_to_20_epochs():
    """
    Complete training to 20 epochs, starting from wherever we left off
    """
    import os
    import sys
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, Dataset
    import numpy as np
    from pathlib import Path
    from tqdm import tqdm
    import segmentation_models_pytorch as smp
    
    print(f"🚀 Completing training to 20 epochs")
    print(f"   Starting time: {datetime.now()}")
    
    # Check current status
    checkpoint_path = Path("/mnt/models/latest_checkpoint.pth")
    best_path = Path("/mnt/models/baseline_best.pth")
    
    if not checkpoint_path.exists() and not best_path.exists():
        print("❌ No checkpoint found! Please run initial training first.")
        return
    
    # Load the latest checkpoint
    if checkpoint_path.exists():
        checkpoint = torch.load(checkpoint_path)
    else:
        checkpoint = torch.load(best_path)
    
    current_epoch = checkpoint.get('epoch', 0)
    print(f"📊 Current status: Completed {current_epoch} epochs")
    
    if current_epoch >= 19:  # 0-indexed, so 19 = 20 epochs
        print("✅ Training already complete! 20 epochs done.")
        return "Training complete"
    
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
                        print(f"Error loading {vid}: {e}")
            
            print(f"Loaded {len(self.volumes)} volumes")
            
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
    print(f"Found {len(volume_ids)} valid volumes")
    
    # Split data
    np.random.seed(42)
    shuffled_ids = np.random.permutation(volume_ids)
    n_val = max(1, int(len(shuffled_ids) * 0.2))
    val_ids = shuffled_ids[:n_val].tolist()
    train_ids = shuffled_ids[n_val:].tolist()
    
    print(f"Train: {len(train_ids)}, Val: {len(val_ids)}")
    
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
    
    # Load model weights
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Loss and optimizer
    criterion = smp.losses.DiceLoss(mode='binary', from_logits=True)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20)
    
    # Load optimizer state
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    # Adjust learning rate for later epochs
    for _ in range(current_epoch + 1):
        scheduler.step()
    
    best_val_loss = checkpoint.get('best_val_loss', float('inf'))
    
    print(f"\n🏃 Continuing training from epoch {current_epoch + 1} to 20")
    print(f"   Best val loss so far: {best_val_loss:.4f}")
    
    # Continue training
    for epoch in range(current_epoch + 1, 20):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch+1}/20 - {datetime.now()}")
        print(f"{'='*60}")
        
        # Training
        model.train()
        train_loss = 0
        progress_bar = tqdm(train_loader, desc=f"Training")
        
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
            progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        avg_train_loss = train_loss / len(train_loader)
        
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for inputs, targets in tqdm(val_loader, desc="Validation"):
                inputs, targets = inputs.cuda(), targets.cuda()
                
                b, c, d, h, w = inputs.shape
                inputs_2d = inputs.squeeze(1).permute(0, 1, 2, 3)
                
                outputs = model(inputs_2d)
                targets = targets.squeeze(2)
                
                loss = criterion(outputs, targets)
                val_loss += loss.item()
        
        avg_val_loss = val_loss / len(val_loader)
        
        print(f"\n📊 Epoch {epoch+1} Results:")
        print(f"   Train Loss: {avg_train_loss:.4f}")
        print(f"   Val Loss: {avg_val_loss:.4f}")
        
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
            torch.save(checkpoint, best_path)
            print(f"   ✅ New best model saved (val_loss: {best_val_loss:.4f})")
        
        # Always save latest
        torch.save(checkpoint, checkpoint_path)
        
        # Update learning rate
        scheduler.step()
        
        # Commit volume every 2 epochs
        if epoch % 2 == 0:
            volume.commit()
            print("   📤 Volume committed")
    
    # Final commit
    volume.commit()
    
    print(f"\n✅ Training completed!")
    print(f"   Final epoch: 20")
    print(f"   Best validation loss: {best_val_loss:.4f}")
    print(f"   Models saved in /mnt/models/")
    
    return f"Training complete! Best val loss: {best_val_loss:.4f}"


@app.local_entrypoint()
def main():
    """
    Complete training to 20 epochs
    """
    print("🎯 Completing training to 20 epochs...")
    result = complete_training_to_20_epochs.remote()
    print(f"\n✅ {result}")


if __name__ == "__main__":
    app.run()