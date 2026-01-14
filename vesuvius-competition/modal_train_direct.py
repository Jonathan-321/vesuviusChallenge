"""
Direct Modal training script for Vesuvius Challenge
"""
import modal

app = modal.App("vesuvius-training-direct")

# Modal volume for data
volume = modal.Volume.from_name("vesuvius-data", create_if_missing=True)

# Docker image with all dependencies
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
def train_baseline():
    """
    Train baseline model with minimal imports
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
    
    print(f"🚀 Starting training on A100-80GB")
    print(f"   GPU: {torch.cuda.get_device_name(0)}")
    print(f"   VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB")
    
    # Simple dataset
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
                    # Load without memory mapping to avoid issues
                    with np.load(vol_path) as vol_data:
                        # Handle different key formats
                        vol = vol_data['data'] if 'data' in vol_data else vol_data['volume'] if 'volume' in vol_data else vol_data[list(vol_data.keys())[0]]
                        vol = vol.copy()  # Make sure it's not a view
                    
                    with np.load(mask_path) as mask_data:
                        mask = mask_data['data'] if 'data' in mask_data else mask_data['mask'] if 'mask' in mask_data else mask_data[list(mask_data.keys())[0]]
                        mask = mask.copy()  # Make sure it's not a view
                        
                    self.volumes.append(vol)
                    self.masks.append(mask)
            
            print(f"Loaded {len(self.volumes)} volumes")
            
        def __len__(self):
            return len(self.volumes) * self.samples_per_volume
        
        def __getitem__(self, idx):
            vol_idx = idx % len(self.volumes)
            volume = self.volumes[vol_idx]
            mask = self.masks[vol_idx]
            
            # Random crop with safety checks
            d, h, w = volume.shape
            pd, ph, pw = 32, 128, 128  # Patch sizes
            
            # Ensure we have enough dimensions
            if d < pd or h < ph or w < pw:
                # Pad if needed
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
            
            # Ensure correct dimensions
            if vol_patch.shape != (pd, ph, pw):
                print(f"Warning: vol_patch shape {vol_patch.shape} != expected {(pd, ph, pw)}")
                # Create a properly sized patch
                correct_patch = np.zeros((pd, ph, pw), dtype=np.float32)
                d_actual, h_actual, w_actual = vol_patch.shape
                correct_patch[:d_actual, :h_actual, :w_actual] = vol_patch
                vol_patch = correct_patch
                
            if mask_patch.shape != (ph, pw):
                print(f"Warning: mask_patch shape {mask_patch.shape} != expected {(ph, pw)}")
                correct_mask = np.zeros((ph, pw), dtype=np.float32)
                if mask_patch.ndim == 2:
                    h_actual, w_actual = mask_patch.shape
                    correct_mask[:h_actual, :w_actual] = mask_patch
                elif mask_patch.ndim == 3:
                    # Should not happen with our clean data, but handle it
                    print(f"Unexpected 3D mask shape: {mask_patch.shape}")
                    if mask_patch.shape[0] > 0:
                        mask_2d = mask_patch[mask_patch.shape[0]//2]
                        h_actual = min(mask_2d.shape[0], ph)
                        w_actual = min(mask_2d.shape[1], pw) if mask_2d.ndim > 1 else 1
                        correct_mask[:h_actual, :w_actual] = mask_2d[:h_actual, :w_actual]
                mask_patch = correct_mask
            
            # Convert to tensors
            vol_tensor = torch.from_numpy(vol_patch).unsqueeze(0)
            mask_tensor = torch.from_numpy(mask_patch).unsqueeze(0).unsqueeze(0)
            
            return vol_tensor, mask_tensor
    
    # Get volume IDs from the correctly uploaded data
    data_path = Path("/mnt/processed")  # Note: changed path
    volume_files = sorted(list(data_path.glob('volume_*.npz')))
    
    # Only use volumes with matching masks and correct shapes
    valid_volume_ids = []
    for vol_file in volume_files:
        vol_id = vol_file.stem.replace('volume_', '')
        mask_file = data_path / f"mask_{vol_id}.npz"
        
        if mask_file.exists():
            # Quick shape check
            try:
                with np.load(vol_file) as vdata:
                    vshape = vdata[list(vdata.keys())[0]].shape
                if vshape == (65, 320, 320):  # Only use standard size
                    valid_volume_ids.append(vol_id)
            except:
                pass
    
    volume_ids = valid_volume_ids
    print(f"\n📊 Found {len(volume_files)} volume files")
    print(f"   Valid volume IDs with correct shapes: {len(volume_ids)}")
    
    # Split data
    np.random.seed(42)
    shuffled_ids = np.random.permutation(volume_ids)
    n_val = max(1, int(len(shuffled_ids) * 0.2))
    val_ids = shuffled_ids[:n_val].tolist()
    train_ids = shuffled_ids[n_val:].tolist()
    
    print(f"   Train: {len(train_ids)} volumes")
    print(f"   Val: {len(val_ids)} volumes")
    
    # Create datasets
    train_dataset = VesuviusDataset(train_ids, data_path, samples_per_volume=200)
    val_dataset = VesuviusDataset(val_ids, data_path, samples_per_volume=50)
    
    # Create dataloaders with num_workers=0 to avoid multiprocessing issues
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
    print(f"\n🏗️  Model: UNet-ResNet34")
    print(f"   Parameters: {total_params:,}")
    
    # Loss and optimizer
    criterion = smp.losses.DiceLoss(mode='binary', from_logits=True)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20)
    
    # Training loop
    print(f"\n🏃 Starting training...")
    best_val_loss = float('inf')
    
    for epoch in range(20):  # Train for 20 epochs
        # Training
        model.train()
        train_loss = 0
        for batch_idx, (inputs, targets) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/20")):
            inputs, targets = inputs.cuda(), targets.cuda()
            
            # Reshape for 2D model - use depth as channels
            b, c, d, h, w = inputs.shape
            # Squeeze channel dimension and use depth slices as channels
            inputs_2d = inputs.squeeze(1).permute(0, 1, 2, 3)  # (b, d, h, w)
            
            outputs = model(inputs_2d)
            
            # outputs is already (b, 1, h, w), just need to match target shape
            # targets is (b, 1, 1, h, w), so squeeze one dimension
            targets = targets.squeeze(2)  # (b, 1, h, w)
            
            loss = criterion(outputs, targets)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
            if batch_idx % 50 == 0:
                print(f"   Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.4f}")
        
        avg_train_loss = train_loss / len(train_loader)
        
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.cuda(), targets.cuda()
                
                # Reshape for 2D model - use depth as channels
                b, c, d, h, w = inputs.shape
                inputs_2d = inputs.squeeze(1).permute(0, 1, 2, 3)  # (b, d, h, w)
                
                outputs = model(inputs_2d)
                targets = targets.squeeze(2)  # (b, 1, h, w)
                
                loss = criterion(outputs, targets)
                val_loss += loss.item()
        
        avg_val_loss = val_loss / len(val_loader)
        
        print(f"\nEpoch {epoch+1}: Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
        
        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            checkpoint_path = Path("/mnt/models/baseline_best.pth")
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_loss': best_val_loss,
            }, checkpoint_path)
            print(f"   ✅ Saved best model (val_loss: {best_val_loss:.4f})")
        
        scheduler.step()
    
    print(f"\n✅ Training complete!")
    print(f"   Best val loss: {best_val_loss:.4f}")
    
    # Commit volume to save checkpoints
    volume.commit()
    
    return f"Training complete! Best validation loss: {best_val_loss:.4f}"


@app.local_entrypoint()
def main():
    """
    Entry point
    """
    print("🚀 Launching Vesuvius baseline training on Modal...")
    result = train_baseline.remote()
    print(f"✅ {result}")


if __name__ == "__main__":
    main()