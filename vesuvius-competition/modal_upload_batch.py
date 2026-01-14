"""
Batch upload processed data to Modal with validation
"""
import modal
import pathlib
import numpy as np
from pathlib import Path

app = modal.App("vesuvius-batch-upload")

# Get or create volume
volume = modal.Volume.from_name("vesuvius-data", create_if_missing=True)

# Image with numpy
image = modal.Image.debian_slim(python_version="3.11").pip_install("numpy")

@app.local_entrypoint()
def upload_and_validate():
    """Upload all processed data and validate"""
    local_root = Path("data/processed")
    
    # Count local files
    volume_files = sorted(local_root.glob("volume_*.npz"))
    mask_files = sorted(local_root.glob("mask_*.npz"))
    
    print(f"\n{'='*60}")
    print("Local Data Summary")
    print(f"{'='*60}")
    print(f"Volume files: {len(volume_files)}")
    print(f"Mask files: {len(mask_files)}")
    
    # Validate pairs exist
    valid_pairs = []
    for vol_path in volume_files:
        vol_id = vol_path.stem.replace('volume_', '')
        mask_path = local_root / f"mask_{vol_id}.npz"
        if mask_path.exists():
            valid_pairs.append((vol_path, mask_path))
    
    print(f"Valid pairs: {len(valid_pairs)}")
    
    # Check shapes of first few pairs
    print(f"\nChecking first 5 pairs:")
    for i, (vol_path, mask_path) in enumerate(valid_pairs[:5]):
        with np.load(vol_path) as vol_data:
            vol_keys = list(vol_data.keys())
            vol_shape = vol_data[vol_keys[0]].shape
        
        with np.load(mask_path) as mask_data:
            mask_keys = list(mask_data.keys())
            mask_shape = mask_data[mask_keys[0]].shape
            
        print(f"{i+1}. ID: {vol_path.stem.replace('volume_', '')}")
        print(f"   Volume: {vol_shape}, Mask: {mask_shape}")
    
    # Clear existing data
    print(f"\n{'='*60}")
    print("Uploading to Modal...")
    print(f"{'='*60}")
    
    # Use batch upload
    with volume.batch_upload(force=True) as batch:
        # Upload entire directory preserving structure
        batch.put_directory(str(local_root), "/processed")
    
    print("✅ Upload complete!")
    
    # Verify upload
    verify_upload.remote()


@app.function(image=image, volumes={"/mnt": volume})
def verify_upload():
    """Verify all files were uploaded correctly"""
    remote_root = Path("/mnt/processed")
    
    print(f"\n{'='*60}")
    print("Verifying Modal Upload")
    print(f"{'='*60}")
    
    # Count files
    all_files = list(remote_root.glob("*.npz"))
    volume_files = sorted(list(remote_root.glob("volume_*.npz")))
    mask_files = sorted(list(remote_root.glob("mask_*.npz")))
    
    print(f"Total .npz files: {len(all_files)}")
    print(f"Volume files: {len(volume_files)}")
    print(f"Mask files: {len(mask_files)}")
    
    # Check pairs
    valid_pairs = []
    shape_issues = []
    
    for vol_path in volume_files:
        vol_id = vol_path.stem.replace('volume_', '')
        mask_path = remote_root / f"mask_{vol_id}.npz"
        
        if mask_path.exists():
            # Load and check shapes
            with np.load(vol_path) as vol_data:
                vol_keys = list(vol_data.keys())
                vol = vol_data[vol_keys[0]]
                vol_shape = vol.shape
            
            with np.load(mask_path) as mask_data:
                mask_keys = list(mask_data.keys()) 
                mask = mask_data[mask_keys[0]]
                mask_shape = mask.shape
            
            # Expected shapes
            if vol_shape[0] == 65 and vol_shape[1] == 320 and vol_shape[2] == 320:
                if mask_shape == (320, 320):  # 2D mask
                    valid_pairs.append(vol_id)
                else:
                    shape_issues.append(f"ID {vol_id}: mask shape {mask_shape} (expected 2D)")
            else:
                shape_issues.append(f"ID {vol_id}: volume shape {vol_shape} (expected 65x320x320)")
    
    print(f"\nValid pairs with correct shapes: {len(valid_pairs)}")
    
    if shape_issues:
        print(f"\nShape issues found:")
        for issue in shape_issues[:5]:
            print(f"  - {issue}")
        if len(shape_issues) > 5:
            print(f"  ... and {len(shape_issues) - 5} more")
    
    print(f"\n✅ Verification complete!")
    print(f"   Ready pairs for training: {len(valid_pairs)}")
    
    return len(valid_pairs)


if __name__ == "__main__":
    app.run()