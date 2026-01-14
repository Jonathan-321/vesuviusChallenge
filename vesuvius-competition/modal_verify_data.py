"""
Verify Modal data upload and check shapes
"""
import modal
import numpy as np
from pathlib import Path

app = modal.App("vesuvius-verify")
volume = modal.Volume.from_name("vesuvius-data")
image = modal.Image.debian_slim(python_version="3.11").pip_install("numpy")


@app.function(image=image, volumes={"/mnt": volume})
def verify_data():
    """Verify all files and shapes"""
    remote_root = Path("/mnt/processed")
    
    print(f"\n{'='*60}")
    print("Modal Data Verification")
    print(f"{'='*60}")
    
    # Count files
    all_files = list(remote_root.glob("*.npz"))
    volume_files = sorted(list(remote_root.glob("volume_*.npz")))
    mask_files = sorted(list(remote_root.glob("mask_*.npz")))
    
    print(f"Total .npz files: {len(all_files)}")
    print(f"Volume files: {len(volume_files)}")
    print(f"Mask files: {len(mask_files)}")
    
    # Check all pairs
    valid_pairs = []
    shape_issues = []
    missing_masks = []
    
    for vol_path in volume_files:
        vol_id = vol_path.stem.replace('volume_', '')
        mask_path = remote_root / f"mask_{vol_id}.npz"
        
        if not mask_path.exists():
            missing_masks.append(vol_id)
            continue
            
        # Load and check shapes
        try:
            with np.load(vol_path) as vol_data:
                vol_keys = list(vol_data.keys())
                vol = vol_data[vol_keys[0]]
                vol_shape = vol.shape
            
            with np.load(mask_path) as mask_data:
                mask_keys = list(mask_data.keys()) 
                mask = mask_data[mask_keys[0]]
                mask_shape = mask.shape
            
            # Expected shapes
            expected_vol = (65, 320, 320)
            expected_mask = (320, 320)
            
            if vol_shape == expected_vol and mask_shape == expected_mask:
                valid_pairs.append(vol_id)
            else:
                shape_issues.append({
                    'id': vol_id,
                    'vol_shape': vol_shape,
                    'mask_shape': mask_shape,
                    'expected_vol': expected_vol,
                    'expected_mask': expected_mask
                })
        except Exception as e:
            shape_issues.append({
                'id': vol_id,
                'error': str(e)
            })
    
    print(f"\n✅ Valid pairs with correct shapes: {len(valid_pairs)}")
    
    if missing_masks:
        print(f"\n❌ Missing masks: {len(missing_masks)}")
        for m in missing_masks[:3]:
            print(f"  - {m}")
    
    if shape_issues:
        print(f"\n⚠️ Shape issues: {len(shape_issues)}")
        for issue in shape_issues[:5]:
            if 'error' in issue:
                print(f"  - ID {issue['id']}: Error - {issue['error']}")
            else:
                print(f"  - ID {issue['id']}: Vol {issue['vol_shape']} (expected {issue['expected_vol']}), "
                      f"Mask {issue['mask_shape']} (expected {issue['expected_mask']})")
    
    # Show some valid examples
    if valid_pairs:
        print(f"\n📊 Sample valid pairs:")
        for vid in valid_pairs[:3]:
            print(f"  - {vid}")
    
    print(f"\n{'='*60}")
    print(f"Summary: {len(valid_pairs)}/{len(volume_files)} volumes ready for training")
    print(f"{'='*60}")
    
    return len(valid_pairs)


@app.local_entrypoint()
def main():
    result = verify_data.remote()
    print(f"\nTotal valid pairs: {result}")


if __name__ == "__main__":
    app.run()