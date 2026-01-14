"""
Check what data is actually in Modal volume
"""
import modal

app = modal.App("vesuvius-check-data")
volume = modal.Volume.from_name("vesuvius-data", create_if_missing=False)

image = modal.Image.debian_slim(python_version="3.11").pip_install("numpy")

@app.function(image=image, volumes={"/mnt": volume})
def check_data():
    from pathlib import Path
    import numpy as np
    
    data_path = Path("/mnt/data/processed")
    
    print(f"\n{'='*60}")
    print(f"Checking Modal Volume Data")
    print(f"{'='*60}")
    
    # List all files
    all_files = sorted(list(data_path.glob("*.npz")))
    print(f"\nTotal .npz files: {len(all_files)}")
    
    # List all filenames
    print("\nAll files:")
    for f in all_files[:10]:
        print(f"  - {f.name}")
    if len(all_files) > 10:
        print(f"  ... and {len(all_files) - 10} more")
    
    # Check volume files
    volume_files = sorted(list(data_path.glob("volume_*.npz")))
    mask_files = sorted(list(data_path.glob("mask_*.npz")))
    
    print(f"\nVolume files: {len(volume_files)}")
    print(f"Mask files: {len(mask_files)}")
    
    # Check some volumes
    print(f"\nChecking first 5 volumes:")
    for i, vol_file in enumerate(volume_files[:5]):
        vol_id = vol_file.stem.replace('volume_', '')
        mask_file = data_path / f"mask_{vol_id}.npz"
        
        print(f"\n{i+1}. Volume ID: {vol_id}")
        print(f"   Volume file: {vol_file.name}")
        print(f"   Mask exists: {mask_file.exists()}")
        
        try:
            # Load and check shapes
            with np.load(vol_file) as data:
                keys = list(data.keys())
                print(f"   Volume keys: {keys}")
                vol = data[keys[0]]
                print(f"   Volume shape: {vol.shape}")
                
            if mask_file.exists():
                with np.load(mask_file) as data:
                    keys = list(data.keys())
                    print(f"   Mask keys: {keys}")
                    mask = data[keys[0]]
                    print(f"   Mask shape: {mask.shape}")
        except Exception as e:
            print(f"   Error: {e}")
    
    # List all volume IDs
    valid_ids = []
    for vol_file in volume_files:
        vol_id = vol_file.stem.replace('volume_', '')
        if vol_id.isdigit():
            mask_file = data_path / f"mask_{vol_id}.npz"
            if mask_file.exists():
                valid_ids.append(vol_id)
    
    print(f"\n{'='*60}")
    print(f"Valid volume-mask pairs: {len(valid_ids)}")
    print(f"IDs: {valid_ids[:10]}{'...' if len(valid_ids) > 10 else ''}")
    print(f"{'='*60}\n")
    
    return len(valid_ids)

@app.local_entrypoint()
def main():
    result = check_data.remote()
    print(f"Found {result} valid volume-mask pairs in Modal")

if __name__ == "__main__":
    main()