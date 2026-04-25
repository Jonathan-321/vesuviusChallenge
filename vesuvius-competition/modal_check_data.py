#!/usr/bin/env python3
"""
Check what data is already on Modal
"""
import modal
from pathlib import Path

app = modal.App("vesuvius-check")
volume = modal.Volume.from_name("vesuvius-data", create_if_missing=False)

# Create image with numpy
image = modal.Image.debian_slim(python_version="3.11").pip_install(["numpy>=1.24.0"])

@app.function(
    image=image,
    volumes={"/mnt": volume},
    timeout=300,
)
def check_modal_data():
    """Check what's already uploaded to Modal"""
    import os
    import numpy as np
    
    print("=== CHECKING MODAL DATA ===\n")
    
    # Check for proper_training
    proper_path = Path("/mnt/proper_training")
    if proper_path.exists():
        print("✅ Found /mnt/proper_training/")
        images = list((proper_path / "images").glob("*.npz")) if (proper_path / "images").exists() else []
        masks = list((proper_path / "masks").glob("*.npz")) if (proper_path / "masks").exists() else []
        print(f"   Images: {len(images)} files")
        print(f"   Masks: {len(masks)} files")
        
        # Check a sample mask
        if masks:
            sample_mask = np.load(masks[0])['data']
            print(f"   Sample mask {masks[0].name}:")
            print(f"     Shape: {sample_mask.shape}")
            print(f"     Values: {np.unique(sample_mask)}")
            print(f"     Coverage: {(sample_mask > 0).sum() / sample_mask.size:.2%}")
    else:
        print("❌ /mnt/proper_training/ not found")
    
    # Check for all_training
    all_path = Path("/mnt/all_training")
    if all_path.exists():
        print("\n✅ Found /mnt/all_training/")
        images = list((all_path / "images").glob("*.npz")) if (all_path / "images").exists() else []
        masks = list((all_path / "masks").glob("*.npz")) if (all_path / "masks").exists() else []
        print(f"   Images: {len(images)} files")
        print(f"   Masks: {len(masks)} files")
    else:
        print("\n❌ /mnt/all_training/ not found")
    
    # Check for models
    models_path = Path("/mnt/models")
    if models_path.exists():
        print("\n✅ Found /mnt/models/")
        model_files = list(models_path.glob("*.pth"))
        for model_file in model_files:
            size_mb = model_file.stat().st_size / (1024 * 1024)
            print(f"   {model_file.name}: {size_mb:.1f} MB")
    
    # Check for old processed data
    processed_path = Path("/mnt/data/processed")
    if processed_path.exists():
        print("\n⚠️  Found OLD /mnt/data/processed/ (with corrupted labels)")
        masks = list(processed_path.glob("mask_*.npz"))
        if masks:
            try:
                with np.load(masks[0]) as data:
                    keys = list(data.keys())
                    sample = data[keys[0]]
                    print(f"   Sample values: {np.unique(sample)} (should be [0, 255])")
            except:
                print(f"   Could not read sample mask")
    
    # Check for processed folder in root
    processed_root = Path("/mnt/processed")
    if processed_root.exists():
        print("\n⚠️  Found /mnt/processed/ folder")
        files = list(processed_root.glob("*.npz"))
        print(f"   Contains {len(files)} files")
    
    return "Check complete"

@app.local_entrypoint()
def main():
    print("Checking Modal data...")
    result = check_modal_data.remote()
    print(f"\n{result}")
    
    print("\n" + "="*60)
    print("RECOMMENDATION:")
    print("If proper_training/ or all_training/ are missing:")
    print("  Run: modal run modal_upload_data.py")
    print("\nIf they exist with correct data:")
    print("  Run: modal run modal_train_balanced.py")
    print("="*60)

if __name__ == "__main__":
    app.run()