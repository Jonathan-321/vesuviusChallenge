"""
Check training progress and download models
"""
import modal
from pathlib import Path

app = modal.App("vesuvius-check-training")
volume = modal.Volume.from_name("vesuvius-data")
image = modal.Image.debian_slim(python_version="3.11").pip_install("torch")


@app.function(image=image, volumes={"/mnt": volume})
def check_training_status():
    """Check what models have been saved"""
    import torch
    
    print(f"\n{'='*60}")
    print("Checking Training Status")
    print(f"{'='*60}")
    
    # Check models directory
    models_dir = Path("/mnt/models")
    if not models_dir.exists():
        print("❌ No models directory found")
        return None
    
    # List all files in models directory
    all_files = list(models_dir.rglob("*"))
    print(f"\nFiles in /mnt/models:")
    for f in all_files:
        if f.is_file():
            size_mb = f.stat().st_size / (1024 * 1024)
            print(f"  - {f.relative_to(models_dir)} ({size_mb:.1f} MB)")
    
    # Check for .pth files
    model_files = list(models_dir.glob("*.pth"))
    if not model_files:
        print("\n❌ No .pth model files found")
        return None
    
    print(f"\n✅ Found {len(model_files)} model file(s)")
    
    # Load and check the latest model
    latest_model = max(model_files, key=lambda p: p.stat().st_mtime)
    print(f"\nLatest model: {latest_model.name}")
    
    try:
        checkpoint = torch.load(latest_model, map_location='cpu')
        print(f"  - Epoch: {checkpoint.get('epoch', 'unknown')}")
        print(f"  - Best val loss: {checkpoint.get('best_val_loss', 'unknown'):.4f}")
        
        # Check model state dict
        if 'model_state_dict' in checkpoint:
            n_params = sum(p.numel() for p in checkpoint['model_state_dict'].values())
            print(f"  - Model parameters: {n_params:,}")
    except Exception as e:
        print(f"  - Error loading model: {e}")
    
    return str(latest_model) if model_files else None


@app.function(image=image, volumes={"/mnt": volume})
def download_best_model(local_path: str = "./models/from_modal"):
    """Download the best model from Modal"""
    from pathlib import Path
    import shutil
    
    models_dir = Path("/mnt/models")
    if not models_dir.exists():
        print("❌ No models directory found")
        return False
    
    # Find best model
    best_model = models_dir / "baseline_best.pth"
    if not best_model.exists():
        # Try to find any .pth file
        model_files = list(models_dir.glob("*.pth"))
        if not model_files:
            print("❌ No model files found")
            return False
        best_model = model_files[0]
    
    print(f"📥 Preparing to download: {best_model.name}")
    
    # Copy to a location that can be downloaded
    temp_path = Path("/tmp") / best_model.name
    shutil.copy(best_model, temp_path)
    
    print(f"✅ Model ready for download: {temp_path}")
    print(f"   Size: {temp_path.stat().st_size / (1024*1024):.1f} MB")
    
    # Return the path for Modal to handle the download
    return str(temp_path)


@app.local_entrypoint()
def main():
    # Check status
    model_path = check_training_status.remote()
    
    if model_path:
        print("\n📥 Downloading model...")
        # Download the model
        local_path = download_best_model.remote()
        if local_path:
            print(f"✅ Model downloaded successfully!")
            
            # Actually download the file using modal volume get
            import subprocess
            result = subprocess.run([
                "modal", "volume", "get", "vesuvius-data",
                "/mnt/models/baseline_best.pth",
                "./models/from_modal/baseline_best.pth"
            ], capture_output=True, text=True)
            
            if result.returncode == 0:
                print("✅ Model saved to ./models/from_modal/baseline_best.pth")
            else:
                print(f"❌ Download failed: {result.stderr}")


if __name__ == "__main__":
    app.run()