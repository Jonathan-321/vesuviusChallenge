"""
Modal training setup - Upload from local machine, train on A100s
Perfect for: Local preprocessing + Cloud training
"""

import modal
from pathlib import Path

# Create Modal app
app = modal.App("vesuvius-training")

# Modal volume for persistent storage
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
        "monai>=1.3.0",
        "segmentation-models-pytorch>=0.3.3",
        "wandb>=0.16.0",
        "pandas>=2.0.0",
        "scikit-learn>=1.3.0",
        "matplotlib>=3.7.0",
    ])
)

# Mount points
DATA_DIR = "/mnt/data"
MODEL_DIR = "/mnt/models"


@app.function(
    image=image,
    volumes={DATA_DIR: volume},
    timeout=7200,
)
def upload_processed_data(local_path: str = "data/processed"):
    """
    Upload preprocessed .npz files from local machine to Modal volume
    Run this ONCE after local preprocessing
    
    Usage:
        modal run src/modal_training.py::upload_processed_data
    """
    import os
    import shutil
    from pathlib import Path
    
    local_path = Path(local_path)
    remote_path = Path(DATA_DIR) / "processed"
    remote_path.mkdir(parents=True, exist_ok=True)
    
    print(f"📤 Uploading processed data from {local_path} to Modal volume...")
    
    # Get list of files
    npz_files = list(local_path.glob("*.npz"))
    
    if not npz_files:
        print(f"❌ No .npz files found in {local_path}")
        print("   Make sure you've run the preprocessing script first:")
        print("   python scripts/preprocessing/prepare_data.py")
        return
    
    print(f"Found {len(npz_files)} files to upload")
    
    # Copy files (Modal CLI handles the actual upload)
    for f in npz_files:
        print(f"  Uploading {f.name}...")
        # Modal's volume.put_file will be used by CLI
    
    volume.commit()
    print(f"✅ Upload complete! {len(npz_files)} files uploaded to Modal volume")


@app.function(
    image=image,
    volumes={DATA_DIR: volume},
)
def list_data():
    """
    List all data in Modal volume
    
    Usage:
        modal run src/modal_training.py::list_data
    """
    from pathlib import Path
    
    data_path = Path(DATA_DIR)
    
    print(f"\n{'='*60}")
    print(f"Modal Volume Contents")
    print(f"{'='*60}")
    
    if not data_path.exists():
        print("❌ Data directory not found in Modal volume")
        return
    
    processed_dir = data_path / "processed"
    if processed_dir.exists():
        files = sorted(list(processed_dir.glob("*.npz")))
        print(f"\n📁 Processed data ({len(files)} files):")
        
        total_size = 0
        for f in files:
            size_mb = f.stat().st_size / (1024 * 1024)
            total_size += size_mb
            print(f"  {f.name:40s} {size_mb:8.1f} MB")
        
        print(f"\n  Total size: {total_size/1024:.2f} GB")
    else:
        print("❌ No processed data found")
        print("   Run: modal run src/modal_training.py::upload_processed_data")
    
    print(f"{'='*60}\n")


@app.function(
    image=image,
    gpu="A100-80GB",
    volumes={"/mnt": volume},
    timeout=60 * 60 * 24,  # 24 hours
    # secrets=[modal.Secret.from_name("wandb-secret")],
)
def train_single_model(config_path: str):
    """
    Train a single model on A100
    
    Usage:
        modal run src/modal_training.py::train_single_model --config-path configs/experiments/baseline.yaml
    """
    import sys
    import os
    import yaml
    import torch
    from pathlib import Path
    
    # Add mounted volume paths
    sys.path.append("/mnt")
    print(f"Python path: {sys.path}")
    print(f"Contents of /mnt: {os.listdir('/mnt')}" if os.path.exists('/mnt') else "/mnt doesn't exist")
    
    # Import our modules from mounted volume
    from src.models import get_model
    from src.data.dataset import get_dataloaders
    from src.training.trainer import Trainer, get_optimizer, get_scheduler
    from src.training.losses import get_loss_function
    
    print(f"🚀 Starting training on A100-80GB")
    print(f"   Config: {config_path}")
    
    # Load config from mounted volume
    config_path = f"/mnt/{config_path}"
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Merge with base config if it exists
    base_config_path = Path('/mnt/configs/base/default.yaml')
    if base_config_path.exists():
        with open(base_config_path, 'r') as f:
            base_config = yaml.safe_load(f)
        
        # Deep merge configs
        for key in base_config:
            if key not in config:
                config[key] = base_config[key]
            elif isinstance(base_config[key], dict):
                for subkey in base_config[key]:
                    if key not in config:
                        config[key] = {}
                    if subkey not in config[key]:
                        config[key][subkey] = base_config[key][subkey]
    
    # Update paths for Modal
    config['data']['processed_dir'] = f"{DATA_DIR}/processed"
    config['paths']['checkpoint_dir'] = f"{MODEL_DIR}/checkpoints/{config['experiment']['name']}"
    config['checkpoint_dir'] = config['paths']['checkpoint_dir']
    
    # Create checkpoint directory
    Path(config['checkpoint_dir']).mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*60}")
    print(f"Experiment: {config['experiment']['name']}")
    print(f"{'='*60}")
    
    # Get volume IDs
    import numpy as np
    data_path = Path(config['data']['processed_dir'])
    volume_files = sorted(list(data_path.glob('volume_*.npz')))
    
    if not volume_files:
        raise FileNotFoundError(f"No volume files found in {data_path}")
    
    volume_ids = [f.stem.replace('volume_', '') for f in volume_files]
    
    # Split data
    np.random.seed(config['experiment'].get('seed', 42))
    shuffled_ids = np.random.permutation(volume_ids)
    n_val = max(1, int(len(shuffled_ids) * 0.2))
    val_ids = shuffled_ids[:n_val].tolist()
    train_ids = shuffled_ids[n_val:].tolist()
    
    print(f"\n📊 Dataset Split:")
    print(f"   Training volumes: {len(train_ids)}")
    print(f"   Validation volumes: {len(val_ids)}")
    
    # Create dataloaders
    train_loader, val_loader = get_dataloaders(config, train_ids, val_ids)
    
    # Create model
    device = torch.device('cuda')
    model = get_model(config)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\n🏗️ Model: {config['model']['architecture']}")
    print(f"   Total parameters: {total_params:,}")
    
    # Create loss, optimizer, scheduler
    criterion = get_loss_function(config)
    optimizer = get_optimizer(model, config)
    scheduler = get_scheduler(optimizer, config)
    
    # Create trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        config=config,
        device=device
    )
    
    # Train
    trainer.train()
    
    # Commit volume to persist checkpoints
    volume.commit()
    
    print(f"\n✅ Training complete!")
    print(f"   Best val loss: {trainer.best_val_loss:.4f}")
    print(f"   Checkpoints saved to Modal volume")
    
    return {
        'experiment': config['experiment']['name'],
        'best_val_loss': trainer.best_val_loss,
        'checkpoint_dir': config['checkpoint_dir']
    }


@app.function(
    image=image,
    gpu="A100-80GB",
    volumes={"/mnt": volume},
    timeout=60 * 60 * 24,
)
def train_multiple_models(config_paths: list[str]):
    """
    Train multiple models in parallel
    
    Usage:
        modal run src/modal_training.py::train_multiple_models \
          --config-paths '["configs/experiments/baseline.yaml", "configs/experiments/attention_unet.yaml"]'
    """
    print(f"🚀 Launching {len(config_paths)} parallel training jobs...")
    
    # Use .map() to train all models in parallel
    results = list(train_single_model.map(config_paths))
    
    print(f"\n{'='*60}")
    print(f"All Training Complete!")
    print(f"{'='*60}")
    
    for result in results:
        print(f"{result['experiment']}: Val Loss = {result['best_val_loss']:.4f}")
    
    return results


@app.function(
    image=image,
    volumes={MODEL_DIR: volume},
)
def download_models(local_dir: str = "models/from_modal"):
    """
    Download trained models from Modal volume to local machine
    
    Usage:
        modal run src/modal_training.py::download_models
    """
    from pathlib import Path
    import shutil
    
    model_dir = Path(MODEL_DIR) / "checkpoints"
    local_path = Path(local_dir)
    local_path.mkdir(parents=True, exist_ok=True)
    
    print(f"📥 Downloading models from Modal volume to {local_dir}...")
    
    if not model_dir.exists():
        print("❌ No models found in Modal volume")
        return
    
    # List all experiments
    exp_dirs = [d for d in model_dir.iterdir() if d.is_dir()]
    
    if not exp_dirs:
        print("❌ No experiment directories found")
        return
    
    print(f"Found {len(exp_dirs)} experiments to download")
    
    # Note: Actual file copying is handled by Modal CLI
    for exp_dir in exp_dirs:
        print(f"  📦 {exp_dir.name}/")
        for model_file in exp_dir.glob("*.pth"):
            print(f"     - {model_file.name}")
    
    print(f"\n✅ Ready to download. Modal CLI will handle the transfer.")


@app.local_entrypoint()
def main(
    command: str = "upload",
    config: str = "configs/experiments/baseline.yaml"
):
    """
    Main CLI interface for Modal training
    
    Usage:
        # Upload preprocessed data (run once)
        modal run src/modal_training.py --command upload
        
        # List data in volume
        modal run src/modal_training.py --command list
        
        # Train single model
        modal run src/modal_training.py --command train --config configs/experiments/baseline.yaml
        
        # Train multiple models in parallel
        modal run src/modal_training.py --command parallel
        
        # Download trained models
        modal run src/modal_training.py --command download
    """
    
    if command == "upload":
        print("📤 Uploading data to Modal...")
        upload_processed_data.remote()
        
    elif command == "list":
        list_data.remote()
        
    elif command == "train":
        print(f"🚀 Training model with config: {config}")
        result = train_single_model.remote(config)
        print(f"✅ Training complete! Best val loss: {result['best_val_loss']:.4f}")
        
    elif command == "parallel":
        configs = [
            "configs/experiments/baseline.yaml",
            "configs/experiments/attention_unet.yaml",
        ]
        print(f"🚀 Training {len(configs)} models in parallel...")
        train_multiple_models.remote(configs)
        
    elif command == "download":
        download_models.remote()
        
    else:
        print(f"❌ Unknown command: {command}")
        print("   Valid commands: upload, list, train, parallel, download")


if __name__ == "__main__":
    main()