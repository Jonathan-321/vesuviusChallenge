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
        "tifffile>=2023.9.26",
        "scikit-image>=0.24.0",
    ])
    .add_local_dir(
        ".",
        remote_path="/workspace",
        ignore=[
            ".git",
            "venv",
            "data",
            "models",
            "logs",
            "__pycache__",
        ],
    )
)

# Mount points
DATA_DIR = "/mnt/data"
MODEL_DIR = "/mnt/models"


def upload_processed_data(local_path: str = "data/processed_3d"):
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
    if not local_path.exists():
        print(f"❌ Local path not found: {local_path}")
        return
    
    print(f"📤 Uploading processed data from {local_path} to Modal volume...")
    
    with volume.batch_upload(force=True) as batch:
        batch.put_directory(str(local_path), f"/data/{local_path.name}")

    print(f"✅ Upload complete to /data/{local_path.name} in Modal volume")


@app.function(
    image=image,
    volumes={"/mnt": volume},
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
    
    processed_dir = data_path / "processed_3d"
    if processed_dir.exists():
        files = sorted(list(processed_dir.glob("*.npz"))) + sorted(list((processed_dir / "images").glob("*.npz"))) + sorted(list((processed_dir / "masks").glob("*.npz")))
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
    import torch
    from pathlib import Path
    
    # Add code path mounted into the image
    sys.path.append("/workspace")
    print(f"Python path: {sys.path}")
    print(f"Contents of /workspace: {os.listdir('/workspace')}")
    
    # Import our modules from mounted volume
    from src.models import get_model
    from src.data.dataset import get_dataloaders
    from src.training.trainer import Trainer, get_optimizer, get_scheduler
    from src.training.losses import get_loss_function
    
    print(f"🚀 Starting training on A100-80GB")
    print(f"   Config: {config_path}")
    
    # Load config from mounted volume
    from train import load_config

    config_path = f"/workspace/{config_path}"
    config = load_config(config_path)
    
    # Update paths for Modal
    data_dir = Path("/mnt") / config['data']['processed_dir']
    config['data']['processed_dir'] = str(data_dir)
    config.setdefault('paths', {})
    config['paths']['checkpoint_dir'] = f"{MODEL_DIR}/checkpoints/{config['experiment']['name']}"
    config['checkpoint_dir'] = config['paths']['checkpoint_dir']
    
    # Create checkpoint directory
    Path(config['checkpoint_dir']).mkdir(parents=True, exist_ok=True)

    # Tee stdout/stderr to a log file on the volume for postmortem debugging.
    log_path = Path(config['checkpoint_dir']) / "train.log"
    log_file = open(log_path, "w", buffering=1)

    class _Tee:
        def __init__(self, *streams):
            self.streams = streams

        def write(self, data):
            for stream in self.streams:
                stream.write(data)
                stream.flush()

        def flush(self):
            for stream in self.streams:
                stream.flush()

    sys.stdout = _Tee(sys.stdout, log_file)
    sys.stderr = _Tee(sys.stderr, log_file)
    
    print(f"\n{'='*60}")
    print(f"Experiment: {config['experiment']['name']}")
    print(f"{'='*60}")
    
    # Get volume IDs
    import numpy as np
    data_path = Path(config['data']['processed_dir'])
    volume_root = data_path / "images" if (data_path / "images").exists() else data_path
    volume_files = sorted(list(volume_root.glob('volume_*.npz')))
    
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
    volumes={"/mnt": volume},
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


@app.function(
    image=image,
    volumes={"/mnt": volume},
)
def show_train_log(experiment: str, lines: int = 200):
    """
    Show the tail of a training log from Modal volume

    Usage:
        modal run src/modal_training.py::show_train_log --experiment surface_unet3d_baseline
    """
    from pathlib import Path

    log_path = Path(MODEL_DIR) / "checkpoints" / experiment / "train.log"
    if not log_path.exists():
        print(f"❌ Log file not found: {log_path}")
        return

    log_lines = log_path.read_text().splitlines()
    tail = log_lines[-lines:] if lines > 0 else log_lines
    print("\n".join(tail))


@app.function(
    image=image,
    gpu="A100-80GB",
    volumes={"/mnt": volume},
)
def val_sweep(
    model_path: str = "/mnt/models/checkpoints/surface_unet3d_baseline/final_model.pth",
    config_path: str = "configs/experiments/surface_unet3d.yaml",
    max_volumes: int = 2,
    t_low_values: str = "0.3,0.4,0.5",
    t_high_values: str = "0.8,0.9",
    dust_values: str = "50,100,200",
    z_radius: int = 1,
    xy_radius: int = 0,
    use_tta: bool = True,
):
    """
    Quick val sweep over thresholds/postprocess on a subset of val volumes.

    Usage:
        modal run src/modal_training.py::val_sweep --model-path /mnt/models/checkpoints/.../final_model.pth
    """
    import numpy as np
    import torch
    import sys
    sys.path.append("/workspace")
    from train import load_config, get_volume_ids
    from src.inference.predict import VesuviusPredictor3D
    from src.inference.create_submission import topo_postprocess

    def _parse_floats(s: str):
        return [float(x) for x in s.split(",") if x]

    def _parse_ints(s: str):
        return [int(x) for x in s.split(",") if x]

    tl_values = _parse_floats(t_low_values)
    th_values = _parse_floats(t_high_values)
    dust_vals = _parse_ints(dust_values)

    cfg = load_config(f"/workspace/{config_path}")
    cfg['data']['processed_dir'] = str(Path("/mnt") / cfg['data']['processed_dir'])
    train_ids, val_ids = get_volume_ids(
        cfg['data']['processed_dir'],
        cfg['validation']['split_ratio'],
        cfg['validation']['seed'],
    )
    val_ids = val_ids[:max_volumes]

    predictor = VesuviusPredictor3D(
        model_path=model_path,
        device="cuda" if torch.cuda.is_available() else "cpu",
        roi_size=tuple(cfg['data']['patch_size']),
        overlap=0.5,
        class_index=1,
        config_path=f"/workspace/{config_path}",
    )

    results = []
    for vid in val_ids:
        vol_path = Path(cfg['data']['processed_dir']) / "images" / f"volume_{vid}.npz"
        mask_path = Path(cfg['data']['processed_dir']) / "masks" / f"mask_{vid}.npz"
        volume = np.load(vol_path)['data']
        mask = np.load(mask_path)['data']
        print(f"\n=== Val volume {vid} shape {volume.shape} ===")
        probs = predictor.predict_volume_tta(volume) if use_tta else predictor.predict_volume(volume)
        gt = (mask == 1).astype(np.uint8)

        for tl in tl_values:
            for th in th_values:
                for dust in dust_vals:
                    pred = topo_postprocess(
                        probs,
                        t_low=tl,
                        t_high=th,
                        z_radius=z_radius,
                        xy_radius=xy_radius,
                        dust_min_size=dust,
                    )
                    inter = (pred & gt).sum()
                    dice = (2 * inter) / (pred.sum() + gt.sum() + 1e-6)
                    results.append((vid, tl, th, dust, dice))
                    print(f"vid {vid} tl {tl} th {th} dust {dust} -> Dice {dice:.4f}")

    best = sorted(results, key=lambda x: x[-1], reverse=True)[:5]
    print("\nTop results:")
    for vid, tl, th, dust, dice in best:
        print(f"vid {vid} tl {tl} th {th} dust {dust} -> Dice {dice:.4f}")

    # Persist summary to volume for easy retrieval
    summary_path = Path(MODEL_DIR) / "val_sweep_results.txt"
    with summary_path.open("a") as f:
        f.write(f"\nRun model {model_path}, config {config_path}, vols {val_ids}\n")
        for vid, tl, th, dust, dice in results:
            f.write(f"{vid}, tl={tl}, th={th}, dust={dust}, dice={dice:.6f}\n")
        f.write("Top:\n")
        for vid, tl, th, dust, dice in best:
            f.write(f"{vid}, tl={tl}, th={th}, dust={dust}, dice={dice:.6f}\n")
    print(f"\nResults written to {summary_path}")


@app.local_entrypoint()
def main(
    command: str = "upload",
    config: str = "configs/experiments/baseline.yaml",
    experiment: str = ""
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

        # Show training log (by experiment name)
        modal run src/modal_training.py --command show_log --experiment surface_unet3d_baseline
    """
    
    if command == "upload":
        print("📤 Uploading data to Modal...")
        upload_processed_data()
        
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

    elif command == "show_log":
        if not experiment:
            print("❌ Missing --experiment for show_log")
            return
        show_train_log.remote(experiment)
        
    else:
        print(f"❌ Unknown command: {command}")
        print("   Valid commands: upload, list, train, parallel, download")


if __name__ == "__main__":
    main()
