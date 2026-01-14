"""
Simplified Modal training script that runs everything in one file
"""
import modal

app = modal.App("vesuvius-training-simple")

# Modal volume for data
volume = modal.Volume.from_name("vesuvius-data", create_if_missing=True)

# Since all code is in the volume, we don't need local mount anymore

# Docker image
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
        "matplotlib>=3.7.0",
        "pandas>=2.0.0",
        "scikit-learn>=1.3.0",
    ])
)


@app.function(
    image=image,
    gpu="A100-80GB",
    volumes={"/mnt": volume},
    timeout=60 * 60 * 24,
)
def train_model(config_name: str = "baseline"):
    """
    Train a model with all code mounted
    """
    import sys
    import os
    import torch
    
    # Add volume directory to path
    sys.path.append("/mnt")
    
    # Now we can import our modules
    from train import main as train_main
    import argparse
    
    # Create args for train.py
    args = argparse.Namespace(
        config=f"/mnt/configs/experiments/{config_name}.yaml",
        resume=None,
        device='cuda',
        wandb=False
    )
    
    # Update data paths to use Modal volume
    os.environ['DATA_DIR'] = '/mnt/data'
    
    # Import and modify config loading to use Modal paths
    import yaml
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Update paths
    config['data']['processed_dir'] = '/mnt/data/processed'
    config['paths']['checkpoint_dir'] = f'/mnt/models/checkpoints/{config["experiment"]["name"]}'
    
    # Save modified config
    import tempfile
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        yaml.dump(config, f)
        args.config = f.name
    
    print("🚀 Starting training on Modal A100")
    print(f"   Config: {config_name}")
    print(f"   GPU: {torch.cuda.get_device_name(0)}")
    print(f"   Data dir: {config['data']['processed_dir']}")
    
    # Run training
    train_main(args)
    
    return f"Training complete for {config_name}"


@app.local_entrypoint()
def main(config: str = "baseline"):
    """
    Entry point
    
    Usage:
        modal run modal_train_simple.py --config baseline
    """
    print(f"🚀 Launching training for config: {config}")
    result = train_model.remote(config)
    print(f"✅ {result}")


if __name__ == "__main__":
    main()