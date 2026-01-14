#!/usr/bin/env python3
"""
Main training script for Vesuvius Challenge
Supports both local and Modal training

Usage:
    python train.py --config configs/experiments/baseline.yaml
    python train.py --config configs/experiments/attention_unet.yaml --resume checkpoint.pth
"""

import argparse
import yaml
import torch
from pathlib import Path
import sys

# Add src to path
sys.path.append(str(Path(__file__).parent))

from src.models import get_model
from src.data.dataset import get_dataloaders
from src.training.trainer import Trainer, get_optimizer, get_scheduler
from src.training.losses import get_loss_function


def load_config(config_path):
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Merge with base config
    base_config_path = Path('configs/config.yaml')
    if base_config_path.exists():
        with open(base_config_path, 'r') as f:
            base_config = yaml.safe_load(f)
        
        # Merge configs (experiment config overrides base)
        for key in base_config:
            if key not in config:
                config[key] = base_config[key]
            elif isinstance(base_config[key], dict):
                for subkey in base_config[key]:
                    if subkey not in config[key]:
                        config[key][subkey] = base_config[key][subkey]
    
    return config


def get_volume_ids(data_dir, split_ratio=0.15, seed=42):
    """
    Get train and validation volume IDs
    """
    import numpy as np
    
    data_path = Path(data_dir)
    volume_files = sorted(list(data_path.glob('volume_*.npz')))
    
    if not volume_files:
        raise FileNotFoundError(f"No volume files found in {data_dir}")
    
    # Extract IDs
    volume_ids = [f.stem.replace('volume_', '') for f in volume_files]
    
    # Shuffle with fixed seed
    np.random.seed(seed)
    shuffled_ids = np.random.permutation(volume_ids)
    
    # Split
    n_val = max(1, int(len(shuffled_ids) * split_ratio))
    val_ids = shuffled_ids[:n_val].tolist()
    train_ids = shuffled_ids[n_val:].tolist()
    
    return train_ids, val_ids


def main(args=None):
    if args is None:
        parser = argparse.ArgumentParser(description='Train Vesuvius model')
        parser.add_argument(
            '--config',
            type=str,
            required=True,
            help='Path to experiment config file'
        )
        parser.add_argument(
            '--resume',
            type=str,
            default=None,
            help='Path to checkpoint to resume from'
        )
        parser.add_argument(
            '--device',
            type=str,
            default='cuda',
            help='Device to use (cuda/cpu)'
        )
        parser.add_argument(
            '--wandb',
            action='store_true',
            help='Enable WandB logging'
        )
        
        args = parser.parse_args()
    
    # Load config
    print(f"📄 Loading config from {args.config}")
    config = load_config(args.config)
    
    # Override config with CLI args
    if args.wandb:
        config['logging']['use_wandb'] = True
    
    config['training']['device'] = args.device
    
    # Print configuration
    print(f"\n{'='*60}")
    print(f"Experiment: {config['experiment']['name']}")
    print(f"Description: {config['experiment']['description']}")
    print(f"{'='*60}")
    print(f"Model: {config['model']['architecture']}")
    print(f"Patch Size: {config['data']['patch_size']}")
    print(f"Batch Size: {config['training']['batch_size']}")
    print(f"Epochs: {config['training']['epochs']}")
    print(f"Learning Rate: {config['training']['learning_rate']}")
    print(f"{'='*60}\n")
    
    # Set device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"🖥️  Device: {device}")
    if device.type == 'cuda':
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
        print(f"   VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB")
    
    # Get volume IDs
    data_dir = Path(config['data']['processed_dir'])
    train_ids, val_ids = get_volume_ids(
        data_dir,
        config['validation']['split_ratio'],
        config['validation']['seed']
    )
    
    print(f"\n📊 Dataset Split:")
    print(f"   Training volumes: {len(train_ids)}")
    print(f"   Validation volumes: {len(val_ids)}")
    print(f"   Train IDs: {train_ids[:3]}{'...' if len(train_ids) > 3 else ''}")
    print(f"   Val IDs: {val_ids}")
    
    # Create dataloaders
    print(f"\n📦 Creating dataloaders...")
    train_loader, val_loader = get_dataloaders(config, train_ids, val_ids)
    
    print(f"   Train batches: {len(train_loader)}")
    print(f"   Val batches: {len(val_loader)}")
    
    # Create model
    print(f"\n🏗️  Building model...")
    model = get_model(config)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"   Total parameters: {total_params:,}")
    print(f"   Trainable parameters: {trainable_params:,}")
    
    # Create loss, optimizer, scheduler
    criterion = get_loss_function(config)
    optimizer = get_optimizer(model, config)
    scheduler = get_scheduler(optimizer, config)
    
    print(f"\n⚙️  Training Setup:")
    print(f"   Loss: {config['loss']['type']}")
    print(f"   Optimizer: {config['training'].get('optimizer', 'adamw')}")
    print(f"   Scheduler: {config['training'].get('scheduler', 'cosine_warmup')}")
    
    # Create checkpoint directory
    exp_name = config['experiment']['name']
    checkpoint_dir = Path('models/experiments') / exp_name
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    config['checkpoint_dir'] = str(checkpoint_dir)
    
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
    
    # Resume from checkpoint if specified
    if args.resume:
        print(f"\n🔄 Resuming from checkpoint: {args.resume}")
        trainer.load_checkpoint(args.resume)
    
    # Train
    try:
        trainer.train()
    except KeyboardInterrupt:
        print("\n⚠️  Training interrupted by user")
        print("Saving checkpoint...")
        trainer.save_checkpoint('interrupted.pth')
    except Exception as e:
        print(f"\n❌ Training failed with error: {e}")
        import traceback
        traceback.print_exc()
        raise
    
    print("\n✅ Training complete!")
    print(f"   Best model saved to: {checkpoint_dir / 'best_model.pth'}")
    print(f"   Best validation loss: {trainer.best_val_loss:.4f}")


if __name__ == '__main__':
    main()