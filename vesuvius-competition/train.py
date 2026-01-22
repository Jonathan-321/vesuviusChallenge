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


def _deep_merge(base, override):
    """Recursively merge two dictionaries"""
    merged = dict(base)
    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _normalize_config(config, config_path):
    """Fill in missing fields and normalize naming across configs"""
    config.setdefault('project', {'name': 'vesuvius-challenge'})
    config.setdefault('experiment', {
        'name': Path(config_path).stem,
        'description': ''
    })

    data_cfg = config.setdefault('data', {})
    data_cfg.setdefault('raw_dir', 'data/raw')
    data_cfg.setdefault('processed_dir', 'data/processed')
    data_cfg.setdefault('cache_dir', 'data/cache')
    data_cfg.setdefault('patch_size', [32, 128, 128])
    data_cfg.setdefault('num_workers', 4)
    data_cfg.setdefault('samples_per_volume', 500)

    training_cfg = config.setdefault('training', {})
    if 'epochs' not in training_cfg:
        training_cfg['epochs'] = training_cfg.get('num_epochs', 50)
    training_cfg['optimizer'] = str(training_cfg.get('optimizer', 'adamw')).lower()
    scheduler_raw = str(training_cfg.get('scheduler', 'cosine_warmup')).lower()
    scheduler_map = {
        'cosineannealinglr': 'cosine',
        'cosineannealingwarmrestarts': 'cosine_warmup',
        'cosineannealing': 'cosine',
        'cosine': 'cosine',
    }
    training_cfg['scheduler'] = scheduler_map.get(scheduler_raw, scheduler_raw)
    training_cfg.setdefault('learning_rate', training_cfg.get('lr', 1e-3))
    training_cfg.setdefault('weight_decay', 1e-4)
    training_cfg.setdefault('batch_size', 8)
    training_cfg.setdefault(
        'gradient_accumulation',
        training_cfg.pop('accumulate_grad_batches', 1)
    )
    training_cfg.setdefault(
        'num_workers',
        training_cfg.get('num_workers', data_cfg.get('num_workers', 4))
    )
    training_cfg.setdefault('mixed_precision', True)

    validation_cfg = config.setdefault('validation', {})
    validation_cfg.setdefault('split_ratio', 0.15)
    validation_cfg.setdefault('seed', 42)

    logging_cfg = config.setdefault('logging', {})
    logging_cfg.setdefault('use_wandb', False)
    logging_cfg.setdefault('log_interval', 100)
    logging_cfg.setdefault('save_interval', 5)

    loss_cfg = config.setdefault('loss', {})
    if 'type' not in loss_cfg and 'name' in loss_cfg:
        loss_cfg['type'] = loss_cfg['name']
    loss_cfg.setdefault('type', 'combined')

    model_cfg = config.setdefault('model', {})
    model_cfg.setdefault('architecture', 'unet')
    model_cfg.setdefault('spatial_dims', 2)
    model_cfg.setdefault('encoder', 'resnet34')
    model_cfg.setdefault('encoder_weights', model_cfg.get('encoder_weights', 'imagenet'))
    patch_depth = data_cfg.get('patch_size', [32, 128, 128])[0]
    if str(model_cfg.get('spatial_dims', 2)) == '2':
        if isinstance(patch_depth, int):
            if model_cfg.get('in_channels') != patch_depth:
                model_cfg['in_channels'] = patch_depth
        else:
            model_cfg.setdefault('in_channels', 32)
    else:
        model_cfg.setdefault('in_channels', 1)
    model_cfg.setdefault('out_channels', 1)

    return config


def load_config(config_path):
    """Load configuration from YAML file with base/default support"""
    config_path = Path(config_path)
    with open(config_path, 'r') as f:
        experiment_config = yaml.safe_load(f) or {}

    config_root = config_path.resolve().parent.parent

    # Handle Hydra-style defaults: e.g., defaults: [base/default]
    defaults = experiment_config.pop('defaults', [])
    base_config = {}
    for default in defaults:
        if isinstance(default, str):
            default_path = config_root / f"{default}.yaml"
        elif isinstance(default, dict) and len(default) == 1:
            name = list(default.values())[0]
            default_path = config_root / f"{name}.yaml"
        else:
            continue

        if default_path.exists():
            with open(default_path, 'r') as f:
                base_config = _deep_merge(base_config, yaml.safe_load(f) or {})

    # Fallback to a single base file if no defaults were provided
    if not base_config:
        fallback_base = config_root / 'base/default.yaml'
        if fallback_base.exists():
            with open(fallback_base, 'r') as f:
                base_config = yaml.safe_load(f) or {}

    merged_config = _deep_merge(base_config, experiment_config)
    merged_config = _normalize_config(merged_config, config_path)
    return merged_config


def get_volume_ids(data_dir, split_ratio=0.15, seed=42):
    """
    Get train and validation volume IDs
    """
    import numpy as np
    
    data_path = Path(data_dir)
    volume_root = data_path / "images" if (data_path / "images").exists() else data_path
    volume_files = sorted(list(volume_root.glob('volume_*.npz')))
    
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
    
    # Set device with MPS fallback
    if args.device == 'cuda' and torch.cuda.is_available():
        device = torch.device('cuda')
    elif args.device in ('mps', 'cuda') and torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    print(f"🖥️  Device: {device}")
    if device.type == 'cuda':
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
        print(f"   VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB")
    elif device.type == 'mps':
        print("   Using Apple MPS backend")
    
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
