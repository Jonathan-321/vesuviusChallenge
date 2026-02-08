#!/usr/bin/env python3
"""
Main training script for Vesuvius Challenge
Supports both local and Modal training

Usage:
    python train.py --config configs/experiments/baseline.yaml
    python -m src.training.train --config configs/experiments/attention_unet.yaml --resume checkpoint.pth
"""

import argparse
from pathlib import Path
import random

import torch
import yaml
import numpy as np

from ..data.dataset import get_dataloaders
from ..models import get_model
from .losses import get_loss_function
from .trainer import Trainer, get_optimizer, get_scheduler


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
    training_cfg.setdefault('seed', validation_cfg.get('seed', 42))

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

    # Set seeds for reproducible comparisons across experiments
    seed = int(config['training'].get('seed', config['validation'].get('seed', 42)))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

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

    # Set device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    # Get volume IDs for training/validation
    train_ids, val_ids = get_volume_ids(
        config['data']['processed_dir'],
        config['validation']['split_ratio'],
        config['validation']['seed']
    )

    print(f"\n📊 Dataset Split:")
    print(f"   Training volumes: {len(train_ids)}")
    print(f"   Validation volumes: {len(val_ids)}")
    print(f"   Train IDs: {train_ids[:3]}{'...' if len(train_ids) > 3 else ''}")
    print(f"   Val IDs: {val_ids[:3]}{'...' if len(val_ids) > 3 else ''}")

    # Create dataloaders
    train_loader, val_loader = get_dataloaders(config, train_ids, val_ids)

    print(f"\n📦 Data Loaders:")
    print(f"   Train batches: {len(train_loader)}")
    print(f"   Val batches: {len(val_loader)}")

    # Create model
    model = get_model(config)
    model = model.to(device)

    # Print model info
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n🧠 Model Info:")
    print(f"   Architecture: {config['model']['architecture']}")
    print(f"   Trainable parameters: {trainable_params:,}")

    # Create loss function
    loss_fn = get_loss_function(config)

    # Create optimizer and scheduler
    optimizer = get_optimizer(config, model)
    scheduler = get_scheduler(config, optimizer)

    print(f"\n⚙️ Training Setup:")
    print(f"   Optimizer: {config['training'].get('optimizer', 'adamw')}")
    print(f"   Scheduler: {config['training'].get('scheduler', 'cosine_warmup')}")
    print(f"   Mixed Precision: {config['training'].get('mixed_precision', True)}")

    # Create trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        loss_fn=loss_fn,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device
    )

    # Resume from checkpoint if provided
    if args.resume:
        print(f"\n🔄 Resuming from checkpoint: {args.resume}")
        trainer.load_checkpoint(args.resume)

    # Train model
    try:
        trainer.train()
    except KeyboardInterrupt:
        print("\n⚠️ Training interrupted. Saving checkpoint...")
        trainer.save_checkpoint('interrupted.pth')
        print("Checkpoint saved. Exiting.")
        return

    print(f"\n✅ Training completed!")
    print(f"   Best validation loss: {trainer.best_val_loss:.4f}")
    print(f"   Best model saved to: {trainer.best_model_path}")


if __name__ == '__main__':
    main()
