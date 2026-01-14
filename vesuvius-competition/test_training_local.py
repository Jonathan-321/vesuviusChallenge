#!/usr/bin/env python3
"""
Test training locally with a small subset
"""
import sys
import torch
from pathlib import Path

# Add current directory to path
sys.path.append(str(Path(__file__).parent))

# Test imports
print("Testing imports...")
try:
    from src.models import get_model
    from src.data.dataset import get_dataloaders
    from src.training.losses import get_loss_function
    print("✅ All imports successful!")
except Exception as e:
    print(f"❌ Import error: {e}")
    sys.exit(1)

# Test data loading
print("\nTesting data loading...")
config = {
    'data': {
        'processed_dir': 'data/processed',
        'patch_size': [32, 64, 64],  # Smaller for testing
        'samples_per_volume': 10,  # Just 10 samples per volume
        'num_workers': 0
    },
    'model': {
        'architecture': 'unet',
        'encoder': 'resnet18',  # Smaller encoder
        'encoder_weights': None,  # No pretrained weights
        'in_channels': 32,  # Using 32 slices
        'out_channels': 1,
        'activation': None
    },
    'training': {
        'batch_size': 2,
        'num_workers': 0
    },
    'loss': {
        'type': 'dice'
    }
}

# Get available volumes
from pathlib import Path
data_path = Path(config['data']['processed_dir'])
volume_files = sorted(list(data_path.glob('volume_*.npz')))
volume_ids = [f.stem.replace('volume_', '') for f in volume_files if 'chunk' not in f.stem and 'large' not in f.stem]

print(f"Found {len(volume_ids)} volumes: {volume_ids[:5]}...")

# Split data
train_ids = volume_ids[:8]
val_ids = volume_ids[8:10]

print(f"Train: {len(train_ids)} volumes, Val: {len(val_ids)} volumes")

# Create model
print("\nCreating model...")
model = get_model(config)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
print(f"Model on device: {device}")
print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")

# Create loss
criterion = get_loss_function(config)

# Test forward pass
print("\nTesting forward pass...")
dummy_input = torch.randn(2, 32, 64, 64).to(device)  # batch_size=2, channels=32, H=64, W=64
with torch.no_grad():
    output = model(dummy_input)
    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
    
    # Test loss
    dummy_target = torch.randint(0, 2, (2, 1, 64, 64)).float().to(device)
    loss = criterion(output, dummy_target)
    print(f"Loss value: {loss.item():.4f}")

print("\n✅ All tests passed! Ready to train on Modal.")

# Try actual data loading
print("\nTesting actual data loading...")
try:
    train_loader, val_loader = get_dataloaders(config, train_ids[:2], val_ids[:1])
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    
    # Get one batch
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        print(f"\nBatch {batch_idx}:")
        print(f"  Inputs shape: {inputs.shape}")
        print(f"  Targets shape: {targets.shape}")
        print(f"  Input range: [{inputs.min():.3f}, {inputs.max():.3f}]")
        print(f"  Target unique values: {torch.unique(targets).tolist()}")
        break
        
except Exception as e:
    print(f"❌ Data loading error: {e}")
    import traceback
    traceback.print_exc()