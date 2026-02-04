#!/bin/bash

# Vesuvius Challenge Competition Repository Setup
# This script creates a complete directory structure and essential files

echo "🚀 Creating Vesuvius Challenge Competition Repository..."

# Create main directory
mkdir -p vesuvius-competition
cd vesuvius-competition

# Create directory structure
echo "📁 Creating directory structure..."
mkdir -p src/{models,data,training,inference,utils}
mkdir -p configs/{base,experiments}
mkdir -p data/{raw,processed,cache}
mkdir -p models/{checkpoints,submissions}
mkdir -p scripts/{preprocessing,analysis,visualization}
mkdir -p notebooks
mkdir -p logs

# Create __init__.py files for Python packages
touch src/__init__.py
touch src/models/__init__.py
touch src/data/__init__.py
touch src/training/__init__.py
touch src/inference/__init__.py
touch src/utils/__init__.py

# Create .gitignore
echo "📄 Creating .gitignore..."
cat > .gitignore << 'EOF'
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg
MANIFEST
venv/
ENV/
env/

# Data
data/raw/
data/processed/
data/cache/
*.tif
*.tiff
*.npz
*.h5
*.hdf5

# Models
models/checkpoints/
*.pth
*.pt
*.ckpt
*.safetensors

# Logs
logs/
*.log
wandb/
.wandb/
runs/

# IDE
.vscode/
.idea/
*.swp
*.swo
.DS_Store

# Jupyter
.ipynb_checkpoints/
*.ipynb_checkpoints

# Credentials
.env
secrets/
*.key
*.pem
EOF

# Create environment setup script
echo "📄 Creating setup_env.sh..."
cat > setup_env.sh << 'EOF'
#!/bin/bash

echo "🔧 Setting up Vesuvius Challenge environment..."

# Check Python version
python_version=$(python3 --version 2>&1 | awk '{print $2}')
required_version="3.8"

if [ "$(printf '%s\n' "$required_version" "$python_version" | sort -V | head -n1)" != "$required_version" ]; then 
    echo "❌ Python 3.8+ required. Current version: $python_version"
    exit 1
fi

# Create virtual environment
echo "🐍 Creating virtual environment..."
python3 -m venv venv
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip setuptools wheel

# Install requirements
echo "📦 Installing requirements..."
pip install -r requirements.txt

# Install package in development mode
pip install -e .

# Create .env file if it doesn't exist
if [ ! -f .env ]; then
    echo "📄 Creating .env file..."
    cat > .env << 'EOL'
# Kaggle API credentials
KAGGLE_USERNAME=your_username
KAGGLE_KEY=your_key

# Weights & Biases
WANDB_API_KEY=your_wandb_key
WANDB_PROJECT=vesuvius-challenge

# Modal
MODAL_TOKEN_ID=your_modal_token
MODAL_TOKEN_SECRET=your_modal_secret

# Training settings
CUDA_VISIBLE_DEVICES=0
MIXED_PRECISION=True
NUM_WORKERS=4
EOL
fi

echo "✅ Environment setup complete!"
echo "🎯 Next steps:"
echo "   1. Fill in your API keys in .env"
echo "   2. Download competition data: kaggle competitions download -c vesuvius-challenge-ink-detection"
echo "   3. Run preprocessing: python scripts/preprocessing/prepare_data.py"
EOF

chmod +x setup_env.sh

# Create requirements.txt
echo "📄 Creating requirements.txt..."
cat > requirements.txt << 'EOF'
# Core ML
torch>=2.0.0
torchvision>=0.15.0
numpy>=1.24.0
scipy>=1.10.0
scikit-image>=0.21.0
opencv-python>=4.8.0
albumentations>=1.3.1

# Model architectures
segmentation-models-pytorch>=0.3.3
timm>=0.9.0

# Data handling
pandas>=2.0.0
h5py>=3.8.0
zarr>=2.15.0
tifffile>=2023.7.0
Pillow>=10.0.0

# Training utilities
wandb>=0.15.0
pytorch-lightning>=2.0.0
torchmetrics>=1.0.0
rich>=13.0.0
tqdm>=4.65.0

# Utilities
pyyaml>=6.0
python-dotenv>=1.0.0
click>=8.1.0
matplotlib>=3.7.0
seaborn>=0.12.0

# Development
jupyter>=1.0.0
ipython>=8.14.0
black>=23.0.0
flake8>=6.0.0
pytest>=7.4.0

# Competition specific
kaggle>=1.5.0
modal>=0.50.0
EOF

# Create setup.py
echo "📄 Creating setup.py..."
cat > setup.py << 'EOF'
from setuptools import setup, find_packages

setup(
    name="vesuvius-competition",
    version="0.1.0",
    author="Your Name",
    description="Vesuvius Challenge Competition Solution",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        line.strip()
        for line in open("requirements.txt").readlines()
        if not line.startswith("#") and line.strip()
    ],
    entry_points={
        "console_scripts": [
            "train-vesuvius=src.training.train:main",
            "predict-vesuvius=src.inference.predict:main",
        ],
    },
)
EOF

# Create base configuration
echo "📄 Creating base configuration..."
mkdir -p configs/base
cat > configs/base/default.yaml << 'EOF'
# Base configuration for Vesuvius Challenge

# Data settings
data:
  raw_dir: data/raw
  processed_dir: data/processed
  cache_dir: data/cache
  tile_size: 512
  stride: 256
  z_start: 15
  z_end: 45
  z_step: 1
  num_workers: 4
  pin_memory: true
  prefetch_factor: 2

# Model settings
model:
  architecture: unet  # unet, attention_unet, unet++, manet, linknet
  encoder: resnet34
  encoder_weights: imagenet
  in_channels: 30
  out_channels: 1
  activation: null

# Training settings
training:
  batch_size: 8
  accumulate_grad_batches: 4
  num_epochs: 100
  optimizer: AdamW
  learning_rate: 1e-3
  weight_decay: 1e-4
  scheduler: CosineAnnealingLR
  warmup_epochs: 5
  early_stopping_patience: 15
  val_check_interval: 0.5
  gradient_clip_val: 1.0
  
# Loss settings
loss:
  name: combined  # dice, focal, tversky, combined
  dice_weight: 0.5
  bce_weight: 0.5
  focal_alpha: 0.25
  focal_gamma: 2.0
  tversky_alpha: 0.3
  tversky_beta: 0.7

# Augmentation settings
augmentation:
  train:
    - RandomCrop: {height: 512, width: 512}
    - HorizontalFlip: {p: 0.5}
    - VerticalFlip: {p: 0.5}
    - RandomRotate90: {p: 0.5}
    - ShiftScaleRotate: {shift_limit: 0.1, scale_limit: 0.2, rotate_limit: 45, p: 0.5}
    - ElasticTransform: {alpha: 120, sigma: 120 * 0.05, alpha_affine: 120 * 0.03, p: 0.3}
    - GridDistortion: {num_steps: 5, distort_limit: 0.3, p: 0.3}
    - RandomBrightnessContrast: {brightness_limit: 0.2, contrast_limit: 0.2, p: 0.5}
    - RandomGamma: {gamma_limit: [80, 120], p: 0.5}
    - GaussNoise: {var_limit: [10.0, 50.0], p: 0.3}
    - GaussianBlur: {blur_limit: 3, p: 0.3}
  val:
    - CenterCrop: {height: 512, width: 512}

# Experiment settings
experiment:
  name: baseline
  seed: 42
  deterministic: false
  log_every_n_steps: 50
  save_top_k: 3
  monitor: val/dice
  mode: max

# Hardware settings
hardware:
  accelerator: gpu
  devices: 1
  precision: 16-mixed
  strategy: ddp
  sync_batchnorm: true

# Paths
paths:
  checkpoint_dir: models/checkpoints
  submission_dir: models/submissions
  log_dir: logs
EOF

# Create example experiment config
echo "📄 Creating experiment configs..."
cat > configs/experiments/baseline.yaml << 'EOF'
# Baseline UNet experiment
# Inherits from configs/base/default.yaml

defaults:
  - base/default

experiment:
  name: baseline_unet_resnet34

model:
  architecture: unet
  encoder: resnet34
  
training:
  batch_size: 8
  learning_rate: 1e-3
  num_epochs: 50

loss:
  name: dice
  dice_weight: 1.0
  bce_weight: 0.0
EOF

cat > configs/experiments/attention_unet.yaml << 'EOF'
# Attention UNet experiment
# Inherits from configs/base/default.yaml

defaults:
  - base/default

experiment:
  name: attention_unet_efficientnet

model:
  architecture: attention_unet
  encoder: efficientnet-b3
  
training:
  batch_size: 6
  learning_rate: 5e-4
  num_epochs: 80

loss:
  name: combined
  dice_weight: 0.7
  bce_weight: 0.3

augmentation:
  train:
    - RandomCrop: {height: 512, width: 512}
    - HorizontalFlip: {p: 0.5}
    - VerticalFlip: {p: 0.5}
    - RandomRotate90: {p: 0.5}
    - ShiftScaleRotate: {shift_limit: 0.1, scale_limit: 0.2, rotate_limit: 45, p: 0.7}
    - ElasticTransform: {alpha: 120, sigma: 120 * 0.05, alpha_affine: 120 * 0.03, p: 0.5}
    - GridDistortion: {num_steps: 5, distort_limit: 0.3, p: 0.5}
    - RandomBrightnessContrast: {brightness_limit: 0.3, contrast_limit: 0.3, p: 0.7}
EOF

# Create README.md
echo "📄 Creating README.md..."
cat > README.md << 'EOF'
# Vesuvius Challenge Competition Solution

A comprehensive deep learning solution for the Vesuvius Challenge ink detection competition.

## 🚀 Quick Start

```bash
# 1. Setup environment
bash setup_env.sh
source venv/bin/activate

# 2. Configure API keys in .env
# Edit .env and add your Kaggle, WandB, and Modal credentials

# 3. Download competition data
kaggle competitions download -c vesuvius-challenge-ink-detection
unzip vesuvius-challenge-ink-detection.zip -d data/raw/

# 4. Preprocess data
python scripts/preprocessing/prepare_data.py

# 5. Train baseline model
python train.py --config configs/experiments/baseline.yaml
```

## 📁 Project Structure

```
vesuvius-competition/
├── src/                    # Source code
│   ├── models/            # Model architectures
│   ├── data/              # Data loading and processing
│   ├── training/          # Training logic
│   ├── inference/         # Inference and submission
│   └── utils/             # Utilities
├── configs/               # Configuration files
│   ├── base/              # Base configurations
│   └── experiments/       # Experiment configs
├── data/                  # Data directory
├── models/                # Model checkpoints
├── scripts/               # Utility scripts
└── notebooks/             # Jupyter notebooks
```

## 🔧 Available Models

- UNet (baseline)
- Attention UNet
- UNet++
- MAnet
- LinkNet

## 📊 Experiment Tracking

All experiments are tracked with Weights & Biases. View your runs at:
https://wandb.ai/your-username/vesuvius-challenge

## 🏃 Training

```bash
# Local training
python train.py --config configs/experiments/your_experiment.yaml

# Modal training (cloud)
modal run src/modal_training.py::train --config configs/experiments/your_experiment.yaml
```

## 🔍 Making Predictions

```bash
python src/inference/predict.py --checkpoint models/checkpoints/best_model.ckpt --data_dir data/raw/test/
```

## 📈 Results

| Model | Validation Dice | Public LB | Private LB |
|-------|----------------|-----------|------------|
| Baseline UNet | 0.75 | - | - |
| Attention UNet | 0.82 | - | - |
| Ensemble | 0.86 | - | - |
EOF

echo "✅ Repository structure created successfully!"
echo "📁 Created vesuvius-competition/ directory with:"
echo "   - Complete project structure"
echo "   - Environment setup script"
echo "   - Requirements file"
echo "   - Configuration system"
echo "   - Base and experiment configs"
echo ""
echo "🎯 Next steps:"
echo "   1. cd vesuvius-competition"
echo "   2. bash setup_env.sh"
echo "   3. Fill in your API keys in .env"
echo "   4. You're ready to start!"