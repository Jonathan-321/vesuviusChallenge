# Vesuvius Challenge - Surface Detection Pipeline

A comprehensive deep learning solution for the Vesuvius Challenge surface detection competition using Modal for cloud GPU training.

## 🚀 Quick Start

```bash
# 1. Setup environment
bash setup_env.sh
source venv/bin/activate

# 2. Configure API keys in .env
# Edit .env and add your Kaggle, WandB, and Modal credentials

# 3. Download competition data (surface detection track)
kaggle competitions download -c vesuvius-challenge-surface-detection
unzip vesuvius-challenge-surface-detection.zip -d data/raw/

# 4. Preprocess data
python scripts/preprocessing/prepare_data.py

# 5. Train baseline model
python train.py --config configs/experiments/baseline.yaml
```

## 🧯 Troubleshooting

- If you see `ValueError: <COMPRESSION.LZW: 5> requires the 'imagecodecs' package`,
  install the dependency: `python -m pip install imagecodecs`.
- If you see `sliding_window_inference() got multiple values for argument 'sw_batch_size'`,
  pull the latest repo version (the inference path was updated to call MONAI directly).

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
