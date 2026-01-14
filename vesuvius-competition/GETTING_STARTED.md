# Getting Started - Vesuvius Challenge Competition

## 🚀 Rapid Setup (15 minutes)

### Step 1: Setup Repository

```bash
# Navigate to project
cd vesuvius-competition

# Setup environment
bash setup_env.sh

# Activate environment
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### Step 2: Download Competition Data

**Option A: Using Kaggle CLI (Recommended)**
```bash
# Setup Kaggle credentials
mkdir -p ~/.kaggle
# Download kaggle.json from https://www.kaggle.com/settings
mv ~/Downloads/kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json

# Download competition data
kaggle competitions download -c vesuvius-challenge-ink-detection
unzip vesuvius-challenge-ink-detection.zip -d data/raw/
```

**Option B: Manual Download**
1. Go to https://www.kaggle.com/competitions/vesuvius-challenge-ink-detection/data
2. Download all files
3. Extract to `data/raw/`

### Step 3: Preprocess Data

```bash
# Create processed .npz files from raw TIFFs
python scripts/preprocessing/prepare_data.py

# Verify processed data
python -c "from pathlib import Path; print(f'Processed volumes: {len(list(Path(\"data/processed\").glob(\"volume_*.npz\")))}')"
```

### Step 4: Train Your First Model

```bash
# Train baseline U-Net
python train.py --config configs/experiments/baseline.yaml

# Train with WandB tracking
python train.py --config configs/experiments/baseline.yaml --wandb
```

---

## 📊 Monitor Training

### Using WandB (Recommended)
```bash
# Login to WandB
wandb login

# Your experiments will appear at:
# https://wandb.ai/your_username/vesuvius-challenge
```

### Using TensorBoard
```bash
# In a separate terminal
tensorboard --logdir logs/

# Open http://localhost:6006
```

---

## 🔬 Run Experiments

### Single Experiment
```bash
python train.py --config configs/experiments/attention_unet.yaml
```

### Multiple Experiments in Parallel
```bash
# Edit run_experiments.sh with your desired configs
bash scripts/run_experiments.sh
```

### Using Modal (A100 GPUs)
```bash
# Setup Modal
modal token new

# Upload data to Modal
python scripts/upload_to_modal.py

# Train on A100
modal run src/modal_training.py::train_single_model \
  --config configs/experiments/attention_unet.yaml
```

---

## 📈 Expected Results Timeline

### Day 1: Baseline
```bash
python train.py --config configs/experiments/baseline.yaml
```
- **Expected**: 0.70-0.75 Dice
- **Time**: 6-8 hours on local GPU
- **Time**: 2-3 hours on A100

### Day 3: Improved Architecture
```bash
python train.py --config configs/experiments/attention_unet.yaml
```
- **Expected**: 0.77-0.80 Dice
- **Improvements**: Attention mechanism, better augmentation

### Week 1: Multiple Models
```bash
# Launch 3-5 different architectures
bash scripts/run_experiments.sh
```
- **Expected**: Best model 0.80-0.83 Dice
- **Have**: 5 trained models for ensemble

### Week 2: Optimization
- Fine-tune hyperparameters
- Test different loss functions
- Implement cross-validation
- **Expected**: 0.83-0.86 Dice

### Week 3-4: Ensemble & TTA
- Create ensemble of best models
- Implement test-time augmentation
- **Target**: 0.87-0.90 Dice (Top 10 range)

---

## 🎯 Quick Commands Reference

### Training
```bash
# Basic training
python train.py --config configs/experiments/baseline.yaml

# Resume from checkpoint
python train.py --config configs/experiments/baseline.yaml \
  --resume models/experiments/baseline/checkpoint_epoch_20.pth

# Train on specific GPU
CUDA_VISIBLE_DEVICES=0 python train.py --config configs/experiments/baseline.yaml
```

### Inference
```bash
# Generate predictions for test set
python src/inference/predict.py \
  --model models/experiments/attention_unet/best_model.pth \
  --config configs/experiments/attention_unet.yaml \
  --output predictions/

# Create submission file
python src/inference/create_submission.py \
  --predictions predictions/ \
  --output submissions/submission_v1.csv
```

### Ensemble
```bash
# Create ensemble from multiple models
python src/inference/ensemble.py \
  --models \
    models/experiments/unet/best_model.pth \
    models/experiments/attention_unet/best_model.pth \
    models/experiments/manet/best_model.pth \
  --weights 0.3 0.4 0.3 \
  --output submissions/ensemble_v1.csv
```

---

## 🐛 Troubleshooting

### Out of Memory Errors
```yaml
# Edit your config file:
training:
  batch_size: 2  # Reduce from 4
  
data:
  tile_size: 256  # Reduce from 512
```

### Slow Data Loading
```bash
# Increase workers if you have CPU cores
# In config:
data:
  num_workers: 8  # Increase from 4
```

### CUDA Out of Memory During Validation
```yaml
# Reduce validation batch size separately
training:
  batch_size: 4
  val_batch_size: 2  # Add this
```

### Models Not Improving
1. Check learning rate (try 5e-4 instead of 1e-3)
2. Verify data augmentation is working
3. Check if loss is decreasing but metric isn't (overfitting)
4. Try different loss function (topology_aware instead of combined)

---

## 📁 Project Structure Explained

```
vesuvius-competition/
├── src/
│   ├── models/              # Neural network architectures
│   │   └── __init__.py     # Model factory with 5 architectures
│   │
│   ├── data/                # Data loading and augmentation
│   │   └── dataset.py      # Dataset classes
│   │
│   ├── training/            # Training logic
│   │   ├── trainer.py      # Main training loop
│   │   └── losses.py       # Loss functions (7 types)
│   │
│   ├── inference/           # Inference and submission
│   │   ├── predict.py      # Generate predictions
│   │   ├── ensemble.py     # Ensemble predictions
│   │   └── create_submission.py
│   │
│   └── utils/               # Helper functions
│       └── metrics.py      # Evaluation metrics
│
├── configs/                 # Configuration files
│   ├── config.yaml         # Base config
│   └── experiments/        # Experiment-specific configs
│
├── data/                    # Data directory
│   ├── raw/                # Raw competition data
│   ├── processed/          # Preprocessed .npz files
│   └── external/           # Additional data
│
├── models/                  # Model outputs
│   ├── experiments/        # Experiment checkpoints
│   └── final/              # Final submission models
│
├── train.py                # Main training script
└── README.md               # Project documentation
```

---

## 🔥 Pro Tips

### 1. Start Simple, Iterate Fast
```bash
# Don't train for 50 epochs immediately
# Train for 5 epochs first to verify everything works
python train.py --config configs/experiments/baseline.yaml --epochs 5
```

### 2. Use WandB from Day 1
```bash
# Track everything - you'll thank yourself later
wandb login
python train.py --config configs/experiments/baseline.yaml --wandb
```

### 3. Save Compute with Smart Checkpointing
```yaml
# In config:
logging:
  save_interval: 5  # Save every 5 epochs, not every epoch
```

### 4. Validate on Small Subset First
```python
# In dataset.py, temporarily set:
samples_per_volume = 50  # Instead of 500
# This makes validation 10x faster during development
```

### 5. Use Mixed Precision Training
```yaml
# Already enabled by default, but verify:
training:
  mixed_precision: true  # 2x faster, 50% less memory
```

---

## 🎓 Learning Resources

### Understanding the Competition
- [Vesuvius Challenge Official](https://scrollprize.org/)
- [Competition Discussion Forum](https://www.kaggle.com/competitions/vesuvius-challenge-ink-detection/discussion)

### Deep Learning for Medical Imaging
- [Segmentation Models PyTorch](https://github.com/qubvel/segmentation_models.pytorch)
- [MONAI Documentation](https://docs.monai.io/)
- [Medical Segmentation Best Practices](https://github.com/Project-MONAI/tutorials)

### Competition Strategy
- [Kaggle Grandmaster Tips](https://www.kaggle.com/discussion/getting-started)
- [Winning Solution Writeups](https://www.kaggle.com/competitions/vesuvius-challenge-ink-detection/discussion)

---

## ✅ Daily Checklist

### Week 1
- [ ] Day 1: Setup complete, baseline training started
- [ ] Day 2: Baseline trained, submitted to leaderboard
- [ ] Day 3: Started attention_unet experiment
- [ ] Day 4: Experiment with different tile sizes
- [ ] Day 5: Test different loss functions
- [ ] Day 6: Start 5-fold cross-validation
- [ ] Day 7: Review results, plan week 2

### Week 2
- [ ] Launch 5 different architectures in parallel
- [ ] Analyze which models perform best
- [ ] Implement ensemble prediction
- [ ] Test Test-Time Augmentation (TTA)
- [ ] Submit ensemble to leaderboard

### Week 3
- [ ] Fine-tune best models with optimized hyperparameters
- [ ] Train final models with more epochs
- [ ] Create weighted ensemble
- [ ] Optimize submission format

### Week 4
- [ ] Final ensemble creation
- [ ] Test submission locally
- [ ] Submit and monitor leaderboard
- [ ] Iterate based on public LB feedback

---

## 🆘 Get Help

- **Issues with code**: Check discussion forum or create GitHub issue
- **Modal questions**: modal.com/docs or Modal Slack
- **WandB questions**: docs.wandb.ai
- **Competition questions**: Kaggle discussion forum

---

## 🚀 Ready to Start?

```bash
# 1. Activate environment
source venv/bin/activate

# 2. Download data
kaggle competitions download -c vesuvius-challenge-ink-detection
unzip vesuvius-challenge-ink-detection.zip -d data/raw/

# 3. Preprocess
python scripts/preprocessing/prepare_data.py

# 4. Train
python train.py --config configs/experiments/baseline.yaml --wandb

# 5. Monitor at wandb.ai
```

**Let's go get that Top 10! 🏆**