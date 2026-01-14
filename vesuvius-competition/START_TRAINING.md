# 🎯 Complete Training Pipeline

## 📋 Pre-flight Checklist

### 1. **Data Check**
```bash
# Check how much data you have
echo "Raw TIFFs: $(ls data/raw/train_images/*.tif | wc -l)"
echo "Processed volumes: $(ls data/processed/volume_*.npz | wc -l)"
echo "Processed masks: $(ls data/processed/mask_*.npz | wc -l)"
```

**❗ You need at least 5-10 volumes for good training!**

### 2. **Process More Data**
```bash
# Process 10 random volumes (adjust based on your time/storage)
python scripts/preprocessing/prepare_data.py --volume_ids \
  1006462223 1013184726 1029212680 1033375083 1044587645 \
  105068588 1068988588 1074255988 1081316088 1088334088

# Or process ALL volumes (will take 1-2 hours, ~20-30GB)
python scripts/preprocessing/prepare_data.py
```

### 3. **Verify Processed Data**
```bash
# Should see pairs of volume_X.npz and mask_X.npz
ls -lh data/processed/
```

## 🔧 Modal Setup

### 1. **Set Modal Token**
```bash
# This opens a browser for authentication
modal token new

# Verify
modal profile current
```

### 2. **Create WandB Secret (Optional)**
```bash
# Get your key from https://wandb.ai/settings
modal secret create wandb-secret WANDB_API_KEY=paste-your-key-here

# Verify
modal secret list
```

### 3. **Create Modal Volume**
```bash
# Create persistent storage
modal volume create vesuvius-data

# Verify
modal volume list
```

## 📤 Upload Data to Modal (One-Time)

```bash
# Upload your processed data
bash scripts/upload_to_modal.sh

# Or manually:
modal volume put vesuvius-data data/processed /data/processed

# Verify upload
modal run src/modal_training.py::list_data
```

## 🚀 Start Training!

### Option 1: Train Baseline Model
```bash
modal run src/modal_training.py --command train \
  --config configs/experiments/baseline.yaml
```

### Option 2: Train Better Model
```bash
modal run src/modal_training.py --command train \
  --config configs/experiments/attention_unet.yaml
```

### Option 3: Train Multiple Models in Parallel
```bash
modal run src/modal_training.py --command parallel
```

## 📊 Monitor Training

### WandB Dashboard
```
Visit: https://wandb.ai/your-username/vesuvius-challenge
```

### Modal Dashboard
```
Visit: https://modal.com/apps
```

### CLI Logs
```bash
# Stream logs
modal app list
modal app logs vesuvius-training
```

## 💾 Download Trained Models

```bash
# After training completes
modal run src/modal_training.py --command download

# Models will be in:
ls models/from_modal/
```

## 🐛 Common Issues & Solutions

### "No volume_*.npz files found"
```bash
# You need to process data first
python scripts/preprocessing/prepare_data.py --volume_ids 1006462223 1013184726
```

### "Modal token not set"
```bash
modal token new
```

### "CUDA out of memory"
```yaml
# Edit config file, reduce batch_size:
training:
  batch_size: 2  # from 4
```

### "WandB not logging"
```bash
# Create the secret
modal secret create wandb-secret WANDB_API_KEY=your-key

# Or train without WandB
# Edit config: logging: use_wandb: false
```

## 💰 Cost Estimate

- **A100-80GB**: $3.60/hour
- **Typical training**: 4-6 hours
- **Cost per model**: $15-25
- **5 models**: $75-125

## ✅ Success Looks Like:

```
🚀 Starting training on A100-80GB
   Experiment: baseline_unet

📊 Dataset Split:
   Training volumes: 8
   Validation volumes: 2

🏗️  Model: unet
   Total parameters: 31,042,881

Starting Training
==================================================
Epochs: 0 → 50
Mixed Precision: True

Epoch 1/50
Training: 100%|████████| 2000/2000 [04:23<00:00, 7.58it/s]
Validation: 100%|██████| 200/200 [00:35<00:00, 5.63it/s]

Epoch 1
  Train Loss: 0.8234
  Val Loss:   0.7123
  Dice Score: 0.5234
  🎯 New best validation loss!
```