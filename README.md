# Vesuvius Challenge - Surface Detection Competition

Deep learning solution for detecting ancient papyrus surfaces from 3D X-ray CT scans.

## 🎯 Competition Goal

Segment the surface of carbonized Herculaneum papyrus scrolls from high-resolution 3D X-ray scans to enable ink detection and text recovery from ancient scrolls.

**Competition:** [Kaggle - Vesuvius Challenge Surface Detection](https://www.kaggle.com/competitions/vesuvius-challenge-surface-detection)

## 🏗️ Architecture

- **Models**: 3D U-Net, Attention U-Net, V-Net, Double U-Net
- **Training**: Modal cloud platform with A100 GPUs
- **Framework**: PyTorch + MONAI
- **Strategy**: Ensemble + Test-Time Augmentation

## 📊 Current Results

- **Best Single Model**: TBD
- **Ensemble**: TBD
- **Leaderboard Position**: TBD

## 🚀 Quick Start

### Local Setup
```bash
# Clone repository
git clone https://github.com/Jonathan-321/vesuviusChallenge.git
cd vesuviusChallenge

# Setup environment
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Download competition data
kaggle competitions download -c vesuvius-challenge-surface-detection
unzip vesuvius-challenge-surface-detection.zip -d data/raw/

# Preprocess data locally
python scripts/preprocessing/prepare_data.py
```

### Train on Modal (A100)
```bash
# Upload data to Modal (one time)
bash scripts/upload_to_modal.sh

# Train single model
modal run modal_training.py --command train

# Train multiple models in parallel
modal run modal_training.py --command parallel
```

## 📁 Project Structure
```
vesuviusChallenge/
├── src/
│   ├── models/          # Neural network architectures
│   ├── data/            # Dataset and data loading
│   ├── training/        # Training engine and losses
│   └── inference/       # Prediction and submission
├── configs/             # Experiment configurations
├── scripts/             # Utility scripts
├── data/                # Local data storage
│   ├── raw/            # Raw competition data
│   └── processed/      # Preprocessed .npz files
└── models/             # Trained model checkpoints
```

## 🛠️ Technologies

- **PyTorch** - Deep learning framework
- **MONAI** - Medical imaging library
- **Albumentations** - Data augmentation
- **Modal** - Cloud GPU platform
- **WandB** - Experiment tracking

## 📈 Training Strategy

1. **Week 1**: Baseline models + infrastructure
2. **Week 2**: Architecture exploration (5 models)
3. **Week 3**: Hyperparameter optimization
4. **Week 4**: Ensemble + final submission

## 🏆 Target

**Top 10** on Kaggle leaderboard

## 📝 License

MIT License - See LICENSE file for details

## 🙏 Acknowledgments

- [Vesuvius Challenge](https://scrollprize.org/)
- [Modal Labs](https://modal.com/)
- Competition organizers and community


**Status**: 🚧 In active development
