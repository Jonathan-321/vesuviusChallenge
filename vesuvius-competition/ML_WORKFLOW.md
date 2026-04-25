# 🎯 ML WORKFLOW - The Right Way

## Overview

This document defines the PROPER workflow for ML projects to avoid chaos, wasted time, and 56-file disasters.

## The ML Development Lifecycle

### 1. **UNDERSTAND THE PROBLEM** (Before Writing ANY Code)

#### Questions to Answer:
- What are we trying to predict? (papyrus surface in 3D scans)
- What does success look like? (Dice score > 0.7)
- What's the data format? (3D volumes → 2D predictions)
- What's the submission format? (RLE encoded masks)

#### For Vesuvius Challenge:
- **Input**: 3D CT scan volumes (65 slices × 320 × 320)
- **Output**: 2D binary mask (320 × 320) 
- **Expected coverage**: 5-20% surface pixels
- **Metric**: Dice coefficient

### 2. **DATA ORGANIZATION** (One Source of Truth)

```
vesuvius-competition/
├── data/
│   ├── raw/                 # Original competition data (NEVER modify)
│   ├── processed/           # Processed training data (ONE version)
│   ├── test/               # Test data for submission
│   └── cache/              # Temporary files (can delete)
```

**Rules:**
- NO duplicate data folders
- NO "proper_training", "all_training", "final_training_v2"
- Clear naming: what it is, not when you made it

### 3. **MODEL DEVELOPMENT** (Systematic Approach)

#### Step 1: Baseline First
```python
# Start simple
model = smp.Unet(
    encoder_name="resnet34",
    encoder_weights="imagenet",
    in_channels=32,
    classes=1
)
```

#### Step 2: Validate Pipeline
- Train on 10% of data
- Ensure loss decreases
- Check predictions are reasonable
- Verify metrics make sense

#### Step 3: Scale Up Gradually
- Add more data
- Try better models
- Add augmentations
- Implement advanced losses

### 4. **EXPERIMENT TRACKING** (Know What You Did)

#### Use a Simple Experiment Log:
```markdown
## Experiment 001: Baseline
- Model: UNet with ResNet34
- Data: 10 volumes
- Loss: BCE
- Result: 0.45 Dice (underfitting)

## Experiment 002: Focal Loss
- Change: BCE → Focal Loss
- Result: 0.52 Dice (better class balance)
```

#### Checkpoint Naming:
```
models/
├── exp001_baseline_dice045.pth
├── exp002_focal_dice052.pth
└── exp003_balanced_dice072.pth  # Current best
```

### 5. **DEBUGGING WORKFLOW** (When Things Go Wrong)

#### The Scientific Method:
1. **Observe**: What exactly is wrong?
2. **Hypothesize**: Why might this happen?
3. **Test**: Change ONE thing
4. **Measure**: Did it fix the issue?
5. **Iterate**: If not, try next hypothesis

#### Example: NaN Loss
```python
# Hypothesis 1: Learning rate too high
# Test: Reduce LR from 1e-3 to 1e-4
# Result: Still NaN

# Hypothesis 2: Numerical instability in loss
# Test: Add epsilon to denominators
# Result: Fixed!
```

### 6. **INFERENCE & SUBMISSION** (Get It Right)

#### Pre-Submission Checklist:
- [ ] Model outputs reasonable predictions (5-20% coverage)
- [ ] RLE encoding is correct (not inverted!)
- [ ] Test on sample data first
- [ ] Verify submission format matches requirements
- [ ] Clean up test files

#### RLE Debugging:
```python
# ALWAYS verify your RLE makes sense
def verify_rle(rle_string, shape=(320, 320)):
    mask = rle_to_mask(rle_string, shape)
    coverage = mask.sum() / mask.size
    
    assert 0.01 < coverage < 0.3, f"Coverage {coverage:.2%} is unrealistic!"
    
    # Visual check
    plt.imshow(mask)
    plt.title(f"Coverage: {coverage:.2%}")
    plt.show()
```

## Workflow for Vesuvius Challenge

### Phase 1: Setup ✅
1. Download competition data
2. Organize into clean structure
3. Understand evaluation metric
4. Create basic data loader

### Phase 2: Baseline ✅
1. Simple UNet model
2. Train on small subset
3. Achieve >0 Dice score
4. Create valid submission

### Phase 3: Improve ⚠️ (Current Stage)
1. Fix class imbalance (Focal Loss) ✅
2. Fix data issues (balanced sampling) ✅
3. Add model improvements ✅
4. **Fix RLE encoding** ← YOU ARE HERE
5. Create proper submission

### Phase 4: Optimize
1. Ensemble multiple models
2. Add test-time augmentation
3. Post-processing improvements
4. Final submission

## Current Issues to Fix

1. **Inverted RLE Encoding**
   - Problem: Predicting 99.97% as surface (inverted)
   - Solution: Fix mask generation or RLE encoding
   - Verification: Coverage should be 5-20%

2. **Messy Data Organization**
   - Problem: Multiple data folders with unclear purpose
   - Solution: Consolidate into single processed folder
   - Verification: One source of truth

3. **Missing Pipeline Verification**
   - Problem: Not checking outputs at each step
   - Solution: Add assertions and visual checks
   - Verification: Each step produces expected output

## Commands You'll Actually Use

```bash
# Training
python train.py --config configs/experiments/baseline.yaml

# Evaluation  
python evaluate_model.py --checkpoint models/best_model.pth

# Modal training
modal run modal_train_balanced.py

# Create submission (AFTER fixing RLE)
python -m src.inference.create_submission \
    --model models/best_model.pth \
    --test data/test \
    --output submission.csv
```

## Remember

1. **One experiment at a time**
2. **Verify each step works**
3. **Clean up as you go**
4. **Document what you learn**
5. **No quick fixes - fix root causes**

## Next Actions

1. ✅ Clean up project files
2. ✅ Create guardrail documents
3. 🔄 Fix RLE encoding in Modal inference
4. ⏳ Consolidate data folders
5. ⏳ Create proper submission
6. ⏳ Submit to Kaggle with correct predictions

**Stop creating files. Start fixing the core issue: your RLE is inverted.**