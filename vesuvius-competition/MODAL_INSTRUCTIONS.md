# 🚀 MODAL TRAINING INSTRUCTIONS

## Issue
Modal has an architecture mismatch on your system. Run these commands in a separate terminal.

## Steps to Train with Correct Data

### 1. First, upload the data
```bash
# In a new terminal, navigate to project
cd ~/vesuviusChallenge/vesuvius-competition

# Upload data to Modal
modal run modal_upload_data.py
```

### 2. Start training
```bash
# Train with balanced approach using correct data
modal run modal_train_balanced.py
```

### 3. Monitor training
The training will show:
- Epoch progress
- Loss values  
- Dice scores
- Validation metrics

Expected results with correct data:
- Training loss should decrease
- Dice score should increase to 0.5-0.7
- Coverage predictions should be 10-20%

### 4. Evaluate after training
```bash
# Check model performance
modal run modal_evaluate_simple.py
```

### 5. Download trained model
```bash
# Get the model for local inference
modal volume get vesuvius-data /models/proper_data_best.pth ./models/from_modal/
```

## Key Points
- Training uses `data/proper_training/` with correct 0/255 labels
- No inversion needed - data is correct now
- Model should predict realistic 10-20% coverage
- Check validation Dice score > 0.5 before using

## Troubleshooting
If Modal still has issues:
```bash
# Try reinstalling for your architecture
pip uninstall modal watchfiles
pip install --no-binary :all: watchfiles
pip install modal
```