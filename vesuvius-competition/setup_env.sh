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
