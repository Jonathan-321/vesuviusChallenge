# 🚀 Complete Modal Setup Guide

## 1️⃣ **Install Modal CLI**

```bash
# If not already installed
pip install modal

# Verify installation
modal --version
```

## 2️⃣ **Set Up Modal Token**

### Option A: Interactive Setup (Recommended)
```bash
modal token new
```
This will open a browser window where you can:
1. Log in to your Modal account (or create one)
2. Authorize the CLI
3. Token will be saved automatically

### Option B: Manual Setup
1. Go to https://modal.com/settings/tokens
2. Create a new token
3. Copy the token ID and secret
4. Run:
```bash
modal token set --token-id "your-token-id" --token-secret "your-token-secret"
```

## 3️⃣ **Set Up WandB (Optional but Recommended)**

### Step 1: Get your WandB API key
1. Go to https://wandb.ai/settings
2. Copy your API key

### Step 2: Create Modal Secret
```bash
modal secret create wandb-secret WANDB_API_KEY=your-actual-wandb-key-here
```

### Step 3: Verify secret was created
```bash
modal secret list
```

## 4️⃣ **Verify Modal Setup**

```bash
# Test Modal connection
modal run --help

# Check your Modal dashboard
echo "Visit: https://modal.com/apps"
```

## 5️⃣ **Environment Variables (Local)**

Create or update `.env` file:
```bash
cat > .env << 'EOF'
# Modal (already set via CLI)
MODAL_TOKEN_ID=your-token-id
MODAL_TOKEN_SECRET=your-token-secret

# WandB
WANDB_API_KEY=your-wandb-key
WANDB_PROJECT=vesuvius-challenge

# Kaggle (for data download)
KAGGLE_USERNAME=your-kaggle-username
KAGGLE_KEY=your-kaggle-key
EOF
```

## 🔐 **Where Your Keys Are Stored**

1. **Modal Token**: `~/.modal/config.json`
   ```json
   {
     "token_id": "your-token-id",
     "token_secret": "your-token-secret"
   }
   ```

2. **Modal Secrets**: Stored in Modal cloud (not local)
   - Access via: `modal secret list`

3. **Local .env**: In your project root (git-ignored)

## ✅ **Quick Verification Commands**

```bash
# 1. Check Modal auth
modal profile current

# 2. List Modal secrets
modal secret list

# 3. Test Modal connection
python -c "import modal; print('Modal imported successfully')"

# 4. Quick Modal function test
modal run --quickstart hello_world
```

## 🎯 **You're Ready When You See:**

```bash
$ modal profile current
✅ Logged in as your-email@example.com

$ modal secret list
wandb-secret     Created 2 minutes ago

$ modal volume list
vesuvius-data    Created 5 minutes ago    1.2 GB
```