#!/bin/bash
# Upload preprocessed data from local machine to Modal volume
# Run this ONCE after local preprocessing

echo "📤 Uploading Preprocessed Data to Modal Volume"
echo "=============================================="

# Check if data/processed exists
if [ ! -d "data/processed" ]; then
    echo "❌ Error: data/processed directory not found"
    echo "   Please run preprocessing first:"
    echo "   python scripts/preprocessing/prepare_data.py"
    exit 1
fi

# Count files
FILE_COUNT=$(ls data/processed/*.npz 2>/dev/null | wc -l)
if [ "$FILE_COUNT" -eq 0 ]; then
    echo "❌ Error: No .npz files found in data/processed/"
    echo "   Please run preprocessing first:"
    echo "   python scripts/preprocessing/prepare_data.py"
    exit 1
fi

echo "Found $FILE_COUNT .npz files to upload"
echo ""

# Calculate total size
TOTAL_SIZE=$(du -sh data/processed | cut -f1)
echo "Total data size: $TOTAL_SIZE"
echo ""

# Confirm upload
read -p "Continue with upload? This may take 10-30 minutes (y/n): " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Upload cancelled"
    exit 0
fi

echo ""
echo "🚀 Starting upload..."
echo "This will take approximately 10-30 minutes depending on your connection"
echo ""

# Create Modal volume if it doesn't exist
echo "1️⃣ Creating Modal volume (if needed)..."
modal volume create vesuvius-data 2>/dev/null || echo "   Volume already exists"

# Upload files using Modal CLI
echo "2️⃣ Uploading files to Modal volume..."
modal volume put vesuvius-data data/processed /data/processed

# Verify upload
echo ""
echo "3️⃣ Verifying upload..."
modal run src/modal_training.py::list_data

echo ""
echo "✅ Upload complete!"
echo ""
echo "Next steps:"
echo "  1. Verify data with: modal run src/modal_training.py::list_data"
echo "  2. Start training with: modal run src/modal_training.py --command train"