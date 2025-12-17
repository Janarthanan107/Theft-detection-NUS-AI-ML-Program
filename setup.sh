#!/bin/bash
# Setup script for Shoplifting Detection System

echo "=========================================="
echo "Shoplifting Detection System - Setup"
echo "=========================================="
echo ""

# Check Python version
echo "Checking Python version..."
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "✓ Python $python_version detected"
echo ""

# Create virtual environment
echo "Creating virtual environment..."
if [ ! -d "venv" ]; then
    python3 -m venv venv
    echo "✓ Virtual environment created"
else
    echo "✓ Virtual environment already exists"
fi
echo ""

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate
echo "✓ Virtual environment activated"
echo ""

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip > /dev/null 2>&1
echo "✓ Pip upgraded"
echo ""

# Install requirements
echo "Installing dependencies..."
echo "This may take several minutes..."
pip install -r requirements.txt
echo "✓ Dependencies installed"
echo ""

# Create necessary directories
echo "Creating project directories..."
python3 -c "
import os
dirs = [
    'data/raw_videos/normal',
    'data/raw_videos/shoplifting',
    'data/processed_frames',
    'data/splits',
    'checkpoints',
    'outputs',
    'logs'
]
for d in dirs:
    os.makedirs(d, exist_ok=True)
print('✓ Directories created')
"
echo ""

# Check for CUDA
echo "Checking for CUDA support..."
python3 -c "
import torch
if torch.cuda.is_available():
    print(f'✓ CUDA available: {torch.cuda.get_device_name(0)}')
else:
    print('⚠ CUDA not available, will use CPU')
"
echo ""

echo "=========================================="
echo "Setup Complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "  1. Download the MNNIT dataset:"
echo "     python scripts/download_dataset.py"
echo ""
echo "  2. Place videos in:"
echo "     data/raw_videos/normal/"
echo "     data/raw_videos/shoplifting/"
echo ""
echo "  3. Prepare the dataset:"
echo "     python scripts/prepare_data.py"
echo ""
echo "  4. Train the model:"
echo "     python scripts/train_video_classifier.py"
echo ""
echo "  5. Run the demo:"
echo "     python scripts/demo.py --source webcam"
echo ""
echo "For more information, see README.md"
echo ""
