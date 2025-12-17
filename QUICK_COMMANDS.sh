#!/bin/bash

# Quick Reference - Theft Detection System
# Handy commands for common tasks

echo "🎯 THEFT DETECTION SYSTEM - QUICK REFERENCE"
echo "=" | tr '=' '='  | head -c 60; echo ""
echo ""

cat << 'EOF'

📥 DOWNLOAD DATASET
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# View download instructions
cat DOWNLOAD_REAL_DATASET.md

# Check if dataset is ready
python3 scripts/download_dataset.py --check

# Expected structure:
data/raw_videos/
├── normal/       # 75+ videos
└── shoplifting/  # 75+ videos


🔄 PREPARE DATA
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Create train/val/test splits
python3 scripts/prepare_data.py


🚀 TRAIN MODEL
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Start training (with logging)
python3 scripts/train_video_classifier.py 2>&1 | tee training_real.log

# Monitor training (in another terminal)
tail -f training_real.log

# Resume from checkpoint (if interrupted)
python3 scripts/train_video_classifier.py --resume checkpoints/checkpoint.pth


📊 AFTER TRAINING
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# View results
ls -lh outputs/
# - training_history.png
# - confusion_matrix.png
# - classification_report.txt

# Check model
ls -lh checkpoints/video_classifier_best.pth


🌐 DEPLOY
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Start Flask backend (Terminal 1)
python3 backend/app.py

# Open web UI (Terminal 2)
open web_ui/index.html

# Or serve with HTTP server
cd web_ui && python3 -m http.server 8000
# Then open: http://localhost:8000


📚 DOCUMENTATION
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
DOWNLOAD_REAL_DATASET.md     # Dataset sources& download guide
TRAIN_ON_REAL_DATA.md         # Complete training instructions
HOW_IT_WORKS.md               # Technical architecture deep dive
QUICK_START.md                # Quick reference for all features
STATUS_AND_NEXT_STEPS.md      # Current status & action items


🔧 CONFIGURATION
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Edit training settings
vim configs/config.yaml

# Key settings:
# - epochs: 30 (how many training iterations)
# - batch_size: 8 (samples per batch)
# - device: cpu/cuda/mps (hardware)
# - learning_rate: 0.001 (how fast to learn)


🐛 TROUBLESHOOTING
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Check Python environment
which python3
python3 --version

# Reinstall dependencies
pip install -r requirements.txt --force-reinstall

# Clear cache
rm -rf __pycache__ */__pycache__
rm -rf .pytest_cache

# Check GPU (NVIDIA)
nvidia-smi

# Check logs
tail -n 100 logs/training_*.log


📁 PROJECT STRUCTURE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
backend/            # Flask API server
configs/            # Configuration files
checkpoints/        # Trained models (created during training)
data/               # Dataset (you provide videos here)
datasets/           # PyTorch dataset classes
docs/               # Additional documentation
logs/               # Training logs (auto-generated)
models/             # Model architectures
outputs/            # Results & visualizations (auto-generated)
scripts/            # Training & utility scripts
utils/              # Helper functions
web_ui/             # Beautiful web interface


⚡ QUICK COMMANDS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Activate virtual environment
source venv/bin/activate

# Navigate to project
cd /Users/janatheboss/.gemini/antigravity/scratch/Theft-detection-NUS-AI-ML-Program

# Full pipeline (after dataset download)
python3 scripts/prepare_data.py && \
python3 scripts/train_video_classifier.py && \
python3 backend/app.py


🎯 CURRENT TASK
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
1. Download MNNIT Shoplifting Dataset (~150 videos)
   See: DOWNLOAD_REAL_DATASET.md

2. Organize videos into:
   data/raw_videos/normal/ and data/raw_videos/shoplifting/

3. Run training pipeline (see TRAIN_ON_REAL_DATA.md)


EOF

echo ""
echo "💡 TIP: Bookmark this file for quick reference!"
echo "   Located at: $(pwd)/QUICK_COMMANDS.sh"
echo ""
echo "🚀 Ready to train? Start with: cat DOWNLOAD_REAL_DATASET.md"
echo ""
