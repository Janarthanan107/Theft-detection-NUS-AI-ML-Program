# 🚀 Train on Real MNNIT Dataset - Complete Guide

## ✅ **Current Status**

- ✅ Demo training stopped
- ✅ Demo data cleaned
- ✅ Config updated for production training
- ✅ Ready for real dataset

---

## 📋 **Step-by-Step Training Process**

### **Step 1: Download Real Dataset** 📥

Follow the detailed guide in `DOWNLOAD_REAL_DATASET.md`.

**Quick Links:**
- **Kaggle**: https://www.kaggle.com/datasets (search "shoplifting video")
- **Mendeley**: https://data.mendeley.com (search "MNNIT shoplifting")
- **UCF Crime** (alternative): https://www.crcv.ucf.edu/projects/real-world/

**What you need:**
- 150+ videos total
- ~75 normal behavior videos
- ~75 shoplifting videos

---

### **Step 2: Organize Dataset** 📁

Place videos in this exact structure:

```bash
cd /Users/janatheboss/.gemini/antigravity/scratch/Theft-detection-NUS-AI-ML-Program

# Organize manually or use this template:
data/raw_videos/
├── normal/
│   ├── normal_001.mp4
│   ├── normal_002.mp4
│   └── ... (75+ videos)
└── shoplifting/
    ├── shoplifting_001.mp4
    ├── shoplifting_002.mp4
    └── ... (75+ videos)
```

**Move your downloaded videos:**
```bash
# Example if downloaded to ~/Downloads/shoplifting_dataset/
cp ~/Downloads/shoplifting_dataset/Normal/*.mp4 data/raw_videos/normal/
cp ~/Downloads/shoplifting_dataset/Shoplifting/*.mp4 data/raw_videos/shoplifting/
```

---

### **Step 3: Verify Dataset** ✅

```bash
cd /Users/janatheboss/.gemini/antigravity/scratch/Theft-detection-NUS-AI-ML-Program
source venv/bin/activate

python3 scripts/download_dataset.py --check
```

**Expected output:**
```
✓ Found 78 normal videos
✓ Found 76 shoplifting videos
✓ Total: 154 videos
✓ Dataset ready for training!
```

---

### **Step 4: Prepare Data Splits** 🔄

```bash
python3 scripts/prepare_data.py
```

**This creates:**
- Training set: 70% (~108 videos)
- Validation set: 15% (~23 videos)
- Test set: 15% (~23 videos)

**Expected output:**
```
============================================================
DATASET SPLIT SUMMARY
============================================================
Total videos: 154
Train: 108 (70.0%)
Val: 23 (15.0%)
Test: 23 (15.0%)

Class Distribution:
  Train:
    Normal: 54
    Shoplifting: 54
  Val:
    Normal: 12
    Shoplifting: 11
  Test:
    Normal: 12
    Shoplifting: 11
============================================================
```

---

### **Step 5: Start Training!** 🚀

```bash
python3 scripts/train_video_classifier.py 2>&1 | tee training_real.log
```

**Training Configuration (Updated):**
- Model: CNN-LSTM with ResNet-18 backbone
- Epochs: 30 (will auto-stop early if not improving)
- Batch Size: 8
- Learning Rate: 0.001
- Pretrained: ✅ Yes (ImageNet weights)
- Device: CPU (change config for GPU)

---

### **Step 6: Monitor Training** 📊

**In another terminal:**
```bash
cd /Users/janatheboss/.gemini/antigravity/scratch/Theft-detection-NUS-AI-ML-Program

# Watch live updates
tail -f training_real.log
```

**What to expect:**

```
Epoch 1 [Train]: 100%|██████| 14/14 [02:15<00:00, 9.65s/it, loss=0.68, acc=52.3]
Epoch 1 [Val]: 100%|██████| 3/3 [00:12<00:00, 4.15s/it, loss=0.69, acc=47.8]

2025-12-16 22:30:45 - INFO - Epoch 1/30
2025-12-16 22:30:45 - INFO - Train Loss: 0.6835, Train Acc: 52.31%
2025-12-16 22:30:45 - INFO - Val Loss: 0.6912, Val Acc: 47.83%
✓ Saved best model (Val Acc: 47.83%)

Epoch 2 [Train]: 100%|██████| 14/14 [02:10<00:00, 9.32s/it, loss=0.61, acc=65.7]
Epoch 2 [Val]: 100%|██████| 3/3 [00:11<00:00, 3.89s/it, loss=0.58, acc=69.6]

2025-12-16 22:33:06 - INFO - Epoch 2/30
2025-12-16 22:33:06 - INFO - Train Loss: 0.6142, Train Acc: 65.74%
2025-12-16 22:33:06 - INFO - Val Loss: 0.5789, Val Acc: 69.57%
✓ Saved best model (Val Acc: 69.57%)

...continues improving...
```

---

### **Step 7: Training Timeline** ⏱️

**Estimated Duration (150 videos, CPU):**
- Per epoch: ~2-3 minutes
- Total (30 epochs): ~1.5-2 hours
- With early stopping: ~45-90 minutes (usually stops around epoch 15-20)

**With GPU (if available):**
- Per epoch: ~30-45 seconds
- Total: ~15-30 minutes

**Tip:** Run overnight or during meal break!

---

## 📊 **Expected Results (Real Dataset)**

### **Performance Metrics:**

With proper training on real data:

```
              precision    recall  f1-score   support

      Normal     0.8750    0.9167    0.8954        12
Shoplifting     0.9091    0.8636    0.8857        11

    accuracy                         0.8913        23
```

**Translation:**
- **Overall Accuracy**: 85-92%
- **Normal Precision**: 87.5% (when predicts normal, usually right)
- **Shoplifting Recall**: 86.4% (catches 86% of thefts)
- **NO OVERFITTING** ✅

---

## 📁 **Files Generated**

After training completes:

```
checkpoints/
└── video_classifier_best.pth  (~50 MB) - Your trained model!

outputs/
├── training_history.png        - Loss & accuracy curves
├── confusion_matrix.png         - Performance heatmap
└── classification_report.txt    - Detailed metrics

logs/
└── training_*.log              - Full training log

training_real.log               - Live output
```

---

## 🎯 **What Happens After Training**

### **Automatic Outputs:**

1. **Training History Plot** (`outputs/training_history.png`):
   - Shows loss decreasing over epochs
   - Shows accuracy increasing
   - Separate curves for train vs validation

2. **Confusion Matrix** (`outputs/confusion_matrix.png`):
   ```
               Predicted
            Normal  Shoplifting
   Normal      11        1
   Shoplifting  2        9
   ```

3. **Classification Report** (`outputs/classification_report.txt`):
   - Precision, recall, F1-score per class
   - Overall accuracy
   - Support (number of samples)

---

## 🚀 **After Training: Deploy!**

Once training completes successfully:

### **Step 1: Start Backend**

```bash
cd /Users/janatheboss/.gemini/antigravity/scratch/Theft-detection-NUS-AI-ML-Program
source venv/bin/activate
python3 backend/app.py
```

Keep this running!

### **Step 2: Update UI**

```bash
# Run automated setup
./setup_real_detection.sh
```

Or manually update `web_ui/script.js`:
- Change `simulateDetection()` to `realDetectionLoop()`

### **Step 3: Test Live Detection**

```bash
# Open UI
open web_ui/index.html

# Or serve with HTTP server
cd web_ui
python3 -m http.server 8000
# Then open: http://localhost:8000
```

1. Click "Live Webcam"
2. Allow camera access
3. Click "Start Detection"
4. **See real AI predictions!** 🎉

---

## 💡 **Tips for Best Results**

### **1. Hardware Acceleration**

If you have:
- **NVIDIA GPU**: Edit `configs/config.yaml`, set `device: "cuda"`
- **Apple M-series**: Try `device: "mps"` (may have issues, fallback to CPU)
- **CPU only**: Keep `device: "cpu"` (slower but stable)

### **2. Optimize Training Time**

```yaml
# In configs/config.yaml, adjust:
training:
  video_classifier:
    batch_size: 16        # Increase if you have good GPU/RAM
    num_workers: 4        # More workers = faster data loading
    epochs: 50            # More epochs = better learning
```

### **3. Monitor GPU Usage**

```bash
# NVIDIA GPU
watch -n 1 nvidia-smi

# Apple Silicon
sudo powermetrics --samplers gpu_power
```

### **4. Resume Interrupted Training**

```bash
# If training stops/crashes, resume from checkpoint
python3 scripts/train_video_classifier.py --resume checkpoints/video_classifier_checkpoint.pth
```

---

## 🐛 **Troubleshooting**

### **"Not enough videos" error**
- Need minimum 30 videos (15 per class)
- Recommended: 100+ videos for good results

### **Out of memory error**
- Reduce `batch_size` in config (try 4 or 2)
- Reduce `num_workers` to 0
- Close other applications

### **Training very slow**
- Check if using GPU (should see CUDA messages)
- Reduce video resolution
- Use fewer frames per clip (edit `clip_length`)

### **Validation accuracy not improving**
- Let it train longer (increase `epochs`)
- Adjust learning rate (try 0.0001 or 0.005)
- Check if data is balanced (equal normal/shoplifting)

### **SSL certificate errors**
- Already fixed in code
- If persists, set `pretrained: false` in config

---

## 📚 **Understanding the Training Process**

### **What's Happening:**

1. **Epoch 1-5**: Model learning basic patterns
   - Accuracy: 50-70%
   - Loss high, decreasing slowly

2. **Epoch 6-15**: Model refining understanding
   - Accuracy: 70-85%
   - Loss decreasing faster
   - Best model usually saved here

3. **Epoch 16-30**: Fine-tuning
   - Accuracy: 85-92%
   - Loss plateauing
   - Early stopping may trigger

### **Key Metrics:**

- **Training Loss**: How wrong model is on training data (lower = better)
- **Validation Accuracy**: Performance on unseen data (higher = better)
- **Gap between train/val**: Overfitting indicator (small gap = good)

---

## ✅ **Success Checklist**

Before considering training complete:

- [ ] Training finished without errors
- [ ] Validation accuracy > 80%
- [ ] Test accuracy > 75%
- [ ] Training/validation gap < 10%
- [ ] Confusion matrix looks balanced
- [ ] Model file exists: `checkpoints/video_classifier_best.pth`
- [ ] Plots generated in `outputs/`

---

## 🎉 **You're Done When:**

```
2025-12-16 23:45:12 - INFO - ============================================================
2025-12-16 23:45:12 - INFO - TRAINING COMPLETED
2025-12-16 23:45:12 - INFO - ============================================================
2025-12-16 23:45:12 - INFO - Best Validation Accuracy: 88.75%
2025-12-16 23:45:12 - INFO - Test Accuracy: 86.96%
2025-12-16 23:45:12 - INFO - Model saved to: checkpoints/video_classifier_best.pth
```

**Next:** Deploy and test with real-time webcam detection! 🚀

---

## 📞 **Need Help?**

- Check `training_real.log` for detailed output
- Review `outputs/training_history.png` for learning curves
- See `DOWNLOAD_REAL_DATASET.md` for dataset help
- Read `HOW_IT_WORKS.md` for technical details

---

**Good luck with training! This will take 1-2 hours, so grab a coffee! ☕** 

**I'll be ready to help you deploy once training completes!** 🎯
