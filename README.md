# Shoplifting Detection System using Ensemble Deep Learning

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

A complete, production-ready shoplifting prevention system using an **ensemble model** trained on the **MNNIT Allahabad Shoplifting Dataset**. This project combines multiple deep learning approaches for robust real-time theft detection in retail environments.

---

## 🎯 Overview

This system implements a multi-stream ensemble architecture:

- **Stream 1**: Video Classifier (CNN-LSTM / 3D-CNN) for temporal activity recognition
- **Stream 2**: Object Detector (YOLOv8) for frame-level suspicious behavior detection  
- **Stream 3**: Anomaly Detector (Motion-based) for unusual movement patterns

The ensemble combines predictions from all streams to produce robust, accurate classifications: **NORMAL**, **SUSPICIOUS**, or **THEFT**.

---

## 📊 Dataset: MNNIT Allahabad Shoplifting Dataset

### Why This Dataset?

We chose the **MNNIT Allahabad Shoplifting Video Dataset** as our primary training source because:

✅ **Small but focused**: Manageable size perfect for student capstone projects  
✅ **Clean binary labels**: Clear distinction between Normal and Shoplifting behaviors  
✅ **Realistic setup**: Lab-simulated retail environment with authentic surveillance conditions  
✅ **Consistent quality**: ~640×480 resolution, 30 FPS, recorded with 32 MP camera  
✅ **Easy preprocessing**: Video-level labels make training straightforward for CNN-LSTM and 3D-CNN models  
✅ **Well-documented**: Created by CV Laboratory at MNNIT Allahabad with clear class definitions

### Dataset Details

- **Classes**: 
  - `Normal`: Walking, browsing items, inspecting products normally
  - `Shoplifting`: Concealing items in pockets, bags, or under clothing
  
- **Source References**:
  - [Mendeley Data](https://data.mendeley.com/datasets) - "Shoplifting Dataset (2022)"
  - [Innovatiana](https://innovatiana.com/shoplifting-dataset) - Dataset description page
  - [Kaggle](https://www.kaggle.com/) - Community reupload (search "Shoplifting Video Dataset")

- **Properties**:
  - Resolution: ~640×480 pixels
  - Frame rate: 30 FPS
  - Format: MP4 videos
  - Camera: 32 MP surveillance-style

---

## 🏗️ Project Structure

```
Theft-detection-NUS-AI-ML-Program/
├── configs/
│   └── config.yaml              # Configuration file
├── data/
│   ├── raw_videos/              # Original videos (manual placement)
│   │   ├── normal/
│   │   └── shoplifting/
│   ├── processed_frames/        # Extracted frames (auto-generated)
│   └── splits/                  # Train/val/test splits (auto-generated)
├── datasets/
│   ├── __init__.py
│   └── shoplifting_dataset.py   # PyTorch dataset classes
├── models/
│   ├── __init__.py
│   ├── video_classifier.py      # CNN-LSTM & 3D-CNN models
│   ├── anomaly_detector.py      # Motion-based anomaly detection
│   └── ensemble.py              # Ensemble fusion model
├── scripts/
│   ├── download_dataset.py      # Dataset download instructions
│   ├── prepare_data.py          # Create train/val/test splits
│   ├── train_video_classifier.py # Train main video classifier
│   └── demo.py                  # Real-time inference demo
├── utils/
│   ├── __init__.py
│   ├── helpers.py               # General utilities
│   ├── video_processing.py      # Video/frame processing
│   └── visualization.py         # Visualization tools
├── checkpoints/                 # Saved model checkpoints (auto-generated)
├── outputs/                     # Results, plots, reports (auto-generated)
├── logs/                        # Training logs (auto-generated)
├── requirements.txt
└── README.md
```

---

## 🚀 Quick Start

### 1. Installation

```bash
# Clone the repository
cd /path/to/Theft-detection-NUS-AI-ML-Program

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Download Dataset

```bash
# See download instructions and sources
python scripts/download_dataset.py

# After manual download, check structure
python scripts/download_dataset.py --check
```

**Manual Steps**:
1. Visit one of the dataset sources (Mendeley, Kaggle, or Innovatiana)
2. Download the dataset ZIP file
3. Extract and organize videos into:
   ```
   data/raw_videos/normal/       # Normal behavior videos
   data/raw_videos/shoplifting/  # Shoplifting videos
   ```

### 3. Prepare Data

```bash
# Create train/val/test splits (70/15/15)
python scripts/prepare_data.py
```

This will create stratified splits and save them to `data/splits/`.

### 4. Train Video Classifier

```bash
# Train CNN-LSTM model on the dataset
python scripts/train_video_classifier.py
```

Training will:
- Use data augmentation for robustness
- Apply class weighting for imbalanced data
- Save best model to `checkpoints/video_classifier_best.pth`
- Generate training plots and metrics in `outputs/`

### 5. Run Real-Time Demo

```bash
# Using webcam
python scripts/demo.py --source webcam --checkpoint checkpoints/video_classifier_best.pth

# Using video file
python scripts/demo.py --source video --video path/to/video.mp4 --output results.mp4
```

**Demo Controls**:
- Press `q` to quit
- Press `s` to save screenshot

---

## ⚙️ Configuration

Edit `configs/config.yaml` to customize:

- **Dataset settings**: Resolution, FPS, clip length, split ratios
- **Model architecture**: CNN-LSTM vs 3D-CNN, backbone (ResNet18/34/50)
- **Training hyperparameters**: Batch size, learning rate, epochs
- **Ensemble weights**: Relative importance of each stream
- **Demo settings**: Input source, display options

---

## 📈 Model Architecture

### Stream 1: Video Classifier (CNN-LSTM)

```
Input Video (16 frames, 224×224×3)
    ↓
CNN Feature Extractor (ResNet18/34/50)
    ↓ (per-frame features)
LSTM Temporal Modeling (2 layers, 256 hidden)
    ↓
Fully Connected Layer
    ↓
Output: [Normal, Shoplifting] probabilities
```

**Alternative**: 3D-CNN for end-to-end spatiotemporal learning

### Stream 2: Object Detector (YOLOv8) - Optional

- Detects suspicious activities at frame level
- Classes: Normal, Suspicious, Theft
- Requires Roboflow dataset for training

### Stream 3: Anomaly Detector - Optional

- Analyzes optical flow and motion statistics
- Uses Isolation Forest or simple neural classifier
- Flags unusual movement patterns

### Ensemble Fusion

**Soft Voting**: Weighted average of stream probabilities  
**Meta-Classifier**: Learned fusion using small neural network

---

## 📊 Training Results

After training, you'll find in `outputs/`:

- `training_history.png` - Loss and accuracy curves
- `confusion_matrix.png` - Test set confusion matrix
- `classification_report.txt` - Precision, recall, F1 scores

Example metrics (will vary based on dataset):
```
              precision    recall  f1-score   support

      Normal     0.9200    0.9500    0.9347       100
 Shoplifting     0.9400    0.9100    0.9247        95

    accuracy                         0.9308       195
```

---

## 🎥 Real-Time Inference

The demo script provides:

- ✅ Live webcam or video file input
- ✅ Real-time classification (NORMAL/SUSPICIOUS/THEFT)
- ✅ Confidence scores for each class
- ✅ Optional bounding box detections (if YOLO enabled)
- ✅ FPS counter
- ✅ Video output recording

---

## 🔧 Advanced Usage

### Custom Training

```bash
# Resume training from checkpoint
python scripts/train_video_classifier.py --resume checkpoints/checkpoint.pth

# Use different config
python scripts/train_video_classifier.py --config configs/custom_config.yaml
```

### Model Selection

Edit `config.yaml`:

```yaml
models:
  video_classifier:
    type: "cnn_lstm"  # or "3d_cnn"
    backbone: "resnet18"  # or "resnet34", "resnet50"
```

### Ensemble Configuration

```yaml
models:
  ensemble:
    method: "soft_voting"  # or "meta_classifier"
    weights:
      video_classifier: 0.5
      object_detector: 0.3
      anomaly_detector: 0.2
```

---

## 🤔 Ethical Considerations

### Important Limitations

⚠️ **This system is for educational/research purposes only**

1. **Privacy Concerns**: Surveillance systems must comply with local privacy laws
2. **Bias & Fairness**: Models may exhibit bias based on training data
3. **False Positives**: No system is 100% accurate; human oversight is critical
4. **Misuse Potential**: Technology should not be used for discriminatory practices

### Responsible Use Guidelines

- ✅ Always inform individuals about surveillance in retail spaces
- ✅ Use as a support tool, not sole decision-maker
- ✅ Regularly audit model performance across demographics
- ✅ Implement human review before taking action
- ✅ Store data securely and delete after appropriate period
- ✅ Comply with GDPR, CCPA, and local regulations

---

## 📚 References

### Dataset
- MNNIT Allahabad CV Laboratory. (2022). Shoplifting Dataset. Mendeley Data.
- Innovatiana Shoplifting Video Dataset

### Models & Frameworks
- PyTorch: https://pytorch.org/
- Ultralytics YOLOv8: https://github.com/ultralytics/ultralytics
- Albumentations: https://albumentations.ai/

### Research
- Donahue et al. (2015). Long-term Recurrent Convolutional Networks for Visual Recognition
- Tran et al. (2018). A Closer Look at Spatiotemporal Convolutions for Action Recognition

---

## 🛠️ Troubleshooting

### Common Issues

**No videos found**:
- Ensure videos are in `data/raw_videos/normal/` and `data/raw_videos/shoplifting/`
- Supported formats: `.mp4`, `.avi`, `.mov`, `.mkv`

**CUDA out of memory**:
- Reduce batch size in `config.yaml`
- Use smaller backbone (`resnet18` instead of `resnet50`)
- Reduce clip length or input resolution

**Low FPS in demo**:
- Use GPU for inference
- Disable object detector and anomaly detector for speed
- Reduce input resolution

**Import errors**:
- Ensure virtual environment is activated
- Reinstall requirements: `pip install -r requirements.txt --force-reinstall`

---

## 📄 License

This project is licensed under the MIT License - see LICENSE file for details.

---

## 🙏 Acknowledgments

- MNNIT Allahabad CV Laboratory for creating the shoplifting dataset
- PyTorch and Ultralytics teams for excellent deep learning frameworks
- Open-source community for various utilities and tools

---

## 📧 Contact & Support

For questions, issues, or contributions:
- Open an issue on GitHub
- Email: janarthananmuthu107@gmail.com

---

**Built with ❤️ for the computer vision and AI community**
