# 📥 Download Real MNNIT Shoplifting Dataset

## 🎯 Dataset Overview

**MNNIT Allahabad Shoplifting Video Dataset**
- **Videos**: 150+ surveillance videos
- **Classes**: Normal (75+), Shoplifting (75+)
- **Quality**: Real surveillance footage from simulated retail environment
- **Resolution**: 640×480 pixels
- **FPS**: 30 frames per second
- **Format**: MP4
- **Size**: ~2-3 GB total

---

## 📍 **Download Sources** (Try in Order)

### **Method 1: Kaggle** ⭐ (Recommended - Easiest)

1. **Create Kaggle Account** (if you don't have one):
   - Go to [kaggle.com](https://www.kaggle.com)
   - Sign up (it's free!)

2. **Search for Dataset**:
   - Go to: https://www.kaggle.com/datasets
   - Search: `"Shoplifting Video Dataset"` or `"MNNIT Shoplifting"`
   - Look for datasets with ~150 videos

3. **Download Dataset**:
   ```bash
   # Option A: Download via browser
   # Click "Download" button on dataset page
   # Save to Downloads folder
   
   # Option B: Use Kaggle API (faster for large files)
   pip install kaggle
   
   # Get API token:
   # 1. Go to kaggle.com/account
   # 2. Click "Create New API Token"
   # 3. Save kaggle.json to ~/.kaggle/
   
   mkdir -p ~/.kaggle
   # Move your downloaded kaggle.json here
   chmod 600 ~/.kaggle/kaggle.json
   
   # Download dataset (replace with actual dataset name)
   kaggle datasets download -d USERNAME/shoplifting-dataset
   ```

4. **Extract**:
   ```bash
   cd ~/Downloads
   unzip shoplifting-dataset.zip -d shoplifting_data
   ```

---

### **Method 2: Mendeley Data** 📚

1. **Visit Mendeley**:
   - URL: https://data.mendeley.com
   - Search: `"Shoplifting Dataset"` or `"MNNIT Allahabad"`

2. **Look for**:
   - Title: "Shoplifting Video Dataset" or similar
   - Author: MNNIT Allahabad researchers
   - Year: 2019-2022

3. **Download**:
   - Click "Download dataset"
   - May require free Mendeley account
   - Extract the ZIP file

---

### **Method 3: GitHub/Academic Repositories** 🔬

1. **Search GitHub**:
   ```bash
   # Search on GitHub:
   https://github.com/search?q=shoplifting+dataset
   
   # Or Google:
   "MNNIT Shoplifting Dataset" site:github.com
   ```

2. **Look for**:
   - Research paper implementations
   - Dataset mirrors
   - Links to original source

3. **Clone/Download**:
   ```bash
   # If found on GitHub
   git clone <repository-url>
   ```

---

### **Method 4: Research Paper/Direct Contact** 📧

1. **Find Research Paper**:
   - Search Google Scholar: "Shoplifting Detection MNNIT Allahabad"
   - Look for papers from 2019-2022

2. **Contact Authors**:
   - Email researchers from MNNIT Allahabad CS Department
   - Request dataset access
   - Usually granted for academic/educational use

3. **Institutional Access**:
   - Check if your university has access
   - Ask your professor/supervisor

---

### **Method 5: Alternative Datasets** 🔄

If MNNIT dataset is unavailable, use these alternatives:

**UCF-Crime Dataset** (larger, includes theft):
```bash
# Download from:
https://www.crcv.ucf.edu/projects/real-world/
# Filter for "Shoplifting" and "Stealing" categories
```

**Custom Dataset Collection**:
```bash
# Create your own using:
# 1. YouTube videos (with proper licensing)
# 2. Recreate scenarios (like our demo generator)
# 3. Public surveillance datasets
```

---

## 📁 **After Download: Organize Files**

Once you have the dataset, organize it:

```bash
cd /Users/janatheboss/.gemini/antigravity/scratch/Theft-detection-NUS-AI-ML-Program

# Create structure
mkdir -p data/raw_videos/normal
mkdir -p data/raw_videos/shoplifting

# Move downloaded videos
# Example if downloaded to ~/Downloads/shoplifting_data/
cp ~/Downloads/shoplifting_data/Normal/*.mp4 data/raw_videos/normal/
cp ~/Downloads/shoplifting_data/Shoplifting/*.mp4 data/raw_videos/shoplifting/

# Or if different structure, manually organize
```

**Expected Structure**:
```
data/raw_videos/
├── normal/
│   ├── normal_001.mp4
│   ├── normal_002.mp4
│   ├── normal_003.mp4
│   └── ... (75+ videos)
└── shoplifting/
    ├── shoplifting_001.mp4
    ├── shoplifting_002.mp4
    ├── shoplifting_003.mp4
    └── ... (75+ videos)
```

---

## ✅ **Verify Dataset**

After organizing, verify:

```bash
cd /Users/janatheboss/.gemini/antigravity/scratch/Theft-detection-NUS-AI-ML-Program
source venv/bin/activate

python3 scripts/download_dataset.py --check
```

**Expected Output**:
```
✓ Found 78 normal videos
✓ Found 76 shoplifting videos
✓ Total: 154 videos
✓ Dataset ready for training!
```

---

## 🚨 **Troubleshooting**

### **Can't find dataset on Kaggle?**
- Try search terms: "shoplifting", "theft detection", "surveillance"
- Look for datasets with 100+ videos
- Check upload date (2019-2022)

### **Dataset requires approval?**
- Some datasets need academic approval
- Use your university email
- Mention educational/research use

### **Different video formats?**
```bash
# Convert to MP4 if needed
for file in *.avi; do
    ffmpeg -i "$file" "${file%.avi}.mp4"
done
```

### **Files too large?**
- Download in batches
- Use good internet connection
- Consider cloud download (Google Colab, AWS)

---

## 📊 **Dataset Statistics**

After organizing, you should have approximately:

| Metric | Value |
|--------|-------|
| Total Videos | 150+ |
| Normal Class | 75+ videos |
| Shoplifting Class | 75+ videos |
| Total Size | 2-3 GB |
| Avg Video Length | 10-30 seconds |
| Resolution | 640×480 |
| FPS | 30 |

---

## 🔗 **Useful Links**

- **Kaggle Datasets**: https://www.kaggle.com/datasets
- **Mendeley Data**: https://data.mendeley.com
- **UCF Crime Dataset**: https://www.crcv.ucf.edu/projects/real-world/
- **Papers with Code**: https://paperswithcode.com (search "shoplifting detection")

---

## ✨ **Next Steps**

Once dataset is downloaded and organized:

1. ✅ Verify structure (see above)
2. ✅ Run: `python3 scripts/prepare_data.py`
3. ✅ Update config for real training
4. ✅ Run: `python3 scripts/train_video_classifier.py`

See `TRAIN_ON_REAL_DATA.md` for detailed training instructions!

---

**Good luck with the download! Let me know when you have the dataset ready!** 🚀
