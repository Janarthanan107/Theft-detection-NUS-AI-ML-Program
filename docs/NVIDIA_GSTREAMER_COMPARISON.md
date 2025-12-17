# NVIDIA Models & GStreamer Integration Guide

## 📋 Table of Contents
1. [Overview](#overview)
2. [NVIDIA Solutions](#nvidia-solutions)
3. [GStreamer Integration](#gstreamer-integration)
4. [Comprehensive Comparison](#comprehensive-comparison)
5. [Implementation Examples](#implementation-examples)
6. [Recommendations](#recommendations)

---

## Overview

This document compares **all available approaches** for building a shoplifting detection system, including:

- **Current PyTorch Implementation** (CNN-LSTM, YOLOv8)
- **NVIDIA TAO Toolkit** (ActionRecognitionNet, PeopleNet, PeopleSemSegNet)
- **NVIDIA DeepStream SDK** (Hardware-accelerated pipeline)
- **GStreamer** (Video pipeline framework)
- **Hybrid Approaches** (Combining multiple technologies)

---

## NVIDIA Solutions

### 1. NVIDIA TAO (Train, Adapt, Optimize) Toolkit

**What is it?**  
TAO Toolkit is NVIDIA's transfer learning toolkit that provides pre-trained models optimized for various computer vision tasks.

#### Available Models for Shoplifting Detection

| Model | Purpose | Input | Output | Best For |
|-------|---------|-------|--------|----------|
| **ActionRecognitionNet** | Video action classification | Video clips | Action labels + confidence | **Primary choice** for shoplifting detection |
| **PeopleNet** | Person detection | Single frames | Bounding boxes | Detecting people in retail |
| **PeopleSemSegNet** | Person segmentation | Single frames | Segmentation masks | Detailed person analysis |
| **Retail Object Detector** | Product detection | Single frames | Object boxes | Inventory tracking |
| **Pose Estimation** | Body pose | Single frames | Keypoints | Suspicious gesture detection |

#### ActionRecognitionNet Details

```yaml
Architecture: 3D ResNet-18 or R2+1D
Input: RGB video clips (16-32 frames)
Pre-training: Kinetics-400 or Kinetics-600
Classes: Customizable (transfer learning)
Inference: TensorRT optimized
Hardware: NVIDIA GPU (Jetson, RTX, Tesla)
```

**Advantages**:
- ✅ Pre-trained on massive action recognition datasets
- ✅ TensorRT optimization for 5-10x faster inference
- ✅ Easy fine-tuning on custom shoplifting data
- ✅ INT8 quantization for edge devices
- ✅ Official NVIDIA support

**Disadvantages**:
- ❌ Requires NVIDIA GPU
- ❌ Less flexibility than PyTorch
- ❌ Steeper learning curve
- ❌ License restrictions for commercial use

#### How to Use TAO for Shoplifting Detection

```bash
# 1. Install TAO Toolkit
pip install nvidia-tao

# 2. Download ActionRecognitionNet pretrained model
tao action_recognition download_specs \
    -o /workspace/specs \
    -r /workspace/specs/default_spec.txt

# 3. Prepare dataset in TAO format
# Directory structure:
# dataset/
#   train/
#     normal/
#       video1.mp4
#     shoplifting/
#       video1.mp4
#   val/
#   test/

# 4. Fine-tune on shoplifting dataset
tao action_recognition train \
    -e /workspace/specs/train_spec.txt \
    -r /workspace/results \
    -k $KEY

# 5. Export to TensorRT
tao action_recognition export \
    -m /workspace/results/model.tlt \
    -k $KEY \
    -e /workspace/specs/export_spec.txt

# 6. Run inference
tao action_recognition inference \
    -m /workspace/results/model.engine \
    -i /path/to/video.mp4
```

---

### 2. NVIDIA DeepStream SDK

**What is it?**  
DeepStream is an SDK for building AI-powered video analytics pipelines, optimized for NVIDIA GPUs.

#### Key Features

- **Hardware Acceleration**: Decode, inference, encode on GPU
- **Multi-stream**: Process 100+ video streams simultaneously
- **Low Latency**: <50ms end-to-end for real-time detection
- **Edge Deploy**: Runs on Jetson Nano, Xavier, Orin
- **GStreamer-based**: Built on GStreamer pipeline architecture

#### DeepStream Pipeline for Shoplifting Detection

```
┌─────────────────────────────────────────────────────────────────┐
│                    DEEPSTREAM PIPELINE                           │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Input Source (RTSP/File/USB)                                   │
│         ↓                                                        │
│  nvstreammux (Batch multiple streams)                           │
│         ↓                                                        │
│  nvinfer (Primary: Person Detection - PeopleNet)                │
│         ↓                                                        │
│  nvtracker (Track individuals across frames)                    │
│         ↓                                                        │
│  nvinfer (Secondary: Action Recognition - ActionRecognitionNet) │
│         ↓                                                        │
│  nvmultistreamtiler (Compose output grid)                       │
│         ↓                                                        │
│  nvdsosd (Draw bounding boxes, labels)                          │
│         ↓                                                        │
│  nvvideoconvert + Encoder                                       │
│         ↓                                                        │
│  Output (Display/File/RTSP)                                     │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

#### DeepStream Configuration Example

```ini
# deepstream_shoplifting_config.txt

[application]
enable-perf-measurement=1
perf-measurement-interval-sec=5

[source0]
enable=1
type=3  # URI
uri=file:///path/to/video.mp4
num-sources=1

[streammux]
gpu-id=0
batch-size=1
width=1920
height=1080
batched-push-timeout=40000

[primary-gie]
enable=1
gpu-id=0
model-engine-file=models/peoplenet.engine
labelfile-path=models/labels_peoplenet.txt
batch-size=1
interval=0
gie-unique-id=1
config-file=config_infer_primary_peoplenet.txt

[secondary-gie0]
enable=1
gpu-id=0
model-engine-file=models/actionrecognition.engine
labelfile-path=models/labels_actions.txt
batch-size=1
gie-unique-id=2
operate-on-gie-id=1  # Operate on primary detector output
config-file=config_infer_secondary_action.txt

[tracker]
enable=1
tracker-width=640
tracker-height=384
ll-lib-file=/opt/nvidia/deepstream/lib/libnvds_nvmultiobjecttracker.so
ll-config-file=config_tracker_NvDCF_perf.yml

[osd]
enable=1
gpu-id=0
border-width=2
text-size=15
text-color=1;1;1;1
text-bg-color=0.3;0.3;0.3;1

[sink0]
enable=1
type=2  # File
codec=1  # H264
output-file=out.mp4
```

**Advantages**:
- ✅ **Extreme performance**: 100+ FPS on RTX 3090
- ✅ **Multi-stream**: Handle multiple cameras simultaneously
- ✅ **Production-ready**: Used in real surveillance systems
- ✅ **Hardware decode/encode**: Full GPU acceleration
- ✅ **Built-in tracking**: Track individuals across frames

**Disadvantages**:
- ❌ **Complex setup**: Steep learning curve
- ❌ **NVIDIA-only**: Requires NVIDIA GPU
- ❌ **Less flexible**: Harder to customize than PyTorch
- ❌ **Large overhead**: Overkill for single-camera prototypes

---

## GStreamer Integration

### What is GStreamer?

GStreamer is a powerful **multimedia framework** for building video/audio processing pipelines. It's hardware-agnostic and supports:

- Hardware acceleration (NVDEC, VAAPI, QuickSync)
- Network streaming (RTSP, RTP, HTTP)
- Format conversion
- Filters and effects
- Plugin architecture

### Why Use GStreamer?

| Feature | Benefit |
|---------|---------|
| **Hardware decode** | Offload video decoding to GPU/dedicated hardware |
| **Network streaming** | Easy RTSP/IP camera integration |
| **Pipeline flexibility** | Chain operations modularly |
| **Cross-platform** | Works on Linux, Mac, Windows, embedded |
| **Real-time** | Low-latency processing |

### GStreamer + PyTorch Integration

You can combine GStreamer for video input/preprocessing and PyTorch for inference:

```python
import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst, GLib
import numpy as np
import torch
import cv2

# Initialize GStreamer
Gst.init(None)

class GStreamerPyTorchDetector:
    def __init__(self, source_uri, model):
        self.model = model
        self.model.eval()
        
        # GStreamer pipeline
        pipeline_str = f"""
        uridecodebin uri={source_uri} !
        videoconvert !
        video/x-raw,format=RGB,width=1280,height=720 !
        appsink name=sink emit-signals=true sync=false max-buffers=1 drop=true
        """
        
        self.pipeline = Gst.parse_launch(pipeline_str)
        self.appsink = self.pipeline.get_by_name("sink")
        self.appsink.connect("new-sample", self.on_new_sample)
        
    def on_new_sample(self, sink):
        # Pull sample from GStreamer
        sample = sink.emit("pull-sample")
        buffer = sample.get_buffer()
        caps = sample.get_caps()
        
        # Extract frame data
        success, map_info = buffer.map(Gst.MapFlags.READ)
        if not success:
            return Gst.FlowReturn.ERROR
        
        # Convert to numpy
        width = caps.get_structure(0).get_value('width')
        height = caps.get_structure(0).get_value('height')
        frame = np.ndarray(
            shape=(height, width, 3),
            dtype=np.uint8,
            buffer=map_info.data
        )
        
        # Run PyTorch inference
        with torch.no_grad():
            tensor = self.preprocess(frame)
            output = self.model(tensor)
            prediction = self.postprocess(output)
        
        # Handle prediction
        print(f"Prediction: {prediction}")
        
        buffer.unmap(map_info)
        return Gst.FlowReturn.OK
    
    def run(self):
        self.pipeline.set_state(Gst.State.PLAYING)
        
        # Run main loop
        loop = GLib.MainLoop()
        try:
            loop.run()
        except KeyboardInterrupt:
            pass
        
        self.pipeline.set_state(Gst.State.NULL)
```

### GStreamer Pipeline Examples

#### 1. RTSP Camera → PyTorch Inference

```bash
gst-launch-1.0 \
  rtspsrc location=rtsp://192.168.1.100:554/stream ! \
  rtph264depay ! \
  h264parse ! \
  nvh264dec ! \
  videoconvert ! \
  video/x-raw,format=RGB,width=1280,height=720 ! \
  appsink
```

#### 2. USB Webcam → Hardware Decode → Inference

```bash
gst-launch-1.0 \
  v4l2src device=/dev/video0 ! \
  video/x-raw,width=1920,height=1080,framerate=30/1 ! \
  nvvidconv ! \
  video/x-raw(memory:NVMM),format=NV12 ! \
  appsink
```

#### 3. Multiple Cameras → Batched Inference

```bash
gst-launch-1.0 \
  videomixer name=mix ! appsink \
  uridecodebin uri=file:///cam1.mp4 ! mix. \
  uridecodebin uri=file:///cam2.mp4 ! mix. \
  uridecodebin uri=file:///cam3.mp4 ! mix.
```

---

## Comprehensive Comparison

### Technical Comparison Matrix

| Feature | **PyTorch (Current)** | **NVIDIA TAO** | **NVIDIA DeepStream** | **GStreamer + PyTorch** |
|---------|----------------------|----------------|----------------------|------------------------|
| **Ease of Use** | ⭐⭐⭐⭐⭐ Very Easy | ⭐⭐⭐ Moderate | ⭐⭐ Complex | ⭐⭐⭐⭐ Easy |
| **Flexibility** | ⭐⭐⭐⭐⭐ Maximum | ⭐⭐⭐ Good | ⭐⭐ Limited | ⭐⭐⭐⭐ High |
| **Performance** | ⭐⭐⭐ Good (15-30 FPS) | ⭐⭐⭐⭐ Fast (50-100 FPS) | ⭐⭐⭐⭐⭐ Fastest (100+ FPS) | ⭐⭐⭐⭐ Fast |
| **Hardware Requirements** | CPU/GPU (any) | NVIDIA GPU required | NVIDIA GPU required | CPU/GPU (any) |
| **Multi-stream Support** | ⭐⭐ Manual | ⭐⭐⭐ Good | ⭐⭐⭐⭐⭐ Excellent | ⭐⭐⭐⭐ Good |
| **Model Customization** | ⭐⭐⭐⭐⭐ Full control | ⭐⭐⭐⭐ Transfer learning | ⭐⭐⭐ Pre-trained | ⭐⭐⭐⭐⭐ Full control |
| **Deployment** | ⭐⭐⭐⭐ Easy | ⭐⭐⭐ Moderate | ⭐⭐⭐⭐⭐ Production | ⭐⭐⭐⭐ Easy |
| **Edge Device Support** | ⭐⭐⭐ Basic | ⭐⭐⭐⭐⭐ Jetson optimized | ⭐⭐⭐⭐⭐ Jetson optimized | ⭐⭐⭐⭐ Good |
| **Learning Curve** | ⭐⭐⭐⭐ Gentle | ⭐⭐⭐ Moderate | ⭐⭐ Steep | ⭐⭐⭐ Moderate |
| **Community Support** | ⭐⭐⭐⭐⭐ Huge | ⭐⭐⭐ Growing | ⭐⭐⭐ NVIDIA forums | ⭐⭐⭐⭐ Large |
| **Cost** | Free | Free (non-commercial) | Free | Free |

### Performance Benchmarks (Estimated)

| Metric | PyTorch | TAO + TensorRT | DeepStream | GStreamer + PyTorch |
|--------|---------|----------------|------------|---------------------|
| **Inference FPS** (RTX 3090) | 20-30 | 60-100 | 100-200 | 30-50 |
| **Latency** (ms) | 50-100 | 20-40 | 10-20 | 30-60 |
| **GPU Memory** | 2-4 GB | 1-2 GB | 1-3 GB | 2-4 GB |
| **CPU Usage** | Medium | Low | Very Low | Low-Medium |
| **Multi-stream** (4 cameras) | 5-10 FPS | 20-30 FPS | 60-100 FPS | 15-25 FPS |

### Use Case Recommendations

| Scenario | Best Choice | Reason |
|----------|-------------|--------|
| **Capstone/Research Project** | **PyTorch (Current)** | Easy to learn, maximum flexibility, great documentation |
| **Single Camera Prototype** | **PyTorch or GStreamer+PyTorch** | Simple setup, no overkill |
| **Production System (1-4 cameras)** | **TAO + TensorRT** | Good balance of performance and ease |
| **Production System (10+ cameras)** | **DeepStream** | Multi-stream excellence, hardware acceleration |
| **Edge Device (Jetson Nano)** | **TAO + TensorRT** | Optimized for edge, INT8 quantization |
| **Cross-platform** | **PyTorch or GStreamer+PyTorch** | Works on any hardware |
| **IP Camera Integration** | **GStreamer + PyTorch or DeepStream** | Native RTSP support |
| **Low Latency Critical** | **DeepStream** | Sub-20ms end-to-end |

---

## Implementation Examples

### Option A: Current PyTorch (Recommended for Students)

**Pros**: Easy, flexible, well-documented  
**Best for**: Learning, prototyping, research

```bash
# Already implemented in this project!
python scripts/train_video_classifier.py
python scripts/demo.py
```

### Option B: NVIDIA TAO ActionRecognitionNet

**Pros**: Fast inference, edge-optimized  
**Best for**: Production with NVIDIA hardware

```bash
# Install TAO
pip install nvidia-tao

# Convert current MNNIT dataset to TAO format
python convert_to_tao_format.py

# Fine-tune ActionRecognitionNet
tao action_recognition train -e specs/train.txt

# Export to TensorRT
tao action_recognition export -m model.tlt -o model.engine

# Inference
python tao_inference.py --model model.engine --video test.mp4
```

### Option C: NVIDIA DeepStream

**Pros**: Multi-camera, ultra-fast  
**Best for**: Large-scale production deployments

```bash
# Install DeepStream
sudo apt install deepstream-6.3

# Run with config
deepstream-app -c deepstream_shoplifting_config.txt
```

### Option D: GStreamer + PyTorch Hybrid

**Pros**: Hardware decode, RTSP support, PyTorch flexibility  
**Best for**: IP camera systems, cross-platform

```python
# See code example in "GStreamer + PyTorch Integration" section above
python gstreamer_pytorch_detector.py --rtsp rtsp://camera-ip/stream
```

---

## Recommendations

### For Your NUS AI/ML Program Project

#### **Primary Recommendation: Stick with PyTorch (Current Implementation)**

**Reasons**:
1. ✅ **Educational Value**: Learn deep learning fundamentals
2. ✅ **Flexibility**: Easy to modify and experiment
3. ✅ **Portability**: Works on any hardware (Mac, Windows, Linux)
4. ✅ **Documentation**: Tons of resources and tutorials
5. ✅ **Time to Complete**: Faster development
6. ✅ **Grading**: Easier for professors to evaluate

#### **Enhancement: Add GStreamer for Video Input**

If you want to show **technical sophistication** without over-engineering:

```python
# Replace OpenCV video capture with GStreamer
# Benefits:
# - Hardware-accelerated decoding
# - RTSP camera support
# - Professional video pipeline
```

**Implementation**:
```bash
# Modify demo.py to use GStreamer backend
python scripts/demo.py --backend gstreamer --source rtsp://camera/stream
```

This shows you understand production systems while keeping PyTorch's simplicity.

### When toChoose Each Approach

#### Choose **PyTorch (Current)** if:
- 👨‍🎓 You're a student/researcher
- 💡 You want to understand deep learning deeply
- 🕒 You have limited time (2-4 weeks)
- 💻 You don't have NVIDIA GPU access guaranteed
- 📚 You want maximum learning resources

#### Choose **NVIDIA TAO** if:
- 🏢 You're building a real product
- 🎯 You have NVIDIA GPUs (RTX, Jetson)
- ⚡ You need 50-100 FPS performance
- 📦 You want easy deployment to edge devices
- 💰 Budget allows for commercial licensing

#### Choose **DeepStream** if:
- 🏭 You're building enterprise surveillance
- 📹 You need to handle 10+ cameras simultaneously
- ⚡ You need sub-20ms latency
- 🔧 You have DevOps support for complex setup
- 💰 Budget for NVIDIA enterprise support

#### Choose **GStreamer + PyTorch** if:
- 📹 You're integrating with IP cameras (RTSP)
- 🖥️ You need cross-platform support
- ⚡ You want hardware decode acceleration
- 🔧 You have moderate technical skills
- 🎯 Single to few cameras (1-4)

---

## Next Steps

### Immediate Actions

1. **Continue with PyTorch implementation** (already 90% complete!)
2. **Optional Enhancement**: Add GStreamer video input backend
3. **Documentation**: Create comparison slides for presentation

### Future Enhancements (Post-Capstone)

1. **Convert to TAO**: For 5-10x speedup
2. **Deploy to Jetson**: For edge demonstration
3. **Add DeepStream**: For multi-camera scaling

### Sample Enhancement: GStreamer Backend

I can add a GStreamer-based video capture option to the current demo.py if you'd like. This would give you:
- Hardware-accelerated video decoding
- RTSP camera support
- Professional pipeline architecture

Would you like me to implement this enhancement?

---

## Summary

| Approach | Complexity | Performance | Flexibility | Best For |
|----------|-----------|-------------|-------------|----------|
| **PyTorch** | ⭐⭐ Easy | ⭐⭐⭐ Good | ⭐⭐⭐⭐⭐ Maximum | **Students, Research** |
| **TAO** | ⭐⭐⭐ Moderate | ⭐⭐⭐⭐ Great | ⭐⭐⭐ Good | **Production (NVIDIA)** |
| **DeepStream** | ⭐⭐⭐⭐⭐ Complex | ⭐⭐⭐⭐⭐ Excellent | ⭐⭐ Limited | **Enterprise Multi-camera** |
| **GStreamer+PyTorch** | ⭐⭐⭐ Moderate | ⭐⭐⭐⭐ Great | ⭐⭐⭐⭐ High | **IP Cameras, Cross-platform** |

**My Recommendation for You**: Keep the current PyTorch implementation and optionally add GStreamer for video input to demonstrate technical depth without complexity overhead.
