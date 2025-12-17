"""
Flask Backend for Theft Detection System
Connects the web UI to the PyTorch model
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
import cv2
import numpy as np
import base64
from io import BytesIO
from PIL import Image
import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.video_classifier import VideoClassifier
from utils.video_processing import preprocess_frame

app = Flask(__name__)
CORS(app)  # Enable CORS for web UI

# Load trained model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = None
frame_buffer = []
BUFFER_SIZE = 16  # Number of frames to analyze

def load_model(checkpoint_path='checkpoints/video_classifier_best.pth'):
    """Load the trained model"""
    global model
    
    try:
        model = VideoClassifier(
            model_type='cnn_lstm',
            backbone='resnet18',
            num_classes=2,
            hidden_dim=256,
            num_layers=2
        ).to(device)
        
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        print(f"✅ Model loaded successfully from {checkpoint_path}")
        return True
        
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return False

@app.route('/api/health', methods=['GET'])
def health_check():
    """Check if backend is running and model is loaded"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'device': str(device)
    })

@app.route('/api/detect', methods=['POST'])
def detect():
    """
    Process a single frame and return detection results
    Expects: { "frame": "base64_encoded_image" }
    Returns: { "prediction": "normal|suspicious|theft", "confidence": {...} }
    """
    global frame_buffer
    
    if model is None:
        return jsonify({'error': 'Model not loaded'}), 500
    
    try:
        # Get frame from request
        data = request.json
        frame_data = data.get('frame')
        
        if not frame_data:
            return jsonify({'error': 'No frame provided'}), 400
        
        # Decode base64 image
        if 'base64,' in frame_data:
            frame_data = frame_data.split('base64,')[1]
        
        image_bytes = base64.b64decode(frame_data)
        image = Image.open(BytesIO(image_bytes))
        frame = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        
        # Preprocess frame
        processed_frame = preprocess_frame(frame, size=(224, 224))
        
        # Add to buffer
        frame_buffer.append(processed_frame)
        
        # Keep only last BUFFER_SIZE frames
        if len(frame_buffer) > BUFFER_SIZE:
            frame_buffer.pop(0)
        
        # Need at least BUFFER_SIZE frames for prediction
        if len(frame_buffer) < BUFFER_SIZE:
            return jsonify({
                'prediction': 'buffering',
                'confidence': {
                    'normal': 0,
                    'suspicious': 0,
                    'theft': 0
                },
                'message': f'Buffering frames: {len(frame_buffer)}/{BUFFER_SIZE}'
            })
        
        # Prepare batch for model
        video_tensor = torch.stack(frame_buffer).unsqueeze(0)  # Shape: (1, 16, C, H, W)
        video_tensor = video_tensor.to(device)
        
        # Run inference
        with torch.no_grad():
            outputs = model(video_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            probs = probabilities[0].cpu().numpy()
        
        # Get prediction
        pred_idx = int(torch.argmax(outputs, dim=1).item())
        pred_label = 'normal' if pred_idx == 0 else 'theft'
        
        # Calculate suspicious as intermediate
        normal_conf = float(probs[0]) * 100
        theft_conf = float(probs[1]) * 100
        suspicious_conf = max(0, 100 - normal_conf - theft_conf)  # Intermediate zone
        
        # Determine final classification
        if theft_conf > 70:
            final_pred = 'theft'
        elif theft_conf > 40 or suspicious_conf > 30:
            final_pred = 'suspicious'
        else:
            final_pred = 'normal'
        
        return jsonify({
            'prediction': final_pred,
            'confidence': {
                'normal': round(normal_conf, 2),
                'suspicious': round(suspicious_conf, 2),
                'theft': round(theft_conf, 2)
            },
            'raw_output': {
                'class_0': round(float(probs[0]), 4),
                'class_1': round(float(probs[1]), 4)
            }
        })
        
    except Exception as e:
        print(f"Error in detection: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/reset', methods=['POST'])
def reset_buffer():
    """Reset the frame buffer"""
    global frame_buffer
    frame_buffer = []
    return jsonify({'status': 'buffer_reset'})

if __name__ == '__main__':
    print("🚀 Starting Theft Detection Backend...")
    print(f"📱 Device: {device}")
    
    # Try to load model
    if os.path.exists('checkpoints/video_classifier_best.pth'):
        load_model()
    else:
        print("⚠️  No trained model found. Train the model first using:")
        print("   python scripts/train_video_classifier.py")
        print("\n   Backend will start but detection won't work until model is trained.")
    
    # Start Flask server
    app.run(host='0.0.0.0', port=5000, debug=True)
