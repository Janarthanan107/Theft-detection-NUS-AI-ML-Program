#!/usr/bin/env python3
"""
Real-time demo script for shoplifting detection.
Runs inference on webcam or video file using the ensemble model.
"""

import os
import sys
import argparse
from pathlib import Path
import cv2
import torch
import numpy as np
from collections import deque
import time

sys.path.append(str(Path(__file__).parent.parent))

from utils import (
    load_config, setup_logging, get_device,
    load_checkpoint, sliding_window_clips, compute_motion_statistics,
    draw_label_on_frame, draw_fps, create_video_from_frames
)
from models import build_video_classifier, build_anomaly_detector, build_ensemble_model

try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None


class ShopliftingDetector:
    """Real-time shoplifting detection system."""
    
    def __init__(self, config_path='configs/config.yaml', checkpoint_path=None):
        """
        Initialize the detector.
        
        Args:
            config_path: Path to configuration file
            checkpoint_path: Path to video classifier checkpoint
        """
        self.config = load_config(config_path)
        self.device = get_device(self.config['device'])
        
        # Load video classifier
        model_config = self.config['models']['video_classifier']
        model_config['num_classes'] = self.config['dataset']['num_classes']
        self.video_classifier = build_video_classifier(model_config)
        
        if checkpoint_path and os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            self.video_classifier.load_state_dict(checkpoint['model_state_dict'])
            print(f"✓ Loaded video classifier from: {checkpoint_path}")
        
        self.video_classifier = self.video_classifier.to(self.device)
        self.video_classifier.eval()
        
        # Load object detector (optional)
        self.object_detector = None
        if self.config['models']['object_detector']['enabled'] and YOLO is not None:
            try:
                model_name = self.config['models']['object_detector']['model']
                self.object_detector = YOLO(f"{model_name}.pt")
                print(f"✓ Loaded object detector: {model_name}")
            except Exception as e:
                print(f"Warning: Could not load YOLO detector: {e}")
        
        # Load anomaly detector (optional)
        self.anomaly_detector = None
        if self.config['models']['anomaly_detector']['enabled']:
            self.anomaly_detector = build_anomaly_detector(
                self.config['models']['anomaly_detector']
            )
            print("✓ Initialized anomaly detector")
        
        # Frame buffer for temporal processing
        self.frame_buffer = deque(maxlen=self.config['dataset']['clip_length'])
        
        # Class names
        self.class_names = ['NORMAL', 'SUSPICIOUS', 'THEFT']
    
    def preprocess_frames(self, frames):
        """Preprocess frames for model input."""
        processed = []
        for frame in frames:
            # Resize
            frame_resized = cv2.resize(
                frame,
                tuple(self.config['dataset']['input_resolution'])
            )
            # Convert to RGB
            frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
            processed.append(frame_rgb)
        
        # Convert to numpy array
        frames_array = np.array(processed, dtype=np.float32) / 255.0
        
        # Normalize
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        frames_array = (frames_array - mean) / std
        
        # Convert to tensor: (T, H, W, C) -> (1, T, C, H, W)
        frames_tensor = torch.from_numpy(frames_array).permute(0, 3, 1, 2).unsqueeze(0)
        
        return frames_tensor.float().to(self.device)
    
    def predict(self, frame):
        """
        Run prediction on a frame.
        
        Args:
            frame: Input frame (BGR format)
            
        Returns:
            Dictionary with prediction results
        """
        # Add frame to buffer
        self.frame_buffer.append(frame.copy())
        
        # Need enough frames for prediction
        if len(self.frame_buffer) < self.config['dataset']['clip_length']:
            return {
                'label': 'BUFFERING',
                'confidence': 0.0,
                'class_probs': [0.0, 0.0, 0.0]
            }
        
        # Preprocess frames
        frames_tensor = self.preprocess_frames(list(self.frame_buffer))
        
        # Video classifier prediction
        with torch.no_grad():
            logits, features = self.video_classifier(frames_tensor)
            probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
        
        # Binary to 3-class mapping
        if len(probs) == 2:
            class_probs = [
                probs[0],  # Normal
                probs[1] * 0.3,  # Suspicious
                probs[1] * 0.7   # Theft
            ]
        else:
            class_probs = probs.tolist()
        
        # Get prediction
        pred_idx = np.argmax(class_probs)
        label = self.class_names[pred_idx]
        confidence = class_probs[pred_idx]
        
        # Optionally run object detector
        detections = []
        if self.object_detector is not None:
            try:
                results = self.object_detector(frame, verbose=False)
                if len(results) > 0 and len(results[0].boxes) > 0:
                    boxes = results[0].boxes
                    for box in boxes:
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        conf = box.conf[0].cpu().numpy()
                        cls = int(box.cls[0].cpu().numpy())
                        detections.append({
                            'bbox': (int(x1), int(y1), int(x2), int(y2)),
                            'label': self.class_names[min(cls, len(self.class_names)-1)],
                            'confidence': float(conf)
                        })
            except Exception as e:
                pass
        
        return {
            'label': label,
            'confidence': confidence,
            'class_probs': class_probs,
            'detections': detections
        }
    
    def visualize_results(self, frame, results, fps=None):
        """
        Draw prediction results on frame.
        
        Args:
            frame: Input frame
            results: Prediction results dictionary
            fps: Current FPS
            
        Returns:
            Annotated frame
        """
        annotated = frame.copy()
        
        # Draw main label
        label = results['label']
        confidence = results['confidence']
        annotated = draw_label_on_frame(
            annotated, label, confidence,
            position=(10, 40),
            font_scale=1.2,
            thickness=3
        )
        
        # Draw class probabilities
        y_offset = 80
        for i, (class_name, prob) in enumerate(zip(self.class_names, results['class_probs'])):
            text = f"{class_name}: {prob:.2%}"
            color = (0, 255, 0) if i == 0 else (0, 165, 255) if i == 1 else (0, 0, 255)
            cv2.putText(
                annotated, text,
                (10, y_offset + i * 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6, color, 2
            )
        
        # Draw detections
        for det in results.get('detections', []):
            x1, y1, x2, y2 = det['bbox']
            det_label = det['label']
            det_conf = det['confidence']
            
            # Draw box
            color = (0, 255, 0) if det_label == 'NORMAL' else (0, 0, 255)
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            
            # Draw label
            label_text = f"{det_label}: {det_conf:.2f}"
            cv2.putText(
                annotated, label_text,
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5, color, 2
            )
        
        # Draw FPS
        if fps is not None:
            annotated = draw_fps(annotated, fps)
        
        return annotated


def main():
    parser = argparse.ArgumentParser(description="Real-time shoplifting detection demo")
    parser.add_argument('--config', type=str, default='configs/config.yaml',
                        help='Path to configuration file')
    parser.add_argument('--checkpoint', type=str,
                        default='checkpoints/video_classifier_best.pth',
                        help='Path to video classifier checkpoint')
    parser.add_argument('--source', type=str, default='webcam',
                        choices=['webcam', 'video'],
                        help='Input source: webcam or video file')
    parser.add_argument('--video', type=str, default=None,
                        help='Path to video file (if source is video)')
    parser.add_argument('--output', type=str, default=None,
                        help='Path to save output video')
    
    args = parser.parse_args()
    
    print("="*60)
    print("SHOPLIFTING DETECTION SYSTEM - REAL-TIME DEMO")
    print("="*60)
    
    # Initialize detector
    detector = ShopliftingDetector(args.config, args.checkpoint)
    
    # Open video source
    if args.source == 'webcam':
        cap = cv2.VideoCapture(0)
        print("\n✓ Opened webcam")
    else:
        if args.video is None:
            print("\nError: --video argument required when source is 'video'")
            sys.exit(1)
        cap = cv2.VideoCapture(args.video)
        print(f"\n✓ Opened video: {args.video}")
    
    if not cap.isOpened():
        print("\nError: Could not open video source")
        sys.exit(1)
    
    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Output video writer
    out_writer = None
    if args.output:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out_writer = cv2.VideoWriter(args.output, fourcc, fps, (width, height))
        print(f"✓ Output will be saved to: {args.output}")
    
    print(f"\nResolution: {width}x{height} @ {fps} FPS")
    print("\nPress 'q' to quit, 's' to screenshot\n")
    
    # FPS calculation
    frame_count = 0
    start_time = time.time()
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("\nEnd of video or camera error")
                break
            
            frame_count += 1
            
            # Run detection
            results = detector.predict(frame)
            
            # Calculate FPS
            elapsed = time.time() - start_time
            current_fps = frame_count / elapsed if elapsed > 0 else 0
            
            # Visualize
            annotated_frame = detector.visualize_results(frame, results, current_fps)
            
            # Display
            cv2.imshow('Shoplifting Detection', annotated_frame)
            
            # Write to output
            if out_writer is not None:
                out_writer.write(annotated_frame)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("\nQuitting...")
                break
            elif key == ord('s'):
                screenshot_path = f"screenshot_{frame_count}.jpg"
                cv2.imwrite(screenshot_path, annotated_frame)
                print(f"Screenshot saved: {screenshot_path}")
    
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
    
    finally:
        # Cleanup
        cap.release()
        if out_writer is not None:
            out_writer.release()
        cv2.destroyAllWindows()
        
        print(f"\n✓ Processed {frame_count} frames")
        print(f"✓ Average FPS: {current_fps:.2f}")
        if args.output:
            print(f"✓ Output saved to: {args.output}")
        
        print("\n" + "="*60)
        print("DEMO COMPLETED")
        print("="*60 + "\n")


if __name__ == '__main__':
    main()
