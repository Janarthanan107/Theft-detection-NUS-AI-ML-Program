"""
Ensemble model that combines predictions from multiple streams.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Optional
from ultralytics import YOLO


class EnsembleModel(nn.Module):
    """
    Ensemble model combining:
    - Stream 1: Video Classifier (CNN-LSTM / 3D-CNN)
    - Stream 2: Object Detector (YOLO)
    - Stream 3: Anomaly Detector (Motion-based)
    """
    
    def __init__(
        self,
        video_classifier: nn.Module,
        object_detector: Optional[YOLO] = None,
        anomaly_detector = None,
        num_classes: int = 3,
        ensemble_method: str = 'soft_voting',
        weights: Dict[str, float] = None
    ):
        """
        Args:
            video_classifier: Trained video classification model
            object_detector: Trained YOLO detector (optional)
            anomaly_detector: Trained anomaly detector (optional)
            num_classes: Number of output classes (NORMAL, SUSPICIOUS, THEFT)
            ensemble_method: 'soft_voting' or 'meta_classifier'
            weights: Dictionary of weights for each stream
        """
        super(EnsembleModel, self).__init__()
        
        self.video_classifier = video_classifier
        self.object_detector = object_detector
        self.anomaly_detector = anomaly_detector
        self.num_classes = num_classes
        self.ensemble_method = ensemble_method
        
        # Default weights
        if weights is None:
            self.weights = {
                'video_classifier': 0.5,
                'object_detector': 0.3,
                'anomaly_detector': 0.2
            }
        else:
            self.weights = weights
        
        # Meta-classifier for learned fusion (optional)
        if ensemble_method == 'meta_classifier':
            # Input: predictions from all streams
            meta_input_size = num_classes * 3  # Assuming 3 streams
            self.meta_classifier = nn.Sequential(
                nn.Linear(meta_input_size, 64),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(64, 32),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(32, num_classes)
            )
    
    def forward(
        self,
        video_clip: torch.Tensor,
        frames: Optional[List[np.ndarray]] = None,
        motion_stats: Optional[Dict] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass combining all streams.
        
        Args:
            video_clip: Video tensor for classifier (batch, T, C, H, W)
            frames: List of frames for object detector
            motion_stats: Motion statistics for anomaly detector
            
        Returns:
            Dictionary with predictions and scores
        """
        outputs = {}
        all_probs = []
        
        # Stream 1: Video Classifier
        with torch.set_grad_enabled(self.training):
            video_logits, video_features = self.video_classifier(video_clip)
            video_probs = torch.softmax(video_logits, dim=1)
            
            outputs['video_logits'] = video_logits
            outputs['video_probs'] = video_probs
            outputs['video_features'] = video_features
            
            # Map binary classification to 3-class (if needed)
            if video_probs.shape[1] == 2:
                # Binary: [Normal, Shoplifting] -> [Normal, Suspicious, Theft]
                # Map shoplifting to both suspicious and theft
                batch_size = video_probs.shape[0]
                video_probs_3class = torch.zeros(batch_size, 3, device=video_probs.device)
                video_probs_3class[:, 0] = video_probs[:, 0]  # Normal
                video_probs_3class[:, 1] = video_probs[:, 1] * 0.3  # Suspicious
                video_probs_3class[:, 2] = video_probs[:, 1] * 0.7  # Theft
                all_probs.append(video_probs_3class * self.weights['video_classifier'])
            else:
                all_probs.append(video_probs * self.weights['video_classifier'])
        
        # Stream 2: Object Detector (if available)
        if self.object_detector is not None and frames is not None:
            detector_probs = self._run_object_detector(frames)
            outputs['detector_probs'] = detector_probs
            all_probs.append(detector_probs * self.weights['object_detector'])
        
        # Stream 3: Anomaly Detector (if available)
        if self.anomaly_detector is not None and motion_stats is not None:
            anomaly_probs = self._run_anomaly_detector(motion_stats, video_clip.device)
            outputs['anomaly_probs'] = anomaly_probs
            all_probs.append(anomaly_probs * self.weights['anomaly_detector'])
        
        # Ensemble fusion
        if self.ensemble_method == 'soft_voting':
            # Weighted average of probabilities
            ensemble_probs = sum(all_probs) / len(all_probs) if all_probs else video_probs
        elif self.ensemble_method == 'meta_classifier':
            # Concatenate all predictions and use meta-classifier
            concat_probs = torch.cat(all_probs, dim=1)
            ensemble_logits = self.meta_classifier(concat_probs)
            ensemble_probs = torch.softmax(ensemble_logits, dim=1)
        
        # Final predictions
        ensemble_predictions = torch.argmax(ensemble_probs, dim=1)
        ensemble_confidence = torch.max(ensemble_probs, dim=1)[0]
        
        outputs['ensemble_probs'] = ensemble_probs
        outputs['predictions'] = ensemble_predictions
        outputs['confidence'] = ensemble_confidence
        
        return outputs
    
    def _run_object_detector(self, frames: List[np.ndarray]) -> torch.Tensor:
        """
        Run YOLO object detector on frames.
        
        Args:
            frames: List of numpy frames
            
        Returns:
            Probability tensor of shape (1, num_classes)
        """
        detection_scores = []
        
        for frame in frames:
            results = self.object_detector(frame, verbose=False)
            
            if len(results) > 0 and len(results[0].boxes) > 0:
                # Get class predictions and confidences
                boxes = results[0].boxes
                classes = boxes.cls.cpu().numpy()
                confs = boxes.conf.cpu().numpy()
                
                # Aggregate scores per class
                class_scores = {0: 0.0, 1: 0.0, 2: 0.0}  # Normal, Suspicious, Theft
                for cls, conf in zip(classes, confs):
                    class_scores[int(cls)] = max(class_scores[int(cls)], conf)
                
                detection_scores.append(list(class_scores.values()))
            else:
                # No detections - assume normal
                detection_scores.append([1.0, 0.0, 0.0])
        
        # Average across frames
        avg_scores = np.mean(detection_scores, axis=0) if detection_scores else [1.0, 0.0, 0.0]
        
        # Convert to tensor
        probs = torch.tensor(avg_scores, dtype=torch.float32).unsqueeze(0)
        probs = probs / probs.sum()  # Normalize
        
        return probs
    
    def _run_anomaly_detector(self, motion_stats: Dict, device: torch.device) -> torch.Tensor:
        """
        Run anomaly detector.
        
        Args:
            motion_stats: Motion statistics dictionary
            device: Target device
            
        Returns:
            Probability tensor
        """
        prediction, anomaly_score = self.anomaly_detector.predict(motion_stats)
        
        # Map anomaly score to 3-class probabilities
        if prediction == 1:  # Anomaly detected
            # High anomaly score -> more likely theft
            probs = [
                max(0.0, 1.0 - anomaly_score),  # Normal
                anomaly_score * 0.3,             # Suspicious
                anomaly_score * 0.7              # Theft
            ]
        else:  # Normal
            probs = [1.0 - anomaly_score, anomaly_score * 0.5, anomaly_score * 0.5]
        
        probs = torch.tensor(probs, dtype=torch.float32, device=device).unsqueeze(0)
        probs = probs / probs.sum()  # Normalize
        
        return probs
    
    def predict_label(self, prediction_idx: int) -> str:
        """
        Convert prediction index to label string.
        
        Args:
            prediction_idx: Predicted class index
            
        Returns:
            Label string
        """
        labels = ['NORMAL', 'SUSPICIOUS', 'THEFT']
        return labels[prediction_idx] if prediction_idx < len(labels) else 'UNKNOWN'


def build_ensemble_model(
    video_classifier: nn.Module,
    config: Dict,
    object_detector: Optional[YOLO] = None,
    anomaly_detector = None
) -> EnsembleModel:
    """
    Build ensemble model from configuration.
    
    Args:
        video_classifier: Trained video classifier
        config: Configuration dictionary
        object_detector: Optional YOLO detector
        anomaly_detector: Optional anomaly detector
        
    Returns:
        Ensemble model instance
    """
    ensemble_config = config.get('ensemble', {})
    
    model = EnsembleModel(
        video_classifier=video_classifier,
        object_detector=object_detector,
        anomaly_detector=anomaly_detector,
        num_classes=3,
        ensemble_method=ensemble_config.get('method', 'soft_voting'),
        weights=ensemble_config.get('weights', None)
    )
    
    return model
