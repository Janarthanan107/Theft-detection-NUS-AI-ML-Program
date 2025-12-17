"""
Anomaly detector based on motion statistics (Stream 3).
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Tuple
from sklearn.ensemble import IsolationForest
import pickle
import os


class MotionAnomalyDetector:
    """
    Anomaly detector using motion statistics.
    Uses optical flow features and classical ML for anomaly detection.
    """
    
    def __init__(
        self,
        threshold: float = 0.7,
        contamination: float = 0.1,
        model_path: str = None
    ):
        """
        Args:
            threshold: Anomaly score threshold
            contamination: Expected proportion of anomalies
            model_path: Path to saved model
        """
        self.threshold = threshold
        self.contamination = contamination
        self.model = IsolationForest(
            contamination=contamination,
            random_state=42,
            n_estimators=100
        )
        self.is_fitted = False
        
        if model_path and os.path.exists(model_path):
            self.load(model_path)
    
    def extract_motion_features(self, motion_stats: Dict) -> np.ndarray:
        """
        Extract feature vector from motion statistics.
        
        Args:
            motion_stats: Dictionary with motion statistics
            
        Returns:
            Feature vector
        """
        features = [
            motion_stats.get('mean_magnitude', 0),
            motion_stats.get('std_magnitude', 0),
            motion_stats.get('max_magnitude', 0),
            motion_stats.get('mean_angle', 0),
            motion_stats.get('std_angle', 0)
        ]
        return np.array(features).reshape(1, -1)
    
    def fit(self, motion_stats_list: list):
        """
        Fit the anomaly detector on training data.
        
        Args:
            motion_stats_list: List of motion statistics dictionaries
        """
        features = []
        for stats in motion_stats_list:
            feat = self.extract_motion_features(stats)
            features.append(feat.flatten())
        
        features = np.array(features)
        self.model.fit(features)
        self.is_fitted = True
    
    def predict(self, motion_stats: Dict) -> Tuple[int, float]:
        """
        Predict if motion pattern is anomalous.
        
        Args:
            motion_stats: Motion statistics dictionary
            
        Returns:
            Tuple of (prediction, anomaly_score)
            prediction: 1 for anomaly, 0 for normal
            anomaly_score: Anomaly score (higher = more anomalous)
        """
        if not self.is_fitted:
            # Return neutral prediction if not fitted
            return 0, 0.5
        
        features = self.extract_motion_features(motion_stats)
        
        # Get anomaly score (lower score = more anomalous for Isolation Forest)
        score = self.model.score_samples(features)[0]
        
        # Normalize score to [0, 1] range
        # Scores are typically in range [-0.5, 0.5], map to [1, 0] for anomaly prob
        anomaly_score = max(0, min(1, 0.5 - score))
        
        # Predict: 1 if anomaly, 0 if normal
        prediction = 1 if anomaly_score > self.threshold else 0
        
        return prediction, anomaly_score
    
    def save(self, filepath: str):
        """Save the model."""
        model_data = {
            'model': self.model,
            'threshold': self.threshold,
            'contamination': self.contamination,
            'is_fitted': self.is_fitted
        }
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
    
    def load(self, filepath: str):
        """Load the model."""
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.model = model_data['model']
        self.threshold = model_data['threshold']
        self.contamination = model_data['contamination']
        self.is_fitted = model_data['is_fitted']


class SimpleAnomalyClassifier(nn.Module):
    """
    Simple neural network for anomaly detection based on motion features.
    Alternative to Isolation Forest.
    """
    
    def __init__(
        self,
        input_size: int = 5,
        hidden_size: int = 32,
        num_classes: int = 2,
        dropout: float = 0.3
    ):
        """
        Args:
            input_size: Number of input features
            hidden_size: Hidden layer size
            num_classes: Number of output classes
            dropout: Dropout probability
        """
        super(SimpleAnomalyClassifier, self).__init__()
        
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, num_classes)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input features (batch, input_size)
            
        Returns:
            Logits (batch, num_classes)
        """
        return self.network(x)


def build_anomaly_detector(config: dict) -> MotionAnomalyDetector:
    """
    Build anomaly detector based on configuration.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Anomaly detector instance
    """
    threshold = config.get('threshold', 0.7)
    contamination = config.get('contamination', 0.1)
    model_path = config.get('model_path', None)
    
    detector = MotionAnomalyDetector(
        threshold=threshold,
        contamination=contamination,
        model_path=model_path
    )
    
    return detector
