"""
CNN-LSTM model for video classification (Stream 1).
Extracts spatial features with CNN and temporal patterns with LSTM.
"""

import ssl
import torch
import torch.nn as nn
import torchvision.models as models
from typing import Tuple

# Fix SSL certificate verification issues
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context


class CNNLSTM(nn.Module):
    """
    CNN-LSTM architecture for video classification.
    Uses a pretrained CNN (ResNet) as feature extractor and LSTM for temporal modeling.
    """
    
    def __init__(
        self,
        num_classes: int = 2,
        backbone: str = 'resnet18',
        lstm_hidden_size: int = 256,
        lstm_num_layers: int = 2,
        dropout: float = 0.5,
        pretrained: bool = True
    ):
        """
        Args:
            num_classes: Number of output classes
            backbone: Backbone CNN architecture ('resnet18', 'resnet34', 'resnet50')
            lstm_hidden_size: Hidden size of LSTM
            lstm_num_layers: Number of LSTM layers
            dropout: Dropout probability
            pretrained: Whether to use pretrained CNN weights
        """
        super(CNNLSTM, self).__init__()
        
        self.num_classes = num_classes
        self.lstm_hidden_size = lstm_hidden_size
        self.lstm_num_layers = lstm_num_layers
        
        # Load pretrained CNN backbone
        if backbone == 'resnet18':
            self.cnn = models.resnet18(pretrained=pretrained)
            cnn_output_size = 512
        elif backbone == 'resnet34':
            self.cnn = models.resnet34(pretrained=pretrained)
            cnn_output_size = 512
        elif backbone == 'resnet50':
            self.cnn = models.resnet50(pretrained=pretrained)
            cnn_output_size = 2048
        else:
            raise ValueError(f"Unknown backbone: {backbone}")
        
        # Remove the final fully connected layer
        self.cnn = nn.Sequential(*list(self.cnn.children())[:-1])
        
        # LSTM for temporal modeling
        self.lstm = nn.LSTM(
            input_size=cnn_output_size,
            hidden_size=lstm_hidden_size,
            num_layers=lstm_num_layers,
            batch_first=True,
            dropout=dropout if lstm_num_layers > 1 else 0,
            bidirectional=False
        )
        
        # Classifier head
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(lstm_hidden_size, num_classes)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (batch, num_frames, channels, height, width)
            
        Returns:
            Tuple of (logits, features)
            logits: Class logits of shape (batch, num_classes)
            features: LSTM features for ensemble
        """
        batch_size, num_frames, c, h, w = x.shape
        
        # Extract CNN features for each frame
        # Reshape: (batch * num_frames, c, h, w)
        x = x.reshape(batch_size * num_frames, c, h, w)
        
        # CNN feature extraction
        cnn_features = self.cnn(x)  # (batch * num_frames, cnn_output_size, 1, 1)
        cnn_features = cnn_features.reshape(cnn_features.size(0), -1)  # (batch * num_frames, cnn_output_size)
        
        # Reshape back to sequence: (batch, num_frames, cnn_output_size)
        cnn_features = cnn_features.reshape(batch_size, num_frames, -1)
        
        # LSTM temporal modeling
        lstm_out, (hn, cn) = self.lstm(cnn_features)
        
        # Use the last hidden state
        lstm_features = hn[-1]  # (batch, lstm_hidden_size)
        
        # Classification
        x = self.dropout(lstm_features)
        logits = self.fc(x)  # (batch, num_classes)
        
        return logits, lstm_features


class CNN3D(nn.Module):
    """
    3D CNN model for video classification.
    Alternative to CNN-LSTM using 3D convolutions.
    """
    
    def __init__(
        self,
        num_classes: int = 2,
        dropout: float = 0.5,
        pretrained: bool = False
    ):
        """
        Args:
            num_classes: Number of output classes
            dropout: Dropout probability
            pretrained: Whether to use pretrained weights
        """
        super(CNN3D, self).__init__()
        
        self.num_classes = num_classes
        
        # 3D Convolutional layers
        self.conv1 = nn.Sequential(
            nn.Conv3d(3, 64, kernel_size=(3, 7, 7), stride=(1, 2, 2), padding=(1, 3, 3)),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1))
        )
        
        self.conv2 = nn.Sequential(
            nn.Conv3d(64, 128, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1)),
            nn.BatchNorm3d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))
        )
        
        self.conv3 = nn.Sequential(
            nn.Conv3d(128, 256, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1)),
            nn.BatchNorm3d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))
        )
        
        self.conv4 = nn.Sequential(
            nn.Conv3d(256, 512, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1)),
            nn.BatchNorm3d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))
        )
        
        # Adaptive pooling
        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        
        # Classifier
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(512, num_classes)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (batch, num_frames, channels, height, width)
            
        Returns:
            Tuple of (logits, features)
        """
        # Reshape to 3D CNN format: (batch, channels, num_frames, height, width)
        x = x.permute(0, 2, 1, 3, 4)
        
        # 3D convolutions
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        
        # Global pooling
        x = self.avgpool(x)
        features = x.view(x.size(0), -1)
        
        # Classification
        x = self.dropout(features)
        logits = self.fc(x)
        
        return logits, features


def build_video_classifier(config: dict) -> nn.Module:
    """
    Build video classifier model based on configuration.
    
    Args:
        config: Model configuration dictionary
        
    Returns:
        Video classifier model
    """
    model_type = config.get('type', 'cnn_lstm')
    num_classes = config.get('num_classes', 2)
    
    if model_type == 'cnn_lstm':
        model = CNNLSTM(
            num_classes=num_classes,
            backbone=config.get('backbone', 'resnet18'),
            lstm_hidden_size=config.get('lstm_hidden_size', 256),
            lstm_num_layers=config.get('lstm_num_layers', 2),
            dropout=config.get('dropout', 0.5),
            pretrained=config.get('pretrained', True)
        )
    elif model_type == '3d_cnn':
        model = CNN3D(
            num_classes=num_classes,
            dropout=config.get('dropout', 0.5),
            pretrained=config.get('pretrained', False)
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    return model
