"""
Utility functions for the shoplifting detection system.
"""

import os
import yaml
import logging
import random
import numpy as np
import torch
from typing import Dict, Any


def load_config(config_path: str = "configs/config.yaml") -> Dict[str, Any]:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Dictionary containing configuration parameters
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def setup_logging(log_dir: str = "logs", log_level: str = "INFO") -> logging.Logger:
    """
    Setup logging configuration.
    
    Args:
        log_dir: Directory to save log files
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        
    Returns:
        Configured logger instance
    """
    os.makedirs(log_dir, exist_ok=True)
    
    # Create logger
    logger = logging.getLogger("shoplifting_detection")
    logger.setLevel(getattr(logging, log_level))
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(getattr(logging, log_level))
    console_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    console_handler.setFormatter(console_formatter)
    
    # File handler
    file_handler = logging.FileHandler(os.path.join(log_dir, 'training.log'))
    file_handler.setLevel(getattr(logging, log_level))
    file_handler.setFormatter(console_formatter)
    
    # Add handlers
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    
    return logger


def set_seed(seed: int = 42):
    """
    Set random seed for reproducibility.
    
    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device(device_name: str = "cuda") -> torch.device:
    """
    Get the appropriate device for training/inference.
    
    Args:
        device_name: Desired device ('cuda', 'cpu', 'mps')
        
    Returns:
        torch.device object
    """
    if device_name == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using CUDA device: {torch.cuda.get_device_name(0)}")
    elif device_name == "mps" and torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using Apple Metal Performance Shaders (MPS)")
    else:
        device = torch.device("cpu")
        print("Using CPU")
    
    return device


def create_dirs(config: Dict[str, Any]):
    """
    Create necessary directories for the project.
    
    Args:
        config: Configuration dictionary
    """
    dirs = [
        config['dataset']['raw_videos_dir'],
        config['dataset']['processed_frames_dir'],
        config['dataset']['splits_dir'],
        config['paths']['checkpoints_dir'],
        config['paths']['logs_dir'],
        config['paths']['outputs_dir'],
        os.path.join(config['dataset']['raw_videos_dir'], 'normal'),
        os.path.join(config['dataset']['raw_videos_dir'], 'shoplifting'),
    ]
    
    for dir_path in dirs:
        os.makedirs(dir_path, exist_ok=True)
        

def count_parameters(model: torch.nn.Module) -> int:
    """
    Count the number of trainable parameters in a model.
    
    Args:
        model: PyTorch model
        
    Returns:
        Number of trainable parameters
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    loss: float,
    accuracy: float,
    filepath: str
):
    """
    Save model checkpoint.
    
    Args:
        model: PyTorch model
        optimizer: Optimizer
        epoch: Current epoch
        loss: Current loss
        accuracy: Current accuracy
        filepath: Path to save checkpoint
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'accuracy': accuracy
    }
    torch.save(checkpoint, filepath)
    

def load_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    filepath: str,
    device: torch.device
):
    """
    Load model checkpoint.
    
    Args:
        model: PyTorch model
        optimizer: Optimizer
        filepath: Path to checkpoint file
        device: Device to load model to
        
    Returns:
        Tuple of (model, optimizer, epoch, loss, accuracy)
    """
    checkpoint = torch.load(filepath, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    accuracy = checkpoint['accuracy']
    
    return model, optimizer, epoch, loss, accuracy
