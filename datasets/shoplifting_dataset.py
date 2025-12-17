"""
PyTorch Dataset class for the MNNIT Shoplifting Video Dataset.
"""

import os
import torch
import numpy as np
from torch.utils.data import Dataset
from typing import Dict, Tuple, Optional, List
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2

from utils.video_processing import load_video_frames


class ShopliftingVideoDataset(Dataset):
    """
    Dataset class for MNNIT Shoplifting Video Dataset.
    Loads video clips and returns them as tensors for training/evaluation.
    """
    
    def __init__(
        self,
        video_paths: List[str],
        labels: List[int],
        num_frames: int = 16,
        resize: Tuple[int, int] = (224, 224),
        augment: bool = False,
        config: Optional[Dict] = None
    ):
        """
        Args:
            video_paths: List of paths to video files
            labels: List of integer labels (0=Normal, 1=Shoplifting)
            num_frames: Number of frames to sample per video
            resize: Target resolution (height, width)
            augment: Whether to apply data augmentation
            config: Configuration dictionary
        """
        self.video_paths = video_paths
        self.labels = labels
        self.num_frames = num_frames
        self.resize = resize
        self.augment = augment
        self.config = config or {}
        
        # Define augmentation pipeline
        if augment:
            self.transform = A.Compose([
                A.HorizontalFlip(p=0.5),
                A.RandomBrightnessContrast(
                    brightness_limit=0.2,
                    contrast_limit=0.2,
                    p=0.5
                ),
                A.HueSaturationValue(
                    hue_shift_limit=10,
                    sat_shift_limit=20,
                    val_shift_limit=10,
                    p=0.5
                ),
                A.ShiftScaleRotate(
                    shift_limit=0.05,
                    scale_limit=0.05,
                    rotate_limit=15,
                    p=0.5
                ),
            ])
        else:
            self.transform = None
    
    def __len__(self) -> int:
        return len(self.video_paths)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """
        Load and preprocess a video clip.
        
        Returns:
            Tuple of (video_tensor, label)
            video_tensor shape: (num_frames, channels, height, width)
        """
        video_path = self.video_paths[idx]
        label = self.labels[idx]
        
        # Load frames
        try:
            frames = load_video_frames(
                video_path,
                num_frames=self.num_frames,
                resize=self.resize
            )
        except Exception as e:
            print(f"Error loading video {video_path}: {e}")
            # Return a dummy tensor
            frames = np.zeros((self.num_frames, self.resize[0], self.resize[1], 3), dtype=np.uint8)
        
        # Apply augmentation to each frame
        if self.transform is not None:
            augmented_frames = []
            for frame in frames:
                transformed = self.transform(image=frame)
                augmented_frames.append(transformed['image'])
            frames = np.array(augmented_frames)
        
        # Normalize frames
        frames = frames.astype(np.float32) / 255.0
        
        # ImageNet normalization
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        frames = (frames - mean) / std
        
        # Convert to tensor: (T, H, W, C) -> (T, C, H, W)
        frames = torch.from_numpy(frames).permute(0, 3, 1, 2).float()
        
        return frames, label


class FrameDataset(Dataset):
    """
    Dataset class for frame-level object detection.
    Used with YOLO detector for Roboflow theft detection dataset.
    """
    
    def __init__(
        self,
        frame_paths: List[str],
        annotations: List[Dict],
        img_size: int = 640,
        augment: bool = False
    ):
        """
        Args:
            frame_paths: List of paths to frame images
            annotations: List of annotation dictionaries with bounding boxes
            img_size: Target image size for YOLO
            augment: Whether to apply augmentation
        """
        self.frame_paths = frame_paths
        self.annotations = annotations
        self.img_size = img_size
        self.augment = augment
        
        # Define augmentation for detection
        if augment:
            self.transform = A.Compose([
                A.HorizontalFlip(p=0.5),
                A.RandomBrightnessContrast(p=0.2),
                A.Blur(blur_limit=3, p=0.1),
            ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))
        else:
            self.transform = None
    
    def __len__(self) -> int:
        return len(self.frame_paths)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict]:
        """
        Load and preprocess a frame with annotations.
        
        Returns:
            Tuple of (image_tensor, annotations_dict)
        """
        frame_path = self.frame_paths[idx]
        annotation = self.annotations[idx]
        
        # Load image
        image = cv2.imread(frame_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Get bounding boxes and labels
        bboxes = annotation.get('bboxes', [])
        labels = annotation.get('labels', [])
        
        # Apply augmentation
        if self.transform is not None and len(bboxes) > 0:
            transformed = self.transform(
                image=image,
                bboxes=bboxes,
                class_labels=labels
            )
            image = transformed['image']
            bboxes = transformed['bboxes']
            labels = transformed['class_labels']
        
        # Resize
        image = cv2.resize(image, (self.img_size, self.img_size))
        
        # Normalize and convert to tensor
        image = image.astype(np.float32) / 255.0
        image = torch.from_numpy(image).permute(2, 0, 1).float()
        
        return image, {'bboxes': bboxes, 'labels': labels}


def get_class_weights(labels: List[int]) -> torch.Tensor:
    """
    Calculate class weights for imbalanced dataset.
    
    Args:
        labels: List of integer labels
        
    Returns:
        Tensor of class weights
    """
    unique_labels, counts = np.unique(labels, return_counts=True)
    total = len(labels)
    weights = total / (len(unique_labels) * counts)
    
    # Create weight tensor (use float32 for MPS compatibility)
    weight_dict = dict(zip(unique_labels, weights))
    class_weights = torch.tensor([weight_dict.get(i, 1.0) for i in range(len(unique_labels))], dtype=torch.float32)
    
    return class_weights


def collate_video_batch(batch: List[Tuple[torch.Tensor, int]]) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Custom collate function for video batches.
    
    Args:
        batch: List of (video_tensor, label) tuples
        
    Returns:
        Tuple of (batched_videos, batched_labels)
    """
    videos = torch.stack([item[0] for item in batch])
    labels = torch.tensor([item[1] for item in batch])
    
    return videos, labels
