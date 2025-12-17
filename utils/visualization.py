"""
Visualization utilities for displaying predictions and metrics.
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Tuple, Dict, Optional
from sklearn.metrics import confusion_matrix, classification_report
import os


# Color mapping for labels
COLOR_MAP = {
    'NORMAL': (0, 255, 0),      # Green
    'SUSPICIOUS': (0, 165, 255), # Orange
    'THEFT': (0, 0, 255)         # Red
}


def draw_label_on_frame(
    frame: np.ndarray,
    label: str,
    confidence: float,
    position: Tuple[int, int] = (10, 30),
    font_scale: float = 1.0,
    thickness: int = 2
) -> np.ndarray:
    """
    Draw prediction label and confidence on frame.
    
    Args:
        frame: Input frame (RGB or BGR)
        label: Prediction label
        confidence: Confidence score (0-1)
        position: Text position (x, y)
        font_scale: Font scale
        thickness: Text thickness
        
    Returns:
        Frame with label drawn
    """
    frame_copy = frame.copy()
    color = COLOR_MAP.get(label, (255, 255, 255))
    
    # Draw background rectangle
    text = f"{label}: {confidence:.2%}"
    (text_width, text_height), baseline = cv2.getTextSize(
        text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness
    )
    
    cv2.rectangle(
        frame_copy,
        (position[0] - 5, position[1] - text_height - 5),
        (position[0] + text_width + 5, position[1] + baseline + 5),
        (0, 0, 0),
        -1
    )
    
    # Draw text
    cv2.putText(
        frame_copy,
        text,
        position,
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        color,
        thickness
    )
    
    return frame_copy


def draw_bounding_boxes(
    frame: np.ndarray,
    boxes: List[Tuple[int, int, int, int]],
    labels: List[str],
    confidences: List[float]
) -> np.ndarray:
    """
    Draw bounding boxes with labels on frame.
    
    Args:
        frame: Input frame
        boxes: List of bounding boxes [(x1, y1, x2, y2), ...]
        labels: List of labels
        confidences: List of confidence scores
        
    Returns:
        Frame with bounding boxes drawn
    """
    frame_copy = frame.copy()
    
    for box, label, conf in zip(boxes, labels, confidences):
        x1, y1, x2, y2 = box
        color = COLOR_MAP.get(label, (255, 255, 255))
        
        # Draw box
        cv2.rectangle(frame_copy, (x1, y1), (x2, y2), color, 2)
        
        # Draw label
        label_text = f"{label}: {conf:.2f}"
        (text_width, text_height), _ = cv2.getTextSize(
            label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
        )
        
        cv2.rectangle(
            frame_copy,
            (x1, y1 - text_height - 5),
            (x1 + text_width, y1),
            color,
            -1
        )
        
        cv2.putText(
            frame_copy,
            label_text,
            (x1, y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 0),
            1
        )
    
    return frame_copy


def draw_fps(
    frame: np.ndarray,
    fps: float,
    position: Tuple[int, int] = None
) -> np.ndarray:
    """
    Draw FPS counter on frame.
    
    Args:
        frame: Input frame
        fps: Frames per second
        position: Text position (if None, top-right)
        
    Returns:
        Frame with FPS drawn
    """
    frame_copy = frame.copy()
    
    if position is None:
        position = (frame.shape[1] - 150, 30)
    
    text = f"FPS: {fps:.1f}"
    cv2.putText(
        frame_copy,
        text,
        position,
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2
    )
    
    return frame_copy


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: List[str],
    save_path: Optional[str] = None,
    normalize: bool = False
):
    """
    Plot confusion matrix.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        class_names: List of class names
        save_path: Path to save plot
        normalize: Whether to normalize values
    """
    cm = confusion_matrix(y_true, y_pred)
    
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        fmt = '.2f'
    else:
        fmt = 'd'
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt=fmt,
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names,
        cbar_kws={'label': 'Count' if not normalize else 'Proportion'}
    )
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def plot_training_history(
    history: Dict[str, List[float]],
    save_path: Optional[str] = None
):
    """
    Plot training history (loss and accuracy).
    
    Args:
        history: Dictionary with keys 'train_loss', 'val_loss', 'train_acc', 'val_acc'
        save_path: Path to save plot
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot loss
    epochs = range(1, len(history['train_loss']) + 1)
    ax1.plot(epochs, history['train_loss'], 'b-', label='Training Loss')
    ax1.plot(epochs, history['val_loss'], 'r-', label='Validation Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()
    ax1.grid(True)
    
    # Plot accuracy
    ax2.plot(epochs, history['train_acc'], 'b-', label='Training Accuracy')
    ax2.plot(epochs, history['val_acc'], 'r-', label='Validation Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Training and Validation Accuracy')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def print_classification_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: List[str],
    save_path: Optional[str] = None
):
    """
    Print and optionally save classification report.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        class_names: List of class names
        save_path: Path to save report
    """
    report = classification_report(
        y_true,
        y_pred,
        target_names=class_names,
        digits=4
    )
    
    print("\n" + "="*60)
    print("CLASSIFICATION REPORT")
    print("="*60)
    print(report)
    print("="*60 + "\n")
    
    if save_path:
        with open(save_path, 'w') as f:
            f.write(report)


def create_side_by_side_comparison(
    original_frame: np.ndarray,
    annotated_frame: np.ndarray,
    title1: str = "Original",
    title2: str = "Detection"
) -> np.ndarray:
    """
    Create side-by-side comparison of original and annotated frames.
    
    Args:
        original_frame: Original frame
        annotated_frame: Annotated frame
        title1: Title for original frame
        title2: Title for annotated frame
        
    Returns:
        Combined frame
    """
    # Ensure same size
    h, w = original_frame.shape[:2]
    annotated_frame = cv2.resize(annotated_frame, (w, h))
    
    # Add titles
    cv2.putText(original_frame, title1, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(annotated_frame, title2, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    # Concatenate horizontally
    combined = np.hstack([original_frame, annotated_frame])
    
    return combined
