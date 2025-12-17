"""
Video processing utilities for frame extraction and preprocessing.
"""

import cv2
import os
import numpy as np
from typing import List, Tuple, Optional
from pathlib import Path
from tqdm import tqdm


def extract_frames(
    video_path: str,
    output_dir: str,
    fps: Optional[int] = None,
    max_frames: Optional[int] = None
) -> List[str]:
    """
    Extract frames from a video file.
    
    Args:
        video_path: Path to input video file
        output_dir: Directory to save extracted frames
        fps: Target FPS (if None, uses original FPS)
        max_frames: Maximum number of frames to extract
        
    Returns:
        List of paths to extracted frame images
    """
    os.makedirs(output_dir, exist_ok=True)
    
    cap = cv2.VideoCapture(video_path)
    original_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Calculate frame sampling interval
    if fps is not None and fps < original_fps:
        frame_interval = int(original_fps / fps)
    else:
        frame_interval = 1
    
    frame_paths = []
    frame_count = 0
    saved_count = 0
    
    with tqdm(total=total_frames, desc=f"Extracting frames from {Path(video_path).name}") as pbar:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Sample frames based on interval
            if frame_count % frame_interval == 0:
                frame_filename = f"frame_{saved_count:06d}.jpg"
                frame_path = os.path.join(output_dir, frame_filename)
                cv2.imwrite(frame_path, frame)
                frame_paths.append(frame_path)
                saved_count += 1
                
                if max_frames is not None and saved_count >= max_frames:
                    break
            
            frame_count += 1
            pbar.update(1)
    
    cap.release()
    return frame_paths


def load_video_frames(
    video_path: str,
    num_frames: int = 16,
    resize: Tuple[int, int] = (224, 224)
) -> np.ndarray:
    """
    Load a fixed number of frames from a video file.
    
    Args:
        video_path: Path to video file
        num_frames: Number of frames to extract
        resize: Target resolution (height, width)
        
    Returns:
        Numpy array of shape (num_frames, height, width, channels)
    """
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Calculate frame indices to sample uniformly
    if total_frames <= num_frames:
        frame_indices = list(range(total_frames))
        # Pad with last frame if needed
        frame_indices += [total_frames - 1] * (num_frames - total_frames)
    else:
        frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
    
    frames = []
    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            # Resize and convert BGR to RGB
            frame = cv2.resize(frame, (resize[1], resize[0]))
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
    
    cap.release()
    
    return np.array(frames)


def preprocess_frame(
    frame: np.ndarray,
    resize: Tuple[int, int] = (224, 224),
    normalize: bool = True
) -> np.ndarray:
    """
    Preprocess a single frame.
    
    Args:
        frame: Input frame (BGR format)
        resize: Target resolution
        normalize: Whether to normalize pixel values
        
    Returns:
        Preprocessed frame
    """
    # Resize
    frame = cv2.resize(frame, (resize[1], resize[0]))
    
    # Convert BGR to RGB
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Normalize
    if normalize:
        frame = frame.astype(np.float32) / 255.0
        # ImageNet normalization
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        frame = (frame - mean) / std
    
    return frame


def compute_optical_flow(
    prev_gray: np.ndarray,
    curr_gray: np.ndarray
) -> np.ndarray:
    """
    Compute optical flow between two frames using Farneback method.
    
    Args:
        prev_gray: Previous grayscale frame
        curr_gray: Current grayscale frame
        
    Returns:
        Optical flow field
    """
    flow = cv2.calcOpticalFlowFarneback(
        prev_gray, curr_gray, None,
        pyr_scale=0.5, levels=3, winsize=15,
        iterations=3, poly_n=5, poly_sigma=1.2,
        flags=0
    )
    return flow


def compute_motion_statistics(frames: np.ndarray) -> dict:
    """
    Compute motion statistics from a sequence of frames.
    
    Args:
        frames: Array of frames (T, H, W, C)
        
    Returns:
        Dictionary containing motion statistics
    """
    gray_frames = [cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY) for frame in frames]
    
    motion_magnitudes = []
    motion_angles = []
    
    for i in range(len(gray_frames) - 1):
        flow = compute_optical_flow(gray_frames[i], gray_frames[i + 1])
        magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        motion_magnitudes.append(np.mean(magnitude))
        motion_angles.append(np.mean(angle))
    
    stats = {
        'mean_magnitude': np.mean(motion_magnitudes),
        'std_magnitude': np.std(motion_magnitudes),
        'max_magnitude': np.max(motion_magnitudes),
        'mean_angle': np.mean(motion_angles),
        'std_angle': np.std(motion_angles)
    }
    
    return stats


def create_video_from_frames(
    frames: List[np.ndarray],
    output_path: str,
    fps: int = 30,
    codec: str = 'mp4v'
):
    """
    Create a video file from a list of frames.
    
    Args:
        frames: List of frames (RGB format)
        output_path: Output video file path
        fps: Frames per second
        codec: Video codec
    """
    if len(frames) == 0:
        return
    
    height, width = frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*codec)
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    for frame in frames:
        # Convert RGB to BGR for OpenCV
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        out.write(frame_bgr)
    
    out.release()


def sliding_window_clips(
    video_path: str,
    clip_length: int = 16,
    stride: int = 8,
    resize: Tuple[int, int] = (224, 224)
) -> List[np.ndarray]:
    """
    Extract clips from video using sliding window approach.
    
    Args:
        video_path: Path to video file
        clip_length: Number of frames per clip
        stride: Number of frames to move window
        resize: Target resolution
        
    Returns:
        List of video clips
    """
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    clips = []
    current_frame = 0
    
    while current_frame + clip_length <= total_frames:
        frames = []
        cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame)
        
        for _ in range(clip_length):
            ret, frame = cap.read()
            if ret:
                frame = cv2.resize(frame, (resize[1], resize[0]))
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame)
        
        if len(frames) == clip_length:
            clips.append(np.array(frames))
        
        current_frame += stride
    
    cap.release()
    return clips
