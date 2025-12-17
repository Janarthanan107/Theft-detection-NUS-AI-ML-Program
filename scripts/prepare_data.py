#!/usr/bin/env python3
"""
Script to download and prepare the MNNIT Shoplifting Dataset.
This script helps organize the dataset for training.
"""

import os
import sys
import argparse
from pathlib import Path
import json
from sklearn.model_selection import train_test_split

sys.path.append(str(Path(__file__).parent.parent))

from utils import load_config, create_dirs, setup_logging


def scan_video_directory(video_dir: str, class_name: str, class_id: int):
    """
    Scan a directory for video files.
    
    Args:
        video_dir: Directory containing videos
        class_name: Class name ('normal' or 'shoplifting')
        class_id: Class ID (0 or 1)
        
    Returns:
        List of (video_path, label) tuples
    """
    video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.flv']
    video_data = []
    
    if not os.path.exists(video_dir):
        print(f"Warning: Directory {video_dir} does not exist")
        return video_data
    
    for filename in os.listdir(video_dir):
        filepath = os.path.join(video_dir, filename)
        if os.path.isfile(filepath):
            ext = os.path.splitext(filename)[1].lower()
            if ext in video_extensions:
                video_data.append({
                    'path': filepath,
                    'filename': filename,
                    'class_name': class_name,
                    'class_id': class_id
                })
    
    print(f"Found {len(video_data)} videos in {video_dir}")
    return video_data


def create_dataset_splits(video_data: list, config: dict):
    """
    Create train/val/test splits and save to JSON files.
    
    Args:
        video_data: List of video data dictionaries
        config: Configuration dictionary
    """
    # Extract paths and labels
    video_paths = [item['path'] for item in video_data]
    labels = [item['class_id'] for item in video_data]
    
    # Get split ratios
    train_ratio = config['dataset']['train_ratio']
    val_ratio = config['dataset']['val_ratio']
    test_ratio = config['dataset']['test_ratio']
    
    # First split: train and temp (val + test)
    train_paths, temp_paths, train_labels, temp_labels = train_test_split(
        video_paths, labels,
        test_size=(val_ratio + test_ratio),
        stratify=labels,
        random_state=42
    )
    
    # Second split: val and test
    val_size = val_ratio / (val_ratio + test_ratio)
    val_paths, test_paths, val_labels, test_labels = train_test_split(
        temp_paths, temp_labels,
        test_size=(1 - val_size),
        stratify=temp_labels,
        random_state=42
    )
    
    # Create split dictionaries
    splits = {
        'train': {
            'video_paths': train_paths,
            'labels': train_labels,
            'size': len(train_paths)
        },
        'val': {
            'video_paths': val_paths,
            'labels': val_labels,
            'size': len(val_paths)
        },
        'test': {
            'video_paths': test_paths,
            'labels': test_labels,
            'size': len(test_paths)
        }
    }
    
    # Save splits to JSON
    splits_dir = config['dataset']['splits_dir']
    os.makedirs(splits_dir, exist_ok=True)
    
    for split_name, split_data in splits.items():
        split_file = os.path.join(splits_dir, f'{split_name}_split.json')
        with open(split_file, 'w') as f:
            json.dump(split_data, f, indent=2)
        print(f"Saved {split_name} split: {split_data['size']} videos -> {split_file}")
    
    # Print statistics
    print("\n" + "="*60)
    print("DATASET SPLIT SUMMARY")
    print("="*60)
    print(f"Total videos: {len(video_data)}")
    print(f"Train: {len(train_paths)} ({train_ratio*100:.1f}%)")
    print(f"Val: {len(val_paths)} ({val_ratio*100:.1f}%)")
    print(f"Test: {len(test_paths)} ({test_ratio*100:.1f}%)")
    
    # Class distribution
    print("\nClass Distribution:")
    for split_name in ['train', 'val', 'test']:
        split_labels = splits[split_name]['labels']
        normal_count = split_labels.count(0)
        shoplifting_count = split_labels.count(1)
        print(f"  {split_name.capitalize()}:")
        print(f"    Normal: {normal_count}")
        print(f"    Shoplifting: {shoplifting_count}")
    print("="*60 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Prepare MNNIT Shoplifting Dataset for training"
    )
    parser.add_argument(
        '--config',
        type=str,
        default='configs/config.yaml',
        help='Path to configuration file'
    )
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Create directories
    create_dirs(config)
    
    # Setup logging
    logger = setup_logging(config['paths']['logs_dir'])
    logger.info("Starting dataset preparation...")
    
    # Scan video directories
    raw_videos_dir = config['dataset']['raw_videos_dir']
    normal_dir = os.path.join(raw_videos_dir, 'normal')
    shoplifting_dir = os.path.join(raw_videos_dir, 'shoplifting')
    
    print("\n" + "="*60)
    print("SCANNING VIDEO DIRECTORIES")
    print("="*60)
    
    # Collect all videos
    all_videos = []
    all_videos.extend(scan_video_directory(normal_dir, 'normal', 0))
    all_videos.extend(scan_video_directory(shoplifting_dir, 'shoplifting', 1))
    
    if len(all_videos) == 0:
        print("\n" + "!"*60)
        print("ERROR: No videos found!")
        print("!"*60)
        print("\nPlease ensure videos are placed in:")
        print(f"  Normal videos: {normal_dir}")
        print(f"  Shoplifting videos: {shoplifting_dir}")
        print("\nSupported formats: .mp4, .avi, .mov, .mkv, .flv")
        print("!"*60 + "\n")
        sys.exit(1)
    
    # Create splits
    print("\n" + "="*60)
    print("CREATING DATASET SPLITS")
    print("="*60)
    create_dataset_splits(all_videos, config)
    
    logger.info("Dataset preparation completed successfully!")
    print("\n✓ Dataset preparation completed!")
    print(f"\nSplit files saved to: {config['dataset']['splits_dir']}")
    print("\nNext steps:")
    print("  1. Run 'python scripts/train_video_classifier.py' to train the video classifier")
    print("  2. Run 'python scripts/train_ensemble.py' to train the ensemble model")
    print("  3. Run 'python scripts/demo.py' to test the system\n")


if __name__ == '__main__':
    main()
