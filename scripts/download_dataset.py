#!/usr/bin/env python3
"""
Download script for the MNNIT Shoplifting Dataset.
Provides instructions and helpers for obtaining the dataset.
"""

import os
import sys
import argparse
from pathlib import Path

# Dataset sources
DATASET_SOURCES = {
    'kaggle': {
        'name': 'Kaggle - Shoplifting Video Dataset',
        'url': 'https://www.kaggle.com/datasets/your-username/shoplifting-video-dataset',
        'description': 'Reupload of MNNIT dataset on Kaggle',
        'command': 'kaggle datasets download -d your-username/shoplifting-video-dataset'
    },
    'mendeley': {
        'name': 'Mendeley Data - Shoplifting Dataset',
        'url': 'https://data.mendeley.com/datasets/[dataset-id]',
        'description': 'Original MNNIT Allahabad CV Lab dataset',
        'command': 'Manual download from website'
    },
    'innovatiana': {
        'name': 'Innovatiana - Shoplifting Dataset',
        'url': 'https://innovatiana.com/shoplifting-dataset',
        'description': 'Dataset description and download page',
        'command': 'Manual download from website'
    }
}


def print_instructions():
    """Print download instructions."""
    print("="*80)
    print("MNNIT SHOPLIFTING DATASET - DOWNLOAD INSTRUCTIONS")
    print("="*80)
    print("\nThis dataset was created by the CV Laboratory at MNNIT Allahabad and is available")
    print("from multiple sources. Choose one of the following options:\n")
    
    for i, (key, info) in enumerate(DATASET_SOURCES.items(), 1):
        print(f"{i}. {info['name']}")
        print(f"   URL: {info['url']}")
        print(f"   Description: {info['description']}")
        print(f"   Download: {info['command']}")
        print()
    
    print("\n" + "="*80)
    print("DATASET STRUCTURE")
    print("="*80)
    print("\nThe dataset contains two classes:")
    print("  • Normal: People walking, browsing, inspecting items normally")
    print("  • Shoplifting: People concealing items in pockets, bags, or under clothing")
    print("\nVideo specifications:")
    print("  • Resolution: ~640×480 pixels")
    print("  • Frame rate: 30 FPS")
    print("  • Format: MP4")
    print("  • Camera: 32 MP surveillance-style camera")
    print()
    
    print("\n" + "="*80)
    print("MANUAL DOWNLOAD STEPS")
    print("="*80)
    print("\n1. Visit one of the URLs above")
    print("2. Accept any terms/licenses if required")
    print("3. Download the dataset ZIP file")
    print("4. Extract the ZIP file")
    print("5. Organize videos into the following structure:")
    print()
    print("   data/raw_videos/")
    print("   ├── normal/")
    print("   │   ├── video1.mp4")
    print("   │   ├── video2.mp4")
    print("   │   └── ...")
    print("   └── shoplifting/")
    print("       ├── video1.mp4")
    print("       ├── video2.mp4")
    print("       └── ...")
    print()
    print("6. Run: python scripts/prepare_data.py")
    print()
    
    print("\n" + "="*80)
    print("KAGGLE API DOWNLOAD (if using Kaggle)")
    print("="*80)
    print("\n1. Install Kaggle API: pip install kaggle")
    print("2. Set up credentials: https://www.kaggle.com/docs/api")
    print("3. Run the download command shown above")
    print("4. Extract and organize as described")
    print()
    
    print("\n" + "="*80)
    print("OPTIONAL: ROBOFLOW DATASETS")
    print("="*80)
    print("\nFor additional object detection training, you can optionally use:")
    print("  • Roboflow 'Theft detection in retail' dataset")
    print("  • URL: https://universe.roboflow.com/[workspace]/theft-detection")
    print("\nThis is OPTIONAL and only needed if you want to train the YOLO detector.")
    print()


def check_dataset_structure(data_dir='data/raw_videos'):
    """Check if dataset is properly organized."""
    normal_dir = os.path.join(data_dir, 'normal')
    shoplifting_dir = os.path.join(data_dir, 'shoplifting')
    
    print("\n" + "="*80)
    print("CHECKING DATASET STRUCTURE")
    print("="*80)
    
    # Check directories
    normal_exists = os.path.exists(normal_dir)
    shoplifting_exists = os.path.exists(shoplifting_dir)
    
    print(f"\nNormal directory: {normal_dir}")
    print(f"  Exists: {'✓' if normal_exists else '✗'}")
    
    if normal_exists:
        normal_videos = [f for f in os.listdir(normal_dir) 
                        if f.endswith(('.mp4', '.avi', '.mov', '.mkv'))]
        print(f"  Videos found: {len(normal_videos)}")
    
    print(f"\nShoplifting directory: {shoplifting_dir}")
    print(f"  Exists: {'✓' if shoplifting_exists else '✗'}")
    
    if shoplifting_exists:
        shoplifting_videos = [f for f in os.listdir(shoplifting_dir) 
                             if f.endswith(('.mp4', '.avi', '.mov', '.mkv'))]
        print(f"  Videos found: {len(shoplifting_videos)}")
    
    print()
    
    if normal_exists and shoplifting_exists:
        total = len(normal_videos) + len(shoplifting_videos)
        if total > 0:
            print("✓ Dataset structure looks good!")
            print(f"  Total videos: {total}")
            print("\nNext step: Run 'python scripts/prepare_data.py'")
        else:
            print("✗ No videos found in the directories")
            print("  Please add video files to the directories")
    else:
        print("✗ Dataset directories not found")
        print("  Please create the directory structure and add videos")
    
    print()


def main():
    parser = argparse.ArgumentParser(
        description="Download and setup MNNIT Shoplifting Dataset"
    )
    parser.add_argument(
        '--check',
        action='store_true',
        help='Check if dataset is properly organized'
    )
    
    args = parser.parse_args()
    
    if args.check:
        check_dataset_structure()
    else:
        print_instructions()


if __name__ == '__main__':
    main()
