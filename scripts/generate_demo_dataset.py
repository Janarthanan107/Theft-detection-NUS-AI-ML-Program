"""
Generate Demo Dataset for Quick Testing
Creates synthetic videos for immediate training without downloading real dataset
"""

import cv2
import numpy as np
from pathlib import Path
import random
from tqdm import tqdm

def create_synthetic_video(output_path, video_type='normal', duration=5, fps=30):
    """
    Create a synthetic video with simple animations
    
    Args:
        output_path: Where to save the video
        video_type: 'normal' or 'shoplifting'
        duration: Video duration in seconds
        fps: Frames per second
    """
    width, height = 640, 480
    num_frames = duration * fps
    
    # Video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
    
    # Initial position
    x, y = width // 4, height // 2
    
    for frame_idx in range(num_frames):
        # Create blank frame
        frame = np.ones((height, width, 3), dtype=np.uint8) * 30  # Dark gray background
        
        # Add grid pattern (simulating store shelves)
        for i in range(0, width, 80):
            cv2.line(frame, (i, 0), (i, height), (50, 50, 50), 1)
        for i in range(0, height, 80):
            cv2.line(frame, (0, i), (width, i), (50, 50, 50), 1)
        
        # Add "shelves" (rectangles)
        cv2.rectangle(frame, (width-150, 100), (width-50, 200), (80, 80, 80), -1)
        cv2.rectangle(frame, (width-150, 250), (width-50, 350), (80, 80, 80), -1)
        
        # Animate based on type
        if video_type == 'normal':
            # Normal behavior: smooth walking
            x = 100 + int((frame_idx / num_frames) * (width - 200))
            y = height // 2 + int(20 * np.sin(frame_idx / 10))
            
            # Draw person (circle)
            cv2.circle(frame, (x, y), 30, (100, 150, 200), -1)
            cv2.circle(frame, (x-10, y-10), 5, (255, 255, 255), -1)  # Eye
            cv2.circle(frame, (x+10, y-10), 5, (255, 255, 255), -1)  # Eye
            
            # Add text
            cv2.putText(frame, 'Normal Browsing', (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (100, 200, 100), 2)
            
        else:  # shoplifting
            # Suspicious behavior: quick movements, looking around
            if frame_idx < num_frames // 3:
                # Phase 1: Approach
                x = 100 + int((frame_idx / (num_frames // 3)) * (width - 300))
                y = height // 2
            elif frame_idx < 2 * num_frames // 3:
                # Phase 2: Suspicious looking around
                x = width - 250
                y = height // 2 + int(30 * np.sin(frame_idx * 0.5))
            else:
                # Phase 3: Quick exit
                progress = (frame_idx - 2 * num_frames // 3) / (num_frames // 3)
                x = width - 250 + int(progress * 200)
                y = height // 2
            
            # Draw person with different color (suspicious)
            cv2.circle(frame, (x, y), 30, (50, 100, 200), -1)
            cv2.circle(frame, (x-10, y-10), 5, (255, 255, 255), -1)
            cv2.circle(frame, (x+10, y-10), 5, (255, 255, 255), -1)
            
            # Draw "hand reaching" when near shelf
            if frame_idx >= num_frames // 3 and frame_idx < 2 * num_frames // 3:
                # Draw reaching hand
                cv2.line(frame, (x+30, y), (x+50, y-20), (200, 150, 100), 5)
                cv2.circle(frame, (x+50, y-20), 8, (200, 150, 100), -1)
            
            # Add warning text
            cv2.putText(frame, 'Suspicious Behavior', (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (100, 100, 200), 2)
        
        # Add frame number
        cv2.putText(frame, f'Frame: {frame_idx+1}/{num_frames}', (10, height-20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)
        
        # Add noise for realism
        noise = np.random.randint(0, 10, (height, width, 3), dtype=np.uint8)
        frame = cv2.add(frame, noise)
        
        out.write(frame)
    
    out.release()

def generate_demo_dataset(num_videos_per_class=10):
    """
    Generate a complete demo dataset
    
    Args:
        num_videos_per_class: Number of videos to generate for each class
    """
    data_dir = Path('data/raw_videos')
    normal_dir = data_dir / 'normal'
    shoplifting_dir = data_dir / 'shoplifting'
    
    # Create directories
    normal_dir.mkdir(parents=True, exist_ok=True)
    shoplifting_dir.mkdir(parents=True, exist_ok=True)
    
    print("🎬 Generating Demo Dataset...")
    print(f"Creating {num_videos_per_class} videos per class\n")
    
    # Generate normal videos
    print("📹 Generating NORMAL behavior videos...")
    for i in tqdm(range(num_videos_per_class)):
        video_path = normal_dir / f'normal_demo_{i:03d}.mp4'
        duration = random.randint(3, 7)  # 3-7 seconds
        create_synthetic_video(video_path, 'normal', duration=duration)
    
    # Generate shoplifting videos
    print("\n🚨 Generating SHOPLIFTING behavior videos...")
    for i in tqdm(range(num_videos_per_class)):
        video_path = shoplifting_dir / f'shoplifting_demo_{i:03d}.mp4'
        duration = random.randint(3, 7)  # 3-7 seconds
        create_synthetic_video(video_path, 'shoplifting', duration=duration)
    
    print(f"\n✅ Demo dataset created!")
    print(f"   Normal videos: {len(list(normal_dir.glob('*.mp4')))}")
    print(f"   Shoplifting videos: {len(list(shoplifting_dir.glob('*.mp4')))}")
    print(f"   Total: {len(list(normal_dir.glob('*.mp4'))) + len(list(shoplifting_dir.glob('*.mp4')))} videos")
    print(f"\nDataset location: {data_dir.absolute()}")
    
    return True

if __name__ == '__main__':
    print("=" * 60)
    print("  DEMO DATASET GENERATOR")
    print("  For Quick Testing Without Real Dataset")
    print("=" * 60)
    print()
    
    # Generate dataset
    generate_demo_dataset(num_videos_per_class=15)  # 15 videos per class = 30 total
    
    print("\n" + "=" * 60)
    print("🎯 Next Steps:")
    print("=" * 60)
    print("1. Run: python scripts/prepare_data.py")
    print("2. Run: python scripts/train_video_classifier.py")
    print("3. Run: python backend/app.py")
    print("\n⚠️  NOTE: This is synthetic data for quick testing.")
    print("   For real results, download the MNNIT Shoplifting Dataset.")
    print("=" * 60)
