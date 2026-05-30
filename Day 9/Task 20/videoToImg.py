import cv2
import os
import sys
from pathlib import Path

def convert_video_to_frames(video_path, output_dir="convertedImages", frame_interval=None, resize_dim=(1280, 720), prefix="frame"):
    
    # Validate video file
    if not os.path.exists(video_path):
        print(f"Error: Video file not found at {video_path}")
        return 0
    
    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {os.path.abspath(output_dir)}")
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}")
        return 0
    
    # Get video properties
    framerate = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Set frame extraction interval
    if frame_interval is None:
        frame_interval = max(1, framerate // 4)  # Extract at 1/4 framerate
    
    print(f"Video: {os.path.basename(video_path)}")
    print(f"FPS: {framerate}, Total frames: {total_frames}")
    print(f"Extracting every {frame_interval} frames...")
    
    frame_count = 0
    extracted_count = 0
    
    while True:
        success, frame = cap.read()
        
        if not success:
            break
        
        # Extract frame at specified interval
        if frame_count % frame_interval == 0:
            # Resize if specified
            if resize_dim:
                frame = cv2.resize(frame, resize_dim)
            
            # Save frame
            output_path = os.path.join(output_dir, f"{prefix}_{extracted_count:04d}.jpg")
            cv2.imwrite(output_path, frame)
            extracted_count += 1
            
            if extracted_count % 10 == 0:
                print(f"  Extracted {extracted_count} frames...")
        
        frame_count += 1
    
    cap.release()
    
    print(f"✓ Successfully extracted {extracted_count} frames from {os.path.basename(video_path)}")
    return extracted_count


if __name__ == "__main__":
    # Configuration
    working_dir = r"/home/sanjay/Documents/MLOps training/Day 9"
    os.chdir(working_dir)
    
    # Process video files
    video_files = [
        (r"/home/sanjay/Documents/MLOps training/Day 9/PollinatorDataset/1102.mp4", "118"),
    ]
    
    for video_path, prefix in video_files:
        if os.path.exists(video_path):
            convert_video_to_frames(
                video_path=video_path,
                output_dir="convertedImages",
                frame_interval=None,  # Extracts at 1/4 framerate
                resize_dim=(1280, 720),
                prefix=prefix
            )
            print()
