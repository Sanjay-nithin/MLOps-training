import cv2
import os
from pathlib import Path

# Directory path where videos are stored
VIDEO_DIR = r"/home/sanjay/Documents/MLOps training/dataset/video"
os.chdir(VIDEO_DIR)

# Create output directory for extracted frames
OUTPUT_DIR = os.path.join(os.path.dirname(VIDEO_DIR), "extracted_frames")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Supported video formats
VIDEO_FORMATS = ('.avi', '.mp4', '.mov', '.mkv', '.flv', '.wmv', '.webm')

def extract_frames_from_video(video_path, output_folder, frame_skip=4):
    """
    Extract frames from a video file and save them as images.
    
    Args:
        video_path: Path to the video file
        output_folder: Directory to save extracted frames
        frame_skip: Extract every nth frame (default: every 4th frame)
    """
    # Create output folder for this video
    os.makedirs(output_folder, exist_ok=True)
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return False
    
    # Get video properties
    framerate = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"\nProcessing: {os.path.basename(video_path)}")
    print(f"  FPS: {framerate}, Total frames: {total_frames}")
    
    framecount = 0
    count = 0
    
    while True:
        success, frame = cap.read()
        
        if not success:
            break
        
        # Resize frame to standardized size
        frame = cv2.resize(frame, (1280, 720))
        framecount += 1
        
        # Extract every nth frame
        if framecount == frame_skip:
            framecount = 0
            frame_name = f"{Path(video_path).stem}_frame_{count:04d}.jpg"
            frame_path = os.path.join(output_folder, frame_name)
            cv2.imwrite(frame_path, frame)
            count += 1
    
    cap.release()
    print(f"  Extracted {count} frames -> {output_folder}")
    return True

# Main execution
def main():
    print("Starting video to frame conversion...")
    print(f"Video directory: {VIDEO_DIR}")
    print(f"Output directory: {OUTPUT_DIR}\n")
    
    video_count = 0
    processed_count = 0
    
    # Process all subdirectories and files
    for root, dirs, files in os.walk(VIDEO_DIR):
        for file in files:
            if file.lower().endswith(VIDEO_FORMATS):
                video_path = os.path.join(root, file)
                
                # Create output folder with video name
                video_name = Path(file).stem
                relative_path = os.path.relpath(root, VIDEO_DIR)
                
                if relative_path == '.':
                    output_folder = os.path.join(OUTPUT_DIR, video_name)
                else:
                    output_folder = os.path.join(OUTPUT_DIR, relative_path, video_name)
                
                video_count += 1
                if extract_frames_from_video(video_path, output_folder, frame_skip=4):
                    processed_count += 1
    
    print(f"\n{'='*60}")
    print(f"Conversion Complete!")
    print(f"Total videos found: {video_count}")
    print(f"Successfully processed: {processed_count}")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()
