#!/usr/bin/env python3
"""
Convert HIGHLIGHTS frames to videos
This script can be used when the main video generation fails
"""

import os
import glob
import cv2
import argparse
from pathlib import Path

def create_videos_opencv(frames_dir, output_dir, fps=5, num_highlights=5):
    """Create videos using OpenCV (fallback method)"""
    print(f"Creating videos using OpenCV...")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    for hl in range(num_highlights):
        hl_str = str(hl).zfill(2)  # Use 2-digit format
        
        # Find all frames for this highlight
        pattern = os.path.join(frames_dir, f"{hl_str}_*.png")
        frame_files = sorted(glob.glob(pattern))
        
        if not frame_files:
            print(f"No frames found for highlight {hl}")
            continue
            
        print(f"Creating video for highlight {hl} with {len(frame_files)} frames")
        
        # Read first frame to get dimensions
        first_frame = cv2.imread(frame_files[0])
        if first_frame is None:
            print(f"Could not read first frame: {frame_files[0]}")
            continue
            
        height, width, channels = first_frame.shape
        
        # Create video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_path = os.path.join(output_dir, f'HL_{hl}.mp4')
        out = cv2.VideoWriter(video_path, fourcc, fps, (width, height))
        
        # Write frames
        for frame_file in frame_files:
            frame = cv2.imread(frame_file)
            if frame is not None:
                out.write(frame)
            else:
                print(f"Warning: Could not read frame {frame_file}")
        
        out.release()
        print(f"Created video: {video_path}")

def create_videos_av(frames_dir, output_dir, fps=5, num_highlights=5):
    """Create videos using av library (original method)"""
    import av
    print(f"Creating videos using av library...")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    for hl in range(num_highlights):
        hl_str = str(hl).zfill(2)  # Use 2-digit format
        
        # Find all frames for this highlight
        pattern = os.path.join(frames_dir, f"{hl_str}_*.png")
        frame_files = sorted(glob.glob(pattern))
        
        if not frame_files:
            print(f"No frames found for highlight {hl}")
            continue
            
        print(f"Creating video for highlight {hl} with {len(frame_files)} frames")
        
        # Read first frame to get dimensions
        first_frame = cv2.imread(frame_files[0])
        if first_frame is None:
            print(f"Could not read first frame: {frame_files[0]}")
            continue
            
        height, width, channels = first_frame.shape
        
        # Create video using av
        video_path = os.path.join(output_dir, f'HL_{hl}.mp4')
        
        try:
            output = av.open(video_path, 'w')
            stream = output.add_stream('h264', rate=fps)
            stream.width = width
            stream.height = height
            stream.pix_fmt = 'yuv420p'
            
            for frame_file in frame_files:
                frame = cv2.imread(frame_file)
                if frame is not None:
                    # Convert BGR to RGB
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    av_frame = av.VideoFrame.from_ndarray(frame_rgb, format='rgb24')
                    packet = stream.encode(av_frame)
                    output.mux(packet)
                else:
                    print(f"Warning: Could not read frame {frame_file}")
            
            # Flush
            packet = stream.encode(None)
            output.mux(packet)
            output.close()
            print(f"Created video: {video_path}")
            
        except Exception as e:
            print(f"Error creating video with av: {e}")
            # Fall back to OpenCV
            create_videos_opencv(frames_dir, output_dir, fps, 1)

def main():
    parser = argparse.ArgumentParser(description="Convert HIGHLIGHTS frames to videos")
    parser.add_argument("--frames_dir", required=True, help="Directory containing frames")
    parser.add_argument("--output_dir", required=True, help="Output directory for videos", default="highlights_videos")
    parser.add_argument("--fps", type=int, default=5, help="Frames per second")
    parser.add_argument("--num_highlights", type=int, default=5, help="Number of highlights")
    parser.add_argument("--method", choices=['opencv', 'av', 'auto'], default='av', 
                       help="Video creation method")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.frames_dir):
        print(f"Error: Frames directory not found: {args.frames_dir}")
        return
    
    print(f"Converting frames from {args.frames_dir} to videos in {args.output_dir}")
    print(f"Parameters: fps={args.fps}, num_highlights={args.num_highlights}")
    
    if args.method == 'opencv':
        create_videos_opencv(args.frames_dir, args.output_dir, args.fps, args.num_highlights)
    elif args.method == 'av':
        create_videos_av(args.frames_dir, args.output_dir, args.fps, args.num_highlights)
    else:  # auto
        try:
            create_videos_av(args.frames_dir, args.output_dir, args.fps, args.num_highlights)
        except Exception as e:
            print(f"av method failed: {e}")
            print("Falling back to OpenCV method...")
            create_videos_opencv(args.frames_dir, args.output_dir, args.fps, args.num_highlights)

if __name__ == "__main__":
    main() 