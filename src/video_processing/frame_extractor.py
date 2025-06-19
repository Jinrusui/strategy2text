"""
Frame extraction utility for video processing.
"""

import cv2
import numpy as np
from typing import List, Tuple, Optional
from pathlib import Path


class FrameExtractor:
    """Extracts and processes frames from video files."""
    
    def __init__(
        self, 
        sampling_rate: int = 1,
        max_frames: int = 50,
        resize_resolution: Optional[Tuple[int, int]] = None
    ):
        """
        Initialize frame extractor.
        
        Args:
            sampling_rate: Extract every Nth frame (1 = every frame)
            max_frames: Maximum number of frames to extract
            resize_resolution: Target resolution (width, height) for resizing
        """
        self.sampling_rate = sampling_rate
        self.max_frames = max_frames
        self.resize_resolution = resize_resolution
    
    def extract_frames(self, video_path: str) -> List[np.ndarray]:
        """
        Extract frames from video file.
        
        Args:
            video_path: Path to video file
            
        Returns:
            List of frame arrays
        """
        video_path = Path(video_path)
        if not video_path.exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")
        
        frames = []
        cap = cv2.VideoCapture(str(video_path))
        
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")
        
        frame_count = 0
        extracted_count = 0
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Sample frames according to sampling rate
                if frame_count % self.sampling_rate == 0:
                    # Convert BGR to RGB
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    
                    # Resize if specified
                    if self.resize_resolution:
                        frame_rgb = cv2.resize(frame_rgb, self.resize_resolution)
                    
                    frames.append(frame_rgb)
                    extracted_count += 1
                    
                    # Stop if we've extracted enough frames
                    if extracted_count >= self.max_frames:
                        break
                
                frame_count += 1
                
        finally:
            cap.release()
        
        return frames
    
    def extract_frames_at_timestamps(
        self, 
        video_path: str, 
        timestamps: List[float]
    ) -> List[np.ndarray]:
        """
        Extract frames at specific timestamps.
        
        Args:
            video_path: Path to video file
            timestamps: List of timestamps in seconds
            
        Returns:
            List of frame arrays
        """
        video_path = Path(video_path)
        if not video_path.exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")
        
        frames = []
        cap = cv2.VideoCapture(str(video_path))
        
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        try:
            for timestamp in timestamps:
                # Convert timestamp to frame number
                frame_number = int(timestamp * fps)
                
                # Seek to frame
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
                
                ret, frame = cap.read()
                if ret:
                    # Convert BGR to RGB
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    
                    # Resize if specified
                    if self.resize_resolution:
                        frame_rgb = cv2.resize(frame_rgb, self.resize_resolution)
                    
                    frames.append(frame_rgb)
                
        finally:
            cap.release()
        
        return frames
    
    def get_video_info(self, video_path: str) -> dict:
        """
        Get basic information about a video file.
        
        Args:
            video_path: Path to video file
            
        Returns:
            Dictionary with video information
        """
        video_path = Path(video_path)
        if not video_path.exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")
        
        cap = cv2.VideoCapture(str(video_path))
        
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")
        
        try:
            info = {
                'width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                'fps': cap.get(cv2.CAP_PROP_FPS),
                'frame_count': int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
                'duration': cap.get(cv2.CAP_PROP_FRAME_COUNT) / cap.get(cv2.CAP_PROP_FPS)
            }
            
        finally:
            cap.release()
        
        return info 