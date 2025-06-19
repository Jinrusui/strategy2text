"""
Video loader utility for handling video files and metadata.
"""

import os
import json
from pathlib import Path
from typing import Dict, List, Optional, Any
import cv2


class VideoLoader:
    """Utility class for loading and managing video files."""
    
    def __init__(self, video_dir: str):
        """
        Initialize video loader.
        
        Args:
            video_dir: Directory containing video files
        """
        self.video_dir = Path(video_dir)
        self.supported_formats = ['.mp4', '.avi', '.mov', '.mkv', '.wmv']
    
    def load_video_list(self) -> List[Dict[str, Any]]:
        """
        Load list of all video files in directory.
        
        Returns:
            List of video metadata dictionaries
        """
        videos = []
        
        for video_file in self.video_dir.rglob('*'):
            if video_file.suffix.lower() in self.supported_formats:
                metadata = self._get_video_metadata(video_file)
                videos.append(metadata)
        
        return videos
    
    def _get_video_metadata(self, video_path: Path) -> Dict[str, Any]:
        """
        Extract metadata from video file.
        
        Args:
            video_path: Path to video file
            
        Returns:
            Dictionary containing video metadata
        """
        metadata = {
            'filepath': str(video_path),
            'filename': video_path.name,
            'size_bytes': video_path.stat().st_size,
            'extension': video_path.suffix.lower()
        }
        
        # Try to get video properties
        try:
            cap = cv2.VideoCapture(str(video_path))
            if cap.isOpened():
                metadata.update({
                    'width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                    'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                    'fps': cap.get(cv2.CAP_PROP_FPS),
                    'frame_count': int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
                    'duration': cap.get(cv2.CAP_PROP_FRAME_COUNT) / cap.get(cv2.CAP_PROP_FPS)
                })
                cap.release()
        except Exception:
            # If we can't read video properties, set defaults
            metadata.update({
                'width': None,
                'height': None,
                'fps': None,
                'frame_count': None,
                'duration': None
            })
        
        return metadata
    
    def validate_video(self, video_path: str) -> bool:
        """
        Validate that a video file can be opened and read.
        
        Args:
            video_path: Path to video file
            
        Returns:
            True if video is valid, False otherwise
        """
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                return False
            
            # Try to read first frame
            ret, frame = cap.read()
            cap.release()
            
            return ret and frame is not None
            
        except Exception:
            return False
    
    def filter_videos(
        self, 
        videos: List[Dict[str, Any]], 
        **criteria
    ) -> List[Dict[str, Any]]:
        """
        Filter videos based on criteria.
        
        Args:
            videos: List of video metadata
            **criteria: Filtering criteria
            
        Returns:
            Filtered list of videos
        """
        filtered = videos.copy()
        
        # Filter by minimum duration
        if 'min_duration' in criteria:
            min_dur = criteria['min_duration']
            filtered = [v for v in filtered if v.get('duration', 0) >= min_dur]
        
        # Filter by maximum duration
        if 'max_duration' in criteria:
            max_dur = criteria['max_duration']
            filtered = [v for v in filtered if v.get('duration', float('inf')) <= max_dur]
        
        # Filter by resolution
        if 'min_width' in criteria:
            min_w = criteria['min_width']
            filtered = [v for v in filtered if v.get('width', 0) >= min_w]
        
        if 'min_height' in criteria:
            min_h = criteria['min_height']
            filtered = [v for v in filtered if v.get('height', 0) >= min_h]
        
        # Filter by file extension
        if 'extensions' in criteria:
            exts = criteria['extensions']
            filtered = [v for v in filtered if v.get('extension') in exts]
        
        return filtered 