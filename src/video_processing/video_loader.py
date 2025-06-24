"""
Simple video loader utility for RL agent recordings.
"""

import os
import logging
from typing import List, Optional, Dict
from pathlib import Path


class VideoLoader:
    """Utility for loading and validating RL agent video files."""
    
    SUPPORTED_FORMATS = {'.mp4', '.avi', '.mov', '.mkv', '.webm'}
    
    def __init__(self):
        """Initialize the video loader."""
        self.logger = logging.getLogger(__name__)
    
    def find_videos(self, directory: str, recursive: bool = True) -> List[str]:
        """
        Find all video files in a directory.
        
        Args:
            directory: Directory path to search
            recursive: Whether to search subdirectories
            
        Returns:
            List of video file paths
        """
        directory = Path(directory)
        if not directory.exists():
            self.logger.warning(f"Directory does not exist: {directory}")
            return []
        
        video_files = []
        
        if recursive:
            pattern = "**/*"
        else:
            pattern = "*"
        
        for file_path in directory.glob(pattern):
            if file_path.is_file() and file_path.suffix.lower() in self.SUPPORTED_FORMATS:
                video_files.append(str(file_path))
        
        self.logger.info(f"Found {len(video_files)} video files in {directory}")
        return sorted(video_files)
    
    def validate_video(self, video_path: str) -> bool:
        """
        Validate that a video file exists and has a supported format.
        
        Args:
            video_path: Path to the video file
            
        Returns:
            True if video is valid, False otherwise
        """
        video_path = Path(video_path)
        
        if not video_path.exists():
            self.logger.error(f"Video file does not exist: {video_path}")
            return False
        
        if not video_path.is_file():
            self.logger.error(f"Path is not a file: {video_path}")
            return False
        
        if video_path.suffix.lower() not in self.SUPPORTED_FORMATS:
            self.logger.error(f"Unsupported video format: {video_path.suffix}")
            return False
        
        # Check file size (should be > 0)
        if video_path.stat().st_size == 0:
            self.logger.error(f"Video file is empty: {video_path}")
            return False
        
        return True
    
    def get_video_info(self, video_path: str) -> Optional[Dict[str, any]]:
        """
        Get basic information about a video file.
        
        Args:
            video_path: Path to the video file
            
        Returns:
            Dictionary with video information or None if invalid
        """
        if not self.validate_video(video_path):
            return None
        
        video_path = Path(video_path)
        stat = video_path.stat()
        
        return {
            "path": str(video_path),
            "name": video_path.name,
            "size_bytes": stat.st_size,
            "size_mb": round(stat.st_size / (1024 * 1024), 2),
            "format": video_path.suffix.lower(),
            "modified_time": stat.st_mtime
        }
    
    def filter_by_pattern(self, video_paths: List[str], pattern: str) -> List[str]:
        """
        Filter video paths by a filename pattern.
        
        Args:
            video_paths: List of video file paths
            pattern: Pattern to match (e.g., "ppo", "breakout")
            
        Returns:
            Filtered list of video paths
        """
        pattern = pattern.lower()
        filtered = []
        
        for video_path in video_paths:
            filename = Path(video_path).name.lower()
            if pattern in filename:
                filtered.append(video_path)
        
        self.logger.info(f"Filtered {len(video_paths)} videos to {len(filtered)} matching '{pattern}'")
        return filtered
    
    def group_by_environment(self, video_paths: List[str]) -> Dict[str, List[str]]:
        """
        Group video paths by detected environment name (Breakout only).
        
        Args:
            video_paths: List of video file paths
            
        Returns:
            Dictionary mapping environment names to video paths
        """
        # Only Breakout environment patterns
        env_patterns = {
            'breakout': ['breakout', 'break-out', 'break_out'],
        }
        
        groups = {}
        unmatched = []
        
        for video_path in video_paths:
            filename = Path(video_path).name.lower()
            matched = False
            
            for env_name, patterns in env_patterns.items():
                if any(pattern in filename for pattern in patterns):
                    if env_name not in groups:
                        groups[env_name] = []
                    groups[env_name].append(video_path)
                    matched = True
                    break
            
            if not matched:
                unmatched.append(video_path)
        
        if unmatched:
            groups['unknown'] = unmatched
        
        return groups 