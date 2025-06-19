"""
Video sampling module for diverse gameplay analysis.

Implements sampling strategies for typical gameplay, edge cases, and longitudinal analysis.
"""

import os
import random
from enum import Enum
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from pathlib import Path


class SamplingStrategy(Enum):
    """Enumeration of video sampling strategies."""
    TYPICAL = "typical"
    EDGE_CASE = "edge_case"
    LONGITUDINAL = "longitudinal"
    RANDOM = "random"
    BALANCED = "balanced"


class VideoSampler:
    """
    Handles diverse video sampling for comprehensive strategy analysis.
    
    Implements the sampling methodology described in the dissertation:
    - Typical Gameplay: Standard/modal behavior
    - Edge Cases: Unusual situations, near-loss scenarios
    - Longitudinal: Different training stages/checkpoints
    """
    
    def __init__(self, video_dir: str, seed: int = 42):
        """
        Initialize video sampler.
        
        Args:
            video_dir: Directory containing video files
            seed: Random seed for reproducible sampling
        """
        self.video_dir = Path(video_dir)
        self.seed = seed
        random.seed(seed)
        np.random.seed(seed)
        
        # Video metadata storage
        self.video_metadata = {}
        self._load_video_metadata()
    
    def _load_video_metadata(self):
        """Load or create metadata for videos in the directory."""
        metadata_file = self.video_dir / "video_metadata.json"
        
        if metadata_file.exists():
            import json
            with open(metadata_file, 'r') as f:
                self.video_metadata = json.load(f)
        else:
            # Create basic metadata from filenames
            self._create_basic_metadata()
    
    def _create_basic_metadata(self):
        """Create basic metadata from video filenames."""
        video_files = list(self.video_dir.glob("*.mp4")) + list(self.video_dir.glob("*.avi"))
        
        for video_file in video_files:
            filename = video_file.name
            
            # Extract information from filename patterns
            # Expected patterns: checkpoint_X_seed_Y_type.mp4
            metadata = {
                'filepath': str(video_file),
                'filename': filename,
                'checkpoint': self._extract_checkpoint(filename),
                'seed': self._extract_seed(filename),
                'type': self._extract_type(filename),
                'duration': None,  # To be filled when loaded
                'score': None,     # To be filled from game logs
                'is_edge_case': self._is_edge_case(filename)
            }
            
            self.video_metadata[filename] = metadata
    
    def _extract_checkpoint(self, filename: str) -> Optional[int]:
        """Extract checkpoint number from filename."""
        import re
        match = re.search(r'checkpoint[_-](\d+)', filename.lower())
        return int(match.group(1)) if match else None
    
    def _extract_seed(self, filename: str) -> Optional[int]:
        """Extract seed number from filename."""
        import re
        match = re.search(r'seed[_-](\d+)', filename.lower())
        return int(match.group(1)) if match else None
    
    def _extract_type(self, filename: str) -> str:
        """Extract video type from filename."""
        filename_lower = filename.lower()
        if 'edge' in filename_lower or 'unusual' in filename_lower:
            return 'edge_case'
        elif 'typical' in filename_lower or 'normal' in filename_lower:
            return 'typical'
        else:
            return 'unknown'
    
    def _is_edge_case(self, filename: str) -> bool:
        """Determine if video represents an edge case."""
        filename_lower = filename.lower()
        edge_indicators = ['edge', 'unusual', 'loss', 'fail', 'rare', 'novel']
        return any(indicator in filename_lower for indicator in edge_indicators)
    
    def sample_videos(
        self, 
        strategy: SamplingStrategy,
        num_videos: int,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """
        Sample videos according to the specified strategy.
        
        Args:
            strategy: Sampling strategy to use
            num_videos: Number of videos to sample
            **kwargs: Additional strategy-specific parameters
            
        Returns:
            List of video metadata dictionaries
        """
        if strategy == SamplingStrategy.TYPICAL:
            return self._sample_typical(num_videos, **kwargs)
        elif strategy == SamplingStrategy.EDGE_CASE:
            return self._sample_edge_cases(num_videos, **kwargs)
        elif strategy == SamplingStrategy.LONGITUDINAL:
            return self._sample_longitudinal(num_videos, **kwargs)
        elif strategy == SamplingStrategy.RANDOM:
            return self._sample_random(num_videos, **kwargs)
        elif strategy == SamplingStrategy.BALANCED:
            return self._sample_balanced(num_videos, **kwargs)
        else:
            raise ValueError(f"Unknown sampling strategy: {strategy}")
    
    def _sample_typical(self, num_videos: int, **kwargs) -> List[Dict[str, Any]]:
        """Sample typical/modal gameplay videos."""
        typical_videos = [
            video for video in self.video_metadata.values()
            if video['type'] == 'typical' or not video['is_edge_case']
        ]
        
        if len(typical_videos) < num_videos:
            print(f"Warning: Only {len(typical_videos)} typical videos available, requested {num_videos}")
        
        sampled = random.sample(typical_videos, min(num_videos, len(typical_videos)))
        return sampled
    
    def _sample_edge_cases(self, num_videos: int, **kwargs) -> List[Dict[str, Any]]:
        """Sample edge case videos."""
        edge_videos = [
            video for video in self.video_metadata.values()
            if video['is_edge_case'] or video['type'] == 'edge_case'
        ]
        
        if len(edge_videos) < num_videos:
            print(f"Warning: Only {len(edge_videos)} edge case videos available, requested {num_videos}")
        
        sampled = random.sample(edge_videos, min(num_videos, len(edge_videos)))
        return sampled
    
    def _sample_longitudinal(self, num_videos: int, **kwargs) -> List[Dict[str, Any]]:
        """Sample videos from different training checkpoints."""
        checkpoints_per_stage = kwargs.get('checkpoints_per_stage', num_videos // 3)
        
        # Group videos by checkpoint
        checkpoint_groups = {}
        for video in self.video_metadata.values():
            checkpoint = video['checkpoint']
            if checkpoint is not None:
                if checkpoint not in checkpoint_groups:
                    checkpoint_groups[checkpoint] = []
                checkpoint_groups[checkpoint].append(video)
        
        if not checkpoint_groups:
            print("Warning: No checkpoint information found, falling back to random sampling")
            return self._sample_random(num_videos)
        
        # Sort checkpoints and divide into early, mid, late stages
        sorted_checkpoints = sorted(checkpoint_groups.keys())
        num_checkpoints = len(sorted_checkpoints)
        
        early_checkpoints = sorted_checkpoints[:num_checkpoints//3]
        mid_checkpoints = sorted_checkpoints[num_checkpoints//3:2*num_checkpoints//3]
        late_checkpoints = sorted_checkpoints[2*num_checkpoints//3:]
        
        sampled_videos = []
        
        # Sample from each stage
        for stage_checkpoints in [early_checkpoints, mid_checkpoints, late_checkpoints]:
            stage_videos = []
            for checkpoint in stage_checkpoints:
                stage_videos.extend(checkpoint_groups[checkpoint])
            
            if stage_videos:
                stage_sample = random.sample(
                    stage_videos, 
                    min(checkpoints_per_stage, len(stage_videos))
                )
                sampled_videos.extend(stage_sample)
        
        # If we need more videos, sample randomly from remaining
        if len(sampled_videos) < num_videos:
            remaining_videos = [
                v for v in self.video_metadata.values() 
                if v not in sampled_videos
            ]
            additional = random.sample(
                remaining_videos,
                min(num_videos - len(sampled_videos), len(remaining_videos))
            )
            sampled_videos.extend(additional)
        
        return sampled_videos[:num_videos]
    
    def _sample_random(self, num_videos: int, **kwargs) -> List[Dict[str, Any]]:
        """Sample videos randomly."""
        all_videos = list(self.video_metadata.values())
        sampled = random.sample(all_videos, min(num_videos, len(all_videos)))
        return sampled
    
    def _sample_balanced(self, num_videos: int, **kwargs) -> List[Dict[str, Any]]:
        """Sample a balanced mix of typical, edge case, and longitudinal videos."""
        typical_ratio = kwargs.get('typical_ratio', 0.6)
        edge_ratio = kwargs.get('edge_ratio', 0.2)
        longitudinal_ratio = kwargs.get('longitudinal_ratio', 0.2)
        
        num_typical = int(num_videos * typical_ratio)
        num_edge = int(num_videos * edge_ratio)
        num_longitudinal = num_videos - num_typical - num_edge
        
        sampled_videos = []
        
        # Sample typical videos
        if num_typical > 0:
            typical_videos = self._sample_typical(num_typical)
            sampled_videos.extend(typical_videos)
        
        # Sample edge case videos
        if num_edge > 0:
            edge_videos = self._sample_edge_cases(num_edge)
            sampled_videos.extend(edge_videos)
        
        # Sample longitudinal videos (ensuring no duplicates)
        if num_longitudinal > 0:
            remaining_videos = [
                v for v in self.video_metadata.values()
                if v not in sampled_videos
            ]
            longitudinal_sample = random.sample(
                remaining_videos,
                min(num_longitudinal, len(remaining_videos))
            )
            sampled_videos.extend(longitudinal_sample)
        
        return sampled_videos
    
    def get_video_distribution(self) -> Dict[str, int]:
        """Get distribution of video types in the dataset."""
        distribution = {
            'total': len(self.video_metadata),
            'typical': 0,
            'edge_case': 0,
            'with_checkpoint': 0,
            'unknown_type': 0
        }
        
        for video in self.video_metadata.values():
            if video['type'] == 'typical' or not video['is_edge_case']:
                distribution['typical'] += 1
            if video['is_edge_case']:
                distribution['edge_case'] += 1
            if video['checkpoint'] is not None:
                distribution['with_checkpoint'] += 1
            if video['type'] == 'unknown':
                distribution['unknown_type'] += 1
        
        return distribution
    
    def save_metadata(self):
        """Save video metadata to file."""
        import json
        metadata_file = self.video_dir / "video_metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(self.video_metadata, f, indent=2) 