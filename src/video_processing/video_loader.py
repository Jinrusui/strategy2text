"""
Video processing utilities for HVA-X hierarchical video analysis.
Supports trajectory sampling, performance stratification, and video management.
"""

import os
import logging
import random
from typing import List, Optional, Dict, Tuple, Any
from pathlib import Path
from dataclasses import dataclass


@dataclass
class TrajectoryData:
    """Data structure for storing trajectory information."""
    video_path: str
    score: float
    episode_id: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert trajectory data to a dictionary."""
        return {
            "video_path": self.video_path,
            "score": self.score,
            "episode_id": self.episode_id,
            "metadata": self.metadata,
        }


class TrajectoryProcessor:
    """Processes agent trajectories for HVA-X analysis including sampling and stratification."""
    
    def __init__(self):
        """Initialize the trajectory processor."""
        self.logger = logging.getLogger(__name__)
    
    def create_trajectory_data(self, video_paths: List[str], scores: List[float], 
                              episode_ids: Optional[List[str]] = None) -> List[TrajectoryData]:
        """
        Create trajectory data from video paths and scores.
        
        Args:
            video_paths: List of video file paths
            scores: List of corresponding scores
            episode_ids: Optional list of episode identifiers
            
        Returns:
            List of TrajectoryData objects
        """
        if len(video_paths) != len(scores):
            raise ValueError("Number of video paths must match number of scores")
        
        if episode_ids and len(episode_ids) != len(video_paths):
            raise ValueError("Number of episode IDs must match number of video paths")
        
        trajectories = []
        for i, (video_path, score) in enumerate(zip(video_paths, scores)):
            episode_id = episode_ids[i] if episode_ids else f"episode_{i}"
            trajectories.append(TrajectoryData(
                video_path=video_path,
                score=score,
                episode_id=episode_id
            ))
        
        self.logger.info(f"Created {len(trajectories)} trajectory data entries")
        return trajectories
    
    def stratify_trajectories(self, trajectories: List[TrajectoryData], 
                             low_percentile: float = 10, 
                             high_percentile: float = 90) -> Dict[str, List[TrajectoryData]]:
        """
        Stratify trajectories into performance tiers as per HVA-X algorithm.
        
        Args:
            trajectories: List of trajectory data
            low_percentile: Percentile threshold for low tier (default: 10%)
            high_percentile: Percentile threshold for high tier (default: 90%)
            
        Returns:
            Dictionary with 'low_tier', 'mid_tier', 'high_tier' keys
        """
        if len(trajectories) < 10:
            self.logger.warning("Less than 10 trajectories available for stratification")
        
        # Sort trajectories by score
        sorted_trajectories = sorted(trajectories, key=lambda t: t.score)
        n_total = len(sorted_trajectories)
        
        # Calculate tier boundaries
        low_boundary = int(n_total * low_percentile / 100)
        high_boundary = int(n_total * high_percentile / 100)
        
        # Calculate middle tier boundaries (45th to 55th percentile)
        mid_low = int(n_total * 0.45)
        mid_high = int(n_total * 0.55)
        
        # Create tiers
        tiers = {
            'low_tier': sorted_trajectories[:low_boundary],
            'mid_tier': sorted_trajectories[mid_low:mid_high],
            'high_tier': sorted_trajectories[high_boundary:]
        }
        
        self.logger.info(f"Stratified {n_total} trajectories into tiers: "
                        f"low={len(tiers['low_tier'])}, "
                        f"mid={len(tiers['mid_tier'])}, "
                        f"high={len(tiers['high_tier'])}")
        
        return tiers
    
    def sample_from_tiers(self, stratified_tiers: Dict[str, List[TrajectoryData]], 
                         samples_per_tier: int = 3) -> Dict[str, List[TrajectoryData]]:
        """
        Sample trajectories from each tier for HVA-X analysis.
        
        Args:
            stratified_tiers: Dictionary of stratified trajectories
            samples_per_tier: Number of samples to take from each tier
            
        Returns:
            Dictionary with sampled trajectories from each tier
        """
        sampled_tiers = {}
        
        for tier_name, trajectories in stratified_tiers.items():
            if len(trajectories) == 0:
                self.logger.warning(f"No trajectories in {tier_name}")
                sampled_tiers[tier_name] = []
                continue
            
            # Sample up to samples_per_tier trajectories
            n_samples = min(samples_per_tier, len(trajectories))
            sampled = random.sample(trajectories, n_samples)
            sampled_tiers[tier_name] = sampled
            
            self.logger.info(f"Sampled {n_samples} trajectories from {tier_name}")
        
        total_samples = sum(len(samples) for samples in sampled_tiers.values())
        self.logger.info(f"Total sampled trajectories: {total_samples}")
        
        return sampled_tiers
    
    def run_trajectory_sampling(self, trajectories: List[TrajectoryData], 
                               samples_per_tier: int = 3) -> Dict[str, List[TrajectoryData]]:
        """
        Complete trajectory sampling pipeline for HVA-X Phase 1.
        
        Args:
            trajectories: List of trajectory data
            samples_per_tier: Number of samples per tier
            
        Returns:
            Dictionary with sampled trajectories ready for analysis
        """
        self.logger.info("Starting HVA-X Phase 1: Trajectory Sampling and Stratification")
        
        # Phase 1: Stratify trajectories
        stratified = self.stratify_trajectories(trajectories)
        
        # Phase 1: Sample from each tier
        sampled = self.sample_from_tiers(stratified, samples_per_tier)
        
        self.logger.info("Phase 1 completed successfully")
        return sampled


class VideoLoader:
    """Enhanced video loader with HVA-X trajectory processing capabilities."""
    
    SUPPORTED_FORMATS = {'.mp4', '.avi', '.mov', '.mkv', '.webm'}
    
    def __init__(self):
        """Initialize the video loader."""
        self.logger = logging.getLogger(__name__)
        self.trajectory_processor = TrajectoryProcessor()
    
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
    
    def get_video_info(self, video_path: str) -> Optional[Dict[str, Any]]:
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
        Group video paths by detected environment name.
        
        Args:
            video_paths: List of video file paths
            
        Returns:
            Dictionary mapping environment names to video paths
        """
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
    
    def load_trajectory_data_from_files(self, video_dir: str, score_file: str) -> List[TrajectoryData]:
        """
        Load trajectory data from video directory and score file.
        
        Args:
            video_dir: Directory containing video files
            score_file: Path to file containing scores (one per line)
            
        Returns:
            List of TrajectoryData objects
        """
        videos = self.find_videos(video_dir)
        
        # Load scores from file
        scores = []
        try:
            with open(score_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        scores.append(float(line))
        except (FileNotFoundError, ValueError) as e:
            self.logger.error(f"Error loading scores from {score_file}: {e}")
            raise
        
        if len(videos) != len(scores):
            self.logger.error(f"Mismatch: {len(videos)} videos vs {len(scores)} scores")
            raise ValueError("Number of videos must match number of scores")
        
        return self.trajectory_processor.create_trajectory_data(videos, scores)
    
    def load_trajectory_data_from_csv(self, csv_file: str) -> List[TrajectoryData]:
        """
        Load trajectory data from CSV file.
        
        Args:
            csv_file: Path to CSV file with columns: video_path, score, episode_id
            
        Returns:
            List of TrajectoryData objects
        """
        import csv
        trajectories = []
        
        try:
            with open(csv_file, 'r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    trajectory = TrajectoryData(
                        video_path=row['video_path'],
                        score=float(row['score']),
                        episode_id=row.get('episode_id'),
                        metadata=row.get('metadata')
                    )
                    trajectories.append(trajectory)
        except Exception as e:
            self.logger.error(f"Error loading trajectory data from CSV: {e}")
            raise
        
        self.logger.info(f"Loaded {len(trajectories)} trajectories from CSV")
        return trajectories
    
    def save_trajectory_data_to_csv(self, trajectories: List[TrajectoryData], output_file: str):
        """
        Save trajectory data to CSV file.
        
        Args:
            trajectories: List of TrajectoryData objects
            output_file: Path to output CSV file
        """
        import csv
        
        with open(output_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['video_path', 'score', 'episode_id', 'metadata'])
            
            for trajectory in trajectories:
                writer.writerow([
                    trajectory.video_path,
                    trajectory.score,
                    trajectory.episode_id,
                    trajectory.metadata
                ])
        
        self.logger.info(f"Saved {len(trajectories)} trajectories to {output_file}")
    
    def prepare_trajectories_for_hva(self, trajectories: List[TrajectoryData], 
                                   samples_per_tier: int = 3) -> Dict[str, List[TrajectoryData]]:
        """
        Prepare trajectories for HVA-X analysis by running the complete sampling pipeline.
        
        Args:
            trajectories: List of TrajectoryData objects
            samples_per_tier: Number of samples per performance tier
            
        Returns:
            Dictionary with sampled trajectories ready for HVA-X analysis
        """
        return self.trajectory_processor.run_trajectory_sampling(trajectories, samples_per_tier) 