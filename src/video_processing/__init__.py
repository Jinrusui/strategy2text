"""
Video Processing Package - Trajectory sampling and video management for HVA-X analysis.

Provides trajectory sampling, performance stratification, and video loading utilities
for the HVA-X hierarchical video analysis algorithm.
"""

from .video_loader import VideoLoader, TrajectoryProcessor, TrajectoryData

__all__ = [
    'VideoLoader',
    'TrajectoryProcessor', 
    'TrajectoryData'
]

__version__ = "1.0.0" 