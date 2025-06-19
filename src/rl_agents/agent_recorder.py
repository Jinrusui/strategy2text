"""
Agent recorder for generating video datasets from RL agents.

Records agent gameplay, saves videos, and creates metadata for strategy analysis.
"""

import os
import json
import numpy as np
import cv2
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path
from datetime import datetime
import imageio

from .sb3_agent import SB3Agent


class AgentRecorder:
    """
    Records agent gameplay and generates video datasets for strategy analysis.
    
    Supports different recording modes:
    - Typical gameplay: Standard episodes
    - Edge cases: Episodes with unusual events or poor performance
    - Longitudinal: Episodes from different training checkpoints
    """
    
    def __init__(self, output_dir: str = "data/videos"):
        """
        Initialize agent recorder.
        
        Args:
            output_dir: Directory to save recorded videos
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Recording metadata
        self.recording_metadata = {
            'videos': [],
            'agents': [],
            'recording_sessions': []
        }
        
        self._load_existing_metadata()
    
    def _load_existing_metadata(self):
        """Load existing metadata if available."""
        metadata_file = self.output_dir / "recording_metadata.json"
        if metadata_file.exists():
            with open(metadata_file, 'r') as f:
                self.recording_metadata = json.load(f)
    
    def _save_metadata(self):
        """Save recording metadata."""
        metadata_file = self.output_dir / "recording_metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(self.recording_metadata, f, indent=2, default=str)
    
    def record_agent_episodes(
        self,
        agent: SB3Agent,
        num_episodes: int = 10,
        recording_type: str = "typical",
        max_steps_per_episode: int = 10000,
        seeds: Optional[List[int]] = None,
        video_format: str = "mp4",
        fps: int = 30,
        frame_skip: int = 1
    ) -> List[Dict[str, Any]]:
        """
        Record multiple episodes from an agent.
        
        Args:
            agent: SB3Agent to record
            num_episodes: Number of episodes to record
            recording_type: Type of recording ('typical', 'edge_case', 'longitudinal')
            max_steps_per_episode: Maximum steps per episode
            seeds: Random seeds for episodes
            video_format: Video format ('mp4', 'avi')
            fps: Frames per second for video
            frame_skip: Skip every N frames to reduce file size
            
        Returns:
            List of video metadata
        """
        print(f"Recording {num_episodes} {recording_type} episodes...")
        
        # Generate seeds if not provided
        if seeds is None:
            seeds = [np.random.randint(0, 10000) for _ in range(num_episodes)]
        
        # Run episodes and collect data
        episodes_data = agent.run_multiple_episodes(
            num_episodes=num_episodes,
            max_steps_per_episode=max_steps_per_episode,
            seeds=seeds,
            record_frames=True
        )
        
        # Save videos and create metadata
        video_metadata = []
        session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        for i, episode_data in enumerate(episodes_data):
            video_meta = self._save_episode_video(
                episode_data=episode_data,
                agent=agent,
                recording_type=recording_type,
                session_id=session_id,
                episode_index=i,
                video_format=video_format,
                fps=fps,
                frame_skip=frame_skip
            )
            
            if video_meta:
                video_metadata.append(video_meta)
        
        # Update recording metadata
        session_metadata = {
            'session_id': session_id,
            'agent_info': agent.get_agent_info(),
            'recording_type': recording_type,
            'num_episodes': num_episodes,
            'timestamp': datetime.now().isoformat(),
            'video_files': [vm['filename'] for vm in video_metadata]
        }
        
        self.recording_metadata['recording_sessions'].append(session_metadata)
        self._save_metadata()
        
        print(f"Recorded {len(video_metadata)} videos in session {session_id}")
        return video_metadata
    
    def _save_episode_video(
        self,
        episode_data: Dict[str, Any],
        agent: SB3Agent,
        recording_type: str,
        session_id: str,
        episode_index: int,
        video_format: str = "mp4",
        fps: int = 30,
        frame_skip: int = 1
    ) -> Optional[Dict[str, Any]]:
        """
        Save individual episode as video file.
        
        Args:
            episode_data: Episode data from agent
            agent: SB3Agent that generated the episode
            recording_type: Type of recording
            session_id: Recording session ID
            episode_index: Index of episode in session
            video_format: Video format
            fps: Frames per second
            frame_skip: Frame skip factor
            
        Returns:
            Video metadata or None if failed
        """
        frames = episode_data.get('frames', [])
        if not frames:
            print(f"Warning: No frames recorded for episode {episode_index}")
            return None
        
        # Create filename
        agent_info = agent.get_agent_info()
        algorithm = agent_info.get('algorithm', 'unknown')
        env_id = agent_info.get('env_id', 'unknown').replace('/', '_')
        
        filename = f"{algorithm}_{env_id}_{recording_type}_{session_id}_ep{episode_index:03d}.{video_format}"
        video_path = self.output_dir / filename
        
        try:
            # Apply frame skipping
            if frame_skip > 1:
                frames = frames[::frame_skip]
            
            # Save video
            if video_format.lower() == 'mp4':
                self._save_mp4_video(frames, video_path, fps)
            elif video_format.lower() == 'avi':
                self._save_avi_video(frames, video_path, fps)
            else:
                raise ValueError(f"Unsupported video format: {video_format}")
            
            # Create video metadata
            video_metadata = {
                'filename': filename,
                'filepath': str(video_path),
                'session_id': session_id,
                'episode_index': episode_index,
                'recording_type': recording_type,
                'agent_algorithm': algorithm,
                'env_id': agent_info.get('env_id'),
                'agent_path': agent_info.get('agent_path'),
                'seed': episode_data.get('seed'),
                'total_reward': episode_data.get('total_reward'),
                'episode_length': episode_data.get('episode_length'),
                'num_frames': len(frames),
                'fps': fps,
                'frame_skip': frame_skip,
                'video_format': video_format,
                'file_size_bytes': video_path.stat().st_size,
                'timestamp': datetime.now().isoformat(),
                'is_edge_case': self._classify_edge_case(episode_data),
                'performance_metrics': {
                    'reward_per_step': episode_data.get('total_reward', 0) / max(episode_data.get('episode_length', 1), 1),
                    'completion_rate': episode_data.get('episode_length', 0) / 10000  # Assuming max 10k steps
                }
            }
            
            # Add to recording metadata
            self.recording_metadata['videos'].append(video_metadata)
            
            return video_metadata
            
        except Exception as e:
            print(f"Error saving video for episode {episode_index}: {e}")
            return None
    
    def _save_mp4_video(self, frames: List[np.ndarray], video_path: Path, fps: int):
        """Save frames as MP4 video using imageio."""
        with imageio.get_writer(str(video_path), fps=fps) as writer:
            for frame in frames:
                writer.append_data(frame)
    
    def _save_avi_video(self, frames: List[np.ndarray], video_path: Path, fps: int):
        """Save frames as AVI video using OpenCV."""
        if not frames:
            return
        
        height, width = frames[0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        
        out = cv2.VideoWriter(str(video_path), fourcc, fps, (width, height))
        
        try:
            for frame in frames:
                # Convert RGB to BGR for OpenCV
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                out.write(frame_bgr)
        finally:
            out.release()
    
    def _classify_edge_case(self, episode_data: Dict[str, Any]) -> bool:
        """
        Classify whether an episode represents an edge case.
        
        Args:
            episode_data: Episode data
            
        Returns:
            True if episode is classified as edge case
        """
        total_reward = episode_data.get('total_reward', 0)
        episode_length = episode_data.get('episode_length', 0)
        
        # Simple heuristics for edge case classification
        # These can be customized based on the specific environment
        
        # Very low reward
        if total_reward < 10:
            return True
        
        # Very short episode (early termination)
        if episode_length < 100:
            return True
        
        # Check for unusual patterns in rewards
        rewards = episode_data.get('rewards', [])
        if rewards:
            # No positive rewards throughout episode
            if all(r <= 0 for r in rewards):
                return True
            
            # Very inconsistent rewards (high variance)
            if len(rewards) > 10:
                reward_std = np.std(rewards)
                reward_mean = np.mean(rewards)
                if reward_mean > 0 and reward_std / reward_mean > 2.0:
                    return True
        
        return False
    
    def record_longitudinal_dataset(
        self,
        agent_checkpoints: List[str],
        episodes_per_checkpoint: int = 5,
        **recording_kwargs
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Record longitudinal dataset from multiple agent checkpoints.
        
        Args:
            agent_checkpoints: List of paths to agent checkpoints
            episodes_per_checkpoint: Number of episodes per checkpoint
            **recording_kwargs: Additional recording arguments
            
        Returns:
            Dictionary mapping checkpoint paths to video metadata
        """
        longitudinal_data = {}
        
        for i, checkpoint_path in enumerate(agent_checkpoints):
            print(f"Recording checkpoint {i+1}/{len(agent_checkpoints)}: {checkpoint_path}")
            
            try:
                # Load agent from checkpoint
                agent = SB3Agent(agent_path=checkpoint_path)
                
                # Record episodes
                video_metadata = self.record_agent_episodes(
                    agent=agent,
                    num_episodes=episodes_per_checkpoint,
                    recording_type="longitudinal",
                    **recording_kwargs
                )
                
                longitudinal_data[checkpoint_path] = video_metadata
                
            except Exception as e:
                print(f"Error recording checkpoint {checkpoint_path}: {e}")
                longitudinal_data[checkpoint_path] = []
        
        return longitudinal_data
    
    def generate_edge_case_dataset(
        self,
        agent: SB3Agent,
        target_num_edge_cases: int = 10,
        max_attempts: int = 100,
        **recording_kwargs
    ) -> List[Dict[str, Any]]:
        """
        Generate dataset focused on edge cases.
        
        Args:
            agent: SB3Agent to record
            target_num_edge_cases: Target number of edge case videos
            max_attempts: Maximum episodes to try
            **recording_kwargs: Additional recording arguments
            
        Returns:
            List of edge case video metadata
        """
        edge_case_videos = []
        attempts = 0
        
        print(f"Generating edge case dataset (target: {target_num_edge_cases} videos)")
        
        while len(edge_case_videos) < target_num_edge_cases and attempts < max_attempts:
            # Record a batch of episodes
            batch_size = min(10, max_attempts - attempts)
            
            video_metadata = self.record_agent_episodes(
                agent=agent,
                num_episodes=batch_size,
                recording_type="edge_case_search",
                **recording_kwargs
            )
            
            # Filter for edge cases
            for video_meta in video_metadata:
                if video_meta.get('is_edge_case', False):
                    edge_case_videos.append(video_meta)
                    
                    if len(edge_case_videos) >= target_num_edge_cases:
                        break
            
            attempts += batch_size
            print(f"Found {len(edge_case_videos)}/{target_num_edge_cases} edge cases (attempts: {attempts})")
        
        print(f"Edge case dataset generation complete: {len(edge_case_videos)} videos")
        return edge_case_videos
    
    def get_recording_summary(self) -> Dict[str, Any]:
        """Get summary of all recordings."""
        videos = self.recording_metadata.get('videos', [])
        
        if not videos:
            return {'total_videos': 0}
        
        # Compute statistics
        recording_types = [v.get('recording_type', 'unknown') for v in videos]
        algorithms = [v.get('agent_algorithm', 'unknown') for v in videos]
        total_rewards = [v.get('total_reward', 0) for v in videos if v.get('total_reward') is not None]
        edge_cases = [v for v in videos if v.get('is_edge_case', False)]
        
        summary = {
            'total_videos': len(videos),
            'recording_types': {rt: recording_types.count(rt) for rt in set(recording_types)},
            'algorithms': {alg: algorithms.count(alg) for alg in set(algorithms)},
            'edge_cases': len(edge_cases),
            'edge_case_rate': len(edge_cases) / len(videos) if videos else 0,
            'total_file_size_mb': sum(v.get('file_size_bytes', 0) for v in videos) / (1024 * 1024),
            'reward_statistics': {
                'mean': np.mean(total_rewards) if total_rewards else 0,
                'std': np.std(total_rewards) if total_rewards else 0,
                'min': np.min(total_rewards) if total_rewards else 0,
                'max': np.max(total_rewards) if total_rewards else 0
            } if total_rewards else {},
            'recording_sessions': len(self.recording_metadata.get('recording_sessions', []))
        }
        
        return summary 