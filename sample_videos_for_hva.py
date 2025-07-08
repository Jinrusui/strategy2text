#!/usr/bin/env python3
"""
Video Sampling Script for HVA-X Analysis
Generates multiple agent episodes with varying performance for comprehensive analysis.
"""

import os
import sys
import subprocess
import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import argparse
from datetime import datetime
import re

# Remove the problematic import since video_processing doesn't exist yet
# sys.path.append(str(Path(__file__).parent / "src"))
# from video_processing import VideoLoader, TrajectoryData


class HVAVideoSampler:
    """Samples agent videos for HVA-X analysis with performance diversity."""
    
    def __init__(self, python_path: Optional[str] = None):
        """
        Initialize the video sampler.
        
        Args:
            python_path: Path to Python executable in your conda environment
        """
        self.python_path = python_path or sys.executable
        self.logger = logging.getLogger(__name__)
        
        # Default sampling configuration
        self.config = {
            "algo": "ppo",
            "env": "BreakoutNoFrameskip-v4",
            "folder": "rl-trained-agents",
            "base_output_dir": "hva_videos",
            "episodes_per_batch": 10,
            "total_episodes": 100,
            "seeds": list(range(1, 21)),  # 20 different seeds for diversity
            "timesteps_per_episode": 2000
        }
    
    def sample_episodes_for_hva(self, num_episodes: int = 100, 
                               output_dir: str = "hva_videos") -> Dict[str, Any]:
        """
        Sample episodes specifically for HVA-X analysis.
        
        Args:
            num_episodes: Total number of episodes to sample
            output_dir: Directory to save videos
            
        Returns:
            Dictionary with sampling results and metadata
        """
        self.logger.info(f"Starting HVA-X video sampling: {num_episodes} episodes")
        
        # Create output directory
        output_path = Path(output_dir).resolve()
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Sample episodes with diverse seeds for performance variation
        video_files = []
        episode_scores = []
        episode_metadata = []
        
        # Use different seeds to get performance diversity
        seeds = self._generate_diverse_seeds(num_episodes)
        
        for i, seed in enumerate(seeds):
            episode_id = f"episode_{i+1:03d}"
            
            try:
                # Record single episode

                video_path, score, metadata = self._record_single_episode(
                    seed=seed,
                    episode_id=episode_id,
                    output_dir=output_path
                )
                
                video_files.append(video_path)
                episode_scores.append(score)
                episode_metadata.append(metadata)
                
                self.logger.info(f"✓ Episode {i+1}/{num_episodes}: {episode_id} (score: {score})")
                
            except Exception as e:
                self.logger.error(f"✗ Failed to record episode {episode_id}: {e}")
                continue
        
        # Save trajectory data
        self._save_trajectory_data(video_files, episode_scores, episode_metadata, output_path)
        
        # Generate summary
        summary = self._generate_sampling_summary(video_files, episode_scores, output_path)
        
        self.logger.info(f"Sampling complete! {len(video_files)} episodes recorded")
        return summary
    
    def _generate_diverse_seeds(self, num_episodes: int) -> List[int]:
        """Generate diverse seeds to ensure performance variation."""
        # Use a mix of predefined seeds and random seeds
        base_seeds = [1, 42, 123, 456, 789, 999, 1337, 2021, 2023, 2024]
        
        seeds = []
        
        # Add base seeds first
        for seed in base_seeds:
            if len(seeds) < num_episodes:
                seeds.append(seed)
        
        # Add additional seeds if needed
        import random
        random.seed(42)  # For reproducible seed generation
        while len(seeds) < num_episodes:
            new_seed = random.randint(1, 10000)
            if new_seed not in seeds:
                seeds.append(new_seed)
        
        return seeds[:num_episodes]
    
    def _record_single_episode(self, seed: int, episode_id: str, 
                              output_dir: Path) -> Tuple[str, float, Dict[str, Any]]:
        """Record a single episode and extract its score."""
        
        # Create episode-specific directory
        episode_dir = output_dir / episode_id
        episode_dir.mkdir(exist_ok=True)
        
        # Record video using rl-baselines3-zoo
        cmd = [
            self.python_path, "-m", "rl_zoo3.record_video_score",
            "--algo", self.config["algo"],
            "--env", self.config["env"],
            "--folder", self.config["folder"],
            "--output-folder", str(episode_dir),
            "--seed", str(seed),
            "--n-timesteps", str(self.config["timesteps_per_episode"]),
            "--stochastic",  # Use stochastic actions for more variation
            "--no-render"
        ]
        
        # Run recording in rl-baselines3-zoo directory
        zoo_dir = Path("rl-baselines3-zoo")
        if not zoo_dir.exists():
            raise FileNotFoundError("rl-baselines3-zoo directory not found")
        
        result = subprocess.run(cmd, cwd=zoo_dir, capture_output=True, text=True)
        print(result.stdout)
        
        if result.returncode != 0:
            raise RuntimeError(f"Recording failed: {result.stderr}")
        
        # Find the generated video file
        video_files = list(episode_dir.glob("*.mp4"))
        if not video_files:
            raise FileNotFoundError("No video file generated")
        
        video_path = video_files[0]
        
        # Rename video file to episode ID
        new_video_path = episode_dir / f"{episode_id}.mp4"
        video_path.rename(new_video_path)
        
        # Extract score from the rl_zoo3.record_video script's output
        score = self._extract_real_score(result.stdout)
        
        # If score extraction fails, use a default score of 0.0
        if score is None:
            self.logger.warning(f"Could not extract score for {episode_id}, using default score 0.0")
            score = 0.0
        
        # Create metadata
        metadata = {
            "seed": seed,
            "episode_id": episode_id,
            "timestamp": datetime.now().isoformat(),
            "algo": self.config["algo"],
            "env": self.config["env"],
            "timesteps": self.config["timesteps_per_episode"]
        }
        
        return str(new_video_path), score, metadata

    def _extract_real_score(self, stdout: str) -> Optional[float]:
        """Extract the real score from the stdout of the recording script."""
        # Look for reward patterns in the output (prioritize specific patterns)
        reward_patterns = [
            r"Total reward:\s*([-+]?\d*\.?\d+)",  # "Total reward: 123.45" (highest priority)
            r"Final Board Score:\s*([-+]?\d*\.?\d+)",  # "Final Board Score: 123.45"
            r"Episode reward:\s*([-+]?\d*\.?\d+)",  # "Episode reward: 123.45"
            r"Final reward:\s*([-+]?\d*\.?\d+)",  # "Final reward: 123.45"
            r"Reward:\s*([-+]?\d*\.?\d+)",  # "Reward: 123.45" (lowest priority)
        ]
        
        # Find the first match (don't sum multiple matches as they're likely the same score)
        for line in stdout.splitlines():
            for pattern in reward_patterns:
                match = re.search(pattern, line, re.IGNORECASE)
                if match:
                    try:
                        reward = float(match.group(1))
                        self.logger.debug(f"Found reward: {reward} in line: {line.strip()}")
                        return reward  # Return the first match found
                    except (ValueError, IndexError):
                        continue
        
        # If no rewards found, try to extract from array format like [123.45]
        array_pattern = r"\[([-+]?\d*\.?\d+)\]"
        matches = re.findall(array_pattern, stdout)
        if matches:
            try:
                reward = float(matches[0])  # Take the first match, not sum
                self.logger.debug(f"Found reward in array format: {reward}")
                return reward
            except ValueError:
                pass
        
        return None
    
    
    def _save_trajectory_data(self, video_files: List[str], scores: List[float], 
                             metadata: List[Dict[str, Any]], output_dir: Path):
        """Save trajectory data in multiple formats."""
        
        # Save as CSV for HVA-X
        csv_path = output_dir / "trajectory_data.csv"
        with open(csv_path, 'w') as f:
            f.write("video_path,score,episode_id,metadata\n")
            for video, score, meta in zip(video_files, scores, metadata):
                f.write(f"{video},{score},{meta['episode_id']},{json.dumps(meta)}\n")
        
        # Save scores as simple text file
        scores_path = output_dir / "scores.txt"
        with open(scores_path, 'w') as f:
            for score in scores:
                f.write(f"{score}\n")
        
        # Save detailed metadata
        metadata_path = output_dir / "metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump({
                "videos": video_files,
                "scores": scores,
                "metadata": metadata,
                "config": self.config
            }, f, indent=2)
        
        self.logger.info(f"Trajectory data saved to {output_dir}")
    
    def _generate_sampling_summary(self, video_files: List[str], scores: List[float], 
                                  output_dir: Path) -> Dict[str, Any]:
        """Generate a summary of the sampling process."""
        
        if not scores:
            return {"error": "No episodes recorded successfully"}
        
        # Filter out any None scores (shouldn't happen now, but just in case)
        valid_scores = [s for s in scores if s is not None]
        
        if not valid_scores:
            return {"error": "No valid scores found"}
        
        # Calculate score statistics
        sorted_scores = sorted(valid_scores)
        n_scores = len(sorted_scores)
        
        summary = {
            "total_episodes": len(video_files),
            "output_directory": str(output_dir),
            "score_statistics": {
                "min": min(valid_scores),
                "max": max(valid_scores),
                "mean": sum(valid_scores) / len(valid_scores),
                "median": sorted_scores[n_scores // 2],
                "std": self._calculate_std(valid_scores)
            },
            "performance_tiers": {
                "low_tier_threshold": sorted_scores[int(n_scores * 0.1)],
                "mid_tier_range": [sorted_scores[int(n_scores * 0.45)], sorted_scores[int(n_scores * 0.55)]],
                "high_tier_threshold": sorted_scores[int(n_scores * 0.9)]
            },
            "files_created": {
                "trajectory_csv": str(output_dir / "trajectory_data.csv"),
                "scores_txt": str(output_dir / "scores.txt"),
                "metadata_json": str(output_dir / "metadata.json")
            }
        }
        
        return summary
    
    def _calculate_std(self, scores: List[float]) -> float:
        """Calculate standard deviation of scores."""
        if len(scores) <= 1:
            return 0.0
        
        mean = sum(scores) / len(scores)
        variance = sum((x - mean) ** 2 for x in scores) / (len(scores) - 1)
        return variance ** 0.5
    
    def quick_sample(self, num_episodes: int = 20) -> str:
        """Quick sampling for testing (fewer episodes)."""
        self.logger.info(f"Quick sampling: {num_episodes} episodes")
        
        output_dir = f"quick_hva_sample_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        summary = self.sample_episodes_for_hva(num_episodes, output_dir)
        
        return output_dir


def main():
    """Main function for command-line usage."""
    parser = argparse.ArgumentParser(description="Sample videos for HVA-X analysis")
    parser.add_argument("--episodes", type=int, default=100, 
                       help="Number of episodes to sample")
    parser.add_argument("--output-dir", type=str, default="hva_videos",
                       help="Output directory for videos")
    parser.add_argument("--python-path", type=str, 
                       help="Path to Python executable in conda environment")
    parser.add_argument("--quick", action="store_true",
                       help="Quick sampling with fewer episodes (20)")
    parser.add_argument("--verbose", action="store_true",
                       help="Enable verbose logging")
    
    args = parser.parse_args()
    
    # Configure logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # Create sampler
    sampler = HVAVideoSampler(python_path=args.python_path)
    
    try:
        if args.quick:
            output_dir = sampler.quick_sample()
            print(f"\n🎯 Quick sampling complete!")
            print(f"📁 Videos saved to: {output_dir}")
            print(f"🚀 Run HVA-X analysis with:")
            print(f"   python run_hva_analysis.py --video-dir {output_dir}")
        else:
            summary = sampler.sample_episodes_for_hva(args.episodes, args.output_dir)
            
            if "error" in summary:
                print(f"❌ Error: {summary['error']}")
                sys.exit(1)
            
            print(f"\n🎯 Video sampling complete!")
            print(f"📊 Episodes recorded: {summary['total_episodes']}")
            print(f"📁 Output directory: {summary['output_directory']}")
            print(f"🎮 Score range: {summary['score_statistics']['min']:.1f} - {summary['score_statistics']['max']:.1f}")
            print(f"📈 Mean score: {summary['score_statistics']['mean']:.1f}")
            print(f"\n📋 Files created:")
            for file_type, path in summary['files_created'].items():
                print(f"   {file_type}: {path}")
            
            print(f"\n🚀 Run HVA-X analysis with:")
            print(f"   python run_hva_analysis.py --video-dir {args.output_dir}")
            
    except Exception as e:
        print(f"❌ Error during sampling: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main() 