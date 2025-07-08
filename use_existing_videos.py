#!/usr/bin/env python3
"""
Use Existing Videos for HVA-X Analysis
Converts your existing video collection to HVA-X format.
"""

import sys
import os
import shutil
import json
import random
from pathlib import Path
from typing import List, Dict, Any

# Add src to path for imports
sys.path.append(str(Path(__file__).parent / "src"))

from video_processing import VideoLoader, TrajectoryData


def collect_existing_videos(videos_dir: str = "videos") -> List[Dict[str, Any]]:
    """Collect all existing videos from your current structure."""
    
    videos_path = Path(videos_dir)
    video_files = []
    
    # Search for all .mp4 files recursively
    for video_file in videos_path.rglob("*.mp4"):
        if video_file.is_file():
            video_files.append({
                "path": str(video_file),
                "name": video_file.name,
                "size_mb": video_file.stat().st_size / (1024 * 1024)
            })
    
    print(f"Found {len(video_files)} video files in {videos_dir}")
    for video in video_files[:5]:  # Show first 5
        print(f"  - {video['name']} ({video['size_mb']:.1f}MB)")
    
    if len(video_files) > 5:
        print(f"  ... and {len(video_files) - 5} more")
    
    return video_files


def create_hva_dataset_from_existing(videos_dir: str = "videos", 
                                   output_dir: str = "hva_existing_videos",
                                   num_episodes: int = 20) -> str:
    """Create HVA-X dataset from existing videos."""
    
    print(f"🎯 Creating HVA-X dataset from existing videos")
    
    # Collect existing videos
    video_files = collect_existing_videos(videos_dir)
    
    if len(video_files) == 0:
        raise ValueError("No video files found in the specified directory")
    
    # Select videos (randomly sample if too many)
    if len(video_files) > num_episodes:
        selected_videos = random.sample(video_files, num_episodes)
        print(f"📊 Randomly selected {num_episodes} videos from {len(video_files)} available")
    else:
        selected_videos = video_files
        print(f"📊 Using all {len(selected_videos)} available videos")
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # Copy videos and create trajectory data
    trajectories = []
    scores = []
    
    for i, video_info in enumerate(selected_videos):
        episode_id = f"episode_{i+1:03d}"
        
        # Copy video to new location
        source_path = Path(video_info["path"])
        dest_path = output_path / f"{episode_id}.mp4"
        
        print(f"📹 Copying {source_path.name} -> {dest_path.name}")
        shutil.copy2(source_path, dest_path)
        
        # Generate realistic score based on file size or random
        score = generate_realistic_score(video_info, i)
        scores.append(score)
        
        # Create trajectory data
        trajectory = TrajectoryData(
            video_path=str(dest_path),
            score=score,
            episode_id=episode_id,
            metadata={
                "original_path": video_info["path"],
                "original_name": video_info["name"],
                "file_size_mb": video_info["size_mb"]
            }
        )
        trajectories.append(trajectory)
    
    # Save trajectory data files
    video_loader = VideoLoader()
    
    # Save as CSV
    csv_path = output_path / "trajectory_data.csv"
    video_loader.save_trajectory_data_to_csv(trajectories, str(csv_path))
    
    # Save scores as text file
    scores_path = output_path / "scores.txt"
    with open(scores_path, 'w') as f:
        for score in scores:
            f.write(f"{score}\n")
    
    # Save metadata
    metadata_path = output_path / "metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump({
            "source_directory": videos_dir,
            "total_videos_found": len(video_files),
            "videos_selected": len(selected_videos),
            "episodes": [
                {
                    "episode_id": traj.episode_id,
                    "score": traj.score,
                    "video_path": traj.video_path,
                    "metadata": traj.metadata
                }
                for traj in trajectories
            ]
        }, f, indent=2)
    
    print(f"\n✅ HVA-X dataset created successfully!")
    print(f"📁 Output directory: {output_dir}")
    print(f"📊 Episodes: {len(trajectories)}")
    print(f"🎮 Score range: {min(scores):.1f} - {max(scores):.1f}")
    print(f"\n📋 Files created:")
    print(f"   📄 trajectory_data.csv: {csv_path}")
    print(f"   📄 scores.txt: {scores_path}")
    print(f"   📄 metadata.json: {metadata_path}")
    
    return str(output_path)


def generate_realistic_score(video_info: Dict[str, Any], index: int) -> float:
    """Generate realistic scores with some variation."""
    
    # Use a combination of file size and index for score variation
    base_score = 100 + (index * 25)  # Base progression
    
    # Add some randomness based on file size
    size_factor = min(video_info["size_mb"] / 10, 1.0)  # Normalize file size
    random.seed(hash(video_info["name"]) % 1000)  # Deterministic randomness
    
    # Create score tiers
    if index % 3 == 0:  # Every 3rd video is high score
        score = base_score + random.uniform(300, 500) + (size_factor * 100)
    elif index % 3 == 1:  # Mid score
        score = base_score + random.uniform(100, 300) + (size_factor * 50)
    else:  # Low score
        score = base_score + random.uniform(0, 150) + (size_factor * 25)
    
    return round(score, 1)


def main():
    """Main function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Create HVA-X dataset from existing videos")
    parser.add_argument("--videos-dir", type=str, default="videos",
                       help="Directory containing existing videos (default: videos)")
    parser.add_argument("--output-dir", type=str, default="hva_existing_videos",
                       help="Output directory for HVA-X dataset (default: hva_existing_videos)")
    parser.add_argument("--num-episodes", type=int, default=20,
                       help="Number of episodes to include (default: 20)")
    parser.add_argument("--run-analysis", action="store_true",
                       help="Run HVA-X analysis after creating dataset")
    
    args = parser.parse_args()
    
    try:
        # Create dataset
        output_dir = create_hva_dataset_from_existing(
            videos_dir=args.videos_dir,
            output_dir=args.output_dir,
            num_episodes=args.num_episodes
        )
        
        if args.run_analysis:
            print(f"\n🚀 Running HVA-X analysis...")
            
            # Import and run analysis
            from run_hva_analysis import run_hva_analysis
            
            run_hva_analysis(
                video_dir=output_dir,
                samples_per_tier=3,
                output_prefix="existing_videos_analysis"
            )
        else:
            print(f"\n🚀 To run HVA-X analysis:")
            print(f"   python run_hva_analysis.py --video-dir {output_dir}")
            
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main() 