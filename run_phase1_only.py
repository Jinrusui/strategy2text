#!/usr/bin/env python3
"""
HVA-X Phase 1 Only Runner
Runs only Phase 1 (Trajectory Sampling and Stratification) of the HVA-X algorithm.
"""

import sys
import argparse
import logging
import json
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any, List

# Add src to path for imports
sys.path.append(str(Path(__file__).parent / "src"))

from video_processing import VideoLoader, TrajectoryData


def run_phase1_only(video_dir: Optional[str] = None, score_file: Optional[str] = None, 
                   csv_file: Optional[str] = None, samples_per_tier: int = 3, 
                   output_prefix: str = "phase1_sampling") -> Dict[str, Any]:
    """
    Run only Phase 1 (Trajectory Sampling and Stratification) of HVA-X algorithm.
    
    Args:
        video_dir: Directory containing video files
        score_file: Path to scores.txt file
        csv_file: Path to trajectory_data.csv file
        samples_per_tier: Number of samples per performance tier
        output_prefix: Prefix for output files
        
    Returns:
        Dictionary with Phase 1 results
    """
    
    logging.info("🚀 Starting HVA-X Phase 1: Trajectory Sampling and Stratification")
    if video_dir:
        logging.info(f"📁 Video directory: {video_dir}")
    if csv_file:
        logging.info(f"📋 CSV file: {csv_file}")
    logging.info(f"📊 Samples per tier: {samples_per_tier}")
    
    # Initialize video loader
    video_loader = VideoLoader()
    
    try:
        # Load trajectory data
        if csv_file:
            logging.info(f"📋 Loading trajectory data from CSV: {csv_file}")
            trajectories = video_loader.load_trajectory_data_from_csv(csv_file)
        elif score_file and video_dir:
            logging.info(f"📋 Loading trajectory data from directory and score file")
            trajectories = video_loader.load_trajectory_data_from_files(video_dir, score_file)
        elif video_dir:
            # Try to find score files automatically
            video_path = Path(video_dir)
            csv_path = video_path / "trajectory_data.csv"
            scores_path = video_path / "scores.txt"
            
            if csv_path.exists():
                logging.info(f"📋 Found CSV file: {csv_path}")
                trajectories = video_loader.load_trajectory_data_from_csv(str(csv_path))
            elif scores_path.exists():
                logging.info(f"📋 Found scores file: {scores_path}")
                trajectories = video_loader.load_trajectory_data_from_files(video_dir, str(scores_path))
            else:
                raise FileNotFoundError(
                    "No score file found. Please provide --score-file or --csv-file, "
                    "or ensure trajectory_data.csv or scores.txt exists in the video directory."
                )
        else:
            raise ValueError("Must provide either --video-dir or --csv-file")
        
        logging.info(f"📊 Loaded {len(trajectories)} total trajectories")
        
        # Run Phase 1: Trajectory Sampling and Stratification
        logging.info("🔄 Running Phase 1: Trajectory Sampling and Stratification")
        sampled_trajectories = video_loader.prepare_trajectories_for_hva(
            trajectories, samples_per_tier
        )
        
        # Calculate statistics
        total_sampled = sum(len(trajs) for trajs in sampled_trajectories.values())
        tier_stats = {}
        
        for tier_name, trajs in sampled_trajectories.items():
            if trajs:
                scores = [t.score for t in trajs]
                tier_stats[tier_name] = {
                    "count": len(trajs),
                    "min_score": min(scores),
                    "max_score": max(scores),
                    "avg_score": sum(scores) / len(scores),
                    "trajectories": [
                        {
                            "episode_id": t.episode_id,
                            "video_path": t.video_path,
                            "score": t.score
                        } for t in trajs
                    ]
                }
            else:
                tier_stats[tier_name] = {
                    "count": 0,
                    "trajectories": []
                }
        
        # Compile results
        results = {
            "algorithm": "HVA-X",
            "phase": "Phase 1 - Trajectory Sampling and Stratification",
            "timestamp": datetime.now().isoformat(),
            "input_data": {
                "total_trajectories": len(trajectories),
                "samples_per_tier": samples_per_tier,
                "source": "csv" if csv_file else "directory"
            },
            "sampling_results": {
                "total_sampled": total_sampled,
                "tier_statistics": tier_stats
            },
            "sampled_trajectories": sampled_trajectories
        }
        
        logging.info("✅ Phase 1 completed successfully!")
        return results
        
    except Exception as e:
        logging.error(f"❌ Phase 1 failed: {e}")
        raise


def save_phase1_results(results: Dict[str, Any], output_file: str):
    """
    Save Phase 1 results to JSON file.
    
    Args:
        results: Phase 1 results dictionary
        output_file: Path to output JSON file
    """
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Convert TrajectoryData objects to dictionaries for JSON serialization
    json_results = results.copy()
    
    # Convert sampled_trajectories to serializable format
    serializable_trajectories = {}
    for tier_name, trajs in results["sampled_trajectories"].items():
        serializable_trajectories[tier_name] = [t.to_dict() for t in trajs]
    
    json_results["sampled_trajectories"] = serializable_trajectories
    
    with open(output_path, 'w') as f:
        json.dump(json_results, f, indent=2, default=str)
    
    logging.info(f"📄 Phase 1 results saved to: {output_file}")


def print_phase1_summary(results: Dict[str, Any]):
    """
    Print a summary of Phase 1 results.
    
    Args:
        results: Phase 1 results dictionary
    """
    print("\n" + "="*60)
    print("🎯 HVA-X PHASE 1 COMPLETE")
    print("="*60)
    
    print(f"📊 Total trajectories loaded: {results['input_data']['total_trajectories']}")
    print(f"📈 Total trajectories sampled: {results['sampling_results']['total_sampled']}")
    print(f"⏱️  Sampling timestamp: {results['timestamp']}")
    
    print(f"\n🔍 Tier Breakdown:")
    for tier_name, stats in results['sampling_results']['tier_statistics'].items():
        print(f"   {tier_name.replace('_', ' ').title()}: {stats['count']} trajectories")
        if stats['count'] > 0:
            print(f"     Score range: {stats['min_score']:.2f} - {stats['max_score']:.2f}")
            print(f"     Average score: {stats['avg_score']:.2f}")
            print(f"     Episodes: {', '.join([t['episode_id'] for t in stats['trajectories']])}")
    
    print(f"\n📋 Next Steps:")
    print(f"   - Use sampled trajectories for Phase 2 (Individual Analysis)")
    print(f"   - Run full HVA-X analysis with: python run_hva_analysis.py")
    print(f"   - Or continue with manual analysis of selected episodes")


def main():
    """Main function for command-line usage."""
    parser = argparse.ArgumentParser(
        description="Run HVA-X Phase 1 (Trajectory Sampling and Stratification) only",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Sample trajectories with auto-detected files
  python run_phase1_only.py --video-dir hva_videos

  # Sample with specific score file
  python run_phase1_only.py --video-dir videos/ --score-file scores.txt

  # Sample from CSV file
  python run_phase1_only.py --csv-file trajectory_data.csv

  # Sample more trajectories per tier
  python run_phase1_only.py --video-dir hva_videos --samples-per-tier 5
        """
    )
    
    parser.add_argument("--video-dir", type=str, default="hva_videos",
                       help="Directory containing video files")
    parser.add_argument("--score-file", type=str,
                       help="Path to scores.txt file")
    parser.add_argument("--csv-file", type=str,
                       help="Path to trajectory_data.csv file")
    parser.add_argument("--samples-per-tier", type=int, default=3,
                       help="Number of samples per performance tier (default: 3)")
    parser.add_argument("--output-prefix", type=str, default="phase1_sampling",
                       help="Prefix for output files (default: phase1_sampling)")
    parser.add_argument("--verbose", action="store_true",
                       help="Enable verbose logging")
    
    args = parser.parse_args()
    
    # Configure logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # Validate arguments
    if not args.csv_file and not args.video_dir:
        parser.error("Must provide either --video-dir or --csv-file")
    
    try:
        # Run Phase 1
        results = run_phase1_only(
            video_dir=args.video_dir,
            score_file=args.score_file,
            csv_file=args.csv_file,
            samples_per_tier=args.samples_per_tier,
            output_prefix=args.output_prefix
        )
        
        # Generate output filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"{args.output_prefix}_{timestamp}.json"
        
        # Save results
        save_phase1_results(results, output_file)
        
        # Print summary
        print_phase1_summary(results)
        
        print(f"\n🎉 Phase 1 completed successfully!")
        print(f"📄 Detailed results saved to: {output_file}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main() 