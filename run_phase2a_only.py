#!/usr/bin/env python3
"""
HVA-X Phase 2A Only Runner
Runs only Phase 2A (Event Detection) of the HVA-X algorithm.
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

from gemini_analysis import GeminiClient
from video_processing import VideoLoader, TrajectoryData


def run_phase2a_only(video_dir: Optional[str] = None, score_file: Optional[str] = None, 
                    csv_file: Optional[str] = None, phase1_file: Optional[str] = None,
                    samples_per_tier: int = 3, output_prefix: str = "phase2a_events", 
                    direct_mode: bool = False) -> Dict[str, Any]:
    """
    Run only Phase 2A (Event Detection) of HVA-X algorithm.
    
    Args:
        video_dir: Directory containing video files
        score_file: Path to scores.txt file
        csv_file: Path to trajectory_data.csv file
        phase1_file: Path to Phase 1 results JSON file
        samples_per_tier: Number of samples per performance tier (ignored in direct mode)
        output_prefix: Prefix for output files
        direct_mode: If True, process all videos directly without stratification
        
    Returns:
        Dictionary with Phase 2A results
    """
    
    if direct_mode:
        logging.info("🚀 Starting HVA-X Phase 2A: Event Detection (Direct Mode)")
    else:
        logging.info("🚀 Starting HVA-X Phase 2A: Event Detection")
    
    if video_dir:
        logging.info(f"📁 Video directory: {video_dir}")
    if csv_file:
        logging.info(f"📋 CSV file: {csv_file}")
    if phase1_file:
        logging.info(f"📄 Phase 1 results file: {phase1_file}")
    if not direct_mode:
        logging.info(f"📊 Samples per tier: {samples_per_tier}")
    
    # Initialize components
    video_loader = VideoLoader()
    
    try:
        # Load trajectory data
        if phase1_file:
            logging.info(f"📄 Loading sampled trajectories from Phase 1 results: {phase1_file}")
            with open(phase1_file, 'r') as f:
                phase1_results = json.load(f)
            
            # Check if Phase 1 was run in direct mode
            phase1_direct_mode = phase1_results.get("input_data", {}).get("direct_mode", False)
            
            # Extract sampled trajectories from Phase 1 results
            sampled_trajectories = {}
            for tier_name, trajs_data in phase1_results["sampled_trajectories"].items():
                sampled_trajectories[tier_name] = [
                    TrajectoryData(
                        video_path=traj["video_path"],
                        score=traj.get("score", 0.0),  # Handle None scores from direct mode
                        episode_id=traj["episode_id"]
                    ) for traj in trajs_data
                ]
            
            logging.info(f"📊 Loaded {sum(len(trajs) for trajs in sampled_trajectories.values())} trajectories from Phase 1")
            
        elif direct_mode:
            # Direct mode: Load all videos from directory without scores/stratification
            if not video_dir:
                raise ValueError("Direct mode requires --video-dir")
            
            logging.info(f"📋 Loading all videos from directory in direct mode")
            video_files = video_loader.find_videos(video_dir)
            
            if not video_files:
                raise FileNotFoundError(f"No video files found in directory: {video_dir}")
            
            # Create trajectory data with dummy scores (not used in direct mode)
            trajectories = []
            for i, video_path in enumerate(video_files):
                # Extract episode ID from filename
                episode_id = Path(video_path).stem
                trajectories.append(TrajectoryData(
                    video_path=video_path,
                    score=0.0,  # Dummy score for direct mode
                    episode_id=episode_id
                ))
            
            logging.info(f"📊 Loaded {len(trajectories)} videos in direct mode")
            
            # In direct mode, put all trajectories in a single "all_videos" tier
            sampled_trajectories = {
                "all_videos": trajectories
            }
            
        else:
            # Load and sample trajectories if no Phase 1 file provided
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
                        "or ensure trajectory_data.csv or scores.txt exists in the video directory, "
                        "or use --direct-mode to process all videos without scores."
                    )
            else:
                raise ValueError("Must provide either --video-dir, --csv-file, --phase1-file, or use --direct-mode")
            
            logging.info(f"📊 Loaded {len(trajectories)} total trajectories")
            
            # Run Phase 1: Trajectory Sampling and Stratification
            logging.info("🔄 Running Phase 1: Trajectory Sampling and Stratification")
            sampled_trajectories = video_loader.prepare_trajectories_for_hva(
                trajectories, samples_per_tier
            )
        
        # Run Phase 2A: Event Detection
        if direct_mode:
            logging.info("🔄 Running Phase 2A: Event Detection (Direct Mode)")
        else:
            logging.info("🔄 Running Phase 2A: Event Detection")
        
        with GeminiClient() as gemini_client:
            event_detection_results = {}
            
            for tier_name, trajs in sampled_trajectories.items():
                tier_label = "videos" if direct_mode else "tier"
                logging.info(f"  Processing {tier_name} {tier_label} ({len(trajs)} videos)")
                tier_events = []
                
                for i, trajectory in enumerate(trajs):
                    logging.info(f"    Video {i+1}/{len(trajs)}: {trajectory.episode_id}")
                    
                    try:
                        # Detect key events in the video
                        key_events = gemini_client.detect_key_events(trajectory.video_path)
                        
                        # Store event detection result
                        event_result = {
                            "trajectory": {
                                "episode_id": trajectory.episode_id,
                                "video_path": trajectory.video_path,
                                "score": trajectory.score if not direct_mode else None,
                                "tier": tier_name
                            },
                            "key_events": key_events,
                            "event_count": len(key_events),
                            "timestamp": datetime.now().isoformat()
                        }
                        
                        tier_events.append(event_result)
                        logging.info(f"      Detected {len(key_events)} key events")
                        
                    except Exception as e:
                        logging.error(f"      Failed to detect events for {trajectory.episode_id}: {e}")
                        # Store error result
                        error_result = {
                            "trajectory": {
                                "episode_id": trajectory.episode_id,
                                "video_path": trajectory.video_path,
                                "score": trajectory.score if not direct_mode else None,
                                "tier": tier_name
                            },
                            "error": str(e),
                            "timestamp": datetime.now().isoformat()
                        }
                        tier_events.append(error_result)
                
                event_detection_results[tier_name] = tier_events
        
        # Calculate statistics
        total_processed = sum(len(events) for events in event_detection_results.values())
        successful_detections = sum(
            len([e for e in events if "key_events" in e]) 
            for events in event_detection_results.values()
        )
        failed_detections = total_processed - successful_detections
        
        # Compile results
        results = {
            "algorithm": "HVA-X",
            "phase": "Phase 2A - Event Detection (Direct Mode)" if direct_mode else "Phase 2A - Event Detection",
            "timestamp": datetime.now().isoformat(),
            "input_data": {
                "total_trajectories_processed": total_processed,
                "samples_per_tier": samples_per_tier if not direct_mode else None,
                "source": "direct_mode" if direct_mode else ("phase1_file" if phase1_file else ("csv" if csv_file else "directory")),
                "direct_mode": direct_mode
            },
            "detection_results": {
                "successful_detections": successful_detections,
                "failed_detections": failed_detections,
                "success_rate": successful_detections / total_processed if total_processed > 0 else 0
            },
            "event_detection_results": event_detection_results
        }
        
        logging.info("✅ Phase 2A completed successfully!")
        return results
        
    except Exception as e:
        logging.error(f"❌ Phase 2A failed: {e}")
        raise


def save_phase2a_results(results: Dict[str, Any], output_file: str):
    """
    Save Phase 2A results to JSON file.
    
    Args:
        results: Phase 2A results dictionary
        output_file: Path to output JSON file
    """
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    logging.info(f"📄 Phase 2A results saved to: {output_file}")


def print_phase2a_summary(results: Dict[str, Any]):
    """
    Print a summary of Phase 2A results.
    
    Args:
        results: Phase 2A results dictionary
    """
    direct_mode = results['input_data'].get('direct_mode', False)
    
    print("\n" + "="*60)
    if direct_mode:
        print("🎯 HVA-X PHASE 2A COMPLETE (DIRECT MODE)")
    else:
        print("🎯 HVA-X PHASE 2A COMPLETE")
    print("="*60)
    
    print(f"📊 Total trajectories processed: {results['input_data']['total_trajectories_processed']}")
    print(f"✅ Successful event detections: {results['detection_results']['successful_detections']}")
    print(f"❌ Failed event detections: {results['detection_results']['failed_detections']}")
    print(f"📈 Success rate: {results['detection_results']['success_rate']:.1%}")
    print(f"⏱️  Processing timestamp: {results['timestamp']}")
    
    if direct_mode:
        print(f"\n🔍 Event Detection (Direct Mode):")
        for tier_name, tier_events in results['event_detection_results'].items():
            successful = [e for e in tier_events if "key_events" in e]
            failed = [e for e in tier_events if "error" in e]
            
            print(f"   {tier_name.replace('_', ' ').title()}: {len(tier_events)} videos")
            print(f"     ✅ Successful: {len(successful)}")
            print(f"     ❌ Failed: {len(failed)}")
            
            if successful:
                total_events = sum(len(e["key_events"]) for e in successful)
                avg_events = total_events / len(successful)
                print(f"     📊 Total events detected: {total_events}")
                print(f"     📊 Average events per video: {avg_events:.1f}")
                
                # Show sample events from first successful video
                sample_events = successful[0]["key_events"][:3]  # First 3 events
                print(f"     📋 Sample events from {successful[0]['trajectory']['episode_id']}:")
                for event in sample_events:
                    print(f"       - {event['timestamp']}: {event['event']}")
    else:
        print(f"\n🔍 Event Detection by Tier:")
        for tier_name, tier_events in results['event_detection_results'].items():
            successful = [e for e in tier_events if "key_events" in e]
            failed = [e for e in tier_events if "error" in e]
            
            print(f"   {tier_name.replace('_', ' ').title()}: {len(tier_events)} videos")
            print(f"     ✅ Successful: {len(successful)}")
            print(f"     ❌ Failed: {len(failed)}")
            
            if successful:
                total_events = sum(len(e["key_events"]) for e in successful)
                avg_events = total_events / len(successful)
                print(f"     📊 Total events detected: {total_events}")
                print(f"     📊 Average events per video: {avg_events:.1f}")
                
                # Show sample events from first successful video
                sample_events = successful[0]["key_events"][:3]  # First 3 events
                print(f"     📋 Sample events from {successful[0]['trajectory']['episode_id']}:")
                for event in sample_events:
                    print(f"       - {event['timestamp']}: {event['event']}")
    
    print(f"\n📋 Next Steps:")
    print(f"   - Use detected events for Phase 2B (Guided Analysis)")
    print(f"   - Run full HVA-X analysis with: python run_hva_analysis.py")
    print(f"   - Or continue with manual analysis of detected events")


def main():
    """Main function for command-line usage."""
    parser = argparse.ArgumentParser(
        description="Run HVA-X Phase 2A (Event Detection) only",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Detect events from Phase 1 results
  python run_phase2a_only.py --phase1-file phase1_sampling_20240101_120000.json

  # Detect events with auto-detected files
  python run_phase2a_only.py --video-dir hva_videos

  # Detect events with specific score file
  python run_phase2a_only.py --video-dir videos/ --score-file scores.txt

  # Detect events from CSV file
  python run_phase2a_only.py --csv-file trajectory_data.csv

  # Process more trajectories per tier
  python run_phase2a_only.py --video-dir hva_videos --samples-per-tier 5

  # Process all videos directly without stratification
  python run_phase2a_only.py --video-dir video_clips_30s --direct-mode

  # Save output to specific directory
  python run_phase2a_only.py --video-dir hva_videos --output-dir results/phase2a
        """
    )
    
    parser.add_argument("--video-dir", type=str,
                       help="Directory containing video files")
    parser.add_argument("--score-file", type=str,
                       help="Path to scores.txt file")
    parser.add_argument("--csv-file", type=str,
                       help="Path to trajectory_data.csv file")
    parser.add_argument("--phase1-file", type=str,
                       help="Path to Phase 1 results JSON file")
    parser.add_argument("--samples-per-tier", type=int, default=3,
                       help="Number of samples per performance tier (default: 3)")
    parser.add_argument("--output-prefix", type=str, default="phase2a_events",
                       help="Prefix for output files (default: phase2a_events)")
    parser.add_argument("--output-dir", type=str, default=".",
                       help="Directory to save output files (default: current directory)")
    parser.add_argument("--verbose", action="store_true",
                       help="Enable verbose logging")
    parser.add_argument("--direct-mode", action="store_true",
                       help="Run in direct mode, processing all videos from --video-dir without stratification")
    
    args = parser.parse_args()
    
    # Configure logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # Validate arguments
    if args.direct_mode:
        if not args.video_dir:
            parser.error("Direct mode requires --video-dir")
    else:
        if not args.phase1_file and not args.csv_file and not args.video_dir:
            parser.error("Must provide either --phase1-file, --video-dir, or --csv-file")
    
    try:
        # Run Phase 2A
        results = run_phase2a_only(
            video_dir=args.video_dir,
            score_file=args.score_file,
            csv_file=args.csv_file,
            phase1_file=args.phase1_file,
            samples_per_tier=args.samples_per_tier,
            output_prefix=args.output_prefix,
            direct_mode=args.direct_mode
        )
        
        # Generate output filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = Path(args.output_dir) / f"{args.output_prefix}_{timestamp}.json"
        
        # Save results
        save_phase2a_results(results, str(output_file))
        
        # Print summary
        print_phase2a_summary(results)
        
        print(f"\n🎉 Phase 2A completed successfully!")
        print(f"📄 Detailed results saved to: {output_file}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main() 