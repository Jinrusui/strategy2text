"""
Example usage of the HVA-X (Hierarchical Video Analysis for Agent Explainability) system.
Demonstrates how to analyze RL agent behavior using the complete HVA-X algorithm.
"""

import logging
from pathlib import Path
from gemini_analysis import HVAAnalyzer
from video_processing import VideoLoader, TrajectoryData


def example_hva_analysis():
    """Example of running complete HVA-X analysis."""
    
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    # Example 1: Analyze from directory with video files and score file
    logger.info("=== HVA-X Analysis Example 1: From Directory ===")
    
    video_dir = "path/to/agent/videos"  # Directory containing .mp4 files
    score_file = "path/to/scores.txt"   # Text file with scores (one per line)
    
    # Run HVA-X analysis
    try:
        with HVAAnalyzer() as analyzer:
            results = analyzer.analyze_agent_from_directory(
                video_dir=video_dir,
                score_file=score_file,
                samples_per_tier=3  # Sample 3 videos from each performance tier
            )
            
            # Save complete results
            analyzer.save_analysis(results, "hva_analysis_results.json")
            
            # Save final report (human-readable)
            analyzer.save_final_report(results, "hva_final_report.md")
            
            logger.info("Analysis completed successfully!")
            logger.info(f"Total videos analyzed: {results['summary']['total_videos_analyzed']}")
            
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
    
    # Example 2: Analyze from CSV file
    logger.info("\n=== HVA-X Analysis Example 2: From CSV ===")
    
    csv_file = "path/to/trajectory_data.csv"  # CSV with columns: video_path, score, episode_id
    
    try:
        with HVAAnalyzer() as analyzer:
            results = analyzer.analyze_agent_from_csv(
                csv_file=csv_file,
                samples_per_tier=5  # Sample 5 videos from each performance tier
            )
            
            # Save results with different names
            analyzer.save_analysis(results, "hva_csv_analysis.json")
            analyzer.save_final_report(results, "hva_csv_report.md")
            
            logger.info("CSV analysis completed successfully!")
            
    except Exception as e:
        logger.error(f"CSV analysis failed: {e}")


def example_trajectory_processing():
    """Example of trajectory processing and sampling."""
    
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    logger.info("=== Trajectory Processing Example ===")
    
    # Create some example trajectory data
    video_paths = [f"video_{i}.mp4" for i in range(100)]
    scores = [i * 10 + (i % 20) for i in range(100)]  # Example scores
    
    # Create trajectory data
    video_loader = VideoLoader()
    trajectories = video_loader.trajectory_processor.create_trajectory_data(
        video_paths, scores
    )
    
    # Run trajectory sampling (Phase 1 of HVA-X)
    sampled_trajectories = video_loader.prepare_trajectories_for_hva(
        trajectories, samples_per_tier=3
    )
    
    # Display sampling results
    for tier, trajs in sampled_trajectories.items():
        logger.info(f"{tier}: {len(trajs)} trajectories")
        for traj in trajs:
            logger.info(f"  - {traj.episode_id}: score={traj.score}")
    
    # Save trajectory data to CSV
    video_loader.save_trajectory_data_to_csv(trajectories, "trajectory_data.csv")
    logger.info("Trajectory data saved to CSV")


def example_custom_analysis():
    """Example of customizing the analysis for specific needs."""
    
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    logger.info("=== Custom HVA-X Analysis Example ===")
    
    # Create custom trajectory data
    trajectories = [
        TrajectoryData("high_score_video1.mp4", 850, "episode_001"),
        TrajectoryData("high_score_video2.mp4", 820, "episode_002"),
        TrajectoryData("mid_score_video1.mp4", 450, "episode_003"),
        TrajectoryData("mid_score_video2.mp4", 420, "episode_004"),
        TrajectoryData("low_score_video1.mp4", 120, "episode_005"),
        TrajectoryData("low_score_video2.mp4", 100, "episode_006"),
    ]
    
    # Run analysis with custom parameters
    try:
        with HVAAnalyzer(model="gemini-2.5-pro-preview-06-05") as analyzer:
            results = analyzer.analyze_agent_from_trajectories(
                trajectories=trajectories,
                samples_per_tier=2  # Use 2 samples per tier
            )
            
            # Process results
            logger.info("Custom analysis completed!")
            logger.info(f"Algorithm: {results['algorithm']}")
            logger.info(f"Timestamp: {results['timestamp']}")
            logger.info(f"Sampling results: {results['phase1_sampling']}")
            
            # Save with custom naming
            analyzer.save_analysis(results, "custom_hva_analysis.json")
            analyzer.save_final_report(results, "custom_hva_report.md")
            
    except Exception as e:
        logger.error(f"Custom analysis failed: {e}")


if __name__ == "__main__":
    print("HVA-X Analysis Examples")
    print("=" * 50)
    
    # Note: Before running these examples, make sure you have:
    # 1. Set up your Google API key (GEMINI_API_KEY environment variable)
    # 2. Have actual video files and score data
    # 3. Installed the required dependencies
    
    print("\n1. Complete HVA-X Analysis Example")
    print("   (Uncomment the line below to run)")
    # example_hva_analysis()
    
    print("\n2. Trajectory Processing Example")
    print("   (Uncomment the line below to run)")
    # example_trajectory_processing()
    
    print("\n3. Custom Analysis Example")
    print("   (Uncomment the line below to run)")
    # example_custom_analysis()
    
    print("\nTo run examples, uncomment the function calls above and ensure you have:")
    print("- Video files in the specified directories")
    print("- Score files with episode scores")
    print("- GEMINI_API_KEY environment variable set")
    print("- All required dependencies installed") 