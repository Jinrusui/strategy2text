#!/usr/bin/env python3
"""
Example script demonstrating how to use the Gemini agent to analyze Breakout RL agent videos.
"""

import os
import sys
import logging
from pathlib import Path

# Add src to path so we can import our modules
sys.path.insert(0, str(Path(__file__).parent / "src"))

from gemini_analysis import BreakoutStrategyAnalyzer
from video_processing import VideoLoader


def setup_logging():
    """Set up logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )


def analyze_single_video(video_path: str):
    """
    Analyze a single Breakout video file.
    
    Args:
        video_path: Path to the video file
    """
    print(f"\n=== Analyzing Single Breakout Video ===")
    print(f"Video: {video_path}")
    
    try:
        with BreakoutStrategyAnalyzer() as analyzer:
            analysis = analyzer.analyze_breakout_strategy(video_path)
            
            if "error" in analysis:
                print(f"❌ Analysis failed: {analysis['error']}")
                return
            
            print(f"✅ Analysis completed successfully!")
            print(f"\n📊 Strategy Summary:")
            
            strategy = analysis["strategy_summary"]
            for section, content in strategy.items():
                if content:
                    print(f"\n{section.replace('_', ' ').title()}:")
                    print(f"  {content[:200]}..." if len(content) > 200 else f"  {content}")
            
            # Save analysis
            output_file = f"analysis_{Path(video_path).stem}.json"
            analyzer.save_analysis(analysis, output_file)
            print(f"\n💾 Full analysis saved to: {output_file}")
            
    except Exception as e:
        print(f"❌ Error during analysis: {str(e)}")


def analyze_multiple_videos(video_directory: str):
    """
    Analyze multiple Breakout videos in a directory.
    
    Args:
        video_directory: Directory containing video files
    """
    print(f"\n=== Analyzing Multiple Breakout Videos ===")
    print(f"Directory: {video_directory}")
    
    # Find videos
    loader = VideoLoader()
    video_paths = loader.find_videos(video_directory)
    
    # Filter for Breakout videos
    breakout_videos = loader.filter_by_pattern(video_paths, "breakout")
    
    if not breakout_videos:
        print("❌ No Breakout video files found in directory")
        return
    
    print(f"📹 Found {len(breakout_videos)} Breakout video files")
    for path in breakout_videos:
        print(f"  - {Path(path).name}")
    
    try:
        with BreakoutStrategyAnalyzer() as analyzer:
            comparison = analyzer.compare_breakout_strategies(breakout_videos)
            
            print(f"\n✅ Comparison analysis completed!")
            
            # Show individual results
            print(f"\n📊 Individual Analyses:")
            for i, analysis in enumerate(comparison["individual_analyses"]):
                video_name = Path(analysis["video_path"]).name
                print(f"\n{i+1}. {video_name}")
                if "error" in analysis:
                    print(f"   ❌ Error: {analysis['error']}")
                else:
                    strategy = analysis["strategy_summary"]["strategy_description"]
                    preview = strategy[:150] + "..." if len(strategy) > 150 else strategy
                    print(f"   📝 {preview}")
            
            # Show comparison
            if "comparison_analysis" in comparison:
                print(f"\n🔍 Comparative Analysis:")
                comp_text = comparison["comparison_analysis"]
                preview = comp_text[:300] + "..." if len(comp_text) > 300 else comp_text
                print(f"   {preview}")
            
            # Save comparison
            output_file = f"comparison_breakout.json"
            analyzer.save_analysis(comparison, output_file)
            print(f"\n💾 Full comparison saved to: {output_file}")
            
    except Exception as e:
        print(f"❌ Error during analysis: {str(e)}")


def main():
    """Main function with example usage."""
    setup_logging()
    
    print("🎮 Gemini Breakout RL Agent Video Analyzer")
    print("=" * 45)
    
    # Check for API key
    if not os.getenv("GOOGLE_API_KEY"):
        print("❌ Error: GOOGLE_API_KEY environment variable not set")
        print("Please set your Google API key:")
        print("export GOOGLE_API_KEY='your-api-key-here'")
        return
    
    # Example usage - modify these paths for your use case
    
    # Option 1: Analyze a single video
    single_video_path = "videos/breakout_ppo_episode.mp4"  # Replace with your video path
    if Path(single_video_path).exists():
        analyze_single_video(single_video_path)
    else:
        print(f"⚠️  Example video not found: {single_video_path}")
    
    # Option 2: Analyze multiple videos in a directory
    video_directory = "videos"  # Replace with your video directory
    if Path(video_directory).exists():
        analyze_multiple_videos(video_directory)
    else:
        print(f"⚠️  Example directory not found: {video_directory}")
    
    # Option 3: Direct usage example
    print(f"\n=== Direct Usage Example ===")
    print("Here's how to use the components directly:")
    print("""
# Basic usage
from src.gemini_analysis import BreakoutStrategyAnalyzer

# Initialize analyzer
analyzer = BreakoutStrategyAnalyzer()

# Analyze a single Breakout video
analysis = analyzer.analyze_breakout_strategy("path/to/breakout_video.mp4")

# Print results
print(analysis["raw_analysis"])

# Compare multiple Breakout videos
comparison = analyzer.compare_breakout_strategies(
    ["breakout_video1.mp4", "breakout_video2.mp4"]
)
""")


if __name__ == "__main__":
    main() 