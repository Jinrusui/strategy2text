#!/usr/bin/env python3
"""
Analyze the Breakout PPO videos using Gemini AI.
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


def main():
    """Analyze the Breakout videos."""
    setup_logging()
    
    print("🎮 Analyzing Breakout PPO Agent Videos")
    print("=" * 50)
    
    # Set API key from file
    try:
        with open("Gemini_API_KEY.txt", "r") as f:
            api_key = f.read().strip()
        os.environ["GOOGLE_API_KEY"] = api_key
        print("✅ API key loaded successfully")
    except Exception as e:
        print(f"❌ Failed to load API key: {e}")
        return
    
    # Find the Breakout videos
    video_paths = [
        "videos/ppo/BreakoutNoFrameskip-v4/seed_123/final-model-ppo-BreakoutNoFrameskip-v4-step-0-to-step-5000.mp4",
        "videos/ppo/BreakoutNoFrameskip-v4/seed_42/final-model-ppo-BreakoutNoFrameskip-v4-step-0-to-step-5000.mp4"
    ]
    
    # Verify videos exist
    existing_videos = []
    for video_path in video_paths:
        if Path(video_path).exists():
            existing_videos.append(video_path)
            print(f"📹 Found: {Path(video_path).name}")
        else:
            print(f"⚠️  Not found: {video_path}")
    
    if not existing_videos:
        print("❌ No video files found!")
        return
    
    print(f"\n🔍 Analyzing {len(existing_videos)} Breakout videos...")
    
    try:
        with BreakoutStrategyAnalyzer() as analyzer:
            # Analyze each video individually first
            individual_analyses = []
            
            for i, video_path in enumerate(existing_videos):
                print(f"\n--- Analyzing Video {i+1}/{len(existing_videos)} ---")
                print(f"File: {Path(video_path).name}")
                
                analysis = analyzer.analyze_breakout_strategy(video_path)
                
                if "error" in analysis:
                    print(f"❌ Analysis failed: {analysis['error']}")
                    continue
                
                individual_analyses.append(analysis)
                
                print(f"✅ Analysis completed!")
                print(f"📊 Strategy Summary:")
                
                # Show key insights
                strategy = analysis["strategy_summary"]
                for section, content in strategy.items():
                    if content:
                        section_name = section.replace('_', ' ').title()
                        preview = content[:200] + "..." if len(content) > 200 else content
                        print(f"\n{section_name}:")
                        print(f"  {preview}")
                
                # Save individual analysis
                output_file = f"analysis_breakout_seed_{Path(video_path).parent.name}.json"
                analyzer.save_analysis(analysis, output_file)
                print(f"💾 Saved to: {output_file}")
            
            # Compare the two videos if we have both
            if len(individual_analyses) >= 2:
                print(f"\n🔍 Comparing Breakout Strategies Between Seeds...")
                
                comparison = analyzer.compare_breakout_strategies(existing_videos)
                
                if "comparison_analysis" in comparison:
                    print(f"✅ Comparison completed!")
                    print(f"\n📊 Comparative Analysis:")
                    comp_text = comparison["comparison_analysis"]
                    print(f"{comp_text}")
                    
                    # Save comparison
                    analyzer.save_analysis(comparison, "comparison_breakout_seeds.json")
                    print(f"\n💾 Comparison saved to: comparison_breakout_seeds.json")
                else:
                    print(f"⚠️  Comparison analysis not available")
            
            print(f"\n🎉 Analysis Complete!")
            print(f"📁 Check the generated JSON files for detailed results")
            
    except Exception as e:
        print(f"❌ Error during analysis: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 