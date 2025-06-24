#!/usr/bin/env python3
"""
Example script demonstrating the use of both BehaviorAnalyzer and BreakoutStrategyAnalyzer.
"""

import os
import sys
import logging
from pathlib import Path
from typing import Optional

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from gemini_analysis import BehaviorAnalyzer, BreakoutStrategyAnalyzer


def setup_logging():
    """Setup logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )


def analyze_video_comprehensive(video_path: str, api_key: Optional[str] = None):
    """
    Perform comprehensive analysis of a video using both analyzers.
    
    Args:
        video_path: Path to the video file
        api_key: Google API key (optional, can use environment variable)
    """
    print(f"Starting comprehensive analysis of: {video_path}")
    print("=" * 60)
    
    # Check if video exists
    if not Path(video_path).exists():
        print(f"Error: Video file not found: {video_path}")
        return
    
    # Initialize analyzers
    try:
        behavior_analyzer = BehaviorAnalyzer(api_key=api_key)
        strategy_analyzer = BreakoutStrategyAnalyzer(api_key=api_key)
    except Exception as e:
        print(f"Error initializing analyzers: {e}")
        return
    
    print("1. Analyzing LOW-LEVEL BEHAVIORS...")
    print("-" * 40)
    
    # Analyze low-level behaviors
    try:
        behavior_result = behavior_analyzer.analyze_behavior(video_path)
        
        if "error" in behavior_result:
            print(f"Behavior analysis failed: {behavior_result['error']}")
        else:
            print("BEHAVIOR ANALYSIS RESULTS:")
            print(f"File size: {behavior_result.get('file_size_mb', 'Unknown'):.1f}MB")
            print("\nSummary:")
            summary = behavior_result.get('behavior_summary', {}).get('summary', 'No summary available')
            print(summary)
            print()
            
    except Exception as e:
        print(f"Behavior analysis error: {e}")
    
    print("2. Analyzing HIGH-LEVEL STRATEGY...")
    print("-" * 40)
    
    # Analyze high-level strategy
    try:
        strategy_result = strategy_analyzer.analyze_breakout_strategy(video_path)
        
        if "error" in strategy_result:
            print(f"Strategy analysis failed: {strategy_result['error']}")
        else:
            print("STRATEGY ANALYSIS RESULTS:")
            print("Key strategic elements identified:")
            strategy_summary = strategy_result.get('strategy_summary', {})
            
            for key, value in strategy_summary.items():
                if value.strip():
                    print(f"- {key.replace('_', ' ').title()}: {value[:100]}...")
            print()
            
    except Exception as e:
        print(f"Strategy analysis error: {e}")
    
    print("Analysis complete!")
    print("=" * 60)


def main():
    """Main function."""
    setup_logging()
    
    # Check for API key - try file first, then environment variable
    api_key = None
    
    # Try reading from Gemini_API_KEY.txt file
    try:
        with open("Gemini_API_KEY.txt", "r") as f:
            api_key = f.read().strip()
        print("✅ API key loaded from Gemini_API_KEY.txt")
    except FileNotFoundError:
        # Fall back to environment variable
        api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if api_key:
            print("✅ API key loaded from environment variable")
    
    if not api_key:
        print("❌ No API key found!")
        print("Please either:")
        print("  1. Create a Gemini_API_KEY.txt file with your API key, or")
        print("  2. Set GEMINI_API_KEY environment variable")
        return
    
    # Example usage
    if len(sys.argv) > 1:
        video_path = sys.argv[1]
    else:
        # Default example path
        video_path = "/mnt/e/Projects/strategy2text/videos/ppo/BreakoutNoFrameskip-v4/seed_25/final-model-ppo-BreakoutNoFrameskip-v4-step-0-to-step-5000.mp4"
        print(f"No video path provided. Using default: {video_path}")
    
    # Perform comprehensive analysis
    analyze_video_comprehensive(video_path, api_key)


if __name__ == "__main__":
    main() 