#!/usr/bin/env python3
"""
Quick video analysis using Gemini Flash model to avoid quota limits.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from gemini_analysis import StrategyAnalyzer

def main():
    print("🚀 Quick Video Analysis with Gemini Flash")
    print("=" * 50)
    
    # Use the smaller, faster model with higher quotas
    analyzer = StrategyAnalyzer(
        video_dir=".",
        output_dir="quick_results",
        model_name="gemini-1.5-flash"  # Smaller model, higher quotas
    )
    
    video_file = "breakout_ppo_episode_1.mp4"
    
    print(f"🎬 Analyzing: {video_file}")
    print("📊 Using gemini-1.5-flash (higher quota limits)")
    print("🔧 Using 5 frames to minimize token usage")
    
    # Strategy analysis with minimal frames
    print("\n🤖 Strategy Analysis...")
    strategy_result = analyzer.analyze_single_video(
        video_path=video_file,
        analysis_type="strategy",
        max_frames=5,  # Reduced frames to save tokens
        use_cache=False
    )
    
    if 'strategy_summary' in strategy_result:
        print("✅ Strategy Analysis SUCCESS!")
        summary = strategy_result['strategy_summary']
        abstraction = strategy_result['abstraction_score']['abstraction_score']
        
        print(f"\n📝 Strategy Summary ({len(summary)} chars):")
        print(f"   {summary}")
        print(f"\n📊 Abstraction Score: {abstraction:.3f}")
        
        # Save result for comparison
        with open("strategy_result.txt", "w") as f:
            f.write(f"Strategy Summary:\n{summary}\n\nAbstraction Score: {abstraction:.3f}\n")
        print("💾 Results saved to strategy_result.txt")
        
    else:
        print("❌ Strategy analysis failed:")
        print(f"   Error: {strategy_result.get('error', 'Unknown error')}")
    
    # Baseline analysis for comparison
    print("\n🎥 Baseline Analysis...")
    baseline_result = analyzer.analyze_single_video(
        video_path=video_file,
        analysis_type="baseline",
        max_frames=5,
        use_cache=False
    )
    
    if 'strategy_summary' in baseline_result:
        print("✅ Baseline Analysis SUCCESS!")
        summary = baseline_result['strategy_summary']
        abstraction = baseline_result['abstraction_score']['abstraction_score']
        
        print(f"\n📝 Baseline Summary ({len(summary)} chars):")
        print(f"   {summary}")
        print(f"\n📊 Abstraction Score: {abstraction:.3f}")
        
        # Save result for comparison
        with open("baseline_result.txt", "w") as f:
            f.write(f"Baseline Summary:\n{summary}\n\nAbstraction Score: {abstraction:.3f}\n")
        print("💾 Results saved to baseline_result.txt")
        
    else:
        print("❌ Baseline analysis failed:")
        print(f"   Error: {baseline_result.get('error', 'Unknown error')}")
    
    print("\n" + "=" * 50)
    print("✅ Analysis complete!")

if __name__ == "__main__":
    main() 