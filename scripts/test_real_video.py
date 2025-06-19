#!/usr/bin/env python3
"""
Test script for Gemini analysis with a real video file.
"""

import sys
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

def test_with_real_video():
    """Test the analysis with a real video file."""
    
    # Check if API key is available
    api_key_file = Path("Gemini_API_KEY.txt")
    if not api_key_file.exists() and not os.getenv('GEMINI_API_KEY'):
        print("⚠️  Gemini API key not found!")
        print("This demo will show error handling, but won't perform actual analysis.")
        print("To enable full analysis:")
        print("1. Create 'Gemini_API_KEY.txt' with your API key")
        print("2. Or set GEMINI_API_KEY environment variable")
        print()
    
    from gemini_analysis import StrategyAnalyzer
    
    # Find a video file
    video_file = "breakout_ppo_episode_1.mp4"
    if not os.path.exists(video_file):
        print(f"❌ Video file not found: {video_file}")
        return
    
    print(f"🎬 Found video file: {video_file}")
    print(f"📁 File size: {os.path.getsize(video_file) / (1024*1024):.1f} MB")
    
    # Initialize analyzer
    print("\n🔧 Initializing Strategy Analyzer...")
    analyzer = StrategyAnalyzer(
        video_dir=".",
        output_dir="test_results"
    )
    
    # Test strategy analysis
    print(f"\n🤖 Analyzing video with strategy-focused approach...")
    try:
        strategy_result = analyzer.analyze_single_video(
            video_path=video_file,
            analysis_type="strategy",
            max_frames=8,
            use_cache=False
        )
        
        if 'strategy_summary' in strategy_result:
            print("✅ Strategy analysis successful!")
            summary = strategy_result['strategy_summary']
            abstraction = strategy_result['abstraction_score']['abstraction_score']
            
            print(f"\n📊 Results:")
            print(f"   Summary length: {len(summary)} characters")
            print(f"   Abstraction score: {abstraction:.3f}")
            print(f"\n📝 Strategy Summary (first 300 chars):")
            print(f"   {summary[:300]}{'...' if len(summary) > 300 else ''}")
            
        else:
            print("❌ Strategy analysis failed:")
            print(f"   Error: {strategy_result.get('error', 'Unknown error')}")
            
    except Exception as e:
        print(f"❌ Exception during analysis: {e}")
    
    # Test baseline analysis for comparison
    print(f"\n🎥 Analyzing video with baseline approach...")
    try:
        baseline_result = analyzer.analyze_single_video(
            video_path=video_file,
            analysis_type="baseline",
            max_frames=8,
            use_cache=False
        )
        
        if 'strategy_summary' in baseline_result:
            print("✅ Baseline analysis successful!")
            summary = baseline_result['strategy_summary']
            abstraction = baseline_result['abstraction_score']['abstraction_score']
            
            print(f"\n📊 Results:")
            print(f"   Summary length: {len(summary)} characters")
            print(f"   Abstraction score: {abstraction:.3f}")
            print(f"\n📝 Baseline Summary (first 300 chars):")
            print(f"   {summary[:300]}{'...' if len(summary) > 300 else ''}")
            
        else:
            print("❌ Baseline analysis failed:")
            print(f"   Error: {baseline_result.get('error', 'Unknown error')}")
            
    except Exception as e:
        print(f"❌ Exception during analysis: {e}")

def test_evaluation_metrics():
    """Test evaluation metrics without requiring API."""
    print("\n🧮 Testing Evaluation Metrics (no API required)...")
    
    from gemini_analysis import EvaluationMetrics
    
    evaluator = EvaluationMetrics()
    
    # Test data
    strategy_summary = "The agent employs a tunneling strategy on the left side, adapting paddle position based on ball trajectory."
    baseline_summary = "The paddle moves left and right. The ball bounces around."
    
    # Test abstraction scoring
    strategy_abs = evaluator.calculate_abstraction_score(strategy_summary)
    baseline_abs = evaluator.calculate_abstraction_score(baseline_summary)
    
    print(f"✅ Abstraction Scores:")
    print(f"   Strategy: {strategy_abs['abstraction_score']:.3f}")
    print(f"   Baseline: {baseline_abs['abstraction_score']:.3f}")
    
    # Test PFS
    prediction = "Agent will tunnel left side"
    ground_truth = "Agent created tunnel on left side"
    pfs = evaluator.calculate_predictive_faithfulness_score(prediction, ground_truth)
    
    print(f"✅ Predictive Faithfulness Score: {pfs:.3f}")

def main():
    print("🚀 Gemini Strategy Analysis - Real Video Test")
    print("=" * 50)
    
    # Test evaluation metrics (works without API)
    test_evaluation_metrics()
    
    # Test with real video (requires API)
    test_with_real_video()
    
    print("\n" + "=" * 50)
    print("✅ Test completed!")
    print("\n💡 Next steps:")
    print("   1. Add your Gemini API key to enable full analysis")
    print("   2. Run: python scripts/demo_gemini_analysis.py")
    print("   3. See GEMINI_ANALYSIS_GUIDE.md for detailed usage")

if __name__ == "__main__":
    main() 