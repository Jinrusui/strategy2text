#!/usr/bin/env python3
"""
Demo script for Gemini-based RL strategy analysis.

This script demonstrates how to use the strategy analysis framework
described in the dissertation to analyze RL agent gameplay videos.
"""

import sys
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from gemini_analysis import StrategyAnalyzer, GeminiClient, PromptEngineer, EvaluationMetrics
from gemini_analysis.prompt_engineering import AnalysisType
from video_processing.video_sampler import VideoSampler, SamplingStrategy


def demo_single_video_analysis():
    """Demonstrate analysis of a single video."""
    print("=== Single Video Analysis Demo ===")
    
    # Initialize components
    analyzer = StrategyAnalyzer(
        video_dir="videos",  # Replace with your video directory
        output_dir="demo_results"
    )
    
    # Analyze a single video (replace with actual video path)
    video_path = "path/to/your/video.mp4"
    
    if not os.path.exists(video_path):
        print(f"Video not found: {video_path}")
        print("Please update the video_path variable with a valid video file.")
        return
    
    print(f"Analyzing video: {video_path}")
    
    # Strategy analysis
    strategy_result = analyzer.analyze_single_video(
        video_path=video_path,
        analysis_type="strategy",
        max_frames=10
    )
    
    print("\n--- Strategy Analysis Result ---")
    if 'strategy_summary' in strategy_result:
        print(f"Summary: {strategy_result['strategy_summary'][:200]}...")
        print(f"Abstraction Score: {strategy_result['abstraction_score']['abstraction_score']:.3f}")
    else:
        print(f"Error: {strategy_result.get('error', 'Unknown error')}")
    
    # Baseline analysis for comparison
    baseline_result = analyzer.analyze_single_video(
        video_path=video_path,
        analysis_type="baseline",
        max_frames=10
    )
    
    print("\n--- Baseline Analysis Result ---")
    if 'strategy_summary' in baseline_result:
        print(f"Summary: {baseline_result['strategy_summary'][:200]}...")
        print(f"Abstraction Score: {baseline_result['abstraction_score']['abstraction_score']:.3f}")
    else:
        print(f"Error: {baseline_result.get('error', 'Unknown error')}")


def demo_video_set_analysis():
    """Demonstrate analysis of a video set using different sampling strategies."""
    print("\n=== Video Set Analysis Demo ===")
    
    analyzer = StrategyAnalyzer(
        video_dir="videos",  # Replace with your video directory
        output_dir="demo_results"
    )
    
    # Test different sampling strategies
    strategies = [
        SamplingStrategy.TYPICAL,
        SamplingStrategy.EDGE_CASE,
        SamplingStrategy.LONGITUDINAL
    ]
    
    for strategy in strategies:
        print(f"\n--- {strategy.value.title()} Sampling ---")
        
        results = analyzer.analyze_video_set(
            sampling_strategy=strategy,
            num_videos=3,  # Small number for demo
            analysis_type="strategy"
        )
        
        if 'aggregate_metrics' in results:
            metrics = results['aggregate_metrics']
            print(f"Videos analyzed: {metrics.get('successful_analyses', 0)}")
            print(f"Average abstraction score: {metrics.get('average_abstraction_score', 0):.3f}")
        else:
            print(f"Error: {results.get('error', 'Unknown error')}")


def demo_evaluation_metrics():
    """Demonstrate the evaluation metrics."""
    print("\n=== Evaluation Metrics Demo ===")
    
    # Initialize evaluation metrics
    evaluator = EvaluationMetrics()
    
    # Example strategy summaries for comparison
    strategy_summary = """
    The agent employs a tunneling strategy, consistently creating channels on the left side 
    of the screen. It demonstrates adaptive behavior by adjusting paddle position based on 
    ball trajectory and maintains a defensive posture when the tunnel is threatened.
    """
    
    baseline_summary = """
    The paddle moves left and right. The ball bounces around the screen. 
    The agent hits the ball with the paddle multiple times.
    """
    
    # Calculate abstraction scores
    strategy_abstraction = evaluator.calculate_abstraction_score(strategy_summary)
    baseline_abstraction = evaluator.calculate_abstraction_score(baseline_summary)
    
    print("--- Abstraction Score Comparison ---")
    print(f"Strategy Summary Score: {strategy_abstraction['abstraction_score']:.3f}")
    print(f"Baseline Summary Score: {baseline_abstraction['abstraction_score']:.3f}")
    
    # Compare summaries
    comparison = evaluator.compare_summaries(strategy_summary, baseline_summary)
    print(f"\nSemantic Similarity: {comparison['semantic_similarity']:.3f}")
    
    # Demo PFS calculation
    prediction = "The agent will move the paddle to the left to direct the ball into the tunnel."
    ground_truth = "The agent positioned the paddle on the left side and hit the ball toward the tunnel opening."
    
    pfs_score = evaluator.calculate_predictive_faithfulness_score(prediction, ground_truth)
    print(f"\nPredictive Faithfulness Score: {pfs_score:.3f}")


def demo_prompt_engineering():
    """Demonstrate the prompt engineering capabilities."""
    print("\n=== Prompt Engineering Demo ===")
    
    prompt_engineer = PromptEngineer()
    
    # Get different types of prompts
    strategy_prompt = prompt_engineer.get_strategy_prompt(
        game_type="Breakout",
        focus_areas=["tunneling behavior", "defensive strategies"]
    )
    
    baseline_prompt = prompt_engineer.get_prompt(AnalysisType.BASELINE_CAPTIONING)
    
    edge_case_prompt = prompt_engineer.get_edge_case_prompt()
    
    print("--- Strategy Prompt (first 200 chars) ---")
    print(strategy_prompt[:200] + "...")
    
    print("\n--- Baseline Prompt ---")
    print(baseline_prompt)
    
    print("\n--- Edge Case Prompt (first 200 chars) ---")
    print(edge_case_prompt[:200] + "...")
    
    # Validate prompt
    validation = prompt_engineer.validate_prompt(strategy_prompt)
    print(f"\n--- Prompt Validation ---")
    print(f"Valid: {validation['is_valid']}")
    print(f"Word count: {validation['word_count']}")
    print(f"Has instructions: {validation['has_instructions']}")
    print(f"Has context: {validation['has_context']}")


def demo_comparative_experiment():
    """Demonstrate running a comparative experiment."""
    print("\n=== Comparative Experiment Demo ===")
    
    analyzer = StrategyAnalyzer(
        video_dir="videos",  # Replace with your video directory
        output_dir="demo_results"
    )
    
    print("Running comparative experiment (this may take a while)...")
    
    # Run comparative experiment
    experiment_results = analyzer.run_comparative_experiment(
        baseline_type="captioning",
        num_videos_per_strategy=2  # Small number for demo
    )
    
    if 'comparison' in experiment_results:
        comparison = experiment_results['comparison']
        print("\n--- Experiment Results ---")
        
        metrics_comp = comparison.get('metrics_comparison', {})
        for metric, data in metrics_comp.items():
            print(f"\n{metric.replace('_', ' ').title()}:")
            print(f"  Strategy: {data.get('strategy_score', 'N/A')}")
            print(f"  Baseline: {data.get('baseline_score', 'N/A')}")
            print(f"  Winner: {data.get('winner', 'N/A')}")
    else:
        print("Experiment completed but no comparison data available")


def main():
    """Run all demos."""
    print("Gemini Strategy Analysis Framework Demo")
    print("=" * 50)
    
    try:
        # Check if API key is available
        api_key_file = Path("Gemini_API_KEY.txt")
        if not api_key_file.exists() and not os.getenv('GEMINI_API_KEY'):
            print("ERROR: Gemini API key not found!")
            print("Please create a 'Gemini_API_KEY.txt' file or set the GEMINI_API_KEY environment variable.")
            return
        
        # Run demos
        demo_prompt_engineering()
        demo_evaluation_metrics()
        
        # Video-based demos (require actual video files)
        print("\n" + "="*50)
        print("NOTE: The following demos require actual video files.")
        print("Please update the video paths in the demo functions.")
        print("="*50)
        
        # Uncomment these when you have video files available
        # demo_single_video_analysis()
        # demo_video_set_analysis()
        # demo_comparative_experiment()
        
    except Exception as e:
        print(f"Demo error: {e}")
        print("Make sure all dependencies are installed: pip install -r requirements.txt")


if __name__ == "__main__":
    main() 