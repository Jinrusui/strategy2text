#!/usr/bin/env python3
"""
High-level strategy analyzer script for Breakout RL agent videos.
Analyzes strategic decision-making and long-term planning using gemini-2.0-flash with file upload.
"""

import os
import sys
import json
import logging
from pathlib import Path
from typing import List, Dict, Any
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from gemini_analysis.strategy_analyzer import BreakoutStrategyAnalyzer


def setup_logging(verbose: bool = False) -> None:
    """Setup logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    
    # Ensure analysis_results directory exists
    analysis_dir = Path(__file__).parent / "analysis_results"
    analysis_dir.mkdir(exist_ok=True)
    
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(analysis_dir / 'high_level_strategy_analysis.log')
        ]
    )


def find_video_files(video_dir: str) -> List[str]:
    """Find all video files in the given directory recursively."""
    video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.webm'}
    video_files = []
    
    for root, dirs, files in os.walk(video_dir):
        for file in files:
            if Path(file).suffix.lower() in video_extensions:
                video_files.append(os.path.join(root, file))
    
    return sorted(video_files)


def analyze_single_video(video_path: str, api_key: str = None, model: str = "gemini-2.5-pro-preview-06-05") -> Dict[str, Any]:
    """Analyze a single video for high-level strategy using the new API format."""
    logger = logging.getLogger(__name__)
    
    try:
        logger.info(f"Starting high-level strategy analysis for: {video_path}")
        start_time = time.time()
        
        # Initialize analyzer with context manager for proper cleanup
        with BreakoutStrategyAnalyzer(api_key=api_key, model=model) as analyzer:
            result = analyzer.analyze_breakout_strategy(video_path)
        
        end_time = time.time()
        result['analysis_duration_seconds'] = end_time - start_time
        
        logger.info(f"Completed high-level strategy analysis for: {video_path} in {end_time - start_time:.2f}s")
        return result
        
    except Exception as e:
        logger.error(f"Failed to analyze high-level strategy for {video_path}: {str(e)}")
        return {
            "video_path": video_path,
            "environment": "Breakout",
            "analysis_type": "high_level_strategy",
            "model": model,
            "error": str(e),
            "timestamp": time.time()
        }


def analyze_videos_parallel(video_files: List[str], max_workers: int = 4, api_key: str = None, model: str = "gemini-2.5-pro-preview-06-05") -> List[Dict[str, Any]]:
    """Analyze multiple videos in parallel for high-level strategy."""
    logger = logging.getLogger(__name__)
    
    results = []
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_video = {
            executor.submit(analyze_single_video, video_path, api_key): video_path
            for video_path in video_files
        }
        
        # Collect results as they complete
        for future in as_completed(future_to_video):
            video_path = future_to_video[future]
            try:
                result = future.result()
                results.append(result)
                logger.info(f"Completed analysis for {video_path}")
            except Exception as e:
                logger.error(f"Analysis failed for {video_path}: {str(e)}")
                results.append({
                    "video_path": video_path,
                    "environment": "Breakout",
                    "analysis_type": "high_level_strategy",
                    "model": model,
                    "error": str(e),
                    "timestamp": time.time()
                })
    
    return results


def analyze_videos_batch(video_files: List[str], api_key: str = None, model: str = "gemini-2.5-pro-preview-06-05") -> List[Dict[str, Any]]:
    """Analyze multiple videos using the batch method for better efficiency."""
    logger = logging.getLogger(__name__)
    
    try:
        logger.info(f"Starting batch high-level strategy analysis for {len(video_files)} videos")
        start_time = time.time()
        
        # Use batch analysis method
        with BreakoutStrategyAnalyzer(api_key=api_key, model=model) as analyzer:
            results = analyzer.batch_analyze_breakout_strategies(video_files)
        
        end_time = time.time()
        total_time = end_time - start_time
        
        # Add timing information to each result
        for result in results:
            result['batch_analysis_duration_seconds'] = total_time
            result['average_time_per_video'] = total_time / len(video_files)
        
        logger.info(f"Completed batch analysis in {total_time:.2f} seconds")
        return results
        
    except Exception as e:
        logger.error(f"Batch analysis failed: {str(e)}")
        # Return error results for all videos
        return [{
            "video_path": video_path,
            "environment": "Breakout",
            "analysis_type": "high_level_strategy",
            "model": model,
            "error": str(e),
            "timestamp": time.time()
        } for video_path in video_files]


def save_results(results: List[Dict[str, Any]], output_file: str, model: str = "gemini-2.5-pro-preview-06-05") -> None:
    """Save analysis results to JSON file."""
    logger = logging.getLogger(__name__)
    
    # Ensure analysis_results directory exists
    analysis_dir = Path(__file__).parent / "analysis_results"
    analysis_dir.mkdir(exist_ok=True)
    
    # If output_file is just a filename, put it in analysis_results directory
    if not os.path.dirname(output_file):
        output_file = analysis_dir / output_file
    else:
        output_file = Path(output_file)
    
    output_data = {
        "analysis_type": "high_level_strategy",
        "model": model,
        "method": "file_upload",
        "total_videos": len(results),
        "successful_analyses": len([r for r in results if "error" not in r]),
        "failed_analyses": len([r for r in results if "error" in r]),
        "timestamp": time.time(),
        "results": results
    }
    
    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    logger.info(f"Results saved to {output_file}")
    logger.info(f"Successfully analyzed {output_data['successful_analyses']}/{output_data['total_videos']} videos")


def main():
    """Main function to run high-level strategy analysis."""

    parser = argparse.ArgumentParser(description="Analyze high-level strategies of Breakout RL agent videos using gemini-2.5-pro-preview-06-05")
    parser.add_argument("video_dir", help="Directory containing video files")
    parser.add_argument("-o", "--output", default="high_level_strategy_results.json", 
                       help="Output JSON file (default: high_level_strategy_results.json)")
    parser.add_argument("-w", "--workers", type=int, default=4, 
                       help="Number of parallel workers (default: 4)")
    parser.add_argument("--batch", action="store_true",
                       help="Use batch analysis method (more efficient for many videos)")
    parser.add_argument("--api-key", 
                       help="Google API key (if not set via environment variable)")
    parser.add_argument("-v", "--verbose", action="store_true", 
                       help="Enable verbose logging")
    parser.add_argument("--model", default="gemini-2.5-pro-preview-06-05",
                       help="Model to use for analysis (default: gemini-2.5-pro-preview-06-05)")
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.verbose)
    logger = logging.getLogger(__name__)
    
    # Check if video directory exists
    if not os.path.exists(args.video_dir):
        logger.error(f"Video directory not found: {args.video_dir}")
        sys.exit(1)
    
    # Find video files
    logger.info(f"Searching for video files in: {args.video_dir}")
    video_files = find_video_files(args.video_dir)
    
    if not video_files:
        logger.error(f"No video files found in: {args.video_dir}")
        sys.exit(1)
    
    logger.info(f"Found {len(video_files)} video files")
    for video_file in video_files:
        logger.info(f"  - {video_file}")
    
    # Analyze videos
    start_time = time.time()
    
    if args.batch:
        logger.info("Starting batch high-level strategy analysis")
        results = analyze_videos_batch(video_files, api_key=args.api_key, model=args.model)
    else:
        logger.info(f"Starting parallel high-level strategy analysis with {args.workers} workers")
        results = analyze_videos_parallel(video_files, max_workers=args.workers, api_key=args.api_key, model=args.model)
    
    end_time = time.time()
    total_time = end_time - start_time
    
    logger.info(f"Analysis completed in {total_time:.2f} seconds")
    
    # Save results
    save_results(results, args.output, model=args.model )
    
    # Print summary
    successful = len([r for r in results if "error" not in r])
    failed = len([r for r in results if "error" in r])
    
    print(f"\n🧠 High-Level Strategy Analysis Summary:")
    print(f"📁 Total videos: {len(video_files)}")
    print(f"✅ Successful analyses: {successful}")
    print(f"❌ Failed analyses: {failed}")
    print(f"🤖 Model used: {args.model} (file upload method)")
    print(f"⏱️  Total time: {total_time:.2f} seconds")
    print(f"⚡ Average time per video: {total_time/len(video_files):.2f} seconds")
    print(f"📄 Results saved to: {args.output}")
    
    if failed > 0:
        print(f"\n⚠️  {failed} videos failed analysis. Check the log file for details.")


if __name__ == "__main__":
    main() 