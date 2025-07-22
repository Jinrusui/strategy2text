#!/usr/bin/env python3
"""
Script to analyze video files using Gemini's video understanding API.
Uses technical RL policy analysis prompt to diagnose agent behavior.
"""

import os
import sys
import logging
from pathlib import Path
from typing import List, Dict, Any
import argparse
from datetime import datetime

# Import our simplified Gemini client
from simple_gemini_client import SimpleGeminiClient


def setup_logging(verbose: bool = False) -> None:
    """Setup logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )


def get_video_analysis_prompt() -> str:
    """Get the technical RL policy analysis prompt for video understanding."""
    return """You are a technical analyst for Reinforcement Learning policies. Your function is to diagnose an agent's behavior based on raw visual data from its environment.

**Context:**
The provided video shows an agent controlling a horizontal platform at the bottom of the screen. Its objective is to intercept a moving projectile and direct it towards static objects in the upper region.

**Task:**
Analyze the agent's policy execution within this short clip, focusing on low-level control, decision-making, and any interesting strategies or patterns.

1.  **Identify Key Moments:** Pinpoint critical events, especially moments of success, failure, or sub-optimal control or interesting strategies or patterns starting to emerge.
2.  **Provide Timestamps:** For each key moment you identify, specify the time range in the clip (e.g., "From 9-13s...").
3.  **Analyze Control Errors:** If the agent fails to intercept the projectile (leading to a termination event), analyze the cause. Examine the agent-controlled platform's movement relative to the projectile's trajectory and velocity. Was the platform's response delayed? Was its movement inefficient or aimed at the wrong intercept point?
4.  **Observe Policy Formation:** Briefly note any observable action patterns that indicate an emerging behavioral policy or a breakdown in its execution.

**Constraints:**
* **Word Count:** The entire analysis must be under 250 words.
* **Focus:** The analysis should center on the agent's control system and its immediate successes or failures.

Please provide your technical analysis of the agent's policy execution in this video clip."""


def analyze_video_file(video_path: Path, output_dir: Path, gemini_client: SimpleGeminiClient) -> Dict[str, Any]:
    """Analyze a single video file using Gemini's video understanding API."""
    logger = logging.getLogger(__name__)
    
    logger.info(f"Starting analysis of video: {video_path.name}")
    
    # Get the analysis prompt
    prompt = get_video_analysis_prompt()
    
    try:
        # For video analysis, we need to use a different approach since our SimpleGeminiClient 
        # is designed for images. Let me create a video-specific method.
        
        # Read video file
        with open(video_path, 'rb') as f:
            video_data = f.read()
        
        logger.info(f"Video file size: {len(video_data)} bytes")
        
        # Use the generative AI API directly for video
        import google.generativeai as genai
        
        # Create video part
        video_part = {
            'mime_type': 'video/mp4',
            'data': video_data
        }
        
        # Generate analysis
        logger.info("Generating video analysis using Gemini")
        response = gemini_client.model.generate_content(
            [video_part, prompt],
            safety_settings=gemini_client.safety_settings
        )
        
        if response and response.text:
            analysis = response.text.strip()
            logger.info(f"Analysis completed ({len(analysis)} characters)")
        else:
            raise ValueError("Empty response from Gemini")
        
        # Prepare results
        results = {
            "video_file": str(video_path),
            "video_name": video_path.name,
            "file_size_bytes": len(video_data),
            "analysis": analysis,
            "analysis_length": len(analysis),
            "timestamp": datetime.now().isoformat()
        }
        
        # Save individual analysis
        output_file = output_dir / f"{video_path.stem}_analysis.md"
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(f"# Video Analysis: {video_path.name}\n\n")
            f.write(f"**File:** {video_path.name}\n")
            f.write(f"**Size:** {len(video_data):,} bytes\n")
            f.write(f"**Analyzed:** {results['timestamp']}\n\n")
            f.write("---\n\n")
            f.write("## Technical RL Policy Analysis\n\n")
            f.write(analysis)
            f.write("\n")
        
        logger.info(f"Analysis saved to: {output_file}")
        
        return results
        
    except Exception as e:
        logger.error(f"Failed to analyze video {video_path.name}: {str(e)}")
        raise


def analyze_videos(video_paths: List[Path], output_dir: Path) -> Dict[str, Any]:
    """Analyze multiple video files."""
    logger = logging.getLogger(__name__)
    
    logger.info(f"Starting analysis of {len(video_paths)} videos")
    logger.info(f"Output directory: {output_dir}")
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Analyze each video
    video_analyses = []
    
    with SimpleGeminiClient() as gemini_client:
        for i, video_path in enumerate(video_paths, 1):
            try:
                logger.info(f"Processing video {i}/{len(video_paths)}: {video_path.name}")
                
                # Analyze the video
                analysis_result = analyze_video_file(video_path, output_dir, gemini_client)
                video_analyses.append(analysis_result)
                
                logger.info(f"✓ Completed analysis for {video_path.name}")
                
            except Exception as e:
                logger.error(f"✗ Failed to process {video_path.name}: {str(e)}")
                # Store error result
                error_result = {
                    "video_file": str(video_path),
                    "video_name": video_path.name,
                    "error": str(e),
                    "timestamp": datetime.now().isoformat()
                }
                video_analyses.append(error_result)
                continue
    
    # Compile results
    results = {
        "analysis_type": "Video Understanding - RL Policy Analysis",
        "total_videos": len(video_paths),
        "successful_analyses": len([r for r in video_analyses if "error" not in r]),
        "failed_analyses": len([r for r in video_analyses if "error" in r]),
        "output_directory": str(output_dir),
        "video_analyses": video_analyses,
        "timestamp": datetime.now().isoformat()
    }
    
    # Save summary results
    summary_file = output_dir / "video_analysis_summary.json"
    import json
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    # Create combined report
    combined_report_file = output_dir / "combined_video_analysis.md"
    with open(combined_report_file, 'w', encoding='utf-8') as f:
        f.write("# Combined Video Analysis Report\n\n")
        f.write(f"**Analysis Type:** Video Understanding - RL Policy Analysis\n")
        f.write(f"**Total Videos:** {results['total_videos']}\n")
        f.write(f"**Successful:** {results['successful_analyses']}\n")
        f.write(f"**Failed:** {results['failed_analyses']}\n")
        f.write(f"**Generated:** {results['timestamp']}\n\n")
        f.write("---\n\n")
        
        for analysis in video_analyses:
            if "error" not in analysis:
                f.write(f"## {analysis['video_name']}\n\n")
                f.write(f"**File Size:** {analysis['file_size_bytes']:,} bytes\n")
                f.write(f"**Analysis Length:** {analysis['analysis_length']} characters\n\n")
                f.write(analysis['analysis'])
                f.write("\n\n---\n\n")
            else:
                f.write(f"## {analysis['video_name']} (FAILED)\n\n")
                f.write(f"**Error:** {analysis['error']}\n\n")
                f.write("---\n\n")
    
    logger.info(f"Analysis complete. Processed {results['successful_analyses']}/{results['total_videos']} videos successfully")
    logger.info(f"Results saved to: {output_dir}")
    logger.info(f"Combined report: {combined_report_file}")
    
    return results


def main():
    """Main function to run the video analysis."""
    parser = argparse.ArgumentParser(description="Analyze videos using Gemini video understanding API")
    parser.add_argument("video_files", nargs="*", help="Video files to analyze")
    parser.add_argument("-v", "--video-dir", type=str, help="Directory containing video files")
    parser.add_argument("-o", "--output-dir", type=str, default="video_analysis_results", 
                       help="Output directory for analysis results")
    parser.add_argument("-i", "--include-pattern", type=str, default="*.mp4", 
                       help="Pattern to match video files (default: *.mp4)")
    parser.add_argument( "--verbose", action="store_true", help="Enable verbose logging")
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.verbose)
    logger = logging.getLogger(__name__)
    
    logger.info("=" * 60)
    logger.info("VIDEO ANALYSIS WITH GEMINI")
    logger.info("=" * 60)
    logger.info(f"Started at: {datetime.now().isoformat()}")
    
    # Collect video files
    video_paths = []
    
    if args.video_files:
        # Use specified video files
        for video_file in args.video_files:
            video_path = Path(video_file)
            if video_path.exists() and video_path.is_file():
                video_paths.append(video_path)
            else:
                logger.warning(f"Video file not found: {video_path}")
    
    if args.video_dir:
        # Use video directory
        video_dir = Path(args.video_dir)
        if video_dir.exists() and video_dir.is_dir():
            found_videos = list(video_dir.glob(args.include_pattern))
            video_paths.extend(found_videos)
            logger.info(f"Found {len(found_videos)} videos in {video_dir}")
        else:
            logger.error(f"Video directory not found: {video_dir}")
            sys.exit(1)
    
    # Default to specific videos if none specified
    if not video_paths:
        default_videos = [
            "video_clips/demo_15-25s.mp4",
            "video_clips/demo_28-41s.mp4"
        ]
        
        for video_file in default_videos:
            video_path = Path(video_file)
            if video_path.exists():
                video_paths.append(video_path)
            else:
                logger.warning(f"Default video file not found: {video_path}")
    
    if not video_paths:
        logger.error("No video files found to analyze")
        sys.exit(1)
    
    logger.info(f"Videos to analyze: {[v.name for v in video_paths]}")
    
    # Setup output directory
    output_dir = Path(args.output_dir)
    
    try:
        # Run the analysis
        results = analyze_videos(video_paths, output_dir)
        
        logger.info("=" * 60)
        logger.info("ANALYSIS SUMMARY")
        logger.info("=" * 60)
        logger.info(f"Total videos: {results['total_videos']}")
        logger.info(f"Successful analyses: {results['successful_analyses']}")
        logger.info(f"Failed analyses: {results['failed_analyses']}")
        logger.info(f"Output directory: {results['output_directory']}")
        
        if results['failed_analyses'] > 0:
            logger.warning(f"Some videos failed to process. Check logs for details.")
            sys.exit(1)
        else:
            logger.info("🎉 All videos processed successfully!")
            
    except Exception as e:
        logger.error(f"Analysis failed: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main() 