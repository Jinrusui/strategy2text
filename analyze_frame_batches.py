#!/usr/bin/env python3
"""
Script to analyze video frames in batches of 10 using Gemini 2.5-pro.
Each batch of 10 frames is analyzed individually to describe ball trajectory and paddle behavior.
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


def get_frame_analysis_prompt() -> str:
    """Get the frame analysis prompt based on the example."""
    return """You are a precise motion analyst for robotic agents. Your task is to describe object trajectories from a sequence of images from the game Breakout.

Your goal is to describe the **trajectory of the ball** by analyzing both its **short-term, frame-to-frame movement** and its **long-term, overall path** across the entire sequence.

Please provide your analysis in the following two-part structure:

### Overall Trajectory Summary:
Describe the ball's complete path from its starting point (in Image 1) to its ending point (in Image 10).
*(Example: "The ball travels in a wide arc from the top-left, strikes the right wall, and then descends steeply towards the paddle which is positioned at the bottom-center.")*

### Detailed Motion Breakdown:
Provide a chronological, bulleted list of the event. For each phase of the movement, describe the ball's **immediate direction and speed (e.g., 'fast,' 'slowing down')** and the paddle's corresponding reaction.

*(Example Bullet Points):*
* **Images 1-3:** The ball moves quickly downwards and to the right. The paddle is stationary on the far left.
* **Images 4-6:** After striking a brick, the ball's trajectory shifts, now moving slowly downwards and to the left. In response, the paddle begins a smooth, deliberate movement to the right.
* **Images 7-8:** The ball accelerates as it falls. The paddle also accelerates, trying to position itself for interception.
* **Images 9-10:** The paddle successfully intercepts the fast-moving ball, causing a sharp rebound upwards and to the right."""


def get_frame_files(frames_dir: Path) -> List[Path]:
    """Get all frame files from the directory, sorted by frame number."""
    frame_files = []
    
    # Look for common image formats
    for ext in ['*.png', '*.jpg', '*.jpeg']:
        frame_files.extend(frames_dir.glob(ext))
    
    # Sort by frame number (assuming format like frame_0001.png)
    frame_files.sort(key=lambda x: x.name)
    
    return frame_files


def create_frame_batches(frame_files: List[Path], batch_size: int = 10) -> List[List[Path]]:
    """Create batches of frames."""
    batches = []
    for i in range(0, len(frame_files), batch_size):
        batch = frame_files[i:i + batch_size]
        if len(batch) == batch_size:  # Only include complete batches
            batches.append(batch)
    
    return batches


def analyze_frame_batch(gemini_client: SimpleGeminiClient, frame_batch: List[Path], batch_num: int) -> str:
    """Analyze a single batch of frames using Gemini."""
    logger = logging.getLogger(__name__)
    
    logger.info(f"Analyzing batch {batch_num} with {len(frame_batch)} frames")
    
    try:
        # Create the analysis prompt
        prompt = get_frame_analysis_prompt()
        
        # Generate analysis using the simplified client
        logger.info(f"Generating analysis for batch {batch_num}")
        analysis = gemini_client.analyze_images_with_prompt(frame_batch, prompt)
        
        logger.info(f"Analysis completed for batch {batch_num}")
        return analysis
            
    except Exception as e:
        logger.error(f"Failed to analyze batch {batch_num}: {str(e)}")
        raise


def analyze_frames_directory(frames_dir: Path, output_dir: Path, batch_size: int = 10) -> Dict[str, Any]:
    """Analyze all frames in a directory in batches."""
    logger = logging.getLogger(__name__)
    
    logger.info(f"Starting analysis of frames in: {frames_dir}")
    logger.info(f"Output directory: {output_dir}")
    
    # Get all frame files
    frame_files = get_frame_files(frames_dir)
    if not frame_files:
        raise ValueError(f"No frame files found in {frames_dir}")
    
    logger.info(f"Found {len(frame_files)} frame files")
    
    # Create batches
    batches = create_frame_batches(frame_files, batch_size)
    logger.info(f"Created {len(batches)} batches of {batch_size} frames each")
    
    if len(batches) == 0:
        raise ValueError(f"No complete batches of {batch_size} frames found")
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Analyze each batch
    batch_analyses = []
    
    with SimpleGeminiClient() as gemini_client:
        for i, batch in enumerate(batches, 1):
            try:
                # Determine frame range for this batch
                start_frame = batch[0].stem.split('_')[-1] if '_' in batch[0].stem else str(i * batch_size - batch_size + 1)
                end_frame = batch[-1].stem.split('_')[-1] if '_' in batch[-1].stem else str(i * batch_size)
                
                logger.info(f"Processing batch {i}/{len(batches)}: frames {start_frame}-{end_frame}")
                
                # Analyze the batch
                analysis = analyze_frame_batch(gemini_client, batch, i)
                
                # Store the analysis
                batch_result = {
                    "batch_number": i,
                    "frame_range": f"{start_frame}-{end_frame}",
                    "frame_files": [f.name for f in batch],
                    "analysis": analysis,
                    "timestamp": datetime.now().isoformat()
                }
                
                batch_analyses.append(batch_result)
                
                # Save individual batch analysis
                batch_file = output_dir / f"batch_{i:03d}_frames_{start_frame}-{end_frame}.md"
                with open(batch_file, 'w', encoding='utf-8') as f:
                    f.write(f"# Batch {i}: Frames {start_frame}-{end_frame}\n\n")
                    f.write(f"**Files:** {', '.join([f.name for f in batch])}\n\n")
                    f.write(f"**Timestamp:** {batch_result['timestamp']}\n\n")
                    f.write("---\n\n")
                    f.write(analysis)
                    f.write("\n")
                
                logger.info(f"Saved analysis for batch {i} to {batch_file}")
                
            except Exception as e:
                logger.error(f"Failed to process batch {i}: {str(e)}")
                # Continue with next batch
                continue
    
    # Compile results
    results = {
        "frames_directory": str(frames_dir),
        "output_directory": str(output_dir),
        "total_frames": len(frame_files),
        "batch_size": batch_size,
        "total_batches": len(batches),
        "successful_batches": len(batch_analyses),
        "failed_batches": len(batches) - len(batch_analyses),
        "batch_analyses": batch_analyses,
        "timestamp": datetime.now().isoformat()
    }
    
    # Save summary results
    summary_file = output_dir / "batch_analysis_summary.json"
    import json
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Analysis complete. Processed {len(batch_analyses)}/{len(batches)} batches successfully")
    logger.info(f"Results saved to: {output_dir}")
    
    return results


def main():
    """Main function to run the frame batch analysis."""
    parser = argparse.ArgumentParser(description="Analyze video frames in batches using Gemini 2.5-pro")
    parser.add_argument("frames_dir", type=str, help="Directory containing frame files")
    parser.add_argument("--output-dir", type=str, help="Output directory for analysis results")
    parser.add_argument("--batch-size", type=int, default=10, help="Number of frames per batch (default: 10)")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose logging")
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.verbose)
    logger = logging.getLogger(__name__)
    
    # Validate input directory
    frames_dir = Path(args.frames_dir)
    if not frames_dir.exists():
        logger.error(f"Frames directory does not exist: {frames_dir}")
        sys.exit(1)
    
    if not frames_dir.is_dir():
        logger.error(f"Frames path is not a directory: {frames_dir}")
        sys.exit(1)
    
    # Setup output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = frames_dir / "gemini_analysis"
    
    try:
        # Run the analysis
        results = analyze_frames_directory(frames_dir, output_dir, args.batch_size)
        
        logger.info("=" * 50)
        logger.info("ANALYSIS SUMMARY")
        logger.info("=" * 50)
        logger.info(f"Frames directory: {results['frames_directory']}")
        logger.info(f"Total frames: {results['total_frames']}")
        logger.info(f"Batch size: {results['batch_size']}")
        logger.info(f"Total batches: {results['total_batches']}")
        logger.info(f"Successful batches: {results['successful_batches']}")
        logger.info(f"Failed batches: {results['failed_batches']}")
        logger.info(f"Output directory: {results['output_directory']}")
        
        if results['failed_batches'] > 0:
            logger.warning(f"Some batches failed to process. Check logs for details.")
            sys.exit(1)
        else:
            logger.info("All batches processed successfully!")
            
    except Exception as e:
        logger.error(f"Analysis failed: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main() 