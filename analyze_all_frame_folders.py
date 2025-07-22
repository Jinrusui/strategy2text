#!/usr/bin/env python3
"""
Convenience script to analyze both frame folders with the complete pipeline:
1. Analyze frames in batches of 10 using Gemini 2.5-pro
2. Synthesize all batch analyses into final reports
"""

import os
import sys
import logging
from pathlib import Path
import argparse
from datetime import datetime
import subprocess


def setup_logging(verbose: bool = False) -> None:
    """Setup logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )


def run_command(command: list, description: str) -> bool:
    """Run a command and return True if successful."""
    logger = logging.getLogger(__name__)
    
    logger.info(f"Running: {description}")
    logger.debug(f"Command: {' '.join(command)}")
    
    try:
        result = subprocess.run(command, check=True, capture_output=True, text=True)
        logger.info(f"✓ {description} completed successfully")
        if result.stdout:
            logger.debug(f"Output: {result.stdout}")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"✗ {description} failed with exit code {e.returncode}")
        if e.stdout:
            logger.error(f"Stdout: {e.stdout}")
        if e.stderr:
            logger.error(f"Stderr: {e.stderr}")
        return False


def analyze_frame_folder(frames_dir: Path, output_base_dir: Path, verbose: bool = False) -> tuple[bool, Path]:
    """Analyze a single frame folder and return success status and output directory."""
    logger = logging.getLogger(__name__)
    
    folder_name = frames_dir.name
    logger.info(f"Starting analysis of folder: {folder_name}")
    
    # Create output directory for this folder's analysis
    analysis_output_dir = output_base_dir / f"{folder_name}_analysis"
    
    # Step 1: Analyze frames in batches
    logger.info(f"Step 1: Analyzing frames in batches for {folder_name}")
    batch_command = [
        sys.executable, 
        "analyze_frame_batches.py",
        str(frames_dir),
        "--output-dir", str(analysis_output_dir),
        "--batch-size", "10"
    ]
    if verbose:
        batch_command.append("--verbose")
    
    batch_success = run_command(batch_command, f"Batch analysis for {folder_name}")
    if not batch_success:
        return False, analysis_output_dir
    
    # Step 2: Synthesize batch analyses into final report
    logger.info(f"Step 2: Synthesizing final report for {folder_name}")
    synthesis_command = [
        sys.executable,
        "synthesize_frame_analysis.py",
        str(analysis_output_dir),
        "--output-file", str(analysis_output_dir / "final_report.md")
    ]
    if verbose:
        synthesis_command.append("--verbose")
    
    synthesis_success = run_command(synthesis_command, f"Synthesis for {folder_name}")
    
    return synthesis_success, analysis_output_dir


def get_all_frame_folders(base_dir: Path) -> list:
    """Get all frame folders from the base directory."""
    frame_folders = []
    
    if not base_dir.exists():
        raise ValueError(f"Base directory does not exist: {base_dir}")
    
    for item in base_dir.iterdir():
        if item.is_dir() and item.name.endswith('_frames'):
            frame_folders.append(item)
    
    # Sort by name for consistent processing order
    frame_folders.sort(key=lambda x: x.name)
    
    return frame_folders


def main():
    """Main function to analyze all frame folders."""
    parser = argparse.ArgumentParser(description="Analyze all frame folders with complete pipeline")
    parser.add_argument("--base-dir", type=str, default="video_clips_30s_frames", 
                       help="Base directory containing frame folders (default: video_clips_30s_frames)")
    parser.add_argument("--output-dir", type=str, default="frame_analysis_results",
                       help="Base output directory for all results")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose logging")
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.verbose)
    logger = logging.getLogger(__name__)
    
    logger.info("=" * 60)
    logger.info("FRAME ANALYSIS PIPELINE")
    logger.info("=" * 60)
    logger.info(f"Started at: {datetime.now().isoformat()}")
    
    # Validate base directory
    base_dir = Path(args.base_dir)
    if not base_dir.exists() or not base_dir.is_dir():
        logger.error(f"Base directory not found: {base_dir}")
        sys.exit(1)
    
    # Get all frame folders
    frame_folders = get_all_frame_folders(base_dir)
    
    if not frame_folders:
        logger.error(f"No frame folders found in {base_dir}")
        sys.exit(1)
    
    logger.info(f"Found {len(frame_folders)} frame folders:")
    for folder in frame_folders:
        logger.info(f"  - {folder.name}")
    
    # Create base output directory
    output_base_dir = Path(args.output_dir)
    output_base_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Output directory: {output_base_dir}")
    
    # Process all frame directories
    results = {}
    
    for i, frames_dir in enumerate(frame_folders, 1):
        logger.info(f"\n" + "=" * 50)
        logger.info(f"PROCESSING FOLDER {i}/{len(frame_folders)}: {frames_dir.name}")
        logger.info("=" * 50)
        
        success, output_dir = analyze_frame_folder(frames_dir, output_base_dir, args.verbose)
        results[frames_dir.name] = {
            "success": success,
            "output_directory": str(output_dir),
            "input_directory": str(frames_dir)
        }
    
    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("PIPELINE SUMMARY")
    logger.info("=" * 60)
    
    total_success = 0
    total_folders = len(results)
    
    for folder_name, result in results.items():
        status = "✓ SUCCESS" if result["success"] else "✗ FAILED"
        logger.info(f"{folder_name}: {status}")
        logger.info(f"  Input:  {result['input_directory']}")
        logger.info(f"  Output: {result['output_directory']}")
        if result["success"]:
            total_success += 1
    
    logger.info(f"\nCompleted: {total_success}/{total_folders} folders processed successfully")
    logger.info(f"Finished at: {datetime.now().isoformat()}")
    
    # Save summary
    import json
    summary_file = output_base_dir / "pipeline_summary.json"
    summary_data = {
        "pipeline": "Frame Analysis Pipeline",
        "timestamp": datetime.now().isoformat(),
        "total_folders": total_folders,
        "successful_folders": total_success,
        "failed_folders": total_folders - total_success,
        "results": results
    }
    
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary_data, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Summary saved to: {summary_file}")
    
    if total_success == total_folders:
        logger.info("🎉 All folders processed successfully!")
        sys.exit(0)
    else:
        logger.warning("⚠️  Some folders failed to process. Check logs for details.")
        sys.exit(1)


if __name__ == "__main__":
    main() 