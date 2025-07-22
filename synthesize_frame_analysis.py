#!/usr/bin/env python3
"""
Script to synthesize all batch frame analyses into a final comprehensive report.
Takes the individual batch analyses and creates a unified strategy summary.
"""

import os
import sys
import json
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


def get_synthesis_prompt() -> str:
    """Get the synthesis prompt for creating the final report."""
    return """You are an expert AI analyst specializing in summarizing agent behavior from observational data.

The following text contains a series of descriptions. Each description details the events of a 10-frame segment from a single, continuous video clip of an agent playing the game Breakout. The descriptions are presented in chronological order.

Your task is to synthesize all of this sequential information into a single, cohesive narrative that clearly explains what happened in the full video clip.

**Provided Sequential Descriptions:**

**[Description from Frames 1-10]**
(Paste the description for the first 10-frame collection here.)

**[Description from Frames 11-20]**
(Paste the description for the second 10-frame collection here.)

**[Description from Frames 21-30]**
(Paste the description for the third 10-frame collection here.)

**(Continue pasting all subsequent descriptions in order...)**

**Synthesis Task:**

Generate a final summary of the entire video clip. The summary must be:
1.  **Readable and Cohesive:** Write it as a smooth, easy-to-understand narrative.
2.  **Insightful:** Clearly explain the agent's overall course of action and pinpoint the key moments that determined the final outcome (whether success or failure).
3.  **Concise:** Keep the entire summary under 200 words.

The final output should allow a user to quickly understand the agent's approach and the significant events of the clip without needing to read the detailed descriptions."""


def load_batch_analyses(analysis_dir: Path) -> List[Dict[str, Any]]:
    """Load all batch analysis files from the analysis directory."""
    logger = logging.getLogger(__name__)
    
    batch_analyses = []
    
    # Look for the summary JSON file first
    summary_file = analysis_dir / "batch_analysis_summary.json"
    if summary_file.exists():
        logger.info(f"Loading batch analyses from summary file: {summary_file}")
        with open(summary_file, 'r', encoding='utf-8') as f:
            summary_data = json.load(f)
            batch_analyses = summary_data.get("batch_analyses", [])
    else:
        # Fallback: load individual batch files
        logger.info("Summary file not found, loading individual batch files")
        batch_files = sorted(analysis_dir.glob("batch_*.md"))
        
        for batch_file in batch_files:
            logger.debug(f"Loading batch file: {batch_file}")
            try:
                with open(batch_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                    
                # Extract analysis content (everything after the "---" separator)
                if "---" in content:
                    analysis_content = content.split("---", 1)[1].strip()
                else:
                    analysis_content = content
                
                # Create batch analysis entry
                batch_analysis = {
                    "batch_file": batch_file.name,
                    "analysis": analysis_content,
                    "timestamp": datetime.now().isoformat()
                }
                
                batch_analyses.append(batch_analysis)
                
            except Exception as e:
                logger.warning(f"Failed to load batch file {batch_file}: {str(e)}")
                continue
    
    logger.info(f"Loaded {len(batch_analyses)} batch analyses")
    return batch_analyses


def create_synthesis_content(batch_analyses: List[Dict[str, Any]]) -> str:
    """Create the content for synthesis by combining all batch analyses."""
    logger = logging.getLogger(__name__)
    
    synthesis_parts = []
    synthesis_parts.append("# Individual Segment Analyses\n")
    synthesis_parts.append("Below are the detailed analyses of each gameplay segment:\n\n")
    
    for i, batch in enumerate(batch_analyses, 1):
        # Extract frame range or batch identifier
        if "frame_range" in batch:
            identifier = f"Frames {batch['frame_range']}"
        elif "batch_file" in batch:
            # Try to extract frame range from filename
            filename = batch["batch_file"]
            if "frames_" in filename:
                frame_part = filename.split("frames_")[1].split(".")[0]
                identifier = f"Frames {frame_part}"
            else:
                identifier = f"Segment {i}"
        else:
            identifier = f"Segment {i}"
        
        synthesis_parts.append(f"## {identifier}\n\n")
        synthesis_parts.append(batch["analysis"])
        synthesis_parts.append("\n\n---\n\n")
    
    content = "".join(synthesis_parts)
    logger.debug(f"Created synthesis content with {len(content)} characters")
    
    return content


def synthesize_analyses(analysis_dir: Path, output_file: Path) -> Dict[str, Any]:
    """Synthesize all batch analyses into a final report."""
    logger = logging.getLogger(__name__)
    
    logger.info(f"Starting synthesis of analyses from: {analysis_dir}")
    
    # Load all batch analyses
    batch_analyses = load_batch_analyses(analysis_dir)
    if not batch_analyses:
        raise ValueError(f"No batch analyses found in {analysis_dir}")
    
    # Create synthesis content
    synthesis_content = create_synthesis_content(batch_analyses)
    
    # Get synthesis prompt
    synthesis_prompt = get_synthesis_prompt()
    
    # Combine prompt and content
    full_prompt = f"{synthesis_prompt}\n\n{synthesis_content}"
    
    # Generate synthesis using Gemini
    with SimpleGeminiClient() as gemini_client:
        logger.info("Generating final synthesis report using Gemini")
        
        try:
            final_report = gemini_client.analyze_text_with_prompt(synthesis_content, synthesis_prompt)
            logger.info(f"Synthesis completed ({len(final_report)} characters)")
                
        except Exception as e:
            logger.error(f"Failed to generate synthesis: {str(e)}")
            raise
    
    # Prepare results
    results = {
        "analysis_directory": str(analysis_dir),
        "output_file": str(output_file),
        "total_batch_analyses": len(batch_analyses),
        "synthesis_length": len(final_report),
        "final_report": final_report,
        "timestamp": datetime.now().isoformat()
    }
    
    # Save the final report
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(f"# Final Frame Analysis Report\n\n")
        f.write(f"**Analysis Directory:** {analysis_dir}\n")
        f.write(f"**Total Batch Analyses:** {len(batch_analyses)}\n")
        f.write(f"**Generated:** {results['timestamp']}\n\n")
        f.write("---\n\n")
        f.write(final_report)
        f.write("\n")
    
    logger.info(f"Final report saved to: {output_file}")
    
    # Also save results as JSON
    json_file = output_file.with_suffix('.json')
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Results metadata saved to: {json_file}")
    
    return results


def process_multiple_directories(base_dir: Path, output_dir: Path) -> Dict[str, Any]:
    """Process multiple analysis directories and create final reports for each."""
    logger = logging.getLogger(__name__)
    
    logger.info(f"Processing multiple directories in: {base_dir}")
    
    # Find all analysis directories (those containing batch analysis files)
    analysis_dirs = []
    for subdir in base_dir.iterdir():
        if subdir.is_dir():
            # Check if this directory contains batch analyses
            batch_files = list(subdir.glob("batch_*.md"))
            summary_file = subdir / "batch_analysis_summary.json"
            
            if batch_files or summary_file.exists():
                analysis_dirs.append(subdir)
    
    if not analysis_dirs:
        raise ValueError(f"No analysis directories found in {base_dir}")
    
    logger.info(f"Found {len(analysis_dirs)} analysis directories")
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Process each analysis directory
    all_results = {}
    
    for analysis_dir in analysis_dirs:
        try:
            logger.info(f"Processing analysis directory: {analysis_dir.name}")
            
            # Create output file for this analysis
            output_file = output_dir / f"{analysis_dir.name}_final_report.md"
            
            # Synthesize analyses
            results = synthesize_analyses(analysis_dir, output_file)
            all_results[analysis_dir.name] = results
            
            logger.info(f"Completed synthesis for {analysis_dir.name}")
            
        except Exception as e:
            logger.error(f"Failed to process {analysis_dir.name}: {str(e)}")
            all_results[analysis_dir.name] = {"error": str(e)}
            continue
    
    # Save combined results
    combined_results = {
        "base_directory": str(base_dir),
        "output_directory": str(output_dir),
        "total_directories_processed": len(analysis_dirs),
        "successful_syntheses": len([r for r in all_results.values() if "error" not in r]),
        "failed_syntheses": len([r for r in all_results.values() if "error" in r]),
        "individual_results": all_results,
        "timestamp": datetime.now().isoformat()
    }
    
    combined_file = output_dir / "synthesis_summary.json"
    with open(combined_file, 'w', encoding='utf-8') as f:
        json.dump(combined_results, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Combined results saved to: {combined_file}")
    
    return combined_results


def main():
    """Main function to run the synthesis."""
    parser = argparse.ArgumentParser(description="Synthesize frame batch analyses into final reports")
    parser.add_argument("analysis_dir", type=str, help="Directory containing batch analysis results")
    parser.add_argument("--output-file", type=str, help="Output file for final report")
    parser.add_argument("--output-dir", type=str, help="Output directory (for multiple analysis directories)")
    parser.add_argument("--multiple", action="store_true", help="Process multiple analysis directories")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose logging")
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.verbose)
    logger = logging.getLogger(__name__)
    
    # Validate input directory
    analysis_dir = Path(args.analysis_dir)
    if not analysis_dir.exists():
        logger.error(f"Analysis directory does not exist: {analysis_dir}")
        sys.exit(1)
    
    if not analysis_dir.is_dir():
        logger.error(f"Analysis path is not a directory: {analysis_dir}")
        sys.exit(1)
    
    try:
        if args.multiple:
            # Process multiple analysis directories
            if args.output_dir:
                output_dir = Path(args.output_dir)
            else:
                output_dir = analysis_dir.parent / "final_reports"
            
            results = process_multiple_directories(analysis_dir, output_dir)
            
            logger.info("=" * 50)
            logger.info("SYNTHESIS SUMMARY")
            logger.info("=" * 50)
            logger.info(f"Base directory: {results['base_directory']}")
            logger.info(f"Total directories: {results['total_directories_processed']}")
            logger.info(f"Successful syntheses: {results['successful_syntheses']}")
            logger.info(f"Failed syntheses: {results['failed_syntheses']}")
            logger.info(f"Output directory: {results['output_directory']}")
            
        else:
            # Process single analysis directory
            if args.output_file:
                output_file = Path(args.output_file)
            else:
                output_file = analysis_dir / "final_analysis_report.md"
            
            results = synthesize_analyses(analysis_dir, output_file)
            
            logger.info("=" * 50)
            logger.info("SYNTHESIS SUMMARY")
            logger.info("=" * 50)
            logger.info(f"Analysis directory: {results['analysis_directory']}")
            logger.info(f"Total batch analyses: {results['total_batch_analyses']}")
            logger.info(f"Final report length: {results['synthesis_length']} characters")
            logger.info(f"Output file: {results['output_file']}")
        
        logger.info("Synthesis completed successfully!")
        
    except Exception as e:
        logger.error(f"Synthesis failed: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main() 