#!/usr/bin/env python3
"""
HVA-X Complete Pipeline Runner
Runs all four phases of the HVA-X algorithm in sequence to form a complete analysis pipeline.
"""

import sys
import argparse
import logging
import json
import os
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any, List

# Add src to path for imports
sys.path.append(str(Path(__file__).parent / "src"))

# Import phase functions
from run_phase1_only import run_phase1_only, save_phase1_results
from run_phase2a_only import run_phase2a_only, save_phase2a_results
from run_phase2b_only import run_phase2b_only, save_phase2b_results
from run_phase3_only import run_phase3_only, save_phase3_results, save_final_report


def run_hva_pipeline(
    video_dir: Optional[str] = None,
    score_file: Optional[str] = None,
    csv_file: Optional[str] = None,
    samples_per_tier: int = 3,
    output_prefix: str = "hva_pipeline",
    save_intermediate: bool = True,
    save_final_report: bool = True,
    cleanup_intermediate: bool = False
) -> Dict[str, Any]:
    """
    Run the complete HVA-X pipeline (all four phases) in sequence.
    
    Args:
        video_dir: Directory containing video files
        score_file: Path to scores.txt file
        csv_file: Path to trajectory_data.csv file
        samples_per_tier: Number of samples per performance tier
        output_prefix: Prefix for output files
        save_intermediate: Save intermediate results from each phase
        save_final_report: Save final report as markdown file
        cleanup_intermediate: Remove intermediate files after completion
        
    Returns:
        Dictionary with complete pipeline results
    """
    
    pipeline_start_time = datetime.now()
    timestamp = pipeline_start_time.strftime("%Y%m%d_%H%M%S")
    
    logging.info("🚀 Starting HVA-X Complete Pipeline")
    logging.info("="*60)
    
    # Initialize results structure
    pipeline_results = {
        "pipeline_info": {
            "timestamp": pipeline_start_time.isoformat(),
            "output_prefix": output_prefix,
            "samples_per_tier": samples_per_tier,
            "phases_completed": [],
            "intermediate_files": [],
            "final_files": []
        },
        "phase_results": {},
        "pipeline_summary": {}
    }
    
    intermediate_files = []
    
    try:
        # ================================
        # PHASE 1: Trajectory Sampling and Stratification
        # ================================
        logging.info("🔄 PHASE 1: Trajectory Sampling and Stratification")
        logging.info("-" * 50)
        
        phase1_start = datetime.now()
        
        phase1_results = run_phase1_only(
            video_dir=video_dir,
            score_file=score_file,
            csv_file=csv_file,
            samples_per_tier=samples_per_tier,
            output_prefix=f"{output_prefix}_phase1"
        )
        
        phase1_duration = datetime.now() - phase1_start
        pipeline_results["phase_results"]["phase1"] = phase1_results
        pipeline_results["pipeline_info"]["phases_completed"].append("Phase 1")
        
        # Save Phase 1 results
        phase1_file = f"{output_prefix}_phase1_{timestamp}.json"
        save_phase1_results(phase1_results, phase1_file)
        intermediate_files.append(phase1_file)
        
        logging.info(f"✅ Phase 1 completed in {phase1_duration}")
        logging.info(f"📄 Phase 1 results saved to: {phase1_file}")
        
        # ================================
        # PHASE 2A: Event Detection
        # ================================
        logging.info("\n🔄 PHASE 2A: Event Detection")
        logging.info("-" * 50)
        
        phase2a_start = datetime.now()
        
        phase2a_results = run_phase2a_only(
            phase1_file=phase1_file,
            samples_per_tier=samples_per_tier,
            output_prefix=f"{output_prefix}_phase2a"
        )
        
        phase2a_duration = datetime.now() - phase2a_start
        pipeline_results["phase_results"]["phase2a"] = phase2a_results
        pipeline_results["pipeline_info"]["phases_completed"].append("Phase 2A")
        
        # Save Phase 2A results
        phase2a_file = f"{output_prefix}_phase2a_{timestamp}.json"
        save_phase2a_results(phase2a_results, phase2a_file)
        intermediate_files.append(phase2a_file)
        
        logging.info(f"✅ Phase 2A completed in {phase2a_duration}")
        logging.info(f"📄 Phase 2A results saved to: {phase2a_file}")
        
        # ================================
        # PHASE 2B: Guided Analysis
        # ================================
        logging.info("\n🔄 PHASE 2B: Guided Analysis")
        logging.info("-" * 50)
        
        phase2b_start = datetime.now()
        
        phase2b_results = run_phase2b_only(
            phase2a_file=phase2a_file,
            output_prefix=f"{output_prefix}_phase2b"
        )
        
        phase2b_duration = datetime.now() - phase2b_start
        pipeline_results["phase_results"]["phase2b"] = phase2b_results
        pipeline_results["pipeline_info"]["phases_completed"].append("Phase 2B")
        
        # Save Phase 2B results
        phase2b_file = f"{output_prefix}_phase2b_{timestamp}.json"
        save_phase2b_results(phase2b_results, phase2b_file)
        intermediate_files.append(phase2b_file)
        
        logging.info(f"✅ Phase 2B completed in {phase2b_duration}")
        logging.info(f"📄 Phase 2B results saved to: {phase2b_file}")
        
        # ================================
        # PHASE 3: Meta-Synthesis
        # ================================
        logging.info("\n🔄 PHASE 3: Meta-Synthesis")
        logging.info("-" * 50)
        
        phase3_start = datetime.now()
        
        phase3_results = run_phase3_only(
            phase2b_file=phase2b_file,
            output_prefix=f"{output_prefix}_phase3"
        )
        
        phase3_duration = datetime.now() - phase3_start
        pipeline_results["phase_results"]["phase3"] = phase3_results
        pipeline_results["pipeline_info"]["phases_completed"].append("Phase 3")
        
        # Save Phase 3 results
        phase3_file = f"{output_prefix}_phase3_{timestamp}.json"
        save_phase3_results(phase3_results, phase3_file)
        
        # Save final report
        final_report_file = f"{output_prefix}_final_report_{timestamp}.md"
        if save_final_report:
            save_final_report(phase3_results, final_report_file)
            pipeline_results["pipeline_info"]["final_files"].append(final_report_file)
        
        pipeline_results["pipeline_info"]["final_files"].append(phase3_file)
        
        logging.info(f"✅ Phase 3 completed in {phase3_duration}")
        logging.info(f"📄 Phase 3 results saved to: {phase3_file}")
        if save_final_report:
            logging.info(f"📄 Final report saved to: {final_report_file}")
        
        # ================================
        # PIPELINE COMPLETION
        # ================================
        pipeline_duration = datetime.now() - pipeline_start_time
        
        # Calculate pipeline summary
        pipeline_results["pipeline_summary"] = {
            "total_duration": str(pipeline_duration),
            "phase_durations": {
                "phase1": str(phase1_duration),
                "phase2a": str(phase2a_duration),
                "phase2b": str(phase2b_duration),
                "phase3": str(phase3_duration)
            },
            "total_trajectories_loaded": phase1_results["input_data"]["total_trajectories"],
            "total_trajectories_sampled": phase1_results["sampling_results"]["total_sampled"],
            "successful_event_detections": phase2a_results["detection_results"]["successful_detections"],
            "successful_guided_analyses": phase2b_results["analysis_results"]["successful_analyses"],
            "synthesis_completed": phase3_results["synthesis_results"]["synthesis_completed"],
            "final_report_length": phase3_results["synthesis_results"]["synthesis_length"]
        }
        
        # Store intermediate files info
        if save_intermediate:
            pipeline_results["pipeline_info"]["intermediate_files"] = intermediate_files
        
        # Clean up intermediate files if requested
        if cleanup_intermediate:
            logging.info("\n🧹 Cleaning up intermediate files...")
            for file_path in intermediate_files:
                try:
                    os.remove(file_path)
                    logging.info(f"   Removed: {file_path}")
                except Exception as e:
                    logging.warning(f"   Failed to remove {file_path}: {e}")
        
        logging.info("\n" + "="*60)
        logging.info("🎉 HVA-X PIPELINE COMPLETED SUCCESSFULLY!")
        logging.info("="*60)
        
        return pipeline_results
        
    except Exception as e:
        logging.error(f"❌ Pipeline failed: {e}")
        
        # Add error information to results
        pipeline_results["pipeline_summary"]["error"] = str(e)
        pipeline_results["pipeline_summary"]["failed_at"] = pipeline_results["pipeline_info"]["phases_completed"][-1] if pipeline_results["pipeline_info"]["phases_completed"] else "Initialization"
        
        raise


def save_pipeline_results(results: Dict[str, Any], output_file: str):
    """
    Save complete pipeline results to JSON file.
    
    Args:
        results: Complete pipeline results dictionary
        output_file: Path to output JSON file
    """
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    logging.info(f"📄 Complete pipeline results saved to: {output_file}")


def print_pipeline_summary(results: Dict[str, Any]):
    """
    Print a comprehensive summary of the pipeline execution.
    
    Args:
        results: Complete pipeline results dictionary
    """
    print("\n" + "="*70)
    print("🎯 HVA-X COMPLETE PIPELINE SUMMARY")
    print("="*70)
    
    # Basic info
    summary = results["pipeline_summary"]
    info = results["pipeline_info"]
    
    print(f"⏱️  Total execution time: {summary['total_duration']}")
    print(f"📊 Phases completed: {len(info['phases_completed'])}/4")
    print(f"🔗 Phases: {' → '.join(info['phases_completed'])}")
    
    if "error" in summary:
        print(f"❌ Pipeline failed at: {summary['failed_at']}")
        print(f"❌ Error: {summary['error']}")
        return
    
    # Phase timing breakdown
    print(f"\n⏰ Phase Execution Times:")
    durations = summary["phase_durations"]
    print(f"   Phase 1 (Sampling): {durations['phase1']}")
    print(f"   Phase 2A (Event Detection): {durations['phase2a']}")
    print(f"   Phase 2B (Guided Analysis): {durations['phase2b']}")
    print(f"   Phase 3 (Meta-Synthesis): {durations['phase3']}")
    
    # Data flow summary
    print(f"\n📊 Data Flow Summary:")
    print(f"   Total trajectories loaded: {summary['total_trajectories_loaded']}")
    print(f"   Total trajectories sampled: {summary['total_trajectories_sampled']}")
    print(f"   Successful event detections: {summary['successful_event_detections']}")
    print(f"   Successful guided analyses: {summary['successful_guided_analyses']}")
    print(f"   Synthesis completed: {'✅ Yes' if summary['synthesis_completed'] else '❌ No'}")
    print(f"   Final report length: {summary['final_report_length']:,} characters")
    
    # File outputs
    print(f"\n📄 Output Files:")
    if info.get("intermediate_files"):
        print(f"   Intermediate files: {len(info['intermediate_files'])}")
        for file in info["intermediate_files"]:
            print(f"     - {file}")
    
    print(f"   Final files: {len(info['final_files'])}")
    for file in info["final_files"]:
        print(f"     - {file}")
    
    # Success metrics
    if summary["synthesis_completed"]:
        success_rate_2a = (summary["successful_event_detections"] / summary["total_trajectories_sampled"]) * 100
        success_rate_2b = (summary["successful_guided_analyses"] / summary["successful_event_detections"]) * 100 if summary["successful_event_detections"] > 0 else 0
        
        print(f"\n📈 Success Metrics:")
        print(f"   Event detection success rate: {success_rate_2a:.1f}%")
        print(f"   Guided analysis success rate: {success_rate_2b:.1f}%")
        print(f"   Overall pipeline success: ✅ Complete")
    
    print(f"\n🎯 Pipeline Status: {'✅ COMPLETE' if summary['synthesis_completed'] else '❌ INCOMPLETE'}")


def main():
    """Main function for command-line usage."""
    parser = argparse.ArgumentParser(
        description="Run the complete HVA-X pipeline (all four phases) in sequence",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run complete pipeline with auto-detected files
  python run_hva_pipeline.py --video-dir hva_videos

  # Run with specific configuration
  python run_hva_pipeline.py --video-dir videos/ --score-file scores.txt --samples-per-tier 5

  # Run from CSV file with custom output
  python run_hva_pipeline.py --csv-file trajectory_data.csv --output-prefix my_analysis

  # Run with cleanup of intermediate files
  python run_hva_pipeline.py --video-dir hva_videos --cleanup-intermediate

  # Run without saving intermediate files
  python run_hva_pipeline.py --video-dir hva_videos --no-save-intermediate
        """
    )
    
    parser.add_argument("--video-dir", type=str,
                       help="Directory containing video files")
    parser.add_argument("--score-file", type=str,
                       help="Path to scores.txt file")
    parser.add_argument("--csv-file", type=str,
                       help="Path to trajectory_data.csv file")
    parser.add_argument("--samples-per-tier", type=int, default=3,
                       help="Number of samples per performance tier (default: 3)")
    parser.add_argument("--output-prefix", type=str, default="hva_pipeline",
                       help="Prefix for output files (default: hva_pipeline)")
    parser.add_argument("--no-save-intermediate", action="store_true",
                       help="Don't save intermediate phase results")
    parser.add_argument("--no-final-report", action="store_true",
                       help="Don't save final report as markdown")
    parser.add_argument("--cleanup-intermediate", action="store_true",
                       help="Remove intermediate files after completion")
    parser.add_argument("--verbose", action="store_true",
                       help="Enable verbose logging")
    
    args = parser.parse_args()
    
    # Configure logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # Validate arguments
    if not args.csv_file and not args.video_dir:
        parser.error("Must provide either --video-dir or --csv-file")
    
    try:
        # Run complete pipeline
        results = run_hva_pipeline(
            video_dir=args.video_dir,
            score_file=args.score_file,
            csv_file=args.csv_file,
            samples_per_tier=args.samples_per_tier,
            output_prefix=args.output_prefix,
            save_intermediate=not args.no_save_intermediate,
            save_final_report=not args.no_final_report,
            cleanup_intermediate=args.cleanup_intermediate
        )
        
        # Generate pipeline results filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        pipeline_results_file = f"{args.output_prefix}_complete_{timestamp}.json"
        
        # Save complete pipeline results
        save_pipeline_results(results, pipeline_results_file)
        
        # Print comprehensive summary
        print_pipeline_summary(results)
        
        print(f"\n🎉 Complete HVA-X pipeline finished successfully!")
        print(f"📄 Complete results saved to: {pipeline_results_file}")
        
        if results["pipeline_summary"]["synthesis_completed"]:
            print(f"🎯 Your comprehensive agent analysis is ready!")
            final_report = [f for f in results["pipeline_info"]["final_files"] if f.endswith('.md')]
            if final_report:
                print(f"📖 Read the final report: {final_report[0]}")
        
    except Exception as e:
        print(f"❌ Pipeline Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main() 