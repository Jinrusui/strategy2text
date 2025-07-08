#!/usr/bin/env python3
"""
HVA-X Phase 2B Only Runner
Runs only Phase 2B (Guided Analysis) of the HVA-X algorithm using Phase 2A results.
"""

import sys
import argparse
import logging
import json
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any, List

# Add src to path for imports
sys.path.append(str(Path(__file__).parent / "src"))

from gemini_analysis import GeminiClient
from video_processing import VideoLoader, TrajectoryData


def run_phase2b_only(phase2a_file: str, output_prefix: str = "phase2b_analysis") -> Dict[str, Any]:
    """
    Run only Phase 2B (Guided Analysis) of HVA-X algorithm using Phase 2A results.
    
    Args:
        phase2a_file: Path to Phase 2A results JSON file
        output_prefix: Prefix for output files
        
    Returns:
        Dictionary with Phase 2B results
    """
    
    logging.info("🚀 Starting HVA-X Phase 2B: Guided Analysis")
    logging.info(f"📄 Phase 2A results file: {phase2a_file}")
    
    try:
        # Load Phase 2A results
        logging.info(f"📄 Loading Phase 2A results from: {phase2a_file}")
        with open(phase2a_file, 'r') as f:
            phase2a_results = json.load(f)
        
        # Extract successful event detections from Phase 2A
        successful_detections = {}
        failed_detections = {}
        
        for tier_name, tier_events in phase2a_results["event_detection_results"].items():
            successful_detections[tier_name] = []
            failed_detections[tier_name] = []
            
            for event_result in tier_events:
                if "key_events" in event_result:
                    successful_detections[tier_name].append(event_result)
                else:
                    failed_detections[tier_name].append(event_result)
        
        total_successful = sum(len(events) for events in successful_detections.values())
        total_failed = sum(len(events) for events in failed_detections.values())
        
        logging.info(f"📊 Found {total_successful} successful event detections from Phase 2A")
        logging.info(f"📊 Found {total_failed} failed event detections from Phase 2A")
        
        if total_successful == 0:
            logging.warning("⚠️ No successful event detections found in Phase 2A results")
            logging.warning("⚠️ Cannot proceed with Phase 2B - guided analysis requires detected events")
            
            # Return empty results structure
            return {
                "algorithm": "HVA-X",
                "phase": "Phase 2B - Guided Analysis",
                "timestamp": datetime.now().isoformat(),
                "input_data": {
                    "phase2a_file": phase2a_file,
                    "total_successful_detections": 0,
                    "total_failed_detections": total_failed
                },
                "analysis_results": {
                    "successful_analyses": 0,
                    "failed_analyses": 0,
                    "success_rate": 0.0
                },
                "guided_analysis_results": {},
                "error": "No successful event detections found in Phase 2A results"
            }
        
        # Run Phase 2B: Guided Analysis
        logging.info("🔄 Running Phase 2B: Guided Analysis")
        
        with GeminiClient() as gemini_client:
            guided_analysis_results = {}
            
            for tier_name, tier_detections in successful_detections.items():
                logging.info(f"  Processing {tier_name} tier ({len(tier_detections)} videos)")
                tier_analyses = []
                
                for i, detection_result in enumerate(tier_detections):
                    trajectory_info = detection_result["trajectory"]
                    key_events = detection_result["key_events"]
                    
                    logging.info(f"    Video {i+1}/{len(tier_detections)}: {trajectory_info['episode_id']}")
                    logging.info(f"      Using {len(key_events)} detected events")
                    
                    try:
                        # Perform guided analysis using detected events
                        detailed_analysis = gemini_client.guided_analysis(
                            trajectory_info["video_path"], 
                            key_events
                        )
                        
                        # Store guided analysis result
                        analysis_result = {
                            "trajectory": trajectory_info,
                            "phase2a_events": key_events,
                            "phase2a_timestamp": detection_result["timestamp"],
                            "guided_analysis": detailed_analysis,
                            "analysis_length": len(detailed_analysis),
                            "timestamp": datetime.now().isoformat()
                        }
                        
                        tier_analyses.append(analysis_result)
                        logging.info(f"      Guided analysis completed ({len(detailed_analysis)} chars)")
                        
                    except Exception as e:
                        logging.error(f"      Failed guided analysis for {trajectory_info['episode_id']}: {e}")
                        # Store error result
                        error_result = {
                            "trajectory": trajectory_info,
                            "phase2a_events": key_events,
                            "phase2a_timestamp": detection_result["timestamp"],
                            "error": str(e),
                            "timestamp": datetime.now().isoformat()
                        }
                        tier_analyses.append(error_result)
                
                guided_analysis_results[tier_name] = tier_analyses
        
        # Calculate statistics
        total_processed = sum(len(analyses) for analyses in guided_analysis_results.values())
        successful_analyses = sum(
            len([a for a in analyses if "guided_analysis" in a]) 
            for analyses in guided_analysis_results.values()
        )
        failed_analyses = total_processed - successful_analyses
        
        # Compile results
        results = {
            "algorithm": "HVA-X",
            "phase": "Phase 2B - Guided Analysis",
            "timestamp": datetime.now().isoformat(),
            "input_data": {
                "phase2a_file": phase2a_file,
                "total_successful_detections": total_successful,
                "total_failed_detections": total_failed
            },
            "analysis_results": {
                "successful_analyses": successful_analyses,
                "failed_analyses": failed_analyses,
                "success_rate": successful_analyses / total_processed if total_processed > 0 else 0
            },
            "guided_analysis_results": guided_analysis_results
        }
        
        logging.info("✅ Phase 2B completed successfully!")
        return results
        
    except Exception as e:
        logging.error(f"❌ Phase 2B failed: {e}")
        raise


def save_phase2b_results(results: Dict[str, Any], output_file: str):
    """
    Save Phase 2B results to JSON file.
    
    Args:
        results: Phase 2B results dictionary
        output_file: Path to output JSON file
    """
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    logging.info(f"📄 Phase 2B results saved to: {output_file}")


def print_phase2b_summary(results: Dict[str, Any]):
    """
    Print a summary of Phase 2B results.
    
    Args:
        results: Phase 2B results dictionary
    """
    print("\n" + "="*60)
    print("🎯 HVA-X PHASE 2B COMPLETE")
    print("="*60)
    
    print(f"📊 Total videos processed: {results['input_data']['total_successful_detections']}")
    print(f"✅ Successful guided analyses: {results['analysis_results']['successful_analyses']}")
    print(f"❌ Failed guided analyses: {results['analysis_results']['failed_analyses']}")
    print(f"📈 Success rate: {results['analysis_results']['success_rate']:.1%}")
    print(f"⏱️  Processing timestamp: {results['timestamp']}")
    
    if "error" in results:
        print(f"\n⚠️  Warning: {results['error']}")
        print(f"📋 Next Steps:")
        print(f"   - Review Phase 2A results and fix event detection issues")
        print(f"   - Ensure Phase 2A completed successfully before running Phase 2B")
        return
    
    print(f"\n🔍 Guided Analysis by Tier:")
    for tier_name, tier_analyses in results['guided_analysis_results'].items():
        successful = [a for a in tier_analyses if "guided_analysis" in a]
        failed = [a for a in tier_analyses if "error" in a]
        
        print(f"   {tier_name.replace('_', ' ').title()}: {len(tier_analyses)} videos")
        print(f"     ✅ Successful: {len(successful)}")
        print(f"     ❌ Failed: {len(failed)}")
        
        if successful:
            total_length = sum(a["analysis_length"] for a in successful)
            avg_length = total_length / len(successful)
            print(f"     📊 Total analysis text: {total_length:,} characters")
            print(f"     📊 Average analysis length: {avg_length:,.0f} characters")
            
            # Show sample analysis preview from first successful video
            sample_analysis = successful[0]["guided_analysis"][:200] + "..."
            print(f"     📋 Sample analysis from {successful[0]['trajectory']['episode_id']}:")
            print(f"       {sample_analysis}")
    
    print(f"\n📋 Next Steps:")
    print(f"   - Use guided analyses for Phase 3 (Meta-Synthesis)")
    print(f"   - Run full HVA-X analysis with: python run_hva_analysis.py")
    print(f"   - Or continue with manual analysis of guided results")


def main():
    """Main function for command-line usage."""
    parser = argparse.ArgumentParser(
        description="Run HVA-X Phase 2B (Guided Analysis) using Phase 2A results",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run guided analysis using Phase 2A results
  python run_phase2b_only.py --phase2a-file phase2a_events_20240101_120000.json

  # Run with custom output prefix
  python run_phase2b_only.py --phase2a-file phase2a_events_20240101_120000.json --output-prefix custom_analysis
        """
    )
    
    parser.add_argument("--phase2a-file", type=str, required=True,
                       help="Path to Phase 2A results JSON file")
    parser.add_argument("--output-prefix", type=str, default="phase2b_analysis",
                       help="Prefix for output files (default: phase2b_analysis)")
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
    if not Path(args.phase2a_file).exists():
        print(f"❌ Error: Phase 2A file not found: {args.phase2a_file}")
        sys.exit(1)
    
    try:
        # Run Phase 2B
        results = run_phase2b_only(
            phase2a_file=args.phase2a_file,
            output_prefix=args.output_prefix
        )
        
        # Generate output filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"{args.output_prefix}_{timestamp}.json"
        
        # Save results
        save_phase2b_results(results, output_file)
        
        # Print summary
        print_phase2b_summary(results)
        
        print(f"\n🎉 Phase 2B completed successfully!")
        print(f"📄 Detailed results saved to: {output_file}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main() 