#!/usr/bin/env python3
"""
HVA-X Phase 3 Only Runner
Runs only Phase 3 (Meta-Synthesis) of the HVA-X algorithm using Phase 2B results.
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


def run_phase3_only(phase2b_file: str, output_prefix: str = "phase3_synthesis") -> Dict[str, Any]:
    """
    Run only Phase 3 (Meta-Synthesis) of HVA-X algorithm using Phase 2B results.
    
    Args:
        phase2b_file: Path to Phase 2B results JSON file
        output_prefix: Prefix for output files
        
    Returns:
        Dictionary with Phase 3 results
    """
    
    logging.info("🚀 Starting HVA-X Phase 3: Meta-Synthesis")
    logging.info(f"📄 Phase 2B results file: {phase2b_file}")
    
    try:
        # Load Phase 2B results
        logging.info(f"📄 Loading Phase 2B results from: {phase2b_file}")
        with open(phase2b_file, 'r') as f:
            phase2b_results = json.load(f)
        
        # Extract successful guided analyses from Phase 2B
        successful_analyses = {}
        failed_analyses = {}
        
        for tier_name, tier_analyses in phase2b_results["guided_analysis_results"].items():
            successful_analyses[tier_name] = []
            failed_analyses[tier_name] = []
            
            for analysis_result in tier_analyses:
                if "guided_analysis" in analysis_result:
                    successful_analyses[tier_name].append(analysis_result["guided_analysis"])
                else:
                    failed_analyses[tier_name].append(analysis_result)
        
        total_successful = sum(len(analyses) for analyses in successful_analyses.values())
        total_failed = sum(len(analyses) for analyses in failed_analyses.values())
        
        logging.info(f"📊 Found {total_successful} successful guided analyses from Phase 2B")
        logging.info(f"📊 Found {total_failed} failed guided analyses from Phase 2B")
        
        if total_successful == 0:
            logging.warning("⚠️ No successful guided analyses found in Phase 2B results")
            logging.warning("⚠️ Cannot proceed with Phase 3 - meta-synthesis requires guided analyses")
            
            # Return empty results structure
            return {
                "algorithm": "HVA-X",
                "phase": "Phase 3 - Meta-Synthesis",
                "timestamp": datetime.now().isoformat(),
                "input_data": {
                    "phase2b_file": phase2b_file,
                    "total_successful_analyses": 0,
                    "total_failed_analyses": total_failed
                },
                "synthesis_results": {
                    "synthesis_completed": False,
                    "synthesis_length": 0
                },
                "final_report": "",
                "error": "No successful guided analyses found in Phase 2B results"
            }
        
        # Prepare analysis summaries for meta-synthesis
        logging.info("🔄 Preparing guided analyses for meta-synthesis")
        
        # Log tier breakdown
        for tier_name, analyses in successful_analyses.items():
            if analyses:
                total_chars = sum(len(analysis) for analysis in analyses)
                avg_chars = total_chars / len(analyses)
                logging.info(f"  {tier_name}: {len(analyses)} analyses ({total_chars:,} chars, avg: {avg_chars:,.0f})")
            else:
                logging.info(f"  {tier_name}: 0 analyses")
        
        # Run Phase 3: Meta-Synthesis
        logging.info("🔄 Running Phase 3: Meta-Synthesis")
        
        with GeminiClient() as gemini_client:
            try:
                # Perform meta-synthesis using all successful guided analyses
                final_report = gemini_client.meta_synthesis(successful_analyses)
                
                logging.info(f"✅ Meta-synthesis completed ({len(final_report):,} characters)")
                
                # Compile results
                results = {
                    "algorithm": "HVA-X",
                    "phase": "Phase 3 - Meta-Synthesis",
                    "timestamp": datetime.now().isoformat(),
                    "input_data": {
                        "phase2b_file": phase2b_file,
                        "total_successful_analyses": total_successful,
                        "total_failed_analyses": total_failed,
                        "tier_breakdown": {
                            tier: len(analyses) for tier, analyses in successful_analyses.items()
                        }
                    },
                    "synthesis_results": {
                        "synthesis_completed": True,
                        "synthesis_length": len(final_report),
                        "input_analyses_count": total_successful
                    },
                    "final_report": final_report
                }
                
            except Exception as e:
                logging.error(f"❌ Meta-synthesis failed: {e}")
                
                # Return error results structure
                results = {
                    "algorithm": "HVA-X",
                    "phase": "Phase 3 - Meta-Synthesis",
                    "timestamp": datetime.now().isoformat(),
                    "input_data": {
                        "phase2b_file": phase2b_file,
                        "total_successful_analyses": total_successful,
                        "total_failed_analyses": total_failed,
                        "tier_breakdown": {
                            tier: len(analyses) for tier, analyses in successful_analyses.items()
                        }
                    },
                    "synthesis_results": {
                        "synthesis_completed": False,
                        "synthesis_length": 0,
                        "input_analyses_count": total_successful
                    },
                    "final_report": "",
                    "error": str(e)
                }
        
        logging.info("✅ Phase 3 completed successfully!")
        return results
        
    except Exception as e:
        logging.error(f"❌ Phase 3 failed: {e}")
        raise


def save_phase3_results(results: Dict[str, Any], output_file: str):
    """
    Save Phase 3 results to JSON file.
    
    Args:
        results: Phase 3 results dictionary
        output_file: Path to output JSON file
    """
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    logging.info(f"📄 Phase 3 results saved to: {output_file}")


def save_final_report(results: Dict[str, Any], report_file: str):
    """
    Save the final report to a readable markdown file.
    
    Args:
        results: Phase 3 results dictionary
        report_file: Path to output markdown file
    """
    output_path = Path(report_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Create header information
    header = f"""# HVA-X Agent Analysis Report

**Generated:** {results.get('timestamp', 'Unknown')}  
**Algorithm:** {results.get('algorithm', 'HVA-X')}  
**Phase:** {results.get('phase', 'Phase 3 - Meta-Synthesis')}  

## Analysis Summary

- **Input Analyses:** {results.get('input_data', {}).get('total_successful_analyses', 'Unknown')}
- **Failed Analyses:** {results.get('input_data', {}).get('total_failed_analyses', 'Unknown')}
- **Synthesis Status:** {'✅ Completed' if results.get('synthesis_results', {}).get('synthesis_completed', False) else '❌ Failed'}
- **Report Length:** {results.get('synthesis_results', {}).get('synthesis_length', 0):,} characters

### Tier Breakdown
"""
    
    # Add tier breakdown if available
    tier_breakdown = results.get('input_data', {}).get('tier_breakdown', {})
    for tier, count in tier_breakdown.items():
        header += f"- **{tier.replace('_', ' ').title()}:** {count} analyses\n"
    
    header += "\n---\n\n"
    
    # Get the final report
    final_report = results.get("final_report", "")
    
    if not final_report and "error" in results:
        final_report = f"**Error:** {results['error']}\n\nMeta-synthesis could not be completed. Please review the Phase 2B results and ensure successful guided analyses are available."
    
    # Write to file
    with open(output_path, 'w') as f:
        f.write(header)
        f.write(final_report)
    
    logging.info(f"📄 Final report saved to: {report_file}")


def print_phase3_summary(results: Dict[str, Any]):
    """
    Print a summary of Phase 3 results.
    
    Args:
        results: Phase 3 results dictionary
    """
    print("\n" + "="*60)
    print("🎯 HVA-X PHASE 3 COMPLETE")
    print("="*60)
    
    print(f"📊 Input analyses processed: {results['input_data']['total_successful_analyses']}")
    print(f"❌ Failed analyses skipped: {results['input_data']['total_failed_analyses']}")
    print(f"✅ Synthesis completed: {'Yes' if results['synthesis_results']['synthesis_completed'] else 'No'}")
    print(f"📄 Final report length: {results['synthesis_results']['synthesis_length']:,} characters")
    print(f"⏱️  Processing timestamp: {results['timestamp']}")
    
    if "error" in results:
        print(f"\n⚠️  Error: {results['error']}")
        print(f"📋 Next Steps:")
        print(f"   - Review Phase 2B results and fix guided analysis issues")
        print(f"   - Ensure Phase 2B completed successfully before running Phase 3")
        return
    
    print(f"\n🔍 Input Analysis Breakdown:")
    tier_breakdown = results['input_data']['tier_breakdown']
    for tier_name, count in tier_breakdown.items():
        print(f"   {tier_name.replace('_', ' ').title()}: {count} analyses")
    
    if results['synthesis_results']['synthesis_completed']:
        # Show preview of final report
        final_report = results.get('final_report', '')
        if final_report:
            preview_length = min(300, len(final_report))
            preview = final_report[:preview_length]
            if len(final_report) > preview_length:
                preview += "..."
            
            print(f"\n📖 Final Report Preview:")
            print(f"   {preview}")
    
    print(f"\n📋 Next Steps:")
    if results['synthesis_results']['synthesis_completed']:
        print(f"   - Review the comprehensive final report")
        print(f"   - Use insights for agent improvement and development")
        print(f"   - Compare results with other agent analyses")
    else:
        print(f"   - Review Phase 2B results and retry meta-synthesis")
        print(f"   - Ensure sufficient guided analyses are available")


def main():
    """Main function for command-line usage."""
    parser = argparse.ArgumentParser(
        description="Run HVA-X Phase 3 (Meta-Synthesis) using Phase 2B results",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run meta-synthesis using Phase 2B results
  python run_phase3_only.py --phase2b-file phase2b_analysis_20240101_120000.json

  # Run with custom output prefix
  python run_phase3_only.py --phase2b-file phase2b_analysis_20240101_120000.json --output-prefix final_analysis

  # Generate both JSON and markdown report
  python run_phase3_only.py --phase2b-file phase2b_analysis_20240101_120000.json --save-report
        """
    )
    
    parser.add_argument("--phase2b-file", type=str, required=True,
                       help="Path to Phase 2B results JSON file")
    parser.add_argument("--output-prefix", type=str, default="phase3_synthesis",
                       help="Prefix for output files (default: phase3_synthesis)")
    parser.add_argument("--save-report", action="store_true",
                       help="Save final report as markdown file")
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
    if not Path(args.phase2b_file).exists():
        print(f"❌ Error: Phase 2B file not found: {args.phase2b_file}")
        sys.exit(1)
    
    try:
        # Run Phase 3
        results = run_phase3_only(
            phase2b_file=args.phase2b_file,
            output_prefix=args.output_prefix
        )
        
        # Generate output filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        json_output = f"{args.output_prefix}_{timestamp}.json"
        
        # Save results
        save_phase3_results(results, json_output)
        
        # Save final report if requested
        if args.save_report:
            report_output = f"{args.output_prefix}_report_{timestamp}.md"
            save_final_report(results, report_output)
            print(f"📄 Final report saved to: {report_output}")
        
        # Print summary
        print_phase3_summary(results)
        
        print(f"\n🎉 Phase 3 completed successfully!")
        print(f"📄 Detailed results saved to: {json_output}")
        
        if results['synthesis_results']['synthesis_completed']:
            print(f"🎯 HVA-X analysis pipeline complete!")
            print(f"📖 Final agent analysis report ready for review")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main() 