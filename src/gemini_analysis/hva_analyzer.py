"""
HVA-X (Hierarchical Video Analysis for Agent Explainability) Main Analyzer.
Implements the complete multi-pass video analysis algorithm for RL agent behavior summarization.
"""

import json
import logging
from typing import Dict, List, Optional, Any
from pathlib import Path
from datetime import datetime

from .gemini_client import GeminiClient
from video_processing import VideoLoader, TrajectoryData


class HVAAnalyzer:
    """Main analyzer implementing the complete HVA-X algorithm for RL agent video analysis."""
    
    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None):
        """
        Initialize the HVA analyzer.
        
        Args:
            api_key: Google API key for Gemini
            model: Model to use for analysis. If None, uses default model
        """
        self.gemini_client = GeminiClient(api_key=api_key, model=model)
        self.video_loader = VideoLoader()
        self.logger = logging.getLogger(__name__)
    
    def analyze_agent_from_trajectories(self, trajectories: List[TrajectoryData], 
                                      samples_per_tier: int = 3) -> Dict[str, Any]:
        """
        Run the complete HVA-X algorithm on agent trajectories.
        
        Args:
            trajectories: List of trajectory data with videos and scores
            samples_per_tier: Number of samples per performance tier
            
        Returns:
            Complete analysis report
        """
        self.logger.info("Starting HVA-X algorithm for agent analysis")
        
        # Phase 1: Trajectory Sampling and Stratification
        sampled_trajectories = self._phase1_trajectory_sampling(trajectories, samples_per_tier)
        
        # Phase 2: Individual Trajectory Analysis (Two-Pass Method)
        individual_analyses = self._phase2_individual_analysis(sampled_trajectories)
        
        # Phase 3: Meta-Summary Synthesis
        final_report = self._phase3_meta_synthesis(individual_analyses)
        
        # Compile complete results
        return {
            "algorithm": "HVA-X",
            "timestamp": datetime.now().isoformat(),
            "phase1_sampling": {
                "total_trajectories": len(trajectories),
                "sampled_trajectories": {
                    tier: len(trajs) for tier, trajs in sampled_trajectories.items()
                }
            },
            "phase2_individual_analyses": individual_analyses,
            "phase3_final_report": final_report,
            "summary": {
                "total_videos_analyzed": sum(len(trajs) for trajs in sampled_trajectories.values()),
                "analysis_complete": True
            }
        }
    
    def analyze_agent_from_directory(self, video_dir: str, score_file: str, 
                                   samples_per_tier: int = 3) -> Dict[str, Any]:
        """
        Run HVA-X analysis on agent videos from a directory.
        
        Args:
            video_dir: Directory containing video files
            score_file: Path to file containing episode scores
            samples_per_tier: Number of samples per performance tier
            
        Returns:
            Complete analysis report
        """
        # Load trajectory data
        trajectories = self.video_loader.load_trajectory_data_from_files(video_dir, score_file)
        
        # Run analysis
        return self.analyze_agent_from_trajectories(trajectories, samples_per_tier)
    
    def analyze_agent_from_csv(self, csv_file: str, samples_per_tier: int = 3) -> Dict[str, Any]:
        """
        Run HVA-X analysis on agent data from CSV file.
        
        Args:
            csv_file: Path to CSV file with trajectory data
            samples_per_tier: Number of samples per performance tier
            
        Returns:
            Complete analysis report
        """
        # Load trajectory data
        trajectories = self.video_loader.load_trajectory_data_from_csv(csv_file)
        
        # Run analysis
        return self.analyze_agent_from_trajectories(trajectories, samples_per_tier)
    
    def _phase1_trajectory_sampling(self, trajectories: List[TrajectoryData], 
                                  samples_per_tier: int) -> Dict[str, List[TrajectoryData]]:
        """
        Phase 1: Trajectory Sampling and Stratification.
        
        Args:
            trajectories: List of trajectory data
            samples_per_tier: Number of samples per tier
            
        Returns:
            Dictionary with sampled trajectories by tier
        """
        self.logger.info("Phase 1: Trajectory Sampling and Stratification")
        
        # Use video loader's trajectory processing
        sampled_trajectories = self.video_loader.prepare_trajectories_for_hva(
            trajectories, samples_per_tier
        )
        
        # Log sampling results
        for tier, trajs in sampled_trajectories.items():
            self.logger.info(f"  {tier}: {len(trajs)} trajectories")
            for traj in trajs:
                self.logger.debug(f"    - {traj.episode_id}: {traj.score}")
        
        return sampled_trajectories
    
    def _phase2_individual_analysis(self, sampled_trajectories: Dict[str, List[TrajectoryData]]) -> Dict[str, Any]:
        """
        Phase 2: Individual Trajectory Analysis using Two-Pass Method.
        
        Args:
            sampled_trajectories: Dictionary of sampled trajectories by tier
            
        Returns:
            Dictionary with individual analysis results
        """
        self.logger.info("Phase 2: Individual Trajectory Analysis (Two-Pass Method)")
        
        analyses = {}
        
        for tier, trajectories in sampled_trajectories.items():
            self.logger.info(f"  Analyzing {tier} ({len(trajectories)} videos)")
            tier_analyses = []
            
            for i, trajectory in enumerate(trajectories):
                self.logger.info(f"    Video {i+1}/{len(trajectories)}: {trajectory.episode_id}")
                
                try:
                    # Pass 2a: Event Identification
                    self.logger.info("      Pass 2a: Event Detection")
                    key_events = self.gemini_client.detect_key_events(trajectory.video_path)
                    
                    # Pass 2b: Guided Analysis
                    self.logger.info("      Pass 2b: Guided Analysis")
                    detailed_analysis = self.gemini_client.guided_analysis(
                        trajectory.video_path, key_events
                    )
                    
                    # Store analysis result
                    analysis_result = {
                        "trajectory": {
                            "episode_id": trajectory.episode_id,
                            "video_path": trajectory.video_path,
                            "score": trajectory.score,
                            "tier": tier
                        },
                        "pass_2a_events": key_events,
                        "pass_2b_analysis": detailed_analysis,
                        "timestamp": datetime.now().isoformat()
                    }
                    
                    tier_analyses.append(analysis_result)
                    self.logger.info(f"      Analysis completed for {trajectory.episode_id}")
                    
                except Exception as e:
                    self.logger.error(f"      Failed to analyze {trajectory.episode_id}: {e}")
                    # Store error result
                    error_result = {
                        "trajectory": {
                            "episode_id": trajectory.episode_id,
                            "video_path": trajectory.video_path,
                            "score": trajectory.score,
                            "tier": tier
                        },
                        "error": str(e),
                        "timestamp": datetime.now().isoformat()
                    }
                    tier_analyses.append(error_result)
            
            analyses[tier] = tier_analyses
        
        return analyses
    
    def _phase3_meta_synthesis(self, individual_analyses: Dict[str, Any]) -> str:
        """
        Phase 3: Meta-Summary Synthesis.
        
        Args:
            individual_analyses: Dictionary with individual analysis results
            
        Returns:
            Final synthesized report
        """
        self.logger.info("Phase 3: Meta-Summary Synthesis")
        
        # Extract successful analyses with complete context from individual analyses
        analysis_data = {}
        
        for tier, analyses in individual_analyses.items():
            analysis_data[tier] = []
            
            for analysis in analyses:
                if "pass_2b_analysis" in analysis:
                    # Convert to format expected by meta_synthesis
                    formatted_analysis = {
                        "trajectory": analysis["trajectory"],
                        "phase2a_events": analysis["pass_2a_events"],
                        "guided_analysis": analysis["pass_2b_analysis"],
                        "timestamp": analysis["timestamp"]
                    }
                    analysis_data[tier].append(formatted_analysis)
                elif "error" in analysis:
                    # Skip failed analyses for synthesis
                    self.logger.warning(f"Skipping failed analysis for synthesis: {analysis['error']}")
        
        # Perform meta-synthesis with complete analysis data
        final_report = self.gemini_client.meta_synthesis(analysis_data)
        
        self.logger.info("Phase 3: Meta-synthesis completed")
        return final_report
    
    def save_analysis(self, analysis_results: Dict[str, Any], output_path: str):
        """
        Save complete analysis results to file.
        
        Args:
            analysis_results: Complete analysis results from HVA-X
            output_path: Path to output file
        """
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w') as f:
            json.dump(analysis_results, f, indent=2, default=str)
        
        self.logger.info(f"Analysis results saved to {output_file}")
    
    def save_final_report(self, analysis_results: Dict[str, Any], output_path: str):
        """
        Save just the final report (Phase 3) to a readable text file.
        
        Args:
            analysis_results: Complete analysis results from HVA-X
            output_path: Path to output text file
        """
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Extract final report
        final_report = analysis_results.get("phase3_final_report", "")
        
        # Add header information
        header = f"""
# HVA-X Agent Analysis Report

Generated: {analysis_results.get('timestamp', 'Unknown')}
Algorithm: {analysis_results.get('algorithm', 'HVA-X')}

## Analysis Summary
- Total Trajectories: {analysis_results.get('phase1_sampling', {}).get('total_trajectories', 'Unknown')}
- Videos Analyzed: {analysis_results.get('summary', {}).get('total_videos_analyzed', 'Unknown')}
- Sampling: {analysis_results.get('phase1_sampling', {}).get('sampled_trajectories', {})}

---

"""
        
        # Write to file
        with open(output_file, 'w') as f:
            f.write(header)
            f.write(final_report)
        
        self.logger.info(f"Final report saved to {output_file}")
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.gemini_client.cleanup_uploaded_files()


def main():
    """Example usage of the HVA analyzer."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Run HVA-X analysis on RL agent videos")
    parser.add_argument("--video_dir", type=str, help="Directory containing video files")
    parser.add_argument("--score_file", type=str, help="File containing episode scores")
    parser.add_argument("--csv_file", type=str, help="CSV file with trajectory data")
    parser.add_argument("--samples_per_tier", type=int, default=3, help="Samples per performance tier")
    parser.add_argument("--output", type=str, default="hva_analysis.json", help="Output file path")
    parser.add_argument("--report", type=str, default="hva_report.md", help="Final report file path")
    
    args = parser.parse_args()
    
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    # Run analysis
    with HVAAnalyzer() as analyzer:
        if args.csv_file:
            results = analyzer.analyze_agent_from_csv(args.csv_file, args.samples_per_tier)
        elif args.video_dir and args.score_file:
            results = analyzer.analyze_agent_from_directory(
                args.video_dir, args.score_file, args.samples_per_tier
            )
        else:
            print("Error: Must provide either --csv_file or both --video_dir and --score_file")
            return
        
        # Save results
        analyzer.save_analysis(results, args.output)
        analyzer.save_final_report(results, args.report)
        
        print(f"Analysis complete! Results saved to {args.output}")
        print(f"Final report saved to {args.report}")


if __name__ == "__main__":
    main() 