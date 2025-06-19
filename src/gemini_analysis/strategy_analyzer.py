"""
Strategy Analyzer Module

Main orchestrator for the complete strategy analysis pipeline.
Integrates video sampling, Gemini analysis, and evaluation metrics.
"""

import os
import json
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import logging
from datetime import datetime

from .gemini_client import GeminiClient
from .prompt_engineering import PromptEngineer, AnalysisType
from .evaluation_metrics import EvaluationMetrics
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from video_processing.video_sampler import VideoSampler, SamplingStrategy


class StrategyAnalyzer:
    """
    Main strategy analyzer that orchestrates the complete analysis pipeline.
    
    Implements the methodology described in the dissertation:
    1. Video sampling (typical, edge cases, longitudinal)
    2. Strategy analysis using Gemini
    3. Evaluation using PFS, Coverage, and Abstraction metrics
    """
    
    def __init__(
        self,
        video_dir: str,
        api_key: Optional[str] = None,
        output_dir: str = "analysis_results",
        model_name: str = "gemini-1.5-pro"
    ):
        """
        Initialize strategy analyzer.
        
        Args:
            video_dir: Directory containing gameplay videos
            api_key: Gemini API key
            output_dir: Directory for saving analysis results
            model_name: Gemini model to use
        """
        self.video_dir = Path(video_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Initialize components
        self.gemini_client = GeminiClient(api_key=api_key, model_name=model_name)
        self.prompt_engineer = PromptEngineer()
        self.evaluation_metrics = EvaluationMetrics()
        self.video_sampler = VideoSampler(str(video_dir))
        
        # Setup logging
        self._setup_logging()
        
        # Analysis cache
        self.analysis_cache = {}
        self._load_cache()
    
    def _setup_logging(self):
        """Setup logging for analysis tracking."""
        log_file = self.output_dir / "analysis.log"
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def _load_cache(self):
        """Load analysis cache from file."""
        cache_file = self.output_dir / "analysis_cache.json"
        if cache_file.exists():
            try:
                with open(cache_file, 'r') as f:
                    self.analysis_cache = json.load(f)
            except Exception as e:
                self.logger.warning(f"Could not load cache: {e}")
                self.analysis_cache = {}
    
    def _save_cache(self):
        """Save analysis cache to file."""
        cache_file = self.output_dir / "analysis_cache.json"
        try:
            with open(cache_file, 'w') as f:
                json.dump(self.analysis_cache, f, indent=2)
        except Exception as e:
            self.logger.warning(f"Could not save cache: {e}")
    
    def analyze_single_video(
        self,
        video_path: str,
        analysis_type: str = "strategy",
        max_frames: int = 10,
        use_cache: bool = True
    ) -> Dict[str, Any]:
        """
        Analyze a single video file.
        
        Args:
            video_path: Path to video file
            analysis_type: Type of analysis ("strategy", "baseline", "edge_case")
            max_frames: Maximum frames to extract
            use_cache: Whether to use cached results
            
        Returns:
            Analysis results dictionary
        """
        video_path = str(video_path)
        cache_key = f"{video_path}_{analysis_type}_{max_frames}"
        
        # Check cache
        if use_cache and cache_key in self.analysis_cache:
            self.logger.info(f"Using cached result for {video_path}")
            return self.analysis_cache[cache_key]
        
        self.logger.info(f"Analyzing video: {video_path}")
        
        try:
            # Get appropriate prompt
            if analysis_type == "strategy":
                prompt = self.prompt_engineer.get_strategy_prompt()
            elif analysis_type == "baseline":
                prompt = self.prompt_engineer.get_prompt(AnalysisType.BASELINE_CAPTIONING)
            elif analysis_type == "edge_case":
                prompt = self.prompt_engineer.get_edge_case_prompt()
            else:
                prompt = self.prompt_engineer.get_strategy_prompt()
            
            # Analyze video
            summary = self.gemini_client.analyze_video(
                video_path=video_path,
                prompt=prompt,
                max_frames=max_frames
            )
            
            # Calculate abstraction score
            abstraction_result = self.evaluation_metrics.calculate_abstraction_score(summary)
            
            result = {
                'video_path': video_path,
                'analysis_type': analysis_type,
                'timestamp': datetime.now().isoformat(),
                'strategy_summary': summary,
                'abstraction_score': abstraction_result,
                'max_frames_used': max_frames
            }
            
            # Cache result
            if use_cache:
                self.analysis_cache[cache_key] = result
                self._save_cache()
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error analyzing video {video_path}: {e}")
            return {
                'video_path': video_path,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    def analyze_video_set(
        self,
        sampling_strategy: SamplingStrategy,
        num_videos: int = 10,
        analysis_type: str = "strategy",
        **sampling_kwargs
    ) -> Dict[str, Any]:
        """
        Analyze a set of videos using specified sampling strategy.
        
        Args:
            sampling_strategy: Video sampling strategy
            num_videos: Number of videos to analyze
            analysis_type: Type of analysis to perform
            **sampling_kwargs: Additional sampling parameters
            
        Returns:
            Analysis results for the video set
        """
        self.logger.info(f"Starting video set analysis: {sampling_strategy.value}")
        
        # Sample videos
        sampled_videos = self.video_sampler.sample_videos(
            strategy=sampling_strategy,
            num_videos=num_videos,
            **sampling_kwargs
        )
        
        if not sampled_videos:
            return {
                'error': f"No videos found for sampling strategy: {sampling_strategy.value}",
                'timestamp': datetime.now().isoformat()
            }
        
        results = {
            'sampling_strategy': sampling_strategy.value,
            'num_videos_requested': num_videos,
            'num_videos_analyzed': len(sampled_videos),
            'timestamp': datetime.now().isoformat(),
            'video_analyses': [],
            'aggregate_metrics': {}
        }
        
        # Analyze each video
        for video_metadata in sampled_videos:
            video_path = video_metadata['filepath']
            analysis_result = self.analyze_single_video(
                video_path=video_path,
                analysis_type=analysis_type
            )
            results['video_analyses'].append(analysis_result)
        
        # Calculate aggregate metrics
        results['aggregate_metrics'] = self._calculate_aggregate_metrics(
            results['video_analyses']
        )
        
        # Save results
        self._save_analysis_results(results, f"{sampling_strategy.value}_{analysis_type}")
        
        return results
    
    def _calculate_aggregate_metrics(self, analyses: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate aggregate metrics across multiple analyses."""
        successful_analyses = [a for a in analyses if 'strategy_summary' in a]
        
        if not successful_analyses:
            return {'error': 'No successful analyses to aggregate'}
        
        # Aggregate abstraction scores
        abstraction_scores = []
        for analysis in successful_analyses:
            if 'abstraction_score' in analysis:
                score = analysis['abstraction_score'].get('abstraction_score', 0)
                abstraction_scores.append(score)
        
        aggregate_metrics = {
            'total_analyses': len(analyses),
            'successful_analyses': len(successful_analyses),
            'success_rate': len(successful_analyses) / len(analyses) if analyses else 0,
        }
        
        if abstraction_scores:
            aggregate_metrics.update({
                'average_abstraction_score': sum(abstraction_scores) / len(abstraction_scores),
                'min_abstraction_score': min(abstraction_scores),
                'max_abstraction_score': max(abstraction_scores),
                'abstraction_score_std': self._calculate_std(abstraction_scores)
            })
        
        return aggregate_metrics
    
    def _calculate_std(self, values: List[float]) -> float:
        """Calculate standard deviation."""
        if len(values) < 2:
            return 0.0
        mean = sum(values) / len(values)
        variance = sum((x - mean) ** 2 for x in values) / (len(values) - 1)
        return variance ** 0.5
    
    def evaluate_strategy_faithfulness(
        self,
        strategy_summary: str,
        test_video_path: str,
        context_duration: float = 5.0
    ) -> Dict[str, Any]:
        """
        Evaluate strategy faithfulness using Predictive Faithfulness Score.
        
        Args:
            strategy_summary: Generated strategy summary
            test_video_path: Path to test video for evaluation
            context_duration: Duration of context clip in seconds
            
        Returns:
            Faithfulness evaluation results
        """
        self.logger.info(f"Evaluating strategy faithfulness for: {test_video_path}")
        
        try:
            # Extract context frames and ground truth frames
            context_frames = self.gemini_client.extract_frames(
                test_video_path, max_frames=5, method="uniform"
            )
            
            # Get prediction based on strategy summary
            prediction = self.gemini_client.predict_behavior(
                strategy_summary=strategy_summary,
                context_frames=context_frames
            )
            
            # Extract ground truth frames (next 5 seconds)
            ground_truth_frames = self.gemini_client.extract_frames(
                test_video_path, max_frames=5, method="uniform"
            )
            
            # Generate ground truth description
            ground_truth_description = self.gemini_client.analyze_frames(
                frames=ground_truth_frames,
                prompt="Describe exactly what the agent does in these frames. Focus on specific actions and movements."
            )
            
            # Calculate PFS
            pfs_score = self.evaluation_metrics.calculate_predictive_faithfulness_score(
                prediction=prediction,
                ground_truth=ground_truth_description
            )
            
            result = {
                'video_path': test_video_path,
                'strategy_summary': strategy_summary,
                'prediction': prediction,
                'ground_truth': ground_truth_description,
                'pfs_score': pfs_score,
                'timestamp': datetime.now().isoformat()
            }
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error in faithfulness evaluation: {e}")
            return {
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    def evaluate_strategy_coverage(
        self,
        strategy_summary: str,
        test_video_paths: List[str]
    ) -> Dict[str, Any]:
        """
        Evaluate strategy coverage using question-answer methodology.
        
        Args:
            strategy_summary: Generated strategy summary
            test_video_paths: List of test video paths
            
        Returns:
            Coverage evaluation results
        """
        self.logger.info("Evaluating strategy coverage")
        
        try:
            all_questions = []
            all_answers = []
            
            # Generate questions for each test video
            for video_path in test_video_paths:
                frames = self.gemini_client.extract_frames(video_path, max_frames=8)
                questions = self.gemini_client.generate_questions(frames)
                
                # Answer questions using strategy summary
                answers = []
                for question in questions:
                    answer = self.gemini_client.answer_question(question, strategy_summary)
                    answers.append(answer)
                
                all_questions.extend(questions)
                all_answers.extend(answers)
            
            # Calculate coverage score
            coverage_result = self.evaluation_metrics.calculate_coverage_score(
                strategy_summary=strategy_summary,
                questions=all_questions,
                answers=all_answers
            )
            
            result = {
                'strategy_summary': strategy_summary,
                'test_videos': test_video_paths,
                'questions': all_questions,
                'answers': all_answers,
                'coverage_metrics': coverage_result,
                'timestamp': datetime.now().isoformat()
            }
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error in coverage evaluation: {e}")
            return {
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    def run_comparative_experiment(
        self,
        baseline_type: str = "captioning",
        num_videos_per_strategy: int = 5
    ) -> Dict[str, Any]:
        """
        Run comparative experiment between strategy analysis and baseline.
        
        Args:
            baseline_type: Type of baseline ("captioning", "generic")
            num_videos_per_strategy: Number of videos to analyze per strategy
            
        Returns:
            Comparative experiment results
        """
        self.logger.info("Running comparative experiment")
        
        # Analyze with strategy-focused approach
        strategy_results = self.analyze_video_set(
            sampling_strategy=SamplingStrategy.BALANCED,
            num_videos=num_videos_per_strategy,
            analysis_type="strategy"
        )
        
        # Analyze with baseline approach
        baseline_results = self.analyze_video_set(
            sampling_strategy=SamplingStrategy.BALANCED,
            num_videos=num_videos_per_strategy,
            analysis_type="baseline"
        )
        
        # Compare results
        comparison = self._compare_analysis_results(strategy_results, baseline_results)
        
        experiment_results = {
            'experiment_type': 'comparative',
            'baseline_type': baseline_type,
            'timestamp': datetime.now().isoformat(),
            'strategy_results': strategy_results,
            'baseline_results': baseline_results,
            'comparison': comparison
        }
        
        # Save experiment results
        self._save_analysis_results(experiment_results, "comparative_experiment")
        
        return experiment_results
    
    def run_ablation_study(
        self,
        ablation_types: List[str] = ["prompt", "sampling"],
        num_videos: int = 10
    ) -> Dict[str, Any]:
        """
        Run ablation studies to validate framework components.
        
        Args:
            ablation_types: Types of ablation studies to run
            num_videos: Number of videos to use for each study
            
        Returns:
            Ablation study results
        """
        self.logger.info("Running ablation studies")
        
        results = {
            'experiment_type': 'ablation',
            'timestamp': datetime.now().isoformat(),
            'studies': {}
        }
        
        if "prompt" in ablation_types:
            # Compare different prompt types
            prompt_study = self._run_prompt_ablation(num_videos)
            results['studies']['prompt_ablation'] = prompt_study
        
        if "sampling" in ablation_types:
            # Compare different sampling strategies
            sampling_study = self._run_sampling_ablation(num_videos)
            results['studies']['sampling_ablation'] = sampling_study
        
        # Save results
        self._save_analysis_results(results, "ablation_study")
        
        return results
    
    def _run_prompt_ablation(self, num_videos: int) -> Dict[str, Any]:
        """Run prompt ablation study."""
        prompt_types = ["strategy", "baseline", "generic", "minimal"]
        prompt_results = {}
        
        for prompt_type in prompt_types:
            if prompt_type == "strategy":
                analysis_type = "strategy"
            else:
                analysis_type = "baseline"
            
            results = self.analyze_video_set(
                sampling_strategy=SamplingStrategy.RANDOM,
                num_videos=num_videos,
                analysis_type=analysis_type
            )
            prompt_results[prompt_type] = results
        
        return {
            'study_type': 'prompt_ablation',
            'prompt_types': prompt_types,
            'results': prompt_results,
            'comparison': self._compare_prompt_results(prompt_results)
        }
    
    def _run_sampling_ablation(self, num_videos: int) -> Dict[str, Any]:
        """Run sampling ablation study."""
        sampling_strategies = [
            SamplingStrategy.TYPICAL,
            SamplingStrategy.EDGE_CASE,
            SamplingStrategy.LONGITUDINAL,
            SamplingStrategy.BALANCED
        ]
        
        sampling_results = {}
        
        for strategy in sampling_strategies:
            results = self.analyze_video_set(
                sampling_strategy=strategy,
                num_videos=num_videos,
                analysis_type="strategy"
            )
            sampling_results[strategy.value] = results
        
        return {
            'study_type': 'sampling_ablation',
            'sampling_strategies': [s.value for s in sampling_strategies],
            'results': sampling_results,
            'comparison': self._compare_sampling_results(sampling_results)
        }
    
    def _compare_analysis_results(
        self,
        results1: Dict[str, Any],
        results2: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Compare two sets of analysis results."""
        comparison = {
            'metrics_comparison': {},
            'summary': {}
        }
        
        # Compare aggregate metrics
        metrics1 = results1.get('aggregate_metrics', {})
        metrics2 = results2.get('aggregate_metrics', {})
        
        for metric in ['average_abstraction_score', 'success_rate']:
            if metric in metrics1 and metric in metrics2:
                comparison['metrics_comparison'][metric] = {
                    'strategy_score': metrics1[metric],
                    'baseline_score': metrics2[metric],
                    'difference': metrics1[metric] - metrics2[metric],
                    'winner': 'strategy' if metrics1[metric] > metrics2[metric] else 'baseline'
                }
        
        return comparison
    
    def _compare_prompt_results(self, prompt_results: Dict[str, Any]) -> Dict[str, Any]:
        """Compare results from different prompt types."""
        # Extract abstraction scores for comparison
        scores = {}
        for prompt_type, results in prompt_results.items():
            metrics = results.get('aggregate_metrics', {})
            if 'average_abstraction_score' in metrics:
                scores[prompt_type] = metrics['average_abstraction_score']
        
        if not scores:
            return {'error': 'No abstraction scores to compare'}
        
        best_prompt = max(scores.keys(), key=lambda k: scores[k])
        
        return {
            'abstraction_scores': scores,
            'best_prompt_type': best_prompt,
            'best_score': scores[best_prompt]
        }
    
    def _compare_sampling_results(self, sampling_results: Dict[str, Any]) -> Dict[str, Any]:
        """Compare results from different sampling strategies."""
        scores = {}
        for strategy, results in sampling_results.items():
            metrics = results.get('aggregate_metrics', {})
            if 'average_abstraction_score' in metrics:
                scores[strategy] = metrics['average_abstraction_score']
        
        if not scores:
            return {'error': 'No abstraction scores to compare'}
        
        best_strategy = max(scores.keys(), key=lambda k: scores[k])
        
        return {
            'abstraction_scores': scores,
            'best_sampling_strategy': best_strategy,
            'best_score': scores[best_strategy]
        }
    
    def _save_analysis_results(self, results: Dict[str, Any], filename_prefix: str):
        """Save analysis results to file."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{filename_prefix}_{timestamp}.json"
        filepath = self.output_dir / filename
        
        try:
            with open(filepath, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            self.logger.info(f"Results saved to: {filepath}")
        except Exception as e:
            self.logger.error(f"Error saving results: {e}")
    
    def generate_comprehensive_report(self, experiment_results: Dict[str, Any]) -> str:
        """
        Generate a comprehensive analysis report.
        
        Args:
            experiment_results: Results from experiments
            
        Returns:
            Formatted report string
        """
        report = f"""
COMPREHENSIVE STRATEGY ANALYSIS REPORT
=====================================

Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

EXPERIMENT OVERVIEW:
- Experiment Type: {experiment_results.get('experiment_type', 'Unknown')}
- Total Videos Analyzed: {self._count_total_videos(experiment_results)}
- Analysis Framework: Vision-Language Model (Gemini) Strategy Analysis

"""
        
        # Add specific sections based on experiment type
        if experiment_results.get('experiment_type') == 'comparative':
            report += self._generate_comparative_section(experiment_results)
        elif experiment_results.get('experiment_type') == 'ablation':
            report += self._generate_ablation_section(experiment_results)
        
        report += "\n" + "="*60 + "\n"
        
        return report
    
    def _count_total_videos(self, results: Dict[str, Any]) -> int:
        """Count total videos analyzed in experiment."""
        total = 0
        
        def count_recursive(obj):
            nonlocal total
            if isinstance(obj, dict):
                if 'num_videos_analyzed' in obj:
                    total += obj['num_videos_analyzed']
                for value in obj.values():
                    count_recursive(value)
            elif isinstance(obj, list):
                for item in obj:
                    count_recursive(item)
        
        count_recursive(results)
        return total
    
    def _generate_comparative_section(self, results: Dict[str, Any]) -> str:
        """Generate comparative experiment section."""
        comparison = results.get('comparison', {})
        metrics_comp = comparison.get('metrics_comparison', {})
        
        section = """
COMPARATIVE ANALYSIS RESULTS:
============================

Strategy-Focused vs Baseline Comparison:
"""
        
        for metric, data in metrics_comp.items():
            section += f"""
{metric.replace('_', ' ').title()}:
- Strategy Approach: {data.get('strategy_score', 'N/A'):.3f}
- Baseline Approach: {data.get('baseline_score', 'N/A'):.3f}
- Difference: {data.get('difference', 'N/A'):.3f}
- Winner: {data.get('winner', 'N/A').title()}
"""
        
        return section
    
    def _generate_ablation_section(self, results: Dict[str, Any]) -> str:
        """Generate ablation study section."""
        studies = results.get('studies', {})
        
        section = """
ABLATION STUDY RESULTS:
======================
"""
        
        for study_name, study_data in studies.items():
            section += f"""
{study_name.replace('_', ' ').title()}:
"""
            comparison = study_data.get('comparison', {})
            if 'best_prompt_type' in comparison:
                section += f"- Best Prompt Type: {comparison['best_prompt_type']}\n"
                section += f"- Best Score: {comparison['best_score']:.3f}\n"
            elif 'best_sampling_strategy' in comparison:
                section += f"- Best Sampling Strategy: {comparison['best_sampling_strategy']}\n"
                section += f"- Best Score: {comparison['best_score']:.3f}\n"
        
        return section
    
    def run_full_experiment(
        self, 
        experiment_name: str = "dissertation_experiment"
    ) -> Dict[str, Any]:
        """
        Run the complete experiment as described in the dissertation.
        
        Args:
            experiment_name: Name for the experiment
            
        Returns:
            Complete experiment results
        """
        self.logger.info(f"Starting full experiment: {experiment_name}")
        
        experiment_results = {
            'experiment_name': experiment_name,
            'timestamp': datetime.now().isoformat(),
            'methodology': 'VLM Strategy Analysis Framework',
            'results': {}
        }
        
        # 1. Baseline Comparison
        self.logger.info("Running baseline comparison...")
        baseline_comparison = self.run_comparative_experiment()
        experiment_results['results']['baseline_comparison'] = baseline_comparison
        
        # 2. Ablation Studies
        self.logger.info("Running ablation studies...")
        ablation_results = self.run_ablation_study()
        experiment_results['results']['ablation_studies'] = ablation_results
        
        # 3. Longitudinal Analysis
        self.logger.info("Running longitudinal analysis...")
        longitudinal_results = self.analyze_video_set(
            sampling_strategy=SamplingStrategy.LONGITUDINAL,
            num_videos=15,
            analysis_type="strategy"
        )
        experiment_results['results']['longitudinal_analysis'] = longitudinal_results
        
        # Generate comprehensive report
        report = self.generate_comprehensive_report(experiment_results)
        experiment_results['comprehensive_report'] = report
        
        # Save complete results
        self._save_analysis_results(experiment_results, f"full_experiment_{experiment_name}")
        
        return experiment_results 