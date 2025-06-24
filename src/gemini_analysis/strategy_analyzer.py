"""
Strategy analyzer for Breakout RL agent videos using Gemini AI.
"""

import json
import logging
from typing import Dict, List, Optional, Any
from pathlib import Path
import re

from .gemini_client import GeminiClient


class BreakoutStrategyAnalyzer:
    """Analyzes Breakout RL agent strategies from video recordings using Gemini AI."""
    
    def __init__(self, api_key: Optional[str] = None, model: str = "gemini-2.0-flash"):
        """
        Initialize the Breakout strategy analyzer.
        
        Args:
            api_key: Google API key for Gemini
            model: Gemini model to use
        """
        self.gemini_client = GeminiClient(api_key=api_key, model=model)
        self.logger = logging.getLogger(__name__)
        
        # Load the analysis prompt from file
        self.analysis_prompt = self._load_analysis_prompt()
    
    def _load_analysis_prompt(self) -> str:
        """Load the Breakout analysis prompt from the text file."""
        prompt_file = Path("breakout_analysis_prompt.txt")
        try:
            with open(prompt_file, 'r') as f:
                return f.read().strip()
        except FileNotFoundError:
            self.logger.warning(f"Prompt file {prompt_file} not found, using default prompt")
            return self._get_default_prompt()
    
    def _get_default_prompt(self) -> str:
        """Fallback prompt if file is not found."""
        return """
You are an expert in reinforcement learning and strategic game analysis. Analyze the overall strategic approach of this Breakout RL agent.

Focus on HIGH-LEVEL STRATEGIC elements:

1. **Overall Strategy & Approach**:
   - Long-term game plan and strategic thinking
   - Risk vs reward decision making
   - Adaptation to different game phases
   - Strategic positioning and planning ahead

2. **Strategic Brick Destruction**:
   - Prioritization of brick targets
   - Strategic use of ball angles and rebounds
   - Planning for efficient brick clearing patterns
   - Strategic tunnel creation and exploitation

3. **Game State Management**:
   - How the agent handles different game scenarios
   - Strategic response to ball loss risks
   - Adaptation to changing brick configurations
   - Long-term score optimization strategies

4. **Learning & Strategic Evolution**:
   - Evidence of strategic learning and improvement
   - Strategic pattern recognition
   - Adaptation of strategy based on game feedback
   - Strategic consistency across different situations

5. **Performance & Strategic Effectiveness**:
   - Overall strategic success in achieving game objectives
   - Efficiency of strategic choices
   - Strategic weaknesses and strengths
   - Comparison to optimal strategic approaches

Please provide a comprehensive analysis focusing on the STRATEGIC aspects rather than moment-to-moment actions. Evaluate the agent's strategic intelligence and planning capabilities.
"""
    
    def analyze_breakout_strategy(self, video_path: str) -> Dict[str, Any]:
        """
        Analyze the strategy of a Breakout RL agent from a video recording.
        
        Args:
            video_path: Path to the video file
            
        Returns:
            Dictionary containing strategy analysis
        """
        try:
            analysis_text = self.gemini_client.analyze_video(video_path, self.analysis_prompt)
            
            # Parse and structure the analysis
            strategy_summary = self._parse_strategy_analysis(analysis_text)
            
            return {
                "video_path": video_path,
                "environment": "Breakout",
                "analysis_type": "high_level_strategy",
                "raw_analysis": analysis_text,
                "strategy_summary": strategy_summary,
                "timestamp": self._get_timestamp()
            }
            
        except Exception as e:
            self.logger.error(f"Failed to analyze strategy for {video_path}: {str(e)}")
            return {
                "video_path": video_path,
                "environment": "Breakout",
                "analysis_type": "high_level_strategy",
                "error": str(e),
                "timestamp": self._get_timestamp()
            }
    
    def compare_breakout_strategies(self, video_paths: List[str]) -> Dict[str, Any]:
        """
        Compare strategies across multiple Breakout agent videos.
        
        Args:
            video_paths: List of video file paths
            
        Returns:
            Dictionary containing comparative analysis
        """
        individual_analyses = []
        
        # Analyze each video individually
        for video_path in video_paths:
            analysis = self.analyze_breakout_strategy(video_path)
            individual_analyses.append(analysis)
        
        # Create comparison prompt
        comparison_prompt = self._create_comparison_prompt(len(individual_analyses))
        
        # For comparison, we'll use the first video as a reference and ask for comparison
        if video_paths and individual_analyses:
            try:
                comparison_text = self.gemini_client.analyze_video(
                    video_paths[0], 
                    comparison_prompt
                )
                
                return {
                    "environment": "Breakout",
                    "individual_analyses": individual_analyses,
                    "comparison_analysis": comparison_text,
                    "timestamp": self._get_timestamp()
                }
            except Exception as e:
                self.logger.error(f"Failed to create comparison analysis: {str(e)}")
                return {
                    "environment": "Breakout",
                    "individual_analyses": individual_analyses,
                    "comparison_error": str(e),
                    "timestamp": self._get_timestamp()
                }
        
        return {
            "environment": "Breakout",
            "individual_analyses": individual_analyses,
            "timestamp": self._get_timestamp()
        }
    
    def _create_comparison_prompt(self, num_agents: int) -> str:
        """Create a prompt for comparing multiple Breakout agent strategies."""
        return f"""
I have analyzed {num_agents} different Breakout RL agents. 

Based on the individual analyses, please provide a comparative analysis that covers:

1. **Strategy Differences**:
   - How do the agents' paddle control strategies differ?
   - Which agents show better ball tracking abilities?
   - Are there differences in brick targeting approaches?

2. **Performance Comparison**:
   - Which agent(s) perform better at maintaining ball control?
   - What are the key performance differentiators in Breakout gameplay?
   - Which agent clears bricks more efficiently?

3. **Learning Quality**:
   - Which agents show better understanding of Breakout physics?
   - Are there common failure patterns across agents?
   - Which agent demonstrates more sophisticated strategies?

4. **Breakout-Specific Recommendations**:
   - Which strategy would be most effective for Breakout and why?
   - What specific improvements could be made to each agent's approach?
   - Which agent shows the most promise for advanced Breakout strategies?

Please provide a detailed comparative analysis focusing on Breakout-specific strategic and performance differences.
"""
    
    def _parse_strategy_analysis(self, analysis_text: str) -> Dict[str, str]:
        """Parse the analysis text into structured components."""
        sections = {
            "overall_strategy": "",
            "strategic_brick_destruction": "",
            "game_state_management": "",
            "learning_evolution": "",
            "performance_effectiveness": ""
        }
        
        # Try to extract sections based on headers
        current_section = None
        lines = analysis_text.split('\n')
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Check for section headers
            if "overall" in line.lower() and "strategy" in line.lower():
                current_section = "overall_strategy"
            elif "strategic" in line.lower() and "brick" in line.lower():
                current_section = "strategic_brick_destruction"
            elif "game" in line.lower() and "state" in line.lower():
                current_section = "game_state_management"
            elif "learning" in line.lower() and ("evolution" in line.lower() or "strategic" in line.lower()):
                current_section = "learning_evolution"
            elif "performance" in line.lower() and ("effectiveness" in line.lower() or "strategic" in line.lower()):
                current_section = "performance_effectiveness"
            elif current_section and line:
                sections[current_section] += line + " "
        
        # If parsing fails, put everything in strategy_description
        if not any(sections.values()):
            sections["strategy_description"] = analysis_text
        
        return {k: v.strip() for k, v in sections.items()}
    
    def _get_timestamp(self) -> str:
        """Get current timestamp as string."""
        from datetime import datetime
        return datetime.now().isoformat()
    
    def save_analysis(self, analysis: Dict[str, Any], output_path: str):
        """Save analysis results to a JSON file."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(analysis, f, indent=2)
        
        self.logger.info(f"Analysis saved to {output_path}")
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.gemini_client.cleanup_uploaded_files() 