"""
Strategy analyzer for Breakout RL agent videos using Gemini AI.
"""

import json
import logging
from typing import Dict, List, Optional, Any
from pathlib import Path

from .gemini_client import GeminiClient


class BreakoutStrategyAnalyzer:
    """Analyzes Breakout RL agent strategies from video recordings using Gemini AI."""
    
    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None):
        """
        Initialize the Breakout strategy analyzer.
        
        Args:
            api_key: Google API key for Gemini
            model: Model to use for analysis. If None, uses default model
        """
        self.gemini_client = GeminiClient(api_key=api_key, model=model)
        self.logger = logging.getLogger(__name__)
        
        # Load the analysis prompt from file
        self.analysis_prompt = self._load_analysis_prompt()
    
    def _load_analysis_prompt(self) -> str:
        """Load the Breakout analysis prompt from the text file."""
        prompt_file = Path("prompts/strategy_prompt.txt")
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
        Analyze the strategy of a Breakout RL agent from a video recording using high-level analysis.
        
        Args:
            video_path: Path to the video file
            
        Returns:
            Dictionary containing strategy analysis
        """
        try:
            # Use high-level analysis method with file upload
            analysis_text = self.gemini_client.analyze_video_high_level(video_path, self.analysis_prompt)
            
            return {
                "video_path": video_path,
                "environment": "Breakout",
                "analysis_type": "high_level_strategy",
                "model": self.gemini_client.model,
                "raw_analysis": analysis_text,
                "timestamp": self._get_timestamp()
            }
            
        except Exception as e:
            self.logger.error(f"Failed to analyze strategy for {video_path}: {str(e)}")
            return {
                "video_path": video_path,
                "environment": "Breakout",
                "analysis_type": "high_level_strategy",
                "model": self.gemini_client.model,
                "error": str(e),
                "timestamp": self._get_timestamp()
            }
    
    def batch_analyze_breakout_strategies(self, video_paths: List[str]) -> List[Dict[str, Any]]:
        """
        Analyze strategies across multiple Breakout videos using high-level analysis.
        
        Args:
            video_paths: List of video file paths
            
        Returns:
            List of strategy analysis results
        """
        results = []
        
        for video_path in video_paths:
            self.logger.info(f"Analyzing strategy for: {video_path}")
            analysis = self.analyze_breakout_strategy(video_path)
            results.append(analysis)
        
        return results
    
    def _get_timestamp(self) -> str:
        """Get current timestamp as string."""
        from datetime import datetime
        return datetime.now().isoformat()
    
    def save_analysis(self, analysis: Dict[str, Any], output_path: str):
        """Save analysis results to a JSON file."""
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w') as f:
            json.dump(analysis, f, indent=2)
        
        self.logger.info(f"Analysis saved to {output_file}")
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.gemini_client.cleanup_uploaded_files() 