"""
Low-level behavior analyzer for RL agent videos using Gemini AI.
"""

import json
import logging
from typing import Dict, List, Optional, Any
from pathlib import Path
import re

from .gemini_client import GeminiClient


class BreakoutBehaviorAnalyzer:
    """Analyzes low-level behaviors of Breakout RL agents from video recordings using Gemini AI."""
    
    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None):
        """
        Initialize the Breakout behavior analyzer.
        
        Args:
            api_key: Google API key for Gemini
            model: Model to use for analysis. If None, uses default model
        """
        self.gemini_client = GeminiClient(api_key=api_key, model=model)
        self.logger = logging.getLogger(__name__)
        
        # Load the analysis prompt from file
        self.analysis_prompt = self._load_behavior_prompt()
    
    def _load_behavior_prompt(self) -> str:
        """Load the Breakout behavior analysis prompt from the text file."""
        prompt_file = Path("prompts/behavior_prompt.txt")
        try:
            with open(prompt_file, 'r') as f:
                return f.read().strip()
        except FileNotFoundError:
            self.logger.warning(f"Prompt file {prompt_file} not found, using default prompt")
            return self._get_default_prompt()
    
    def _get_default_prompt(self) -> str:
        """Fallback prompt if file is not found."""
        return """
You are an expert in reinforcement learning and Breakout game analysis. Analyze the low-level behaviors and actions of this Breakout RL agent."""


# """
# Focus on LOW-LEVEL BEHAVIORAL elements:

# 1. **Immediate Actions & Reactions**:
#    - Frame-by-frame paddle movements and responses
#    - Reaction times to ball position changes
#    - Precision and timing of paddle positioning
#    - Micro-adjustments and fine-grained paddle control

# 2. **Movement Patterns**:
#    - Paddle speed and acceleration patterns
#    - Direction changes and movement frequency
#    - Hesitation or uncertainty in paddle movements
#    - Repetitive motion patterns or habits

# 3. **Ball Tracking & Response**:
#    - How quickly the agent responds to ball direction changes
#    - Accuracy of paddle positioning relative to ball trajectory
#    - Evidence of predictive vs reactive behavior
#    - Response consistency across different ball speeds

# 4. **Technical Execution**:
#    - Precision of paddle movements within game constraints
#    - Consistency in similar ball-paddle interaction scenarios
#    - Error patterns and recovery behaviors
#    - Micro-level decision making in paddle control

# 5. **Breakout-Specific Behaviors**:
#    - Paddle positioning strategy for different ball angles
#    - Response to ball bouncing off different surfaces
#    - Behavior when ball approaches paddle edges vs center
#    - Timing accuracy for successful ball returns

# Please provide a detailed analysis of these low-level behaviors observed in the video. Focus on what the agent is actually doing moment-to-moment rather than high-level strategic thinking.

# Summarize your findings in exactly 3 sentences that capture the most important low-level behavioral characteristics.
# """


    
    def analyze_breakout_behavior(self, video_path: str, fps: int = 5) -> Dict[str, Any]:
        """
        Analyze the low-level behavior of a Breakout RL agent from a video recording using direct video processing.
        
        Args:
            video_path: Path to the video file
            fps: Frame sampling rate for video analysis (configurable hyperparameter: 5, 10, etc.)
            
        Returns:
            Dictionary containing behavior analysis
        """
        try:
            # Check file size (should be <20MB for direct upload)
            video_file = Path(video_path)
            if not video_file.exists():
                raise FileNotFoundError(f"Video file not found: {video_path}")
            
            file_size_mb = video_file.stat().st_size / (1024 * 1024)
            if file_size_mb >= 20:
                self.logger.warning(f"Video file size ({file_size_mb:.1f}MB) is >= 20MB, may cause issues with low-level analysis")
            
            # Use low-level analysis method with direct video bytes
            analysis_text = self.gemini_client.analyze_video_low_level(
                video_path, 
                self.analysis_prompt, 
                fps=fps
            )
            
            # Parse and structure the analysis
            behavior_summary = self._parse_behavior_analysis(analysis_text)
            
            return {
                "video_path": video_path,
                "environment": "Breakout",
                "analysis_type": "low_level_behavior",
                "model": self.gemini_client.model,
                "fps": fps,
                "raw_analysis": analysis_text,
                "behavior_summary": behavior_summary,
                "file_size_mb": file_size_mb,
                "timestamp": self._get_timestamp()
            }
            
        except Exception as e:
            self.logger.error(f"Failed to analyze behavior for {video_path}: {str(e)}")
            return {
                "video_path": video_path,
                "environment": "Breakout",
                "analysis_type": "low_level_behavior",
                "model": self.gemini_client.model,
                "fps": fps,
                "error": str(e),
                "timestamp": self._get_timestamp()
            }
    

    
    def batch_analyze_breakout_behaviors(self, video_paths: List[str], fps: int = 5) -> List[Dict[str, Any]]:
        """
        Analyze behaviors across multiple Breakout videos using low-level analysis.
        
        Args:
            video_paths: List of video file paths
            fps: Frame sampling rate for video analysis (configurable hyperparameter: 5, 10, etc.)
            
        Returns:
            List of behavior analysis results
        """
        results = []
        
        for video_path in video_paths:
            self.logger.info(f"Analyzing behavior for: {video_path}")
            analysis = self.analyze_breakout_behavior(video_path, fps=fps)
            results.append(analysis)
        
        return results
    

    
    def _parse_behavior_analysis(self, analysis_text: str) -> Dict[str, str]:
        """Parse the analysis text into structured components."""
        sections = {
            "immediate_actions": "",
            "movement_patterns": "",
            "ball_tracking": "",
            "technical_execution": "",
            "breakout_specific": "",
            "summary": ""
        }
        
        # Try to extract sections based on headers
        current_section = None
        lines = analysis_text.split('\n')
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Check for section headers
            if "immediate" in line.lower() and "action" in line.lower():
                current_section = "immediate_actions"
            elif "movement" in line.lower() and "pattern" in line.lower():
                current_section = "movement_patterns"
            elif "ball" in line.lower() and ("tracking" in line.lower() or "response" in line.lower()):
                current_section = "ball_tracking"
            elif "technical" in line.lower() and "execution" in line.lower():
                current_section = "technical_execution"
            elif "breakout" in line.lower() and ("specific" in line.lower() or "behavior" in line.lower()):
                current_section = "breakout_specific"
            elif current_section and line:
                sections[current_section] += line + " "
        
        # Extract summary (last 3 sentences or sentences that appear to be summary)
        sentences = re.split(r'[.!?]+', analysis_text)
        sentences = [s.strip() for s in sentences if s.strip()]
        if len(sentences) >= 3:
            sections["summary"] = '. '.join(sentences[-3:]) + '.'
        else:
            sections["summary"] = analysis_text
        
        return {k: v.strip() for k, v in sections.items()}
    
    def _get_timestamp(self) -> str:
        """Get current timestamp."""
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