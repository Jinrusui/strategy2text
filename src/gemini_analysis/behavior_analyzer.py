"""
Low-level behavior analyzer for RL agent videos using Gemini AI.
"""

import json
import logging
from typing import Dict, List, Optional, Any
from pathlib import Path
import re

from .gemini_client import GeminiClient


class BehaviorAnalyzer:
    """Analyzes low-level behaviors of RL agents from video recordings using Gemini AI."""
    
    def __init__(self, api_key: Optional[str] = None, model: str = "gemini-2.5-flash-preview-05-20"):
        """
        Initialize the behavior analyzer.
        
        Args:
            api_key: Google API key for Gemini
            model: Gemini model to use (using the model from user's example)
        """
        self.gemini_client = GeminiClient(api_key=api_key, model=model)
        self.logger = logging.getLogger(__name__)
        
        # System prompt for low-level behavior analysis
        self.behavior_prompt = self._get_behavior_analysis_prompt()
    
    def _get_behavior_analysis_prompt(self) -> str:
        """Get the system prompt for low-level behavior analysis."""
        return """
You are an expert in reinforcement learning and game analysis. Your task is to analyze the low-level behaviors and actions of an RL agent in this video.

Focus on the following aspects of low-level behavior:

1. **Immediate Actions & Reactions**:
   - Frame-by-frame action sequences
   - Reaction times to environmental changes
   - Precision and timing of individual moves
   - Micro-adjustments and fine-grained control

2. **Movement Patterns**:
   - Speed and acceleration patterns
   - Direction changes and their frequency
   - Hesitation or uncertainty in movements
   - Repetitive motion patterns or habits

3. **Decision-Making Process**:
   - How quickly decisions are made
   - Evidence of exploration vs exploitation
   - Response to immediate threats or opportunities
   - Local optimization behaviors

4. **Technical Execution**:
   - Accuracy of actions relative to game mechanics
   - Consistency in similar situations
   - Error patterns and recovery behaviors
   - Adaptation to changing conditions

Please provide a detailed analysis of these low-level behaviors observed in the video. Focus on what the agent is actually doing moment-to-moment rather than high-level strategic thinking.

Summarize your findings in exactly 3 sentences that capture the most important low-level behavioral characteristics.
"""
    
    def analyze_behavior(self, video_path: str) -> Dict[str, Any]:
        """
        Analyze the low-level behavior of an RL agent from a video recording.
        
        Args:
            video_path: Path to the video file
            
        Returns:
            Dictionary containing behavior analysis
        """
        try:
            # Check file size (should be <20MB as per user's example)
            video_file = Path(video_path)
            if not video_file.exists():
                raise FileNotFoundError(f"Video file not found: {video_path}")
            
            file_size_mb = video_file.stat().st_size / (1024 * 1024)
            if file_size_mb >= 20:
                self.logger.warning(f"Video file size ({file_size_mb:.1f}MB) is >= 20MB, may cause issues")
            
            analysis_text = self.gemini_client.analyze_video_direct(video_path, self.behavior_prompt)
            
            # Parse and structure the analysis
            behavior_summary = self._parse_behavior_analysis(analysis_text)
            
            return {
                "video_path": video_path,
                "analysis_type": "low_level_behavior",
                "raw_analysis": analysis_text,
                "behavior_summary": behavior_summary,
                "file_size_mb": file_size_mb,
                "timestamp": self._get_timestamp()
            }
            
        except Exception as e:
            self.logger.error(f"Failed to analyze behavior for {video_path}: {str(e)}")
            return {
                "video_path": video_path,
                "analysis_type": "low_level_behavior",
                "error": str(e),
                "timestamp": self._get_timestamp()
            }
    
    def batch_analyze_behaviors(self, video_paths: List[str]) -> List[Dict[str, Any]]:
        """
        Analyze behaviors across multiple videos.
        
        Args:
            video_paths: List of video file paths
            
        Returns:
            List of behavior analysis results
        """
        results = []
        
        for video_path in video_paths:
            self.logger.info(f"Analyzing behavior for: {video_path}")
            analysis = self.analyze_behavior(video_path)
            results.append(analysis)
        
        return results
    
    def _parse_behavior_analysis(self, analysis_text: str) -> Dict[str, str]:
        """Parse the analysis text into structured components."""
        sections = {
            "immediate_actions": "",
            "movement_patterns": "",
            "decision_making": "",
            "technical_execution": "",
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
            elif "decision" in line.lower() and "making" in line.lower():
                current_section = "decision_making"
            elif "technical" in line.lower() and "execution" in line.lower():
                current_section = "technical_execution"
            elif current_section and line:
                sections[current_section] += line + " "
        
        # Extract summary (last 3 sentences or sentences that appear to be summary)
        sentences = re.split(r'[.!?]+', analysis_text)
        sentences = [s.strip() for s in sentences if s.strip()]
        if len(sentences) >= 3:
            sections["summary"] = '. '.join(sentences[-3:]) + '.'
        else:
            sections["summary"] = analysis_text
        
        return sections
    
    def _get_timestamp(self) -> str:
        """Get current timestamp."""
        from datetime import datetime
        return datetime.now().isoformat()
    
    def save_analysis(self, analysis: Dict[str, Any], output_path: str):
        """Save analysis results to JSON file."""
        with open(output_path, 'w') as f:
            json.dump(analysis, f, indent=2)
        self.logger.info(f"Analysis saved to {output_path}")
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.gemini_client.cleanup_uploaded_files() 