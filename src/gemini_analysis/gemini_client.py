"""
Gemini API Client for video analysis and strategy generation.

This module handles communication with Google's Gemini API for video-to-text analysis.
"""

import os
import base64
import asyncio
from typing import List, Dict, Any, Optional, Union
from pathlib import Path
import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold
import cv2
import numpy as np
from PIL import Image
import io


class GeminiClient:
    """
    Client for interacting with Google's Gemini API for video analysis.
    
    Handles video upload, frame extraction, and text generation for RL strategy analysis.
    """
    
    def __init__(self, api_key: Optional[str] = None, model_name: str = "gemini-1.5-pro"):
        """
        Initialize Gemini client.
        
        Args:
            api_key: Google API key. If None, loads from environment or file.
            model_name: Gemini model to use for analysis
        """
        self.api_key = api_key or self._load_api_key()
        self.model_name = model_name
        
        # Configure Gemini
        genai.configure(api_key=self.api_key)
        self.model = genai.GenerativeModel(model_name)
        
        # Safety settings for academic research
        self.safety_settings = {
            HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
        }
        
        # Generation configuration
        self.generation_config = genai.types.GenerationConfig(
            temperature=0.7,
            top_p=0.8,
            top_k=40,
            max_output_tokens=2048,
        )
    
    def _load_api_key(self) -> str:
        """Load API key from environment variable or file."""
        # Try environment variable first
        api_key = os.getenv('GEMINI_API_KEY')
        if api_key:
            return api_key
        
        # Try loading from file
        key_file = Path("Gemini_API_KEY.txt")
        if key_file.exists():
            return key_file.read_text().strip()
        
        raise ValueError("Gemini API key not found. Set GEMINI_API_KEY environment variable or create Gemini_API_KEY.txt file.")
    
    def extract_frames(self, video_path: str, max_frames: int = 10, method: str = "uniform") -> List[Image.Image]:
        """
        Extract frames from video for analysis.
        
        Args:
            video_path: Path to video file
            max_frames: Maximum number of frames to extract
            method: Frame extraction method ("uniform", "keyframes", "random")
            
        Returns:
            List of PIL Images
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        duration = total_frames / fps if fps > 0 else 0
        
        frames = []
        
        if method == "uniform":
            # Extract frames uniformly across video duration
            frame_indices = np.linspace(0, total_frames - 1, max_frames, dtype=int)
            
            for frame_idx in frame_indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()
                if ret:
                    # Convert BGR to RGB
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    pil_image = Image.fromarray(frame_rgb)
                    frames.append(pil_image)
        
        elif method == "keyframes":
            # Extract keyframes (simplified - every nth frame)
            step = max(1, total_frames // max_frames)
            frame_idx = 0
            
            while frame_idx < total_frames and len(frames) < max_frames:
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()
                if ret:
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    pil_image = Image.fromarray(frame_rgb)
                    frames.append(pil_image)
                frame_idx += step
        
        elif method == "random":
            # Extract random frames
            frame_indices = np.random.choice(total_frames, min(max_frames, total_frames), replace=False)
            frame_indices = sorted(frame_indices)
            
            for frame_idx in frame_indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()
                if ret:
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    pil_image = Image.fromarray(frame_rgb)
                    frames.append(pil_image)
        
        cap.release()
        return frames
    
    def analyze_video(
        self, 
        video_path: str, 
        prompt: str,
        max_frames: int = 10,
        frame_extraction_method: str = "uniform"
    ) -> str:
        """
        Analyze video using Gemini and return strategy summary.
        
        Args:
            video_path: Path to video file
            prompt: Analysis prompt
            max_frames: Maximum frames to extract
            frame_extraction_method: Method for frame extraction
            
        Returns:
            Generated strategy summary text
        """
        try:
            # Extract frames from video
            frames = self.extract_frames(video_path, max_frames, frame_extraction_method)
            
            if not frames:
                raise ValueError(f"No frames extracted from video: {video_path}")
            
            # Prepare content for Gemini
            content = [prompt]
            content.extend(frames)
            
            # Generate response
            response = self.model.generate_content(
                content,
                generation_config=self.generation_config,
                safety_settings=self.safety_settings
            )
            
            if response.text:
                return response.text.strip()
            else:
                raise ValueError("Empty response from Gemini API")
                
        except Exception as e:
            raise RuntimeError(f"Error analyzing video {video_path}: {str(e)}")
    
    def analyze_frames(self, frames: List[Image.Image], prompt: str) -> str:
        """
        Analyze a list of frames with a given prompt.
        
        Args:
            frames: List of PIL Images
            prompt: Analysis prompt
            
        Returns:
            Generated analysis text
        """
        try:
            content = [prompt]
            content.extend(frames)
            
            response = self.model.generate_content(
                content,
                generation_config=self.generation_config,
                safety_settings=self.safety_settings
            )
            
            if response.text:
                return response.text.strip()
            else:
                raise ValueError("Empty response from Gemini API")
                
        except Exception as e:
            raise RuntimeError(f"Error analyzing frames: {str(e)}")
    
    def predict_behavior(self, strategy_summary: str, context_frames: List[Image.Image]) -> str:
        """
        Predict agent behavior based on strategy summary and context frames.
        Used for Predictive Faithfulness Score calculation.
        
        Args:
            strategy_summary: Previously generated strategy summary
            context_frames: Context frames showing current game state
            
        Returns:
            Predicted behavior description
        """
        prompt = f"""
        Based on this strategy summary of an RL agent:
        "{strategy_summary}"
        
        And looking at these context frames showing the current game state, predict what the agent will do next in the following 5 seconds. Be specific about:
        1. The agent's likely actions (paddle movement, ball direction, etc.)
        2. The reasoning behind these actions based on the established strategy
        3. Expected outcomes of these actions
        
        Provide a detailed prediction of the agent's behavior.
        """
        
        return self.analyze_frames(context_frames, prompt)
    
    def generate_questions(self, video_frames: List[Image.Image]) -> List[str]:
        """
        Generate questions about agent behavior for coverage evaluation.
        
        Args:
            video_frames: Frames from the video to analyze
            
        Returns:
            List of generated questions about agent behavior
        """
        prompt = """
        Looking at these gameplay frames, generate 5-7 specific questions about the agent's behavior and decision-making that would help understand its strategy. Focus on:
        1. Specific actions taken by the agent
        2. Decision points and choices made
        3. Patterns in behavior
        4. Reactions to different game states
        
        Format each question on a new line starting with "Q:".
        """
        
        response = self.analyze_frames(video_frames, prompt)
        
        # Extract questions from response
        questions = []
        for line in response.split('\n'):
            line = line.strip()
            if line.startswith('Q:'):
                questions.append(line[2:].strip())
            elif line and not line.startswith(('A:', 'Answer:', 'Response:')):
                # Handle cases where questions don't have Q: prefix
                if '?' in line:
                    questions.append(line)
        
        return questions
    
    def answer_question(self, question: str, strategy_summary: str) -> str:
        """
        Answer a question about agent behavior using the strategy summary.
        
        Args:
            question: Question about agent behavior
            strategy_summary: Strategy summary to use for answering
            
        Returns:
            Answer to the question
        """
        prompt = f"""
        Strategy Summary: "{strategy_summary}"
        
        Question: {question}
        
        Based on the strategy summary above, provide a concise answer to this question. If the strategy summary doesn't contain enough information to answer the question, state that clearly.
        """
        
        content = [prompt]
        
        response = self.model.generate_content(
            content,
            generation_config=self.generation_config,
            safety_settings=self.safety_settings
        )
        
        return response.text.strip() if response.text else "No response generated"
    
    def evaluate_abstraction(self, summary1: str, summary2: str) -> Dict[str, Any]:
        """
        Compare two summaries for abstraction level using Gemini as judge.
        
        Args:
            summary1: First strategy summary
            summary2: Second strategy summary
            
        Returns:
            Dictionary with comparison results
        """
        prompt = f"""
        Compare these two strategy summaries for their level of abstraction. An abstract summary describes underlying policies and generalizable patterns rather than specific sequences of events.
        
        Summary 1: "{summary1}"
        
        Summary 2: "{summary2}"
        
        Evaluate each summary on a scale of 1-10 for abstraction level, where:
        - 1-3: Very concrete, describes specific events
        - 4-6: Moderately abstract, some generalizable patterns
        - 7-10: Highly abstract, describes underlying policies and invariant rules
        
        Provide your evaluation in this format:
        Summary 1 Score: [score]
        Summary 2 Score: [score]
        Winner: [Summary 1/Summary 2/Tie]
        Reasoning: [brief explanation]
        """
        
        content = [prompt]
        
        response = self.model.generate_content(
            content,
            generation_config=self.generation_config,
            safety_settings=self.safety_settings
        )
        
        # Parse response
        result = {
            'summary1_score': None,
            'summary2_score': None,
            'winner': None,
            'reasoning': ''
        }
        
        if response.text:
            lines = response.text.strip().split('\n')
            for line in lines:
                line = line.strip()
                if line.startswith('Summary 1 Score:'):
                    try:
                        result['summary1_score'] = float(line.split(':')[1].strip())
                    except:
                        pass
                elif line.startswith('Summary 2 Score:'):
                    try:
                        result['summary2_score'] = float(line.split(':')[1].strip())
                    except:
                        pass
                elif line.startswith('Winner:'):
                    result['winner'] = line.split(':')[1].strip()
                elif line.startswith('Reasoning:'):
                    result['reasoning'] = line.split(':', 1)[1].strip()
        
        return result 