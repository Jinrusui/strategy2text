"""
Gemini AI client for hierarchical video analysis (HVA-X) of RL agents.
"""

import os
import logging
from typing import Optional, Dict, Any, List
from pathlib import Path
import time

try:
    from google import genai
    from google.genai import types
except ImportError:
    raise ImportError("Please install google-genai: pip install google-genai")


class GeminiClient:
    """Client for interacting with Gemini AI to analyze RL agent videos using HVA-X algorithm."""
    
    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None):
        """
        Initialize Gemini client.
        
        Args:
            api_key: Google API key. If None, will try to get from GOOGLE_API_KEY env var
            model: Model to use for analysis. If None, uses default model
        """
        
        self.api_key = (
            api_key or 
            os.getenv("GEMINI_API_KEY") or 
            os.getenv("GOOGLE_API_KEY") or 
            self._read_key_from_file("Gemini_API_KEY.txt")
        )
        if not self.api_key:
            raise ValueError(
                "API key required. Set GEMINI_API_KEY or GOOGLE_API_KEY environment variable or pass api_key parameter"
            )
        
        # Initialize client with API key
        self.client = genai.Client(api_key=self.api_key)
        self.logger = logging.getLogger(__name__)
        self.model = model or "gemini-2.5-pro"
        
        # Track uploaded files for cleanup
        self.uploaded_files: List[Any] = []
        
        # HVA-X Algorithm Prompts
        self.prompts = {
            "event_detection": self._get_event_detection_prompt(),
            "guided_analysis": self._get_guided_analysis_prompt(),
            "meta_synthesis": self._get_meta_synthesis_prompt()
        }
    
    def _read_key_from_file(self, filename: str) -> Optional[str]:
        """Read API key from file."""
        try:
            with open(filename, "r") as f:
                return f.read().strip()
        except FileNotFoundError:
            return None
    
    def _get_event_detection_prompt(self) -> str:
        """Get the event detection prompt for Phase 2a of HVA-X."""
        return """
You are an AI assistant analyzing a Breakout RL agent's gameplay video for key moments.

Your task is to identify and timestamp the most important strategic and tactical events. Focus on:

1.  **Critical Decision Points**: Moments where the agent makes important strategic choices
2.  **Tactical Errors & Misses**: Significant mistakes, missed opportunities, or clear failures in ball tracking.
3.  **Strategy Changes**: Points where the agent adapts or changes its brick-clearing approach
4.  **Key Successes & Skilled Shots**: Moments of effective play or shots indicating advanced ball control.
5.  **Game State Transitions**: Important changes in game dynamics (e.g., ball speed increase).

**OUTPUT FORMAT**: Return ONLY a JSON list of events with this exact structure:
[
  {
    "timestamp": "MM:SS-MM:SS",
    "event": "Brief description of what happens",
    "type": "critical_decision|tactical_error|strategy_change|key_success|game_transition|skilled_shot"
  }
]

**IMPORTANT**: 
- Provide timestamps in MM:SS-MM:SS format (e.g., "0:15-0:20")
- Keep event descriptions concise (1-2 sentences max)
- Include 5-10 most important events only
- Return valid JSON format only - no additional text or explanations
"""
    
    def _get_guided_analysis_prompt(self) -> str:
        """Get the guided analysis prompt template for Phase 2b of HVA-X."""
        return """
You are an expert RL analyst conducting a comprehensive analysis of a Breakout agent's gameplay.

**KEY MOMENTS FOR FOCUSED ANALYSIS:**
{key_events}

Using the key moments above as focal points, provide a detailed analysis covering the framework below.

## Strategic & Tactical Analysis Framework

### 1. Overall Brick-Clearing Strategy
- Describe the agent's primary method for clearing bricks (e.g., tunneling, side-clearing, random).
- Analyze its risk management profile (e.g., plays conservatively in the center vs. takes risks at the edges).
- Note any adaptation in strategy as the brick pattern changes.

### 2. Paddle Control & Ball Striking Technique
- **Responsiveness:** How quickly and accurately does the agent react to the ball's speed and trajectory? Note any visible lag or overcorrection.
- **Striking Method:** Does the agent simply position itself under the ball, or does it show evidence of using the paddle's edges to intentionally control the ball's angle?
- **Positioning:** Describe its default paddle position and movement patterns during play.

### 3. Performance Consistency & Failure Analysis
- **Consistency:** How does the agent's performance change across different lives in the video? Is it consistent, or does its skill degrade?
- **Failure Points:** Analyze the agent's misses. Does it fail predictably in certain situations (e.g., high-speed balls, sharp angles)? Use the 'tactical_error' events as evidence.

### 4. Quantitative Performance Summary (Estimate if necessary)
- **Overall Score:** State the final score of the episode.
- **Miss Rate:** How many lives were lost?
- **Brick Clearing Efficiency:** Comment on how effectively it cleared bricks relative to the time played.

### 5. Summary of Strengths & Weaknesses
- Based on the above, what are the agent's primary strategic strengths and tactical limitations?

**CONCLUSION**: End with a 3-sentence summary capturing the agent's strategic profile, its key paddle skill, and its overall effectiveness.
"""
    
    def _get_meta_synthesis_prompt(self) -> str:
        """Get the meta-synthesis prompt for Phase 3 of HVA-X."""
        return """
You are a lead RL analyst synthesizing multiple gameplay analyses to create a comprehensive agent evaluation report.

**ANALYSIS DATA:**
{all_summaries}

Your task is to synthesize these individual analyses into a cohesive, high-level report.

## Synthesis Framework

### 1. Differentiators of Performance
- What specific strategies, paddle techniques, or behaviors separate high-scoring episodes from low-scoring ones?
- Is there a correlation between quantitative metrics (like miss rate) and the final score tier?
- What are the most common failure modes observed in the low-performing episodes?

### 2. Agent Competence Profile
- **Core Strategy:** What is the agent's most common brick-clearing strategy across all episodes?
- **Tactical Skill:** What is the agent's general level of paddle skill? Does it consistently demonstrate advanced techniques like angle control, or does it primarily play reactively?
- **Key Strengths:** What are its most reliable strengths (e.g., consistent defense, effective tunneling)?
- **Key Weaknesses:** What are its most common weaknesses (e.g., handling high speeds, sharp-angle shots)?

### 3. Overall Consistency & Reliability
- How consistent is the agent's performance across the entire sample set?
- Does the agent's skill (e.g., paddle control, responsiveness) appear uniform, or does it vary significantly between games?

## Final Report Format

Provide a comprehensive report structured as:
1.  **Executive Summary** (A 3-sentence overview of the agent's profile, skill, and consistency).
2.  **Detailed Strategic Profile** (Analysis of its core strategies and decision-making).
3.  **Tactical Skill Assessment** (Evaluation of its paddle control, responsiveness, and striking techniques).
4.  **Performance Analysis** (Patterns and key differentiators across performance tiers).
5.  **Conclusion & Recommendations** (Overall assessment and potential areas for improvement).

Focus on insights that emerge from comparing multiple episodes. Contrast the different performance tiers to build a complete picture of the agent's capabilities.
"""

    def _wait_for_file_active(self, uploaded_file: Any, max_wait_time: int = 120) -> None:
        """
        Wait for uploaded file to become active.
        
        Args:
            uploaded_file: The uploaded file object
            max_wait_time: Maximum time to wait in seconds
        """
        self.logger.info(f"Waiting for file {uploaded_file.name} to become active...")
        start_time = time.time()
        
        while time.time() - start_time < max_wait_time:
            try:
                # Get file status using correct API call
                file_info = self.client.files.get(name=uploaded_file.name)
                
                if hasattr(file_info, 'state') and file_info.state == 'ACTIVE':
                    self.logger.info(f"File {uploaded_file.name} is now active")
                    return
                elif hasattr(file_info, 'state') and file_info.state == 'FAILED':
                    raise Exception(f"File processing failed for {uploaded_file.name}")
                
                # Wait before checking again
                time.sleep(2)
                
            except Exception as e:
                self.logger.warning(f"Error checking file status: {e}")
                time.sleep(2)
        
        raise Exception(f"File {uploaded_file.name} did not become active within {max_wait_time} seconds")
    
    def upload_video(self, video_path: str) -> Any:
        """
        Upload a video file to Gemini for analysis.
        
        Args:
            video_path: Path to the video file
            
        Returns:
            Uploaded file object
            
        Raises:
            FileNotFoundError: If video file doesn't exist
            Exception: If upload fails
        """
        video_path_obj = Path(video_path)
        if not video_path_obj.exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")
        
        self.logger.info(f"Uploading video: {video_path}")
        
        try:
            uploaded_file = self.client.files.upload(file=str(video_path_obj))
            self.uploaded_files.append(uploaded_file)
            self.logger.info(f"Successfully uploaded video. File ID: {uploaded_file.name}")
            
            # Wait for file to be processed
            self._wait_for_file_active(uploaded_file)
            
            return uploaded_file
        except Exception as e:
            self.logger.error(f"Failed to upload video {video_path}: {str(e)}")
            raise

    def detect_key_events(self, video_path: str, max_retries: int = 3) -> List[Dict[str, str]]:
        """
        Phase 2a: Detect key strategic events in a video with timestamps.
        
        Args:
            video_path: Path to the video file
            max_retries: Maximum number of retry attempts
            
        Returns:
            List of event dictionaries with timestamps and descriptions
        """
        uploaded_file = None
        
        for attempt in range(max_retries):
            try:
                # Upload video if not already uploaded
                if uploaded_file is None:
                    uploaded_file = self.upload_video(video_path)
                
                self.logger.info(f"Detecting key events in video: {video_path}")
                
                # Generate event detection analysis
                response = self.client.models.generate_content(
                    model=self.model,
                    contents=[uploaded_file, self.prompts["event_detection"]]
                )
                
                if response and response.text:
                    # Log the raw response for debugging
                    self.logger.debug(f"Raw Gemini response: {response.text[:500]}...")
                    
                    # Try to extract JSON from the response
                    response_text = response.text.strip()
                    
                    # Look for JSON array in the response
                    import json
                    import re
                    
                    # Try to find JSON array pattern
                    json_match = re.search(r'\[.*\]', response_text, re.DOTALL)
                    if json_match:
                        json_str = json_match.group(0)
                        try:
                            events = json.loads(json_str)
                            self.logger.info(f"Detected {len(events)} key events")
                            return events
                        except json.JSONDecodeError as e:
                            self.logger.warning(f"Failed to parse extracted JSON: {e}")
                            self.logger.debug(f"Extracted JSON string: {json_str[:200]}...")
                    
                    # If no JSON found, try parsing the whole response
                    try:
                        events = json.loads(response_text)
                        self.logger.info(f"Detected {len(events)} key events")
                        return events
                    except json.JSONDecodeError as e:
                        self.logger.warning(f"Failed to parse response as JSON: {e}")
                        self.logger.debug(f"Full response text: {response_text[:200]}...")
                        raise ValueError(f"Invalid JSON response: {response_text[:100]}...")
                else:
                    raise ValueError("Empty response from Gemini")
                    
            except Exception as e:
                self.logger.warning(f"Event detection attempt {attempt + 1} failed: {str(e)}")
                if attempt == max_retries - 1:
                    self.logger.error(f"All {max_retries} attempts failed")
                    raise
                
                # Wait before retry
                time.sleep(2 ** attempt)
        
        raise Exception("Event detection failed after all retries")

    def guided_analysis(self, video_path: str, key_events: List[Dict[str, str]], max_retries: int = 3) -> str:
        """
        Phase 2b: Perform guided strategic analysis using detected events.
        
        Args:
            video_path: Path to the video file
            key_events: List of key events from event detection phase
            max_retries: Maximum number of retry attempts
            
        Returns:
            Detailed strategic analysis text
        """
        uploaded_file = None
        
        for attempt in range(max_retries):
            try:
                # Upload video if not already uploaded
                if uploaded_file is None:
                    uploaded_file = self.upload_video(video_path)
                
                # Format key events for the prompt
                events_text = "\n".join([
                    f"- {event['timestamp']}: {event['event']} (Type: {event['type']})"
                    for event in key_events
                ])
                
                # Create guided analysis prompt with events
                prompt = self.prompts["guided_analysis"].format(key_events=events_text)
                
                self.logger.info(f"Performing guided analysis with {len(key_events)} key events")
                
                # Generate guided analysis
                response = self.client.models.generate_content(
                    model=self.model,
                    contents=[uploaded_file, prompt]
                )
                
                if response and response.text:
                    self.logger.info("Guided analysis completed successfully")
                    return response.text.strip()
                else:
                    raise ValueError("Empty response from Gemini")
                    
            except Exception as e:
                self.logger.warning(f"Guided analysis attempt {attempt + 1} failed: {str(e)}")
                if attempt == max_retries - 1:
                    self.logger.error(f"All {max_retries} attempts failed")
                    raise
                
                # Wait before retry
                time.sleep(2 ** attempt)
        
        raise Exception("Guided analysis failed after all retries")

    def meta_synthesis(self, all_summaries: Dict[str, List[str]], max_retries: int = 3) -> str:
        """
        Phase 3: Synthesize all individual analyses into a comprehensive report.
        
        Args:
            all_summaries: Dictionary with tier names as keys and lists of summaries as values
            max_retries: Maximum number of retry attempts
            
        Returns:
            Final synthesized report
        """
        for attempt in range(max_retries):
            try:
                # Format all summaries for the prompt
                formatted_summaries = []
                for tier, summaries in all_summaries.items():
                    formatted_summaries.append(f"\n=== {tier.upper()} PERFORMANCE TIER ===\n")
                    for i, summary in enumerate(summaries, 1):
                        formatted_summaries.append(f"\n--- {tier.title()} Tier Analysis {i} ---\n{summary}\n")
                
                summaries_text = "\n".join(formatted_summaries)
                
                # Create meta-synthesis prompt
                prompt = self.prompts["meta_synthesis"].format(all_summaries=summaries_text)
                
                self.logger.info(f"Performing meta-synthesis of {sum(len(s) for s in all_summaries.values())} analyses")
                
                # Generate meta-synthesis (text-only, no video needed)
                response = self.client.models.generate_content(
                    model=self.model,
                    contents=[prompt]
                )
                
                if response and response.text:
                    self.logger.info("Meta-synthesis completed successfully")
                    return response.text.strip()
                else:
                    raise ValueError("Empty response from Gemini")
                    
            except Exception as e:
                self.logger.warning(f"Meta-synthesis attempt {attempt + 1} failed: {str(e)}")
                if attempt == max_retries - 1:
                    self.logger.error(f"All {max_retries} attempts failed")
                    raise
                
                # Wait before retry
                time.sleep(2 ** attempt)
        
        raise Exception("Meta-synthesis failed after all retries")
    
    def upload_video_for_high_level_analysis(self, video_path: str) -> Any:
        """
        Upload a video file to Gemini for high-level strategy analysis.
        
        Args:
            video_path: Path to the video file
            
        Returns:
            Uploaded file object
            
        Raises:
            FileNotFoundError: If video file doesn't exist
            Exception: If upload fails
        """
        video_path_obj = Path(video_path)
        if not video_path_obj.exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")
        
        self.logger.info(f"Uploading video for high-level analysis: {video_path}")
        
        try:
            uploaded_file = self.client.files.upload(file=str(video_path_obj))
            self.uploaded_files.append(uploaded_file)
            self.logger.info(f"Successfully uploaded video. File ID: {uploaded_file.name}")
            
            # Wait for file to be processed
            self._wait_for_file_active(uploaded_file)
            
            return uploaded_file
        except Exception as e:
            self.logger.error(f"Failed to upload video {video_path}: {str(e)}")
            raise
    
    def analyze_video_high_level(self, video_path: str, prompt: str, max_retries: int = 3) -> str:
        """
        Analyze a video for high-level strategy using file upload approach.
        
        Args:
            video_path: Path to the video file
            prompt: Analysis prompt for Gemini
            max_retries: Maximum number of retry attempts
            
        Returns:
            Analysis result as text
        """
        uploaded_file = None
        
        for attempt in range(max_retries):
            try:
                # Upload video if not already uploaded
                if uploaded_file is None:
                    uploaded_file = self.upload_video_for_high_level_analysis(video_path)
                
                self.logger.info(f"Analyzing video with high-level prompt: {prompt[:100]}...")
                
                # Generate analysis using gemini-2.0-flash
                response = self.client.models.generate_content(
                    model=self.model,
                    contents=[uploaded_file, prompt]
                )
                
                if response and response.text:
                    self.logger.info("High-level analysis completed successfully")
                    return response.text.strip()
                else:
                    raise ValueError("Empty response from Gemini")
                    
            except Exception as e:
                self.logger.warning(f"High-level analysis attempt {attempt + 1} failed: {str(e)}")
                if attempt == max_retries - 1:
                    self.logger.error(f"All {max_retries} attempts failed")
                    raise
                
                # Wait before retry
                time.sleep(2 ** attempt)
        
        raise Exception("High-level analysis failed after all retries")
    
    def analyze_video_low_level(self, video_path: str, prompt: str, fps: int = 5, max_retries: int = 3) -> str:
        """
        Analyze a video for low-level behavior using direct inline data approach (for videos <20MB).
        
        Args:
            video_path: Path to the video file
            prompt: Analysis prompt for Gemini
            fps: FPS setting for video processing (hyperparameter)
            max_retries: Maximum number of retry attempts
            
        Returns:
            Analysis result as text
        """
        video_path_obj = Path(video_path)
        if not video_path_obj.exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")
        
        # Check file size
        file_size_mb = video_path_obj.stat().st_size / (1024 * 1024)
        if file_size_mb >= 20:
            self.logger.warning(f"Video file size ({file_size_mb:.1f}MB) is >= 20MB, may cause issues with low-level analysis")
        
        for attempt in range(max_retries):
            try:
                self.logger.info(f"Analyzing video with low-level approach (fps={fps}): {prompt[:100]}...")
                
                # Read video bytes
                with open(video_path_obj, 'rb') as f:
                    video_bytes = f.read()
                
                # Generate analysis using gemini-2.5-flash-preview-05-20 with direct video bytes
                response = self.client.models.generate_content(
                    model=self.model,
                    contents=types.Content(
                        parts=[
                            types.Part(
                                inline_data=types.Blob(
                                    data=video_bytes,
                                    mime_type='video/mp4'
                                ),
                                video_metadata=types.VideoMetadata(fps=fps)
                            ),
                            types.Part(text=prompt)
                        ]
                    )
                )
                
                if response and response.text:
                    self.logger.info("Low-level analysis completed successfully")
                    return response.text.strip()
                else:
                    raise ValueError("Empty response from Gemini")
                    
            except Exception as e:
                self.logger.warning(f"Low-level analysis attempt {attempt + 1} failed: {str(e)}")
                if attempt == max_retries - 1:
                    self.logger.error(f"All {max_retries} attempts failed")
                    raise
                
                # Wait before retry
                time.sleep(2 ** attempt)
        
        raise Exception("Low-level analysis failed after all retries")
    
    def batch_analyze_videos_high_level(self, video_paths: List[str], prompt: str) -> Dict[str, str]:
        """
        Analyze multiple videos with high-level approach.
        
        Args:
            video_paths: List of video file paths
            prompt: Analysis prompt for all videos
            
        Returns:
            Dictionary mapping video paths to analysis results
        """
        results = {}
        
        for video_path in video_paths:
            try:
                self.logger.info(f"Processing video for high-level analysis: {video_path}")
                result = self.analyze_video_high_level(video_path, prompt)
                results[video_path] = result
            except Exception as e:
                self.logger.error(f"Failed to analyze {video_path}: {str(e)}")
                results[video_path] = f"Error: {str(e)}"
        
        return results
    
    def batch_analyze_videos_low_level(self, video_paths: List[str], prompt: str, fps: int = 5) -> Dict[str, str]:
        """
        Analyze multiple videos with low-level approach.
        
        Args:
            video_paths: List of video file paths
            prompt: Analysis prompt for all videos
            fps: FPS setting for video processing
            
        Returns:
            Dictionary mapping video paths to analysis results
        """
        results = {}
        
        for video_path in video_paths:
            try:
                self.logger.info(f"Processing video for low-level analysis: {video_path}")
                result = self.analyze_video_low_level(video_path, prompt, fps=fps)
                results[video_path] = result
            except Exception as e:
                self.logger.error(f"Failed to analyze {video_path}: {str(e)}")
                results[video_path] = f"Error: {str(e)}"
        
        return results
    
    def cleanup_uploaded_files(self):
        """Clean up uploaded files from Gemini storage."""
        for uploaded_file in self.uploaded_files:
            try:
                # Delete the uploaded file
                self.client.files.delete(name=uploaded_file.name)
                self.logger.info(f"Cleaned up uploaded file: {uploaded_file.name}")
            except Exception as e:
                self.logger.warning(f"Failed to cleanup file {uploaded_file.name}: {str(e)}")
        
        self.uploaded_files.clear()
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit with cleanup."""
        self.cleanup_uploaded_files() 