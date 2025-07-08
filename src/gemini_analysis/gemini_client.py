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
You are an AI assistant analyzing a Breakout RL agent's gameplay video for key strategic moments.

Your task is to identify and timestamp the most important strategic events in this video. Focus on:

1. **Critical Decision Points**: Moments where the agent makes important strategic choices
2. **Tactical Errors**: Significant mistakes or missed opportunities
3. **Strategy Changes**: Points where the agent adapts or changes approach
4. **Key Successes**: Moments of particularly effective play or strategic wins
5. **Game State Transitions**: Important changes in game dynamics or brick patterns

**OUTPUT FORMAT**: Return ONLY a JSON list of events with this exact structure:
[
  {
    "timestamp": "MM:SS-MM:SS",
    "event": "Brief description of what happens",
    "type": "critical_decision|tactical_error|strategy_change|key_success|game_transition"
  }
]

**IMPORTANT**: 
- Provide timestamps in MM:SS-MM:SS format (e.g., "0:15-0:20")
- Keep event descriptions concise (1-2 sentences max)
- Include 5-10 most important events only
- Focus on strategic significance, not routine gameplay
- Return valid JSON format only - no additional text or explanations
"""
    
    def _get_guided_analysis_prompt(self) -> str:
        """Get the guided analysis prompt template for Phase 2b of HVA-X."""
        return """
You are an expert RL analyst conducting a comprehensive strategic analysis of a Breakout agent's gameplay.

**KEY MOMENTS FOR FOCUSED ANALYSIS:**
{key_events}

Using the key moments above as focal points, provide a detailed strategic analysis covering:

## Strategic Analysis Framework

### 1. Overall Strategy & Approach
- Long-term game plan and strategic thinking
- Risk vs reward decision making patterns
- Adaptation to different game phases
- Strategic positioning and forward planning

### 2. Strategic Execution Analysis
- How well the agent executes its intended strategy
- Consistency of strategic choices across similar situations
- Evidence of strategic learning and adaptation
- Quality of strategic decision-making under pressure

### 3. Performance-Critical Moments
- Analysis of the key moments identified above
- How these moments reflect the agent's strategic capabilities
- What these moments reveal about strengths and weaknesses
- Strategic implications of successes and failures

### 4. Tactical Competence
- Precision and effectiveness of tactical execution
- Ball control and paddle positioning strategies
- Brick-breaking efficiency and pattern recognition
- Response to changing game dynamics

### 5. Strategic Weaknesses & Strengths
- Primary strategic advantages demonstrated
- Key strategic limitations or blind spots
- Comparison to optimal strategic approaches
- Areas for strategic improvement

**CONCLUSION**: End with a 3-sentence summary capturing the agent's strategic profile, its primary strategic approach, and overall strategic effectiveness.

Focus on STRATEGIC intelligence rather than moment-to-moment actions. Use the key moments to illustrate broader strategic patterns and capabilities.
"""
    
    def _get_meta_synthesis_prompt(self) -> str:
        """Get the meta-synthesis prompt for Phase 3 of HVA-X."""
        return """
You are a lead RL analyst synthesizing multiple video analyses to create a comprehensive agent evaluation report.

**ANALYSIS DATA:**
{all_summaries}

Your task is to synthesize these individual analyses into a cohesive, high-level report that identifies:

## Synthesis Framework

### 1. Performance Patterns Across Tiers
- How does the agent's strategic approach differ between high, medium, and low-performing episodes?
- What strategic elements correlate with successful vs unsuccessful outcomes?
- Are there consistent strategic patterns within each performance tier?

### 2. Strategic Competence Profile
- What is the agent's core strategic approach to Breakout?
- Which strategic elements does the agent execute well consistently?
- What strategic limitations appear across multiple episodes?

### 3. Learning & Adaptation Evidence
- Does the agent show evidence of strategic learning or adaptation?
- Are there strategic capabilities that emerge in higher-performing episodes?
- What does the variation in performance tell us about the agent's strategic flexibility?

### 4. Critical Success & Failure Factors
- What strategic factors most strongly predict episode success?
- What are the common failure modes that lead to poor performance?
- How does the agent handle strategic pressure and critical moments?

### 5. Overall Strategic Assessment
- How would you characterize this agent's strategic intelligence?
- What are its primary strategic strengths and weaknesses?
- How does it compare to what you'd expect from an expert Breakout player?

## Final Report Format

Provide a comprehensive report structured as:
1. **Executive Summary** (2-3 sentences)
2. **Strategic Profile** (detailed analysis using the framework above)
3. **Performance Analysis** (patterns across tiers)
4. **Recommendations** (areas for improvement)
5. **Conclusion** (overall assessment)

Focus on strategic-level insights that can only be gained by analyzing multiple episodes across different performance levels.
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