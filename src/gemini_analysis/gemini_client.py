"""
Gemini AI client for analyzing RL agent videos and extracting strategies.
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
    """Client for interacting with Gemini AI to analyze RL agent videos."""
    
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
        self.model = model or "gemini-2.5-pro-preview-06-05"
        
        # Track uploaded files for cleanup
        self.uploaded_files: List[Any] = []
    
    def _read_key_from_file(self, filename: str) -> Optional[str]:
        """Read API key from file."""
        try:
            with open(filename, "r") as f:
                return f.read().strip()
        except FileNotFoundError:
            return None
    
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
        video_path = Path(video_path)
        if not video_path.exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")
        
        self.logger.info(f"Uploading video for high-level analysis: {video_path}")
        
        try:
            uploaded_file = self.client.files.upload(file=str(video_path))
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
        video_path = Path(video_path)
        if not video_path.exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")
        
        # Check file size
        file_size_mb = video_path.stat().st_size / (1024 * 1024)
        if file_size_mb >= 20:
            self.logger.warning(f"Video file size ({file_size_mb:.1f}MB) is >= 20MB, may cause issues with low-level analysis")
        
        for attempt in range(max_retries):
            try:
                self.logger.info(f"Analyzing video with low-level approach (fps={fps}): {prompt[:100]}...")
                
                # Read video bytes
                with open(video_path, 'rb') as f:
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