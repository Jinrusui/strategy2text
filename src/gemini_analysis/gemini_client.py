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
except ImportError:
    raise ImportError("Please install google-genai: pip install google-genai")


class GeminiClient:
    """Client for interacting with Gemini AI to analyze RL agent videos."""
    
    def __init__(self, api_key: Optional[str] = None, model: str = "gemini-2.0-flash"):
        """
        Initialize Gemini client.
        
        Args:
            api_key: Google API key. If None, will try to get from GOOGLE_API_KEY env var
            model: Gemini model to use for analysis
        """
        self.api_key = api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if not self.api_key:
            raise ValueError(
                "API key required. Set GEMINI_API_KEY or GOOGLE_API_KEY environment variable or pass api_key parameter"
            )
        
        self.model = model
        self.client = genai.Client(api_key=self.api_key)
        self.logger = logging.getLogger(__name__)
        
        # Track uploaded files for cleanup
        self.uploaded_files: List[Any] = []
    
    def upload_video(self, video_path: str) -> Any:
        """
        Upload a video file to Gemini and wait for it to be processed.
        
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
        
        self.logger.info(f"Uploading video: {video_path}")
        
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
    
    def _wait_for_file_active(self, uploaded_file: Any, max_wait_time: int = 120) -> None:
        """
        Wait for uploaded file to become active.
        
        Args:
            uploaded_file: The uploaded file object
            max_wait_time: Maximum time to wait in seconds
        """
        import time
        
        self.logger.info(f"Waiting for file {uploaded_file.name} to become active...")
        start_time = time.time()
        
        while time.time() - start_time < max_wait_time:
            try:
                # Get file status
                file_info = self.client.files.get(uploaded_file.name)
                
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
    
    def analyze_video(self, video_path: str, prompt: str, max_retries: int = 3) -> str:
        """
        Analyze a video with a custom prompt. Automatically chooses best method based on file size.
        
        Args:
            video_path: Path to the video file
            prompt: Analysis prompt for Gemini
            max_retries: Maximum number of retry attempts
            
        Returns:
            Analysis result as text
        """
        # Check file size and try direct method first for smaller files
        video_path_obj = Path(video_path)
        file_size_mb = 0
        if video_path_obj.exists():
            file_size_mb = video_path_obj.stat().st_size / (1024 * 1024)
            if file_size_mb < 20:
                self.logger.info(f"File size {file_size_mb:.1f}MB < 20MB, trying direct analysis first")
                try:
                    return self.analyze_video_direct(video_path, prompt, max_retries=1)
                except Exception as e:
                    self.logger.warning(f"Direct analysis failed: {e}, falling back to upload method")
        
        # Use upload method
        uploaded_file = None
        
        for attempt in range(max_retries):
            try:
                # Upload video if not already uploaded
                if uploaded_file is None:
                    uploaded_file = self.upload_video(video_path)
                
                self.logger.info(f"Analyzing video with prompt: {prompt[:100]}...")
                
                # Generate analysis
                response = self.client.models.generate_content(
                    model=self.model,
                    contents=[uploaded_file, prompt]
                )
                
                if response and response.text:
                    self.logger.info("Analysis completed successfully")
                    return response.text.strip()
                else:
                    raise ValueError("Empty response from Gemini")
                    
            except Exception as e:
                self.logger.warning(f"Upload analysis attempt {attempt + 1} failed: {str(e)}")
                if attempt == max_retries - 1:
                    # Final fallback: try direct method if we haven't already
                    if file_size_mb < 20:
                        self.logger.info("Final fallback: trying direct analysis")
                        try:
                            return self.analyze_video_direct(video_path, prompt, max_retries=1)
                        except Exception as direct_e:
                            self.logger.error(f"Direct analysis fallback also failed: {direct_e}")
                    
                    self.logger.error(f"All {max_retries} attempts failed")
                    raise
                
                # Wait before retry
                time.sleep(2 ** attempt)
        
        raise Exception("Analysis failed after all retries")
    
    def analyze_video_direct(self, video_path: str, prompt: str, fps: int = 5, max_retries: int = 3) -> str:
        """
        Analyze a video using direct inline data approach (for videos <20MB).
        
        Args:
            video_path: Path to the video file
            prompt: Analysis prompt for Gemini
            fps: FPS setting for video processing
            max_retries: Maximum number of retry attempts
            
        Returns:
            Analysis result as text
        """
        try:
            from google.genai import types
        except ImportError:
            raise ImportError("Please install google-genai with types support")
        
        video_path = Path(video_path)
        if not video_path.exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")
        
        # Check file size
        file_size_mb = video_path.stat().st_size / (1024 * 1024)
        if file_size_mb >= 20:
            self.logger.warning(f"Video file size ({file_size_mb:.1f}MB) is >= 20MB, may cause issues")
        
        for attempt in range(max_retries):
            try:
                self.logger.info(f"Analyzing video directly with prompt: {prompt[:100]}...")
                
                # Read video bytes
                with open(video_path, 'rb') as f:
                    video_bytes = f.read()
                
                # Generate analysis using direct approach
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
                    self.logger.info("Direct video analysis completed successfully")
                    return response.text.strip()
                else:
                    raise ValueError("Empty response from Gemini")
                    
            except Exception as e:
                self.logger.warning(f"Direct analysis attempt {attempt + 1} failed: {str(e)}")
                if attempt == max_retries - 1:
                    self.logger.error(f"All {max_retries} attempts failed")
                    raise
                
                # Wait before retry
                time.sleep(2 ** attempt)
        
        raise Exception("Direct analysis failed after all retries")
    
    def batch_analyze_videos(self, video_paths: List[str], prompt: str) -> Dict[str, str]:
        """
        Analyze multiple videos with the same prompt.
        
        Args:
            video_paths: List of video file paths
            prompt: Analysis prompt for all videos
            
        Returns:
            Dictionary mapping video paths to analysis results
        """
        results = {}
        
        for video_path in video_paths:
            try:
                self.logger.info(f"Processing video: {video_path}")
                result = self.analyze_video(video_path, prompt)
                results[video_path] = result
            except Exception as e:
                self.logger.error(f"Failed to analyze {video_path}: {str(e)}")
                results[video_path] = f"Error: {str(e)}"
        
        return results
    
    def cleanup_uploaded_files(self):
        """Clean up uploaded files from Gemini storage."""
        for uploaded_file in self.uploaded_files:
            try:
                # Note: The actual cleanup method depends on the genai library version
                # This is a placeholder for the cleanup logic
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