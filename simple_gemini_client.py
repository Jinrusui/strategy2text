#!/usr/bin/env python3
"""
Simplified Gemini client for image analysis using google-generativeai package.
"""

import os
import logging
from typing import Optional, List
from pathlib import Path
import time
import io
import numpy as np
from PIL import Image

try:
    import google.generativeai as genai
    from google.generativeai.types import HarmCategory, HarmBlockThreshold
except ImportError:
    raise ImportError("Please install google-generativeai: pip install google-generativeai")


class SimpleGeminiClient:
    """Simplified client for image analysis using Gemini."""
    
    def __init__(self, api_key: Optional[str] = None, model: str = "gemini-2.5-pro"):
        """
        Initialize Gemini client.
        
        Args:
            api_key: Google API key. If None, will try to get from env vars
            model: Model to use for analysis
        """
        
        self.api_key = (
            api_key or 
            self._read_key_from_file("GEMINI_API_KEY.txt") or
            os.getenv("GEMINI_API_KEY") or 
            os.getenv("GOOGLE_API_KEY")
        )
        if not self.api_key:
            raise ValueError(
                "API key required. Set GEMINI_API_KEY or GOOGLE_API_KEY environment variable or pass api_key parameter"
            )
        
        # Configure the API
        genai.configure(api_key=self.api_key)
        
        self.model_name = model
        self.model = genai.GenerativeModel(model)
        self.logger = logging.getLogger(__name__)
        
        # Safety settings to be more permissive for game analysis
        self.safety_settings = {
            HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
        }
    
    def _read_key_from_file(self, filename: str) -> Optional[str]:
        """Read API key from file."""
        try:
            with open(filename, "r") as f:
                return f.read().strip()
        except FileNotFoundError:
            return None
    
    def analyze_images_with_prompt(self, image_paths: List[Path], prompt: str, max_retries: int = 3) -> str:
        """
        Analyze a batch of images with a given prompt.
        
        Args:
            image_paths: List of image file paths
            prompt: Analysis prompt
            max_retries: Maximum number of retry attempts
            
        Returns:
            Analysis result as text
        """
        self.logger.info(f"Analyzing {len(image_paths)} images with prompt")
        
        for attempt in range(max_retries):
            try:
                # Load images
                images = []
                for image_path in image_paths:
                    self.logger.debug(f"Loading image: {image_path}")
                    # Read image data
                    with open(image_path, 'rb') as f:
                        image_data = f.read()
                    
                    # Create image part
                    image_part = {
                        'mime_type': 'image/png',  # Assume PNG, could be made more flexible
                        'data': image_data
                    }
                    images.append(image_part)
                
                # Create content with images and prompt
                content = images + [prompt]
                
                # Generate analysis
                self.logger.info(f"Generating analysis (attempt {attempt + 1})")
                response = self.model.generate_content(
                    content,
                    safety_settings=self.safety_settings
                )
                
                if response and response.text:
                    self.logger.info("Analysis completed successfully")
                    return response.text.strip()
                else:
                    raise ValueError("Empty response from Gemini")
                    
            except Exception as e:
                self.logger.warning(f"Analysis attempt {attempt + 1} failed: {str(e)}")
                if attempt == max_retries - 1:
                    self.logger.error(f"All {max_retries} attempts failed")
                    raise
                
                # Wait before retry
                time.sleep(2 ** attempt)
        
        raise Exception("Analysis failed after all retries")
    
    def analyze_image_data_with_prompt(self, images_data: List[np.ndarray], prompt: str, max_retries: int = 3) -> str:
        """
        Analyze a batch of in-memory image data (NumPy arrays) with a given prompt.
        
        Args:
            images_data: List of images as NumPy arrays
            prompt: Analysis prompt
            max_retries: Maximum number of retry attempts
            
        Returns:
            Analysis result as text
        """
        self.logger.info(f"Analyzing {len(images_data)} image arrays with prompt")
        
        for attempt in range(max_retries):
            try:
                # Prepare content
                content = []
                for img_array in images_data:
                    # Encode numpy array to PNG bytes
                    with io.BytesIO() as output:
                        Image.fromarray(img_array).save(output, format="PNG")
                        png_data = output.getvalue()
                    
                    image_part = {
                        'mime_type': 'image/png',
                        'data': png_data
                    }
                    content.append(image_part)
                
                # Add prompt to content
                content.append(prompt)
                
                # Generate analysis
                self.logger.info(f"Generating analysis (attempt {attempt + 1})")
                response = self.model.generate_content(
                    content,
                    safety_settings=self.safety_settings
                )
                
                if response and response.text:
                    self.logger.info("Analysis completed successfully")
                    return response.text.strip()
                else:
                    raise ValueError("Empty response from Gemini")
                    
            except Exception as e:
                self.logger.warning(f"Analysis attempt {attempt + 1} failed: {str(e)}")
                if attempt == max_retries - 1:
                    self.logger.error(f"All {max_retries} attempts failed")
                    raise
                
                # Wait before retry
                time.sleep(2 ** attempt)
        
        raise Exception("Analysis failed after all retries")

    def analyze_text_with_prompt(self, text: str, prompt: str, max_retries: int = 3) -> str:
        """
        Analyze text with a given prompt (for synthesis).
        
        Args:
            text: Text to analyze
            prompt: Analysis prompt
            max_retries: Maximum number of retry attempts
            
        Returns:
            Analysis result as text
        """
        self.logger.info("Analyzing text with prompt")
        
        for attempt in range(max_retries):
            try:
                # Combine prompt and text
                full_prompt = f"{prompt}\n\n{text}"
                
                # Generate analysis
                self.logger.info(f"Generating text analysis (attempt {attempt + 1})")
                response = self.model.generate_content(
                    full_prompt,
                    safety_settings=self.safety_settings
                )
                
                if response and response.text:
                    self.logger.info("Text analysis completed successfully")
                    return response.text.strip()
                else:
                    raise ValueError("Empty response from Gemini")
                    
            except Exception as e:
                self.logger.warning(f"Text analysis attempt {attempt + 1} failed: {str(e)}")
                if attempt == max_retries - 1:
                    self.logger.error(f"All {max_retries} attempts failed")
                    raise
                
                # Wait before retry
                time.sleep(2 ** attempt)
        
        raise Exception("Text analysis failed after all retries")
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        # No cleanup needed for this simple client
        pass 