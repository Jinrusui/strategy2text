# Updated Gemini Analysis Scripts - New API Format

This document explains the updated Gemini analysis scripts that now use the new Google GenAI API format.

## Overview

The scripts have been rewritten to use two different approaches based on the analysis type:

### 1. High-Level Strategy Analysis
- **Model**: `gemini-2.0-flash`
- **Method**: File upload approach
- **Use case**: Strategic analysis, long-term planning, overall game approach
- **API Pattern**:
```python
from google import genai

client = genai.Client(api_key="GOOGLE_API_KEY")
myfile = client.files.upload(file="path/to/sample.mp4")
response = client.models.generate_content(
    model="gemini-2.0-flash", 
    contents=[myfile, "Analyze strategy..."]
)
```

### 2. Low-Level Behavior Analysis
- **Model**: `gemini-2.5-flash-preview-05-20`
- **Method**: Direct video bytes with configurable FPS
- **Use case**: Frame-by-frame analysis, immediate reactions, micro-behaviors
- **File size limit**: <20MB
- **API Pattern**:
```python
from google.genai import types

video_bytes = open("path/to/video.mp4", 'rb').read()
response = client.models.generate_content(
    model='models/gemini-2.5-flash-preview-05-20',
    contents=types.Content(
        parts=[
            types.Part(
                inline_data=types.Blob(
                    data=video_bytes,
                    mime_type='video/mp4'
                ),
                video_metadata=types.VideoMetadata(fps=5)  # Configurable hyperparameter
            ),
            types.Part(text='Analyze behavior...')
        ]
    )
)
```

## Key Changes

### GeminiClient (`src/gemini_analysis/gemini_client.py`)
- **New methods**:
  - `analyze_video_high_level()`: Uses file upload for strategy analysis
  - `analyze_video_low_level()`: Uses direct video bytes for behavior analysis
  - `batch_analyze_videos_high_level()`: Batch processing for strategy
  - `batch_analyze_videos_low_level()`: Batch processing for behavior
- **Removed**: Old unified `analyze_video()` method
- **Simplified**: No more model parameter in constructor

### BreakoutStrategyAnalyzer (`src/gemini_analysis/strategy_analyzer.py`)
- **Updated**: Uses `analyze_video_high_level()` method
- **Model**: Always uses `gemini-2.0-flash`
- **Method**: File upload approach for better strategic analysis
- **New method**: `batch_analyze_breakout_strategies()`

### BreakoutBehaviorAnalyzer (`src/gemini_analysis/behavior_analyzer.py`)
- **Updated**: Uses `analyze_video_low_level()` method
- **Model**: Always uses `gemini-2.5-flash-preview-05-20`
- **Method**: Direct video bytes with configurable FPS
- **Hyperparameter**: FPS setting (5, 10, etc.) for frame sampling rate
- **File size check**: Warns if video is ≥20MB
- **Updated method**: `batch_analyze_breakout_behaviors()` with FPS parameter

## Usage Examples

### Basic High-Level Strategy Analysis
```python
from src.gemini_analysis import BreakoutStrategyAnalyzer

# Initialize analyzer
with BreakoutStrategyAnalyzer() as analyzer:
    # Analyze strategy
    result = analyzer.analyze_breakout_strategy("video.mp4")
    
    # Save results
    analyzer.save_analysis(result, "strategy_analysis.json")
```

### Basic Low-Level Behavior Analysis
```python
from src.gemini_analysis import BreakoutBehaviorAnalyzer

# Initialize analyzer
with BreakoutBehaviorAnalyzer() as analyzer:
    # Analyze behavior with 5 FPS sampling
    result = analyzer.analyze_breakout_behavior("video.mp4", fps=5)
    
    # Try different FPS settings
    result_10fps = analyzer.analyze_breakout_behavior("video.mp4", fps=10)
    
    # Save results
    analyzer.save_analysis(result, "behavior_analysis_5fps.json")
    analyzer.save_analysis(result_10fps, "behavior_analysis_10fps.json")
```

### Batch Analysis
```python
video_paths = ["video1.mp4", "video2.mp4", "video3.mp4"]

# Batch strategy analysis
with BreakoutStrategyAnalyzer() as strategy_analyzer:
    results = strategy_analyzer.batch_analyze_breakout_strategies(video_paths)

# Batch behavior analysis with configurable FPS
with BreakoutBehaviorAnalyzer() as behavior_analyzer:
    results = behavior_analyzer.batch_analyze_breakout_behaviors(video_paths, fps=10)
```

## Configuration

### API Key Setup
Set your Google API key using one of these methods:
1. Environment variable: `export GOOGLE_API_KEY="your_key_here"`
2. Environment variable: `export GEMINI_API_KEY="your_key_here"`
3. File: Create `Gemini_API_KEY.txt` with your key
4. Pass directly: `BreakoutStrategyAnalyzer(api_key="your_key_here")`

### FPS Configuration (Low-Level Analysis)
The FPS parameter controls frame sampling rate for low-level analysis:
- **5 FPS**: Standard sampling, good balance of detail and efficiency
- **10 FPS**: Higher detail, more computational cost
- **Custom**: Experiment with different values based on your needs

### File Size Considerations
- **High-level analysis**: No strict file size limit (uses upload)
- **Low-level analysis**: Recommended <20MB for direct video bytes approach
- Files ≥20MB will show warnings but may still work

## Output Format

### Strategy Analysis Result
```json
{
  "video_path": "path/to/video.mp4",
  "environment": "Breakout",
  "analysis_type": "high_level_strategy",
  "model": "gemini-2.0-flash",
  "raw_analysis": "Full analysis text...",
  "strategy_summary": {
    "overall_strategy": "...",
    "strategic_brick_destruction": "...",
    "game_state_management": "...",
    "learning_evolution": "...",
    "performance_effectiveness": "..."
  },
  "timestamp": "2024-01-01T12:00:00"
}
```

### Behavior Analysis Result
```json
{
  "video_path": "path/to/video.mp4",
  "environment": "Breakout",
  "analysis_type": "low_level_behavior",
  "model": "gemini-2.5-flash-preview-05-20",
  "fps": 5,
  "file_size_mb": 15.2,
  "raw_analysis": "Full analysis text...",
  "behavior_summary": {
    "immediate_actions": "...",
    "movement_patterns": "...",
    "ball_tracking": "...",
    "technical_execution": "...",
    "breakout_specific": "...",
    "summary": "3-sentence summary..."
  },
  "timestamp": "2024-01-01T12:00:00"
}
```

## Dependencies

Make sure you have the latest Google GenAI library:
```bash
pip install google-genai
```

## Running the Example

See `example_usage.py` for a complete working example that demonstrates all features:
```bash
python example_usage.py
```

## Migration from Old API

If you were using the old scripts:
1. **Strategy analysis**: Replace `analyze_breakout_strategy()` calls - they now use high-level analysis automatically
2. **Behavior analysis**: Add FPS parameter to `analyze_breakout_behavior(video_path, fps=5)`
3. **Batch processing**: Use the new batch methods with appropriate parameters
4. **Model specification**: Remove model parameters from constructors - models are now fixed per analysis type 