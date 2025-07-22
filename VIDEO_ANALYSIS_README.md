# Video Analysis with Gemini API

This script analyzes video files using Gemini's video understanding API to provide technical RL policy analysis of agent behavior in Breakout gameplay.

## Overview

The video analysis script (`analyze_videos_with_gemini.py`) uses Gemini 2.5-pro to analyze agent behavior directly from video files, providing technical insights about policy execution, control errors, and behavioral patterns.

## Key Features

- **Direct Video Analysis**: Processes MP4 video files directly without frame extraction
- **Technical RL Focus**: Uses specialized prompt for technical reinforcement learning analysis
- **Batch Processing**: Can analyze multiple videos in one run
- **Comprehensive Output**: Generates individual analyses, combined reports, and JSON summaries
- **Error Handling**: Robust error handling with detailed logging

## Script: `analyze_videos_with_gemini.py`

### Usage

**Basic Usage:**
```bash
# Analyze specific video files
python analyze_videos_with_gemini.py video_clips/demo_15-25s.mp4 video_clips/demo_28-41s.mp4

# Analyze with verbose logging
python analyze_videos_with_gemini.py --verbose video_clips/demo_15-25s.mp4 video_clips/demo_28-41s.mp4

# Analyze all videos in a directory
python analyze_videos_with_gemini.py --video-dir video_clips --include-pattern "*.mp4"

# Custom output directory
python analyze_videos_with_gemini.py --output-dir my_video_analysis video_clips/demo_15-25s.mp4
```

**Default Behavior:**
If no video files are specified, the script will automatically analyze:
- `video_clips/demo_15-25s.mp4`
- `video_clips/demo_28-41s.mp4`

### Command Line Options

- `video_files`: Specific video files to analyze (positional arguments)
- `--video-dir`: Directory containing video files to analyze
- `--output-dir`: Output directory for results (default: `video_analysis_results`)
- `--include-pattern`: File pattern to match (default: `*.mp4`)
- `--verbose`: Enable detailed logging

## Analysis Approach

### Technical RL Policy Analysis Prompt

The script uses a specialized prompt that focuses on:

1. **Key Moments**: Identification of critical events and failures
2. **Timestamps**: Precise time ranges for important events
3. **Control Errors**: Analysis of agent control failures and causes
4. **Policy Formation**: Behavioral patterns and policy characteristics

### Technical Language Requirements

The analysis uses strictly technical terminology:
- "projectile" (not "ball")
- "agent-controlled platform" (not "paddle")
- "static elements" (not "bricks")
- "termination event" (not "life lost")
- "policy" and "action vector" for behavioral analysis

## Output Structure

```
video_analysis_results/
├── demo_15-25s_analysis.md          # Individual analysis for first video
├── demo_28-41s_analysis.md          # Individual analysis for second video
├── combined_video_analysis.md       # Combined report with all analyses
└── video_analysis_summary.json     # JSON summary with metadata
```

### Individual Analysis Format

Each video analysis includes:
- **File metadata**: Name, size, analysis timestamp
- **Technical RL Policy Analysis**: 
  - Key moments with timestamps
  - Control error analysis
  - Policy behavior observations
  - Technical insights (under 200 words)

### Combined Report

The combined report provides:
- Summary statistics (total videos, success/failure counts)
- All individual analyses in one document
- Easy comparison between different videos

## Example Analysis Output

### Key Moment Analysis
```
**Key Moment & Control Error (0:07 - 0:13):**
A termination event occurs at 0:13 due to a failure in the agent's control policy. 
Following projectile contact with a static element at 0:07, it assumes a 
high-velocity, low-angle trajectory toward the bottom-right. The agent's policy 
correctly generates a rightward action vector for the controlled platform. 
However, the execution is suboptimal; the platform's velocity is insufficient 
to reach the intercept point before the projectile passes its position.
```

### Policy Observation
```
**Policy Observation:**
Throughout the clip, the agent exhibits a reactive policy, primarily adjusting 
the platform's position to track the projectile's horizontal coordinate. This 
simple heuristic proves inadequate for the high-speed trajectory in the final 
sequence. The policy fails to demonstrate predictive capability, not accounting 
for the projectile's velocity and the time needed for its own platform to respond.
```

## Prerequisites

### API Key Setup
You need a Google Gemini API key. Set it up using one of these methods:

```bash
# Environment variable
export GEMINI_API_KEY="your-api-key-here"

# Or create a key file
echo "your-api-key-here" > Gemini_API_KEY.txt
```

### Dependencies
Ensure you have the required packages:
```bash
pip install google-generativeai
```

### Video Files
- Supported format: MP4
- Recommended: Short clips (10-60 seconds) for focused analysis
- File size: Should be reasonable for API upload (typically < 50MB)

## Example Workflow

### Quick Analysis of Default Videos
```bash
# Set your API key
export GEMINI_API_KEY="your-api-key-here"

# Run analysis on default videos
python analyze_videos_with_gemini.py --verbose
```

### Custom Video Analysis
```bash
# Analyze specific videos with custom output
python analyze_videos_with_gemini.py \
    --output-dir custom_analysis \
    --verbose \
    path/to/video1.mp4 path/to/video2.mp4
```

### Batch Directory Analysis
```bash
# Analyze all MP4 files in a directory
python analyze_videos_with_gemini.py \
    --video-dir video_clips \
    --include-pattern "demo_*.mp4" \
    --output-dir batch_analysis \
    --verbose
```

## Analysis Quality

The script provides:

- **Precise Timestamps**: Exact time ranges for critical events
- **Technical Accuracy**: Focuses on control systems and policy execution
- **Behavioral Insights**: Identifies patterns in agent decision-making
- **Failure Analysis**: Detailed examination of control errors and causes
- **Concise Reports**: Under 200 words per video for focused insights

## Comparison with Frame Analysis

| Aspect | Video Analysis | Frame Analysis |
|--------|----------------|----------------|
| **Input** | MP4 video files | Individual frame images |
| **Processing** | Direct video upload | Batch frame processing |
| **Analysis Depth** | Policy-level insights | Frame-by-frame trajectory |
| **Temporal Understanding** | Continuous motion | Discrete snapshots |
| **Output Length** | ~200 words per video | Multiple batch analyses + synthesis |
| **Best For** | Policy evaluation | Detailed motion analysis |

## Troubleshooting

### Common Issues

1. **API Key Not Found**
   ```
   Error: API key required
   ```
   Solution: Set `GEMINI_API_KEY` environment variable or create key file

2. **Video File Not Found**
   ```
   Warning: Video file not found: path/to/video.mp4
   ```
   Solution: Check file paths and ensure videos exist

3. **Video Too Large**
   ```
   Error: File size exceeds limit
   ```
   Solution: Compress video or use shorter clips

4. **Empty Response**
   ```
   Error: Empty response from Gemini
   ```
   Solution: Check API key validity and video content

### Debugging

Use verbose logging for detailed information:
```bash
python analyze_videos_with_gemini.py --verbose video1.mp4 video2.mp4
```

### Performance Tips

- **File Size**: Keep videos under 10MB for faster processing
- **Duration**: 10-60 second clips work best for focused analysis
- **Quality**: Good video quality helps with accurate analysis
- **Content**: Clear agent behavior is easier to analyze

## Cost Considerations

- Video analysis uses Gemini's multimodal capabilities
- Larger video files consume more tokens
- Monitor usage through Google Cloud Console
- Consider shorter clips to optimize costs

## Customization

### Modify Analysis Prompt
Edit the `get_video_analysis_prompt()` function in the script to customize:
- Analysis focus areas
- Technical terminology
- Output format requirements
- Word count limits

### Add New Analysis Types
Extend the script to support:
- Different analysis frameworks
- Multiple prompt variations
- Custom output formats
- Additional metadata extraction

### Integration
The script can be integrated with:
- Automated analysis pipelines
- Video processing workflows
- RL training evaluation systems
- Performance monitoring tools 