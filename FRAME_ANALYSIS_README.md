# Frame Analysis Scripts

This directory contains scripts to analyze video frames using Gemini 2.5-pro for Breakout gameplay analysis.

## Overview

The frame analysis pipeline consists of two main steps:
1. **Batch Analysis**: Analyze frames in batches of 10 using Gemini 2.5-pro to describe ball trajectory and paddle behavior
2. **Synthesis**: Combine all batch analyses into a comprehensive final report about the agent's strategy

## Scripts

### 1. `analyze_frame_batches.py`
Analyzes video frames in batches of 10 using Gemini 2.5-pro.

**Usage:**
```bash
python analyze_frame_batches.py <frames_directory> [options]

# Example:
python analyze_frame_batches.py video_clips/frames_15-25s
python analyze_frame_batches.py video_clips/frames_28-41s --output-dir custom_output --verbose
```

**Options:**
- `--output-dir`: Output directory for analysis results (default: `<frames_dir>/gemini_analysis`)
- `--batch-size`: Number of frames per batch (default: 10)
- `--verbose`: Enable verbose logging

**Output:**
- Individual batch analysis files: `batch_001_frames_X-Y.md`
- Summary JSON file: `batch_analysis_summary.json`

### 2. `synthesize_frame_analysis.py`
Synthesizes all batch analyses into a final comprehensive report.

**Usage:**
```bash
python synthesize_frame_analysis.py <analysis_directory> [options]

# Example:
python synthesize_frame_analysis.py video_clips/frames_15-25s/gemini_analysis
python synthesize_frame_analysis.py analysis_results --output-file final_report.md
```

**Options:**
- `--output-file`: Output file for final report (default: `<analysis_dir>/final_analysis_report.md`)
- `--output-dir`: Output directory (for multiple analysis directories)
- `--multiple`: Process multiple analysis directories
- `--verbose`: Enable verbose logging

**Output:**
- Final report: `final_analysis_report.md`
- Results metadata: `final_analysis_report.json`

### 3. `analyze_all_frame_folders.py` (Convenience Script)
Runs the complete pipeline for both frame folders automatically.

**Usage:**
```bash
python analyze_all_frame_folders.py [options]

# Example:
python analyze_all_frame_folders.py --verbose
python analyze_all_frame_folders.py --output-dir my_analysis_results
```

**Options:**
- `--frames-15-25s`: Path to frames_15-25s directory (default: `video_clips/frames_15-25s`)
- `--frames-28-41s`: Path to frames_28-41s directory (default: `video_clips/frames_28-41s`)
- `--output-dir`: Base output directory (default: `frame_analysis_results`)
- `--verbose`: Enable verbose logging

**Output:**
- Analysis results for both folders in separate subdirectories
- Pipeline summary: `pipeline_summary.json`

## Prerequisites

1. **API Key**: You need a Google Gemini API key. Set it up using one of these methods:
   ```bash
   # Environment variable
   export GEMINI_API_KEY="your-api-key-here"
   
   # Or create a key file
   echo "your-api-key-here" > Gemini_API_KEY.txt
   ```

2. **Dependencies**: Make sure you have installed:
   ```bash
   pip install google-genai
   ```

3. **Frame Files**: Ensure your frame directories contain PNG/JPG files named sequentially (e.g., `frame_0001.png`, `frame_0002.png`, etc.)

## Analysis Process

### Step 1: Batch Analysis
- Divides frames into batches of 10
- Uploads each batch to Gemini 2.5-pro
- Analyzes ball trajectory and paddle behavior for each batch
- Saves individual batch analyses

### Step 2: Synthesis
- Loads all batch analyses
- Uses Gemini 2.5-pro to synthesize into a comprehensive report
- Provides strategic insights about the agent's behavior

## Example Workflow

### Quick Start (Analyze Both Folders)
```bash
# Set your API key
export GEMINI_API_KEY="your-api-key-here"

# Run complete analysis for both folders
python analyze_all_frame_folders.py --verbose
```

### Manual Step-by-Step
```bash
# Step 1: Analyze frames_15-25s in batches
python analyze_frame_batches.py video_clips/frames_15-25s --verbose

# Step 2: Synthesize the batch analyses
python synthesize_frame_analysis.py video_clips/frames_15-25s/gemini_analysis --verbose

# Step 3: Repeat for frames_28-41s
python analyze_frame_batches.py video_clips/frames_28-41s --verbose
python synthesize_frame_analysis.py video_clips/frames_28-41s/gemini_analysis --verbose
```

## Output Structure

```
frame_analysis_results/
├── frames_15-25s_analysis/
│   ├── batch_001_frames_0001-0010.md
│   ├── batch_002_frames_0011-0020.md
│   ├── ...
│   ├── batch_analysis_summary.json
│   ├── final_report.md
│   └── final_report.json
├── frames_28-41s_analysis/
│   ├── batch_001_frames_0001-0010.md
│   ├── batch_002_frames_0011-0020.md
│   ├── ...
│   ├── batch_analysis_summary.json
│   ├── final_report.md
│   └── final_report.json
└── pipeline_summary.json
```

## Analysis Format

### Batch Analysis Format
Each batch analysis follows this structure:
- **Overall Trajectory Summary**: Complete ball path description
- **Detailed Motion Breakdown**: Chronological bullet points of ball and paddle behavior

### Final Report Format
The synthesis creates a comprehensive report with:
- **Agent Strategy Overview**: Overall approach and risk management
- **Paddle Control Analysis**: Responsiveness and positioning strategy
- **Performance Consistency**: Reliability and failure patterns
- **Key Behavioral Patterns**: Recurring behavior patterns
- **Final Assessment**: Concise summary of capabilities and limitations

## Troubleshooting

### Common Issues

1. **API Key Not Found**
   ```
   Error: API key required. Set GEMINI_API_KEY or GOOGLE_API_KEY environment variable
   ```
   Solution: Set your API key as described in Prerequisites

2. **No Frame Files Found**
   ```
   Error: No frame files found in directory
   ```
   Solution: Ensure the directory contains PNG/JPG files

3. **Incomplete Batches**
   ```
   Warning: No complete batches of 10 frames found
   ```
   Solution: Check if you have at least 10 frames, or adjust `--batch-size`

4. **Upload Failures**
   ```
   Error: Failed to upload frame
   ```
   Solution: Check internet connection and API key validity

### Debugging

Use the `--verbose` flag to get detailed logging:
```bash
python analyze_frame_batches.py video_clips/frames_15-25s --verbose
```

### File Cleanup

The scripts automatically clean up uploaded files, but if you need to manually clean up:
- Check the Gemini Files API dashboard
- Delete any stuck uploaded files

## Cost Considerations

- Each batch analysis uploads 10 images to Gemini
- Each synthesis uses text-only analysis
- Monitor your API usage through the Google Cloud Console
- Consider using smaller batch sizes for cost optimization

## Customization

### Modify Analysis Prompts
Edit the prompt functions in the scripts:
- `get_frame_analysis_prompt()` in `analyze_frame_batches.py`
- `get_synthesis_prompt()` in `synthesize_frame_analysis.py`

### Adjust Batch Size
Use `--batch-size` parameter to change from default 10 frames:
```bash
python analyze_frame_batches.py frames_dir --batch-size 5
```

### Custom Output Formats
Modify the output writing sections in the scripts to change file formats or add additional metadata. 