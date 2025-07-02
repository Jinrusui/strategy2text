# Breakout Strategy & Behavior Analysis Scripts

This repository contains scripts to analyze high-level strategies and low-level behaviors of Breakout RL agent videos using Gemini AI.

## Scripts Overview

### 1. Individual Analysis Scripts

- **`analyze_high_level_strategy.py`** - Analyzes strategic decision-making and long-term planning
- **`analyze_low_level_behavior.py`** - Analyzes moment-to-moment actions and immediate reactions

### 2. Batch Processing Script

- **`batch_analyze_breakout_strategies.py`** - Runs both analyses in parallel for multiple videos

## Prerequisites

1. **Environment Variables**: Set your Gemini API key:
   ```bash
   export GEMINI_API_KEY="your_api_key_here"
   # or
   export GOOGLE_API_KEY="your_api_key_here"
   ```

2. **Dependencies**: Install required packages:
   ```bash
   pip install -r requirements.txt
   ```

## Usage Examples

### Quick Start - Batch Analysis

Analyze all videos in the `/videos` directory:

```bash
python batch_analyze_breakout_strategies.py videos/
```

### Advanced Usage

#### Batch Analysis with Custom Settings

```bash
# Integrated analysis (recommended)
python batch_analyze_breakout_strategies.py videos/ \
  --output my_analysis_results.json \
  --workers 3 \
  --verbose

# Using separate scripts method
python batch_analyze_breakout_strategies.py videos/ \
  --method separate \
  --workers 4 \
  --output combined_results.json
```

#### Individual Script Usage

```bash
# High-level strategy analysis only
python analyze_high_level_strategy.py videos/ \
  --output strategy_results.json \
  --workers 4

# Low-level behavior analysis only
python analyze_low_level_behavior.py videos/ \
  --output behavior_results.json \
  --workers 4
```

## Command Line Options

### Batch Analysis Script (`batch_analyze_breakout_strategies.py`)

```
positional arguments:
  video_dir             Directory containing video files

optional arguments:
  -h, --help            Show help message
  -o, --output OUTPUT   Output JSON file (default: breakout_analysis_results.json)
  -w, --workers WORKERS Number of parallel workers (default: 2)
  --method {integrated,separate}
                        Analysis method (default: integrated)
  -v, --verbose         Enable verbose logging
  --no-summary          Skip creating summary report
```

### Individual Scripts

Both `analyze_high_level_strategy.py` and `analyze_low_level_behavior.py` support:

```
positional arguments:
  video_dir             Directory containing video files

optional arguments:
  -h, --help            Show help message
  -o, --output OUTPUT   Output JSON file
  -w, --workers WORKERS Number of parallel workers (default: 4)
  -v, --verbose         Enable verbose logging
```

## Output Files

### JSON Results

The scripts generate detailed JSON files containing:

- **Analysis metadata** (timestamps, success rates, etc.)
- **Individual video results** with full analysis text
- **Structured summaries** for each analysis type
- **Error information** for failed analyses

### Summary Report

The batch script also creates a human-readable Markdown summary report (e.g., `breakout_analysis_results_summary.md`) containing:

- Success rates and statistics
- Key findings for each video
- Formatted analysis summaries

## Analysis Types

### High-Level Strategy Analysis

Focuses on:
- Overall strategy and approach
- Strategic brick destruction patterns
- Game state management
- Learning and strategic evolution
- Performance effectiveness

### Low-Level Behavior Analysis

Focuses on:
- Immediate actions and reactions
- Movement patterns and precision
- Decision-making speed
- Technical execution quality

## Performance Considerations

- **File Size**: Videos < 20MB use faster direct analysis
- **Parallel Processing**: Adjust `--workers` based on your system
- **Memory Usage**: Each worker uses ~1-2GB RAM
- **API Limits**: Gemini API has rate limits; reduce workers if needed

## Example Workflow

1. **Prepare videos**: Ensure videos are in supported formats (mp4, avi, mov, mkv, webm)

2. **Set API key**:
   ```bash
   export GEMINI_API_KEY="your_key_here"
   ```

3. **Run batch analysis**:
   ```bash
   python batch_analyze_breakout_strategies.py videos/ -v
   ```

4. **Review results**:
   - Check `breakout_analysis_results.json` for detailed data
   - Read `breakout_analysis_results_summary.md` for overview
   - Check log files for any issues

## Troubleshooting

### Common Issues

1. **API Key Error**: Ensure `GEMINI_API_KEY` or `GOOGLE_API_KEY` is set
2. **File Size Issues**: Large videos (>20MB) may fail; try smaller clips
3. **Rate Limiting**: Reduce `--workers` if getting API rate limit errors
4. **Memory Issues**: Reduce `--workers` on systems with limited RAM

### Log Files

Check these log files for detailed error information:
- `batch_breakout_analysis.log`
- `high_level_strategy_analysis.log`
- `low_level_behavior_analysis.log`

## Sample Output Structure

```json
{
  "analysis_type": "combined_breakout_analysis",
  "total_videos": 7,
  "successful_strategy_analyses": 7,
  "successful_behavior_analyses": 7,
  "results": {
    "path/to/video1.mp4": {
      "strategy_analysis": {
        "video_path": "path/to/video1.mp4",
        "analysis_type": "high_level_strategy",
        "strategy_summary": {
          "overall_strategy": "...",
          "strategic_brick_destruction": "...",
          "game_state_management": "..."
        }
      },
      "behavior_analysis": {
        "video_path": "path/to/video1.mp4",
        "analysis_type": "low_level_behavior",
        "behavior_summary": {
          "immediate_actions": "...",
          "movement_patterns": "...",
          "decision_making": "..."
        }
      }
    }
  }
}
``` 