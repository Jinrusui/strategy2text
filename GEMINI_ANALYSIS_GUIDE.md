# Gemini Analysis Framework Guide

This guide explains how to use the Gemini-based strategy analysis framework for analyzing RL agent gameplay videos, as described in your dissertation.

## Overview

The framework implements a Vision-Language Model (VLM) approach using Google's Gemini API to automatically translate visual gameplay into human-readable strategy summaries. It includes three key evaluation metrics:

1. **Predictive Faithfulness Score (PFS)** - Measures how well the strategy predicts future behavior
2. **Coverage Score** - Evaluates comprehensiveness of the strategy summary
3. **Abstraction Score** - Assesses the level of abstraction vs concrete descriptions

## Installation

1. Install the required dependencies:
```bash
pip install -r requirements.txt
```

2. Set up your Gemini API key:
   - Create a file named `Gemini_API_KEY.txt` in the project root with your API key
   - Or set the environment variable: `export GEMINI_API_KEY=your_api_key_here`

## Quick Start

### Basic Video Analysis

```python
from src.gemini_analysis import StrategyAnalyzer

# Initialize analyzer
analyzer = StrategyAnalyzer(
    video_dir="path/to/your/videos",
    output_dir="analysis_results"
)

# Analyze a single video
result = analyzer.analyze_single_video(
    video_path="path/to/video.mp4",
    analysis_type="strategy",
    max_frames=10
)

print(f"Strategy Summary: {result['strategy_summary']}")
print(f"Abstraction Score: {result['abstraction_score']['abstraction_score']:.3f}")
```

### Running the Demo

```bash
python scripts/demo_gemini_analysis.py
```

This demo will show you how to use all the framework components without requiring actual video files.

## Core Components

### 1. StrategyAnalyzer

The main orchestrator that ties everything together:

```python
from src.gemini_analysis import StrategyAnalyzer
from src.video_processing.video_sampler import SamplingStrategy

analyzer = StrategyAnalyzer(video_dir="videos", output_dir="results")

# Analyze video sets with different sampling strategies
results = analyzer.analyze_video_set(
    sampling_strategy=SamplingStrategy.LONGITUDINAL,
    num_videos=10,
    analysis_type="strategy"
)
```

### 2. GeminiClient

Direct interface to Google's Gemini API:

```python
from src.gemini_analysis import GeminiClient

client = GeminiClient()

# Analyze video with custom prompt
summary = client.analyze_video(
    video_path="video.mp4",
    prompt="Analyze this RL agent's strategy...",
    max_frames=10
)
```

### 3. PromptEngineer

Manages different types of analysis prompts:

```python
from src.gemini_analysis import PromptEngineer, AnalysisType

prompt_engineer = PromptEngineer()

# Get strategy-focused prompt
strategy_prompt = prompt_engineer.get_strategy_prompt(
    game_type="Breakout",
    focus_areas=["tunneling behavior", "defensive strategies"]
)

# Get baseline captioning prompt
baseline_prompt = prompt_engineer.get_prompt(AnalysisType.BASELINE_CAPTIONING)
```

### 4. EvaluationMetrics

Implements the three key evaluation metrics:

```python
from src.gemini_analysis import EvaluationMetrics

evaluator = EvaluationMetrics()

# Calculate Predictive Faithfulness Score
pfs = evaluator.calculate_predictive_faithfulness_score(
    prediction="Agent will tunnel left",
    ground_truth="Agent created tunnel on left side"
)

# Calculate Abstraction Score
abstraction = evaluator.calculate_abstraction_score(summary)

# Calculate Coverage Score (requires questions and answers)
coverage = evaluator.calculate_coverage_score(
    strategy_summary=summary,
    questions=questions_list,
    answers=answers_list
)
```

## Experimental Workflows

### 1. Baseline Comparison Experiment

Compare strategy-focused analysis vs baseline video captioning:

```python
# Run comparative experiment
experiment_results = analyzer.run_comparative_experiment(
    baseline_type="captioning",
    num_videos_per_strategy=10
)

# Results will show which approach produces higher abstraction scores
print(experiment_results['comparison'])
```

### 2. Ablation Studies

Test the importance of different framework components:

```python
# Run ablation studies
ablation_results = analyzer.run_ablation_study(
    ablation_types=["prompt", "sampling"],
    num_videos=15
)

# Compare different prompt types and sampling strategies
print(ablation_results['studies'])
```

### 3. Longitudinal Analysis

Analyze how strategy descriptions change across training checkpoints:

```python
# Analyze different training stages
longitudinal_results = analyzer.analyze_video_set(
    sampling_strategy=SamplingStrategy.LONGITUDINAL,
    num_videos=20,
    analysis_type="strategy"
)

# Shows how strategy sophistication evolves over training
print(longitudinal_results['aggregate_metrics'])
```

### 4. Full Dissertation Experiment

Run the complete experimental pipeline:

```python
# Run all experiments described in the dissertation
full_results = analyzer.run_full_experiment(
    experiment_name="dissertation_validation"
)

# Generates comprehensive report with all metrics
print(full_results['comprehensive_report'])
```

## Video Sampling Strategies

The framework supports different video sampling approaches:

- **TYPICAL**: Standard/modal gameplay behavior
- **EDGE_CASE**: Unusual situations, near-loss scenarios
- **LONGITUDINAL**: Different training stages/checkpoints
- **BALANCED**: Mix of all sampling types
- **RANDOM**: Random sampling

```python
from src.video_processing.video_sampler import VideoSampler, SamplingStrategy

sampler = VideoSampler("video_directory")

# Sample edge case videos
edge_videos = sampler.sample_videos(
    strategy=SamplingStrategy.EDGE_CASE,
    num_videos=5
)
```

## Evaluation Metrics Details

### Predictive Faithfulness Score (PFS)

Measures how well a strategy summary can predict future agent behavior:

1. Generate strategy summary from video
2. Show context frames to model
3. Ask model to predict next actions based on strategy
4. Compare prediction with actual behavior
5. Calculate semantic similarity score

### Coverage Score

Evaluates how comprehensively the strategy explains agent behaviors:

1. Generate questions about agent behavior from video
2. Answer questions using the strategy summary
3. Evaluate answer quality and completeness
4. Calculate percentage of adequately answered questions

### Abstraction Score

Measures whether summary describes underlying policies vs specific events:

- **High abstraction**: "Agent employs tunneling strategy"
- **Low abstraction**: "Agent moves paddle left, then right"

Uses linguistic analysis to count abstract vs concrete indicators.

## Output and Results

All analysis results are saved to JSON files in the output directory:

- `analysis_cache.json` - Cached analysis results
- `comparative_experiment_*.json` - Baseline comparison results
- `ablation_study_*.json` - Ablation study results
- `analysis.log` - Detailed logging information

Results include:
- Strategy summaries
- Evaluation metric scores
- Aggregate statistics
- Comprehensive reports

## Best Practices

1. **Video Quality**: Use clear, well-lit gameplay videos
2. **Frame Selection**: 8-12 frames usually sufficient for analysis
3. **Prompt Engineering**: Customize prompts for your specific game/domain
4. **Caching**: Enable caching to avoid re-analyzing same videos
5. **Batch Processing**: Use video sets rather than individual videos for robust results

## Troubleshooting

### Common Issues

1. **API Key Error**: Ensure Gemini API key is properly set
2. **Video Format**: Use MP4 or AVI formats
3. **Memory Issues**: Reduce max_frames if running out of memory
4. **Rate Limits**: Add delays between API calls if hitting rate limits

### Error Handling

The framework includes comprehensive error handling and logging. Check `analysis.log` for detailed error information.

## Integration with Existing Code

The Gemini analysis modules integrate seamlessly with your existing video processing pipeline:

```python
# Use with existing video sampler
from src.video_processing.video_sampler import VideoSampler
from src.gemini_analysis import StrategyAnalyzer

# Your existing video sampling
sampler = VideoSampler("videos")
videos = sampler.sample_videos(SamplingStrategy.TYPICAL, 10)

# Add Gemini analysis
analyzer = StrategyAnalyzer("videos")
for video in videos:
    result = analyzer.analyze_single_video(video['filepath'])
    # Process results...
```

## Research Applications

This framework enables several research directions:

1. **Cross-game Analysis**: Compare strategies across different games
2. **Architecture Comparison**: Analyze different RL algorithms
3. **Training Dynamics**: Study how strategies emerge during training
4. **Human-AI Comparison**: Compare AI and human strategies
5. **Interpretability Research**: Make RL agents more interpretable

## Citation

If you use this framework in your research, please cite your dissertation and mention the Gemini Strategy Analysis Framework.

---

For more examples and advanced usage, see the demo script: `scripts/demo_gemini_analysis.py` 