# HVA-X: Hierarchical Video Analysis for Agent Explainability

HVA-X is a multi-pass video analysis system that uses AI to generate comprehensive, qualitative analyses of Reinforcement Learning agent behavior. The system analyzes agent gameplay videos to identify strategies, strengths, weaknesses, and behavioral patterns.

## 🎯 Quick Start

```bash
# 1. Set up your API key
export GEMINI_API_KEY="your-google-api-key-here"

# 2. Install dependencies
pip install -r requirements.txt
pip install google-genai

# 3. Run complete analysis (uses existing videos)
python use_existing_videos.py --num-episodes 12 --run-analysis

# 4. Or run full pipeline
python run_hva_pipeline.py --video-dir hva_videos
```

## 📋 Table of Contents

- [Overview](#overview)
- [Installation](#installation)
- [Usage](#usage)
  - [Complete Pipeline](#complete-pipeline)
  - [Individual Phases](#individual-phases)
  - [Video Sampling](#video-sampling)
- [File Formats](#file-formats)
- [Configuration](#configuration)
- [Troubleshooting](#troubleshooting)
- [Examples](#examples)

## 🔍 Overview

### The HVA-X Algorithm

HVA-X implements a four-phase analysis pipeline:

1. **Phase 1: Trajectory Sampling** - Stratifies videos by performance and samples representatives
2. **Phase 2A: Event Detection** - Identifies key strategic moments in each video
3. **Phase 2B: Guided Analysis** - Performs detailed analysis using detected events
4. **Phase 3: Meta-Synthesis** - Synthesizes individual analyses into comprehensive report

### Key Features

- **Multi-tier Analysis**: Analyzes low, medium, and high-performance episodes
- **Strategic Focus**: Identifies patterns, strategies, and decision-making processes
- **Comprehensive Reports**: Generates human-readable insights and recommendations
- **Flexible Input**: Supports various data formats (CSV, directory + scores, etc.)
- **Modular Design**: Run individual phases or complete pipeline

## 🛠️ Installation

### Prerequisites

- Python 3.8+
- Google Gemini API key
- Video files (MP4 format recommended)

### Dependencies

```bash
# Install core dependencies
pip install -r requirements.txt

# Install Gemini API client
pip install google-genai

# For video sampling (if using rl-baselines3-zoo)
pip install stable-baselines3 gymnasium[atari]
```

### API Key Setup

Choose one of these methods:

```bash
# Method 1: Environment variable
export GEMINI_API_KEY="your-api-key-here"

# Method 2: Create key file
echo "your-api-key-here" > Gemini_API_KEY.txt

# Method 3: Set in code (pass to functions)
```

## 🚀 Usage

### Video Sampling

Generate videos for HVA-X analysis:

#### Option 1: Use Existing Videos (Fastest)

```bash
# Quick test with existing videos
python use_existing_videos.py --num-episodes 12 --run-analysis

# Prepare larger dataset
python use_existing_videos.py --num-episodes 20 --output-dir my_hva_dataset
```

#### Option 2: Generate New Videos (Recommended)

```bash
# Quick sampling (20 episodes)
python sample_videos_for_hva.py --quick

# Full sampling (100 episodes)
python sample_videos_for_hva.py --episodes 100 --output-dir hva_full_dataset

# Custom sampling
python sample_videos_for_hva.py \
    --episodes 50 \
    --output-dir custom_dataset \
    --python-path /path/to/conda/env/bin/python
```

### Complete Pipeline

Run all four phases in sequence:

```bash
# Basic usage with video directory
python run_hva_pipeline.py --video-dir hva_videos

# With specific score file
python run_hva_pipeline.py --video-dir videos/ --score-file scores.txt

# From CSV file
python run_hva_pipeline.py --csv-file trajectory_data.csv

# Advanced options
python run_hva_pipeline.py \
    --video-dir hva_videos \
    --samples-per-tier 5 \
    --output-prefix detailed_analysis \
    --cleanup-intermediate \
    --verbose
```

**Pipeline Options:**
- `--video-dir`: Directory containing video files
- `--score-file`: Path to scores.txt file
- `--csv-file`: Path to trajectory_data.csv file
- `--samples-per-tier`: Number of samples per performance tier (default: 3)
- `--output-prefix`: Prefix for output files (default: "hva_pipeline")
- `--cleanup-intermediate`: Remove intermediate files after completion
- `--no-save-intermediate`: Don't save intermediate phase results
- `--no-final-report`: Don't save final report as markdown
- `--verbose`: Enable verbose logging

### Individual Phases

Run specific phases independently:

#### Phase 1: Trajectory Sampling

```bash
# Basic sampling
python run_phase1_only.py --video-dir hva_videos

# With specific files
python run_phase1_only.py --video-dir videos/ --score-file scores.txt

# From CSV
python run_phase1_only.py --csv-file trajectory_data.csv

# More samples per tier
python run_phase1_only.py --video-dir hva_videos --samples-per-tier 5
```

#### Phase 2A: Event Detection

```bash
# Using Phase 1 results
python run_phase2a_only.py --phase1-file phase1_sampling_20240101_120000.json

# Direct from video directory
python run_phase2a_only.py --video-dir hva_videos

# From CSV
python run_phase2a_only.py --csv-file trajectory_data.csv
```

#### Phase 2B: Guided Analysis

```bash
# Using Phase 2A results
python run_phase2b_only.py --phase2a-file phase2a_events_20240101_120000.json

# Custom output prefix
python run_phase2b_only.py \
    --phase2a-file phase2a_events_20240101_120000.json \
    --output-prefix custom_analysis
```

#### Phase 3: Meta-Synthesis

```bash
# Using Phase 2B results
python run_phase3_only.py --phase2b-file phase2b_analysis_20240101_120000.json

# Save final report
python run_phase3_only.py \
    --phase2b-file phase2b_analysis_20240101_120000.json \
    --save-report
```

### Phase-by-Phase Execution

You can also run each phase individually for more control:

```bash
# Phase 1: Trajectory Sampling
python run_phase1_only.py --video-dir hva_videos --output-dir results/phase1

# Phase 2A: Event Detection
python run_phase2a_only.py --phase1-file results/phase1/phase1_sampling_*.json --output-dir results/phase2a

# Phase 2B: Guided Analysis
python run_phase2b_only.py --phase2a-file results/phase2a/phase2a_events_*.json --output-dir results/phase2b

# Phase 3: Meta-Synthesis
python run_phase3_only.py --phase2b-file results/phase2b/phase2b_analysis_*.json --save-report --output-dir results/phase3
```

## Video-Referenced Analysis Reports

The HVA-X Phase 3 meta-synthesis now generates reports with **specific video references and timestamps**, allowing users to verify insights by watching exact video segments.

### Key Features

- **Precise Video References**: Reports cite specific episodes (e.g., "`seed42` at 00:06-00:14")
- **Timestamp Accuracy**: All observations include exact time spans for verification
- **Evidence-Based Claims**: Every major insight is backed by specific video evidence
- **Comparative Analysis**: Direct comparisons between videos with timestamps

### Example Report Format

```markdown
The agent demonstrates advanced tunneling in `seed100` at 00:15-00:20, but fails at 
basic defensive play in `seed42` at 00:22-00:23. This pattern is consistent across 
episodes, with similar failures visible in `seed420` at 00:16-00:19.
```

### Benefits

1. **Verification**: Users can immediately locate and verify any claim in the report
2. **Precision**: Specific timestamps eliminate ambiguity about when events occur
3. **Research Value**: Enables detailed behavioral analysis with exact references
4. **Quality Assurance**: Forces the AI to be specific and evidence-based in its analysis

## 📁 File Formats

### Input Formats

#### scores.txt (Simple format)
```
120.5
450.2
380.7
820.1
...
```

#### trajectory_data.csv (Rich format)
```csv
video_path,score,episode_id,metadata
episode_001/episode_001.mp4,120.5,episode_001,"{""seed"": 123}"
episode_002/episode_002.mp4,450.2,episode_002,"{""seed"": 456}"
...
```

### Output Formats

#### Phase Results (JSON)
Each phase generates detailed JSON results:
- `phase1_sampling_TIMESTAMP.json` - Sampling results
- `phase2a_events_TIMESTAMP.json` - Event detection results
- `phase2b_analysis_TIMESTAMP.json` - Guided analysis results
- `phase3_synthesis_TIMESTAMP.json` - Meta-synthesis results

#### Final Report (Markdown)
Human-readable analysis report:
- Executive summary
- Strategic profile analysis
- Performance tier comparisons
- Recommendations for improvement

## ⚙️ Configuration

### Sampling Configuration

```python
# Adjust sampling parameters
samples_per_tier = 3  # Number of videos per performance tier
total_episodes = 100  # Total episodes to generate

# Performance tiers
# - Low tier: Bottom 10% of scores
# - Mid tier: Middle 10% (45th-55th percentile)
# - High tier: Top 10% of scores
```

### API Configuration

```python
# Gemini model selection
model = "gemini-2.5-pro"  # Default model
# model = "gemini-2.5-pro-preview-06-05"  # Alternative model

# Analysis parameters
event_detection_focus = ["critical_decision", "tactical_error", "strategy_change"]
analysis_depth = "comprehensive"  # or "focused"
```

## 🔧 Troubleshooting

### Common Issues

#### "No videos found"
```bash
# Check your videos directory
ls -la videos/

# Use specific path
python use_existing_videos.py --videos-dir videos/ppo/BreakoutNoFrameskip-v4
```

#### "API key not found"
```bash
# Set your API key
export GEMINI_API_KEY="your-key-here"

# Or check if file exists
cat Gemini_API_KEY.txt
```

#### "rl-baselines3-zoo not found"
```bash
# Make sure you're in the right directory
pwd
ls -la rl-baselines3-zoo/

# Or specify python path
python sample_videos_for_hva.py --python-path /your/conda/env/bin/python
```

#### "Import errors"
```bash
# Install missing dependencies
pip install google-genai
pip install stable-baselines3
pip install gymnasium[atari]
```

### Performance Issues

#### Large video files
- Use compressed MP4 format
- Limit video length to 2-3 minutes
- Consider reducing video resolution

#### API rate limits
- Add delays between API calls
- Use smaller batch sizes
- Monitor API usage

## 📊 Examples

### Example 1: Quick Analysis

```bash
# Fastest way to test the system
python use_existing_videos.py --num-episodes 10 --run-analysis
```

### Example 2: Full Pipeline

```bash
# Generate diverse video dataset
python sample_videos_for_hva.py --episodes 30 --output-dir hva_breakout

# Run complete analysis
python run_hva_pipeline.py --video-dir hva_breakout --samples-per-tier 3

# View results
cat hva_pipeline_final_report_*.md
```

### Example 3: Phase-by-Phase Analysis

```bash
# Phase 1: Sample trajectories
python run_phase1_only.py --video-dir hva_videos --samples-per-tier 4

# Phase 2A: Detect events
python run_phase2a_only.py --phase1-file phase1_sampling_*.json

# Phase 2B: Guided analysis
python run_phase2b_only.py --phase2a-file phase2a_events_*.json

# Phase 3: Meta-synthesis
python run_phase3_only.py --phase2b-file phase2b_analysis_*.json --save-report
```

### Example 4: Custom Configuration

```bash
# High-detail analysis
python run_hva_pipeline.py \
    --video-dir custom_videos \
    --samples-per-tier 7 \
    --output-prefix detailed_analysis \
    --verbose

# Minimal analysis
python run_hva_pipeline.py \
    --csv-file trajectory_data.csv \
    --samples-per-tier 2 \
    --cleanup-intermediate
```

## 📈 Expected Results & Examples

This section shows concrete examples of what each method produces, helping you understand the different types of insights available.

### HVA-X: Comprehensive Strategic Analysis

HVA-X produces detailed written reports analyzing agent behavior at a strategic level. Here's an excerpt from an actual DQN agent analysis (see [full report](phase3_synthesis_report_20250708_194821.md)):

```markdown
### Agent Evaluation Report: Breakout Specialist

#### Executive Summary
The agent exhibits the strategic profile of a highly specialized "glass cannon." 
It has masterfully learned the optimal offensive strategy in Breakout—"tunneling"—
which it pursues with remarkable focus and precision. However, its overall 
performance is dictated by a critical weakness in reactive defense.

#### Strategic Profile
- **Core Strategic Approach: Tunneling Supremacy**
  The agent's single, overarching strategy is to create a vertical channel 
  on one side of the brick wall. This is a sophisticated, high-risk, 
  high-reward approach that prioritizes future automated scoring.

- **Consistent Strengths: Proactive Offensive Planning**
  * Long-Term Goal Identification: Deep understanding that tunneling is most effective
  * Offensive Execution: Precise paddle control when enacting offensive plan
  * Strategic Persistence: Remains committed to plan even after setbacks

- **Consistent Limitations: Reactive Brittleness**
  * Poor Defensive Awareness: Fails to integrate defense into offensive planning
  * State Transition Failure: Struggles when game becomes chaotic
  * Tactical Instability: Unreliable ball interception under pressure

#### Performance Analysis Across Tiers
- **Low-Tier**: Fails to execute tunneling plan due to basic tactical failures
- **Mid-Tier**: Successfully creates tunnel but fails during ball re-entry
- **High-Tier**: Executes strategy well but has specific exploitable blind spots

#### Recommendations
1. Bolster defensive fundamentals through curriculum learning
2. Target state transition brittleness with augmented training data
3. Remediate strategic blind spots using adversarial examples
```

**Key Insights from HVA-X:**
- Identifies the agent as a "specialist" rather than generalist
- Reveals the "tunneling" strategy as core approach
- Explains performance variance through defensive weaknesses
- Provides actionable recommendations for improvement

### HIGHLIGHTS: Important Trajectory Selection

HIGHLIGHTS produces short video clips showing the most important moments in agent gameplay:

**Example Results (from this repository):**
- [HL_0.mp4](HIGHLIGHTS/highlights/results/run_2025-07-08_22:40:12_339986/Highlight_Videos/HL_0.mp4) (8.1KB) - Critical decision point with high state importance
- [HL_1.mp4](HIGHLIGHTS/highlights/results/run_2025-07-08_22:40:12_339986/Highlight_Videos/HL_1.mp4) (6.8KB) - Key strategic moment during tunnel creation
- [HL_2.mp4](HIGHLIGHTS/highlights/results/run_2025-07-08_22:40:12_339986/Highlight_Videos/HL_2.mp4) (7.0KB) - Important defensive failure pattern
- [HL_3.mp4](HIGHLIGHTS/highlights/results/run_2025-07-08_22:40:12_339986/Highlight_Videos/HL_3.mp4) (7.1KB) - Successful tunnel execution sequence
- [HL_4.mp4](HIGHLIGHTS/highlights/results/run_2025-07-08_22:40:12_339986/Highlight_Videos/HL_4.mp4) (7.1KB) - High-value state transition moment

**Typical HIGHLIGHTS Output:**
- 5-10 short video clips (5-15 seconds each)
- Ranked by state importance scores
- Focus on decision points with highest Q-values
- Useful for quickly identifying critical moments

**Key Insights from HIGHLIGHTS:**
- Visual identification of important decision points
- Quick overview of agent's key strategic moments
- Useful for debugging specific tactical failures
- Good for presentations and demonstrations

### GradCAM: Visual Attention Analysis

GradCAM produces videos showing where the neural network focuses its attention during gameplay:

**Example Results (from this repository):**
- [gradcam_BreakoutNoFrameskip-v4_dqn.mp4](gradcam_BreakoutNoFrameskip-v4_dqn.mp4) - Multi-panel attention visualization showing:
  - Original Gameplay (left panel)
  - GradCAM Attention Heatmap (center panel) 
  - Action Probabilities (right panel)

**Typical GradCAM Output:**
- Side-by-side video with original gameplay and attention heatmaps
- Color-coded attention maps (red = high attention, blue = low attention)
- Real-time action probability distributions
- Frame-by-frame analysis of visual focus

**Key Insights from GradCAM:**
- Shows spatial attention patterns (where agent "looks")
- Reveals if agent focuses on ball, paddle, or bricks
- Identifies visual processing anomalies
- Useful for debugging perception issues

### Method Comparison: Same Agent, Different Insights

Here's how each method analyzed the same DQN Breakout agent:

| Method | Primary Insight | Output Type | Time to Generate |
|--------|----------------|-------------|------------------|
| **HVA-X** | "Agent is a tunneling specialist with defensive weaknesses" | Text report (6,788 chars) | ~15 minutes |
| **HIGHLIGHTS** | "5 key moments showing tunnel creation and failures" | 5 video clips (~35KB total) | ~2 minutes |
| **GradCAM** | "Agent focuses on ball and tunnel area, ignores periphery" | Attention video (~2MB) | ~1 minute |

### Complete Results Structure

After running all methods, you'll have:

**Example Results in This Repository:**
```
📁 strategy2text/
├── 📄 phase3_synthesis_report_20250708_194821.md           # HVA-X strategic analysis
├── 📹 gradcam_BreakoutNoFrameskip-v4_dqn.mp4              # GradCAM attention video
├── 📁 HIGHLIGHTS/highlights/results/run_2025-07-08_22:40:12_339986/
│   └── 📁 Highlight_Videos/                                # HIGHLIGHTS output
│       ├── 📹 HL_0.mp4 → HL_4.mp4                         # Important moments
└── 📄 README.md                                            # This documentation
```

### Using Results for Agent Improvement

**From HVA-X Analysis:**
```python
# Identified issues: defensive weaknesses, state transition failures
# Recommended fixes:
1. Increase penalty for losing lives in reward function
2. Add curriculum learning starting with defensive scenarios  
3. Augment training data with more "ball re-entry" situations
4. Use adversarial training for corner defense scenarios
```

**From HIGHLIGHTS Analysis:**
```python
# Use important moments for:
1. Creating targeted training scenarios
2. Debugging specific failure patterns
3. Validating model improvements
4. Demonstrating agent capabilities
```

**From GradCAM Analysis:**
```python
# Use attention patterns for:
1. Verifying visual processing is correct
2. Identifying perception blind spots
3. Debugging CNN layer issues
4. Ensuring agent sees relevant game elements
```

## 🎯 Best Practices

### For Video Sampling
1. **Use diverse seeds** to get varied performance
2. **Include enough episodes** (50-100) for good stratification
3. **Ensure video quality** is sufficient for analysis
4. **Balance episode length** (not too short, not too long)

### For Analysis
1. **Start with fewer samples** (2-3 per tier) for initial testing
2. **Use verbose logging** to monitor progress
3. **Save intermediate results** for debugging
4. **Review Phase 1 results** before proceeding

### For Interpretation
1. **Focus on strategic patterns** rather than individual actions
2. **Compare across performance tiers** to identify key differences
3. **Look for consistent themes** in the final report
4. **Use insights to guide agent training** improvements

## 🔬 Baseline Methods

HVA-X can be compared against several baseline methods for agent analysis and explainability. This section covers two main baseline approaches: HIGHLIGHTS for trajectory summarization and GradCAM for visual attention analysis.

### HIGHLIGHTS: Trajectory Summarization

HIGHLIGHTS is designed for value-based RL algorithms and creates video summaries by selecting important state trajectories.

### Running HIGHLIGHTS

```bash
# Basic HIGHLIGHTS usage
cd HIGHLIGHTS
python run.py --env BreakoutNoFrameskip-v4 --algo dqn

# With custom configuration
python run.py \
    --env BreakoutNoFrameskip-v4 \
    --algo dqn \
    --n_traces 20 \
    --num_highlights 10 \
    --trajectory_length 15 \
    --output_dir highlights_results

# Load specific model
python run.py \
    --env BreakoutNoFrameskip-v4 \
    --algo dqn \
    --folder ../rl-baselines3-zoo/rl-trained-agents \
    --exp-id 0 \
    --load-best
```

### HIGHLIGHTS Options

**Core Parameters:**
- `--env`: Environment ID (default: BreakoutNoFrameskip-v4)
- `--algo`: RL Algorithm (dqn, qrdqn, ddqn)
- `--n_traces`: Number of traces to obtain (default: 10)
- `--num_highlights`: Number of highlight trajectories (default: 5)
- `--trajectory_length`: Length of highlight trajectories (default: 10)

**Model Loading:**
- `--folder`: Path to trained models (default: ../rl-baselines3-zoo/rl-trained-agents)
- `--exp-id`: Experiment ID (default: 0)
- `--load-best`: Load best model instead of last model
- `--seed`: Random generator seed (default: 0)

**Advanced Options:**
- `--state_importance`: Method for calculating state importance (default: 'second')
- `--highlights_div`: Use diversity measures (default: False)
- `--div_coefficient`: Diversity coefficient (default: 2)
- `--fps`: Summary video fps (default: 5)
- `--output_dir`: Output directory for results

### HIGHLIGHTS vs HVA-X Comparison

| Aspect | HIGHLIGHTS | HVA-X |
|--------|------------|-------|
| **Approach** | State importance ranking | Multi-phase video analysis |
| **Algorithm Support** | Value-based RL (DQN, etc.) | Any RL algorithm |
| **Output** | Video highlights | Comprehensive text analysis |
| **Analysis Depth** | Trajectory selection | Strategic behavior analysis |
| **Human Interpretation** | Visual summary | Detailed written report |
| **Customization** | Importance metrics | Analysis prompts |

### Example Workflow: Comparing Methods

```bash
# 1. Generate agent videos for both methods
python sample_videos_for_hva.py --episodes 50 --output-dir comparison_videos

# 2. Run HIGHLIGHTS baseline
cd HIGHLIGHTS
python run.py \
    --env BreakoutNoFrameskip-v4 \
    --algo dqn \
    --n_traces 15 \
    --num_highlights 8 \
    --output_dir ../highlights_baseline_results


### GradCAM: Visual Attention Analysis

GradCAM (Gradient-weighted Class Activation Mapping) is a saliency map method that visualizes what parts of the input the neural network is focusing on when making decisions. This baseline creates videos with attention heatmaps overlaid on the gameplay.

### Running GradCAM

```bash
# Basic GradCAM usage
python record_with_gradcam.py --env BreakoutNoFrameskip-v4 --algo ppo

# With custom configuration
python record_with_gradcam.py \
    --env BreakoutNoFrameskip-v4 \
    --algo dqn \
    --model-path path/to/model.zip \
    --output-folder gradcam_analysis \
    --n-timesteps 1000

# Load specific model
python record_with_gradcam.py \
    --env BreakoutNoFrameskip-v4 \
    --algo ppo \
    --folder rl-trained-agents \
    --exp-id 0 \
    --load-best \
    --deterministic
```

### GradCAM Options

**Core Parameters:**
- `--env`: Environment ID (default: BreakoutNoFrameskip-v4)
- `--algo`: RL Algorithm (ppo, dqn, etc.)
- `--model-path`: Path to specific model file
- `--output-folder`: Output folder for videos (default: gradcam_videos)
- `--n-timesteps`: Number of timesteps to record (default: 500)

**Model Loading:**
- `--folder`: Path to trained models (default: rl-trained-agents)
- `--exp-id`: Experiment ID (default: 0)
- `--load-best`: Load best model instead of last model
- `--deterministic`: Use deterministic actions

**Video Options:**
- `--seed`: Random generator seed (default: 42)

### Method Comparison

| Aspect | HIGHLIGHTS | GradCAM | HVA-X |
|--------|------------|---------|-------|
| **Approach** | State importance ranking | Visual attention mapping | Multi-phase video analysis |
| **Algorithm Support** | Value-based RL (DQN, etc.) | CNN-based policies | Any RL algorithm |
| **Output** | Video highlights | Attention heatmap videos | Comprehensive text analysis |
| **Analysis Focus** | Trajectory selection | Visual attention patterns | Strategic behavior analysis |
| **Interpretability** | Important moments | Spatial attention | Strategic reasoning |
| **Real-time** | No | Yes (during recording) | No |
| **Customization** | Importance metrics | Attention layers | Analysis prompts |

### Example Workflow: Comparing All Methods

```bash
# 1. Generate agent videos for analysis
python sample_videos_for_hva.py --episodes 50 --output-dir comparison_videos

# 2. Run HIGHLIGHTS baseline
cd HIGHLIGHTS
python run.py \
    --env BreakoutNoFrameskip-v4 \
    --algo dqn \
    --n_traces 15 \
    --num_highlights 8 \
    --output_dir ../highlights_baseline_results

# 3. Run GradCAM baseline
cd ..
python record_with_gradcam.py \
    --env BreakoutNoFrameskip-v4 \
    --algo ppo \
    --output-folder gradcam_baseline_results \
    --n-timesteps 1000

# 4. Run HVA-X analysis
python run_hva_pipeline.py \
    --video-dir comparison_videos \
    --samples-per-tier 3 \
    --output-prefix hva_comparison

# 5. Compare results
# - HIGHLIGHTS: Video summaries in highlights_baseline_results/
# - GradCAM: Attention videos in gradcam_baseline_results/
# - HVA-X: Text analysis in hva_comparison_final_report_*.md
```

### When to Use Each Method

**Use HIGHLIGHTS when:**
- You need visual summaries of agent behavior
- Working with value-based RL algorithms
- Want to identify key decision points visually
- Need quick trajectory highlights

**Use GradCAM when:**
- You want to understand visual attention patterns
- Working with CNN-based policies
- Need to debug visual perception issues
- Want to see real-time decision-making focus
- Investigating spatial reasoning capabilities

**Use HVA-X when:**
- You need detailed strategic analysis
- Want comprehensive written reports
- Working with any RL algorithm type
- Need insights for agent improvement
- Want to understand failure modes and success patterns
- Require high-level behavioral understanding

## 🤝 Contributing

To contribute to HVA-X:
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Google Gemini API for video understanding capabilities
- RL-Baselines3-Zoo for agent training infrastructure
- OpenAI Gymnasium for environment support 