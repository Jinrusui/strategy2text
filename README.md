# Strategy2Text: RL Agent Strategy Analysis Framework

## Overview

This repository provides a comprehensive framework for analyzing Reinforcement Learning agent strategies using Vision-Language Models (VLMs). It combines pre-trained RL agents from the [RL Baselines3 Zoo](https://github.com/DLR-RM/rl-baselines3-zoo) with Google's Gemini API to automatically translate visual gameplay into human-readable strategy summaries.

The framework implements the methodology described in the dissertation, providing quantitative evaluation metrics for strategy analysis quality including Predictive Faithfulness Score (PFS), Coverage Score, and Abstraction Score.

## Key Features

- **Pre-trained Agent Support**: Download and run PPO, DQN, A2C, and QR-DQN agents trained on Breakout
- **Gemini-based Strategy Analysis**: Automated strategy extraction from gameplay videos using VLM
- **Comprehensive Evaluation Metrics**: PFS, Coverage, and Abstraction scoring
- **Video Sampling Framework**: Diverse sampling strategies (typical, edge cases, longitudinal)
- **Experimental Pipeline**: Baseline comparisons and ablation studies
- **Simple Interface**: Easy-to-use scripts for downloading models and running analyses
- **Video Recording**: Option to save gameplay videos for analysis
- **No Training Required**: Focus on analyzing existing trained agents

## Quick Start

### 1. Install Dependencies

**⚠️ Important**: If you encounter segmentation faults during installation (common in WSL2), see our [Installation Guide](INSTALLATION.md) for stable installation methods.

```bash
# Quick installation (may cause segmentation faults in some environments)
pip install -r requirements.txt

# OR use the stable installation script (recommended)
python scripts/setup.py
```

### 2. Download Pre-trained Models

```bash
# Download PPO and DQN models (default)
python scripts/download_rl_zoo_models.py

# Or download all available Breakout models
python scripts/download_rl_zoo_models.py --all

# Or download a specific algorithm
python scripts/download_rl_zoo_models.py --algorithm ppo
python scripts/download_rl_zoo_models.py --algorithm dqn
```

### 3. Run Agent Demo

```bash
# Run PPO agent (default)
python scripts/demo_breakout.py

# Run DQN agent
python scripts/demo_breakout.py --algorithm dqn

# Run with video recording
python scripts/demo_breakout.py --algorithm ppo --save-video

# Run multiple episodes
python scripts/demo_breakout.py --algorithm ppo --episodes 5
```

### 4. Strategy Analysis with Gemini

```bash
# Run strategy analysis demo (requires Gemini API key)
python scripts/demo_gemini_analysis.py

# Analyze a single video
python -c "
from src.gemini_analysis import StrategyAnalyzer
analyzer = StrategyAnalyzer('videos')
result = analyzer.analyze_single_video('path/to/video.mp4')
print(result['strategy_summary'])
"
```

### 5. Check Available Models

```bash
# List downloaded models
python scripts/demo_breakout.py --list

# Verify downloads
python scripts/download_rl_zoo_models.py --verify
```

## Available Algorithms

The following pre-trained algorithms are available for Breakout:

- **PPO** (Proximal Policy Optimization) - Generally good performance, stable
- **DQN** (Deep Q-Network) - Classic deep RL algorithm
- **A2C** (Advantage Actor-Critic) - Fast and efficient
- **QR-DQN** (Quantile Regression DQN) - Advanced DQN variant

## Project Structure

```
strategy2text/
├── scripts/                     # Main scripts
│   ├── download_rl_zoo_models.py    # Download pre-trained models
│   ├── demo_breakout.py             # Run agent demos
│   └── demo_gemini_analysis.py      # Strategy analysis demo
├── src/                         # Source code
│   ├── gemini_analysis/         # Gemini-based strategy analysis
│   │   ├── gemini_client.py         # Gemini API interface
│   │   ├── strategy_analyzer.py     # Main analysis orchestrator
│   │   ├── prompt_engineering.py    # Prompt management
│   │   └── evaluation_metrics.py    # PFS, Coverage, Abstraction metrics
│   ├── rl_agents/               # Agent loading and execution
│   ├── video_processing/        # Video utilities and sampling
│   └── utils/                   # Utility functions
├── rl-trained-agents/           # Downloaded models (created automatically)
│   ├── ppo/
│   ├── dqn/
│   ├── a2c/
│   └── qr-dqn/
├── requirements.txt             # Dependencies
├── GEMINI_ANALYSIS_GUIDE.md     # Detailed strategy analysis guide
└── README.md                    # This file
```

## Usage Examples

### Basic Demo
```bash
# Run 3 episodes with PPO agent
python scripts/demo_breakout.py --algorithm ppo --episodes 3
```

### Performance Comparison
```bash
# Compare different algorithms
python scripts/demo_breakout.py --algorithm ppo --episodes 5
python scripts/demo_breakout.py --algorithm dqn --episodes 5
```

### Video Recording
```bash
# Record gameplay videos
python scripts/demo_breakout.py --algorithm ppo --save-video --episodes 2
```

### Headless Mode (No Display)
```bash
# Run without rendering (faster, for performance testing)
python scripts/demo_breakout.py --algorithm ppo --no-render --episodes 10
```

## Model Information

The pre-trained models are downloaded from the official RL Baselines3 Zoo repository:
- **Source**: https://github.com/DLR-RM/rl-trained-agents
- **Environment**: BreakoutNoFrameskip-v4 (Atari)
- **Training**: Models are pre-trained with tuned hyperparameters
- **Performance**: Varies by algorithm, generally good performance on Breakout

## Requirements

- Python 3.8+
- PyTorch
- Stable Baselines3
- OpenAI Gymnasium
- ALE (Arcade Learning Environment)
- OpenCV (for video recording)

## Troubleshooting

### Segmentation Faults
If you encounter segmentation faults during package imports:
1. See the detailed [Installation Guide](INSTALLATION.md)
2. Use the stable installation script: `python scripts/setup.py`
3. Or manually install packages in the correct order (see INSTALLATION.md)

### Common Issues

1. **Segmentation fault during import**:
   ```bash
   # Use stable installation order
   python scripts/setup.py
   ```

2. **ImportError for ALE environments**:
   ```bash
   pip install ale-py shimmy
   ```

3. **Model not found**:
   ```bash
   python scripts/download_rl_zoo_models.py --algorithm ppo
   ```

4. **Display issues on headless systems**:
   ```bash
   python scripts/demo_breakout.py --no-render
   ```

5. **Run the troubleshooter**:
   ```bash
   python scripts/troubleshoot.py
   ```

### Performance Notes

- **PPO**: Generally provides stable and good performance
- **DQN**: Classic approach, may be less stable than PPO
- **A2C**: Faster training, good for quick demos
- **QR-DQN**: Advanced DQN variant, often better than standard DQN

## Extension for Future Agents

The codebase is designed to be easily extensible for additional algorithms and environments. To add support for new agents:

1. Add algorithm to `BREAKOUT_MODELS` in `download_rl_zoo_models.py`
2. Update the choices in `demo_breakout.py`
3. Ensure the algorithm is supported in `src/rl_agents/sb3_agent.py`

## Credits

- **RL Baselines3 Zoo**: https://github.com/DLR-RM/rl-baselines3-zoo
- **Stable Baselines3**: https://github.com/DLR-RM/stable-baselines3
- **OpenAI Gymnasium**: https://github.com/Farama-Foundation/Gymnasium
- **Arcade Learning Environment**: https://github.com/mgbellemare/Arcade-Learning-Environment

## License

This project uses pre-trained models from RL Baselines3 Zoo. Please refer to their licensing terms for model usage. 