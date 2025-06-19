# Quick Start Guide: Breakout Agent Demo

## 🚀 Get Started in 3 Steps

### Step 1: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 2: Download Pre-trained Models
```bash
# Download PPO and DQN models (recommended)
python scripts/download_rl_zoo_models.py

# Or download all available models
python scripts/download_rl_zoo_models.py --all
```

### Step 3: Run a Demo
```bash
# Run PPO agent (default)
python scripts/demo_breakout.py

# Run DQN agent
python scripts/demo_breakout.py --algorithm dqn
```

## 🎮 Available Commands

### Download Models
```bash
# Download specific algorithm
python scripts/download_rl_zoo_models.py --algorithm ppo
python scripts/download_rl_zoo_models.py --algorithm dqn

# Download all Breakout models
python scripts/download_rl_zoo_models.py --all

# Verify downloads
python scripts/download_rl_zoo_models.py --verify
```

### Run Demos
```bash
# Basic demo (3 episodes)
python scripts/demo_breakout.py

# Specify algorithm
python scripts/demo_breakout.py --algorithm ppo
python scripts/demo_breakout.py --algorithm dqn
python scripts/demo_breakout.py --algorithm a2c

# More episodes
python scripts/demo_breakout.py --episodes 5

# Save videos
python scripts/demo_breakout.py --save-video

# Run without display (faster)
python scripts/demo_breakout.py --no-render --episodes 10

# List available models
python scripts/demo_breakout.py --list
```

## 🤖 Available Algorithms

- **PPO** (Proximal Policy Optimization) - Stable, good performance
- **DQN** (Deep Q-Network) - Classic deep RL
- **A2C** (Advantage Actor-Critic) - Fast and efficient
- **QR-DQN** (Quantile Regression DQN) - Advanced DQN variant

## 📁 Project Structure After Setup

```
breakout-agent-demo/
├── scripts/
│   ├── download_rl_zoo_models.py    # Download models
│   ├── demo_breakout.py             # Run demos
│   └── setup.py                     # Setup script
├── src/
│   └── rl_agents/
│       └── sb3_agent.py             # Agent wrapper
├── rl-trained-agents/               # Downloaded models
│   ├── ppo/
│   │   └── BreakoutNoFrameskip-v4_1/
│   │       └── BreakoutNoFrameskip-v4.zip
│   ├── dqn/
│   └── a2c/
└── requirements.txt
```

## 🔧 Troubleshooting

### Import Errors
```bash
pip install ale-py shimmy
```

### Model Not Found
```bash
python scripts/download_rl_zoo_models.py --algorithm ppo
```

### Display Issues (Headless Systems)
```bash
python scripts/demo_breakout.py --no-render
```

## 🎯 Example Output

```
Loading pre-trained PPO agent for Breakout...
Found model: rl-trained-agents\ppo\BreakoutNoFrameskip-v4_1\BreakoutNoFrameskip-v4.zip
✓ Successfully loaded PPO agent

Running 3 episodes...
==================================================

Episode 1/3
------------------------------
  Total Reward: 432.0
  Steps: 1847
  Average Reward: 0.234

Episode 2/3
------------------------------
  Total Reward: 378.0
  Steps: 1692
  Average Reward: 0.223

Episode 3/3
------------------------------
  Total Reward: 456.0
  Steps: 1923
  Average Reward: 0.237

==================================================
SUMMARY
==================================================
Algorithm: PPO
Episodes completed: 3
Average reward: 422.00 ± 32.11
Best reward: 456.0
Average episode length: 1820.7 steps
👍 Good performance!
```

## 🚀 Next Steps

1. **Compare Algorithms**: Try different algorithms to see performance differences
2. **Record Videos**: Use `--save-video` to capture gameplay
3. **Performance Analysis**: Run longer evaluations with `--episodes 10`
4. **Extend**: Add support for other games or algorithms

## 📚 Learn More

- [RL Baselines3 Zoo](https://github.com/DLR-RM/rl-baselines3-zoo)
- [Stable Baselines3](https://stable-baselines3.readthedocs.io/)
- [OpenAI Gymnasium](https://gymnasium.farama.org/)

---

**Ready to play? Run your first demo:**
```bash
python scripts/demo_breakout.py
``` 