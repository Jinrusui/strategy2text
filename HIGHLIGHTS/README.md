# HIGHLIGHTS for rl-baselines3-zoo

An implementation of the HIGHLIGHTS algorithm for Agent Policy Summarization, adapted for **rl-baselines3-zoo** and **stable-baselines3**: 

#### Original Paper: Amir, Ofra, Finale Doshi-Velez, and David Sarne. "Summarizing agent strategies." Autonomous Agents and Multi-Agent Systems 33.5 (2019): 628-644.

## Description

Strategy summarization techniques convey agent behavior by demonstrating the actions taken by the agent in a selected set of world states. The key question in this approach is then how to recognize meaningful agent situations.

The HIGHLIGHTS algorithm extracts *important* states from execution traces of the agent based on some importance metric. Intuitively, a state is considered important if the decision made in that state has a substantial impact on the agent's utility.

## Requirements

This implementation works with **trained agents from rl-baselines3-zoo** and requires:

- **Value-based RL algorithms** (DQN, DDQN, QR-DQN) with access to Q-values
- **stable-baselines3** compatible models
- **rl-baselines3-zoo** format for model loading

## Setup

1. Install requirements:
```bash
pip install -r requirements.txt
```

2. Make sure you have trained models available in the rl-baselines3-zoo format, or train your own:
```bash
# Example: Train a DQN model
python -m rl_zoo3.train --algo dqn --env CartPole-v1 --save-freq 1000
```

## Usage

### Basic Usage

Run HIGHLIGHTS on a trained DQN agent:

```bash
python run.py --env CartPole-v1 --algo dqn --folder rl-trained-agents
```

### Advanced Usage

```bash
python run.py \
  --env CartPole-v1 \
  --algo dqn \
  --folder rl-trained-agents \
  --n_traces 20 \
  --num_highlights 10 \
  --trajectory_length 15 \
  --state_importance second \
  --highlights_div \
  --seed 42
```

### Key Parameters

**Model Loading:**
- `--env`: Environment ID (e.g., CartPole-v1, DQN-Atari)
- `--algo`: RL Algorithm (dqn, qrdqn, ddqn)
- `--folder`: Path to trained models (default: rl-trained-agents)
- `--exp-id`: Experiment ID (default: 0 for latest)
- `--load-best`: Load best model instead of last checkpoint

**HIGHLIGHTS Configuration:**
- `--n_traces`: Number of execution traces to collect (default: 10)
- `--num_highlights`: Number of highlight videos to generate (default: 5)
- `--trajectory_length`: Length of each highlight trajectory (default: 10)
- `--state_importance`: Method for computing state importance:
  - `second`: Difference between best and second-best Q-value (default)
  - `worst`: Difference between best and worst Q-value
  - `variance`: Variance of Q-values
- `--highlights_div`: Enable diversity-based selection (default: False)
- `--minimum_gap`: Minimum gap between selected trajectories (default: 0)

**Video Generation:**
- `--fps`: Video frames per second (default: 5)
- `--pause`: Add pause frames at start/end of videos (default: 0)

**Seeding for Fair Method Comparison:**
- `--seed`: Base environment seed (default: 0)
- `--seed-mode`: Seeding strategy for method comparison:
  - `fixed`: All traces use same seed (default, original behavior)
  - `sequential`: Sequential seeds starting from base seed
  - `random`: Random seeds (reproducible with base seed)
  - `trace-specific`: Use specific seeds for each trace
- `--base-seed`: Base seed for sequential/random modes (default: same as --seed)
- `--trace-seeds`: Comma-separated specific seeds for trace-specific mode
- `--deterministic-eval`: Use deterministic policy evaluation (default: True)
- `--save-seeds`: Save seed information in metadata for reproducibility (default: True)

### Fair Method Comparison

To compare different algorithms on the same game scenarios:

```bash
# Compare DQN, DDQN, and QR-DQN on identical scenarios
for algo in dqn ddqn qrdqn; do
  python run.py \
    --env BreakoutNoFrameskip-v4 \
    --algo $algo \
    --seed-mode sequential \
    --base-seed 42 \
    --n-traces 10 \
    --deterministic-eval \
    --save-seeds
done
```

Or use specific interesting scenarios:

```bash
# Define specific game scenarios to analyze
python run.py \
  --env BreakoutNoFrameskip-v4 \
  --algo dqn \
  --seed-mode trace-specific \
  --trace-seeds 123,456,789,101112,131415 \
  --n-traces 5 \
  --deterministic-eval
```

### Output

The algorithm generates:
1. **Highlight videos** (`HL_0.mp4`, `HL_1.mp4`, etc.) showing important decision points
2. **Trace data** (`Traces.pkl`) containing execution traces
3. **State data** (`States.pkl`) containing state information and Q-values
4. **Trajectory data** (`Trajectories.pkl`) containing selected highlight trajectories
5. **Metadata** (`metadata.json`) with run configuration

Results are saved in `highlights/results/run_YYYY-MM-DD_HH:MM:SS_XXXXXX/`

## Supported Algorithms

Currently tested with:
- **DQN** (Deep Q-Network)
- **DDQN** (Double Deep Q-Network)  
- **QR-DQN** (Quantile Regression DQN)

## Supported Environments

Any environment supported by rl-baselines3-zoo:
- **Classic Control**: CartPole-v1, MountainCar-v0, etc.
- **Atari**: ALE/Breakout-v5, ALE/Pong-v5, etc.
- **Box2D**: LunarLander-v2, etc.
- **Custom environments** with proper registration

## How It Works

1. **Trace Collection**: Runs the trained agent in the environment to collect execution traces
2. **Q-value Extraction**: Extracts Q-values from the value-based model for each state
3. **Importance Calculation**: Computes state importance based on Q-value differences
4. **Highlight Selection**: Selects most important states while avoiding redundancy
5. **Video Generation**: Creates videos showing selected important decision points

## Example: Analyzing a DQN Agent

```bash
# Train a DQN agent (if not already trained)
python -m rl_zoo3.train --algo dqn --env CartPole-v1 --total-timesteps 50000

# Run HIGHLIGHTS analysis
python run.py \
  --env CartPole-v1 \
  --algo dqn \
  --n_traces 15 \
  --num_highlights 8 \
  --trajectory_length 12 \
  --state_importance second \
  --verbose
```

This will generate 8 highlight videos showing the most important decision points where the DQN agent's Q-values indicated critical choices.

## Method Comparison Examples

### Using the Example Script

The `example_usage.py` script provides convenient examples:

```bash
# Compare multiple methods on the same scenarios
python example_usage.py --compare-methods

# Analyze specific predefined scenarios
python example_usage.py --specific-scenarios

# Custom method comparison
python example_usage.py \
  --env BreakoutNoFrameskip-v4 \
  --algo dqn \
  --seed-mode sequential \
  --base-seed 42 \
  --n-traces 10
```

### Manual Method Comparison

1. **Analyze DQN with sequential seeding:**
```bash
python run.py \
  --env BreakoutNoFrameskip-v4 \
  --algo dqn \
  --seed-mode sequential \
  --base-seed 42 \
  --n-traces 5 \
  --save-seeds
```

2. **Analyze QR-DQN on the same scenarios:**
```bash
python run.py \
  --env BreakoutNoFrameskip-v4 \
  --algo qrdqn \
  --seed-mode sequential \
  --base-seed 42 \
  --n-traces 5 \
  --save-seeds
```

Both runs will analyze the agents on identical game scenarios (seeds 42, 43, 44, 45, 46), enabling fair comparison of their decision-making strategies.

### Reproducibility

Each run saves complete seeding information in `metadata.json`:
- Seeds used for each trace
- Reproduction command
- Seeding configuration

This ensures that interesting scenarios can be re-analyzed or shared with other researchers.

## Dependencies

- **stable-baselines3[extra]>=1.6.0**: RL algorithms
- **rl-baselines3-zoo>=1.6.0**: Model zoo and utilities
- **huggingface-sb3>=2.2.0**: Model loading
- **PyTorch>=1.12.1**: Deep learning framework
- **OpenCV**: Video generation
- **Gym>=0.21.0**: Environment interface
- **NumPy, Pandas, Matplotlib**: Data processing and visualization

## Citation

If you use this implementation in your research, please cite:

```bibtex
@article{amir2019summarizing,
  title={Summarizing agent strategies},
  author={Amir, Ofra and Doshi-Velez, Finale and Sarne, David},
  journal={Autonomous Agents and Multi-Agent Systems},
  volume={33},
  number={5},
  pages={628--644},
  year={2019},
  publisher={Springer}
}
```

## Troubleshooting

**Q-value extraction fails:**
- Ensure your model is value-based (DQN, DDQN, QR-DQN)
- Check that the model was trained with rl-baselines3-zoo

**Video generation fails:**
- Ensure environment supports rendering
- Check that OpenCV is properly installed

**Model loading fails:**
- Verify model path and experiment ID
- Use `--custom-objects` flag for compatibility issues

## License

This project is licensed under the MIT License - see the LICENSE file for details.







