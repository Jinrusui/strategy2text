#!/bin/bash

# Example usage of record_video_checkpoints.py
# This script shows common usage patterns for recording videos from checkpoints

echo "Examples of using record_video_checkpoints.py"
echo "============================================="

# Change to the rl-baselines3-zoo directory
cd /mnt/e/Projects/strategy2text/rl-baselines3-zoo

echo "Available algorithms in logs directory:"
ls -la logs/
echo ""

# Example 1: Basic usage - record from DQN checkpoints
echo "Example 1: Basic DQN checkpoint video recording"
echo "Command:"
echo "python -m rl_zoo3.record_video_checkpoints --algo dqn --num-videos-per-cpt 3 --checkpoint-interval 5"
echo ""

# Example 2: More frequent sampling
echo "Example 2: More frequent checkpoint sampling"
echo "Command:"
echo "python -m rl_zoo3.record_video_checkpoints --algo dqn --num-videos-per-cpt 5 --checkpoint-interval 2 --n-timesteps 2000"
echo ""

# Example 3: Custom output directory and stochastic actions
echo "Example 3: Custom output with stochastic actions"
echo "Command:"
echo "python -m rl_zoo3.record_video_checkpoints --algo dqn --output-folder my_videos --stochastic --num-videos-per-cpt 10"
echo ""

# Example 4: Single video per checkpoint, all checkpoints
echo "Example 4: Single video per checkpoint, sample all checkpoints"
echo "Command:"
echo "python -m rl_zoo3.record_video_checkpoints --algo dqn --num-videos-per-cpt 1 --checkpoint-interval 1"
echo ""

echo "Key parameters:"
echo "  --algo: Algorithm name (required) - must match folder in logs/"
echo "  --num-videos-per-cpt: Number of videos per checkpoint (default: 3)"
echo "    - Uses seeds 0, 1, 2, ... (num_videos_per_cpt - 1)"
echo "  --checkpoint-interval: Every nth checkpoint (default: 5)"
echo "    - E.g., with 50 checkpoints and interval=10, selects checkpoints 1, 11, 21, 31, 41, 50"
echo "  --output-folder: Where to save videos (default: checkpoint_videos)"
echo "  --n-timesteps: Length of each video (default: 1000)"
echo "  --deterministic/--stochastic: Action selection mode"
echo ""

echo "Output structure:"
echo "checkpoint_videos/"
echo "├── steps_00199992/"
echo "│   ├── dqn_BreakoutNoFrameskip-v4_steps_199992_seed_0-episode-0.mp4"
echo "│   ├── dqn_BreakoutNoFrameskip-v4_steps_199992_seed_1-episode-0.mp4"
echo "│   └── dqn_BreakoutNoFrameskip-v4_steps_199992_seed_2-episode-0.mp4"
echo "├── steps_01999920/"
echo "│   └── ..."
echo "└── ..."
echo ""

# Actually run a quick test to show it works
echo "Running a quick test with 1 video per checkpoint, interval=10, 200 timesteps:"
python -m rl_zoo3.record_video_checkpoints \
    --algo dqn \
    --logs-dir logs \
    --output-folder example_output \
    --num-videos-per-cpt 1 \
    --checkpoint-interval 10 \
    --n-timesteps 200 \
    --deterministic \
    --no-render

echo ""
echo "Test completed! Check the example_output/ directory for results."
