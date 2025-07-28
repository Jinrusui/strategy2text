#!/bin/bash

# Test script for record_video_checkpoints.py
# This script demonstrates how to use the new checkpoint video recording functionality

echo "Testing record_video_checkpoints.py"
echo "===================================="

# Change to the rl-baselines3-zoo directory
cd /mnt/e/Projects/strategy2text/rl-baselines3-zoo

# Test 1: Record videos from DQN checkpoints with default settings
echo "Test 1: Recording videos from DQN checkpoints (interval=5, 3 videos per checkpoint)"
python -m rl_zoo3.record_video_checkpoints \
    --algo dqn \
    --logs-dir logs \
    --output-folder checkpoint_videos/test1 \
    --num-videos-per-cpt 3 \
    --checkpoint-interval 5 \
    --n-timesteps 500 \
    --deterministic

echo ""
echo "Test 1 completed. Check checkpoint_videos/test1/ for results."
echo ""

# Test 2: Record videos with smaller interval and more videos per checkpoint
echo "Test 2: Recording videos with smaller interval (interval=2, 2 videos per checkpoint)"
python -m rl_zoo3.record_video_checkpoints \
    --algo dqn \
    --logs-dir logs \
    --output-folder checkpoint_videos/test2 \
    --num-videos-per-cpt 2 \
    --checkpoint-interval 2 \
    --n-timesteps 300 \
    --deterministic

echo ""
echo "Test 2 completed. Check checkpoint_videos/test2/ for results."
echo ""

# Test 3: Dry run to see what checkpoints would be selected
echo "Test 3: Showing available checkpoints (this will fail at video recording but show checkpoint selection)"
python -c "
import sys
sys.path.append('.')
from rl_zoo3.record_video_checkpoints import find_checkpoint_files, select_checkpoints_by_interval

try:
    checkpoints = find_checkpoint_files('logs', 'dqn')
    print(f'Found {len(checkpoints)} total checkpoints:')
    for i, (path, steps) in enumerate(checkpoints[:10]):  # Show first 10
        print(f'  {i+1:2d}. {steps:8,} steps')
    if len(checkpoints) > 10:
        print(f'  ... and {len(checkpoints) - 10} more')
    
    print()
    selected = select_checkpoints_by_interval(checkpoints, 5)
    print(f'With interval=5, would select {len(selected)} checkpoints:')
    for i, (path, steps) in enumerate(selected):
        print(f'  {i+1:2d}. {steps:8,} steps')
        
except Exception as e:
    print(f'Error: {e}')
"

echo ""
echo "All tests completed!"
echo "Videos are saved in checkpoint_videos/ directory with subdirectories for each checkpoint step count."
