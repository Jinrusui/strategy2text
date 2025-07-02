#!/bin/bash

PYTHON="/home/sjr116/miniconda3/envs/s2t/bin/python"  # 你的 conda s2t 环境的 Python
ALGO="ppo"
ENV="BreakoutNoFrameskip-v4"
FOLDER="rl-trained-agents"
OUTPUT_DIR="videos/${ALGO}/${ENV}"
TIMESTEPS=2000
SEEDS=(123 56 78 94 25)

mkdir -p "$OUTPUT_DIR"

for SEED in "${SEEDS[@]}"; do
    SEED_DIR="${OUTPUT_DIR}/seed_${SEED}"
    mkdir -p "$SEED_DIR"
    
    echo "Recording: Seed $SEED"

    (cd rl-baselines3-zoo && "$PYTHON" -m rl_zoo3.record_video \
        --algo "$ALGO" \
        --env "$ENV" \
        --folder "rl-trained-agents" \
        --output-folder "../$SEED_DIR" \
        --seed "$SEED" \
        --n-timesteps "$TIMESTEPS" \
        --deterministic \
        --no-render)

    if [ $? -eq 0 ]; then
        echo "✓ Success for seed $SEED"
    else
        echo "✗ Failed for seed $SEED"
    fi
done

echo "All done. Check $OUTPUT_DIR for videos."
