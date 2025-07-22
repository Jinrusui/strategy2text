# Grad-CAM Visualization for DQN Agents in Breakout

This implementation provides Grad-CAM (Gradient-weighted Class Activation Mapping) visualization for Deep Q-Network (DQN) agents playing Atari Breakout. The system ensures **perfect reproducibility** with the original `record_video.py` script, meaning the agent will take the exact same actions and achieve the same rewards under identical seeds.

## 🎯 Features

- **Reproducible Behavior**: Identical action sequences and environment states compared to original `record_video.py`
- **Real-time Grad-CAM Visualization**: See what parts of the screen the DQN agent focuses on for each action
- **Multiple Output Formats**: Generate both original videos and Grad-CAM overlay videos
- **Frame-by-Frame Analysis**: Save individual frames with Grad-CAM visualizations
- **Action Annotations**: Each frame shows the action taken and its name (NOOP, FIRE, RIGHT, LEFT)
- **Automatic Layer Detection**: Automatically finds the best convolutional layers for Grad-CAM

## 🔧 Requirements

Before using this implementation, ensure you have:

1. **pytorch-grad-cam** library:
   ```bash
   pip install grad-cam
   ```

2. **rl-baselines3-zoo** setup with a trained DQN model for Breakout

3. **Required Python packages**:
   ```bash
   pip install torch torchvision opencv-python matplotlib numpy
   ```

## 📁 File Structure

```
strategy2text/
├── record_video_with_gradcam.py      # Main Grad-CAM implementation
├── test_gradcam_reproducibility.py   # Reproducibility testing
├── README_GRADCAM.md                 # This documentation
├── rl-baselines3-zoo/                # SB3 Zoo directory
│   └── rl-trained-agents/            # Trained models location
└── videos/                           # Output videos directory
    ├── original/                     # Original videos (no Grad-CAM)
    └── gradcam/                      # Grad-CAM visualized videos
```

## 🚀 Quick Start

### 1. Basic Usage

Record a video with Grad-CAM visualization:

```bash
python record_video_with_gradcam.py \
    --env BreakoutNoFrameskip-v4 \
    --algo dqn \
    --seed 42 \
    -n 1000 \
    --deterministic \
    -o ./gradcam_videos
```

### 2. Advanced Options

```bash
python record_video_with_gradcam.py \
    --env BreakoutNoFrameskip-v4 \
    --algo dqn \
    --seed 42 \
    -n 1000 \
    --deterministic \
    --save-frames \
    --exp-id 1 \
    --load-best \
    -o ./gradcam_videos
```

### 3. Command Line Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--env` | Environment ID | `BreakoutNoFrameskip-v4` |
| `--algo` | Algorithm (dqn, qrdqn) | `dqn` |
| `--seed` | Random seed for reproducibility | `0` |
| `-n, --n-timesteps` | Number of timesteps to record | `1000` |
| `--deterministic` | Use deterministic actions | `False` |
| `--save-frames` | Save individual Grad-CAM frames | `False` |
| `--exp-id` | Experiment ID (0=latest) | `0` |
| `--load-best` | Load best model instead of last | `False` |
| `-o, --output-folder` | Output directory | Auto-detected |

## 🔬 Testing Reproducibility

### Quick Reproducibility Test

Generate a minimal test script and run it:

```bash
# Create the minimal test script
python test_gradcam_reproducibility.py --create-minimal

# Run the reproducibility test
python test_minimal_reproducibility.py
```

Expected output:
```
============================================================
REPRODUCIBILITY TEST
============================================================

🎯 Run 1:
Testing action reproducibility with seed 42
...
🎯 Run 2 (same seed):
Testing action reproducibility with seed 42
...

📊 RESULTS:
Actions 1: [1, 2, 2, 1, 0, 3, 2, ...]
Actions 2: [1, 2, 2, 1, 0, 3, 2, ...]
✅ ACTIONS ARE IDENTICAL - Reproducibility confirmed!
✅ REWARDS ARE IDENTICAL - Environment consistency confirmed!
```

### Full Reproducibility Test

Compare complete video recordings:

```bash
python test_gradcam_reproducibility.py \
    --seed 42 \
    --n-steps 200 \
    --algo dqn
```

## 📊 Understanding Grad-CAM Output

### Video Structure

The Grad-CAM implementation generates two types of videos:

1. **Original Video** (`original/`): Standard gameplay video (same as `record_video.py`)
2. **Grad-CAM Video** (`gradcam/`): Overlaid with heat maps showing agent attention

### Heat Map Interpretation

- **Red regions**: Areas the agent considers most important for the current action
- **Blue/Cool regions**: Areas the agent considers less relevant
- **Action overlay**: Shows the current action being taken (NOOP, FIRE, RIGHT, LEFT)

### Example Frame Analysis

```
🔥 Grad-CAM Analysis for Breakout:

Frame 156: Action = RIGHT (2)
├─ Red focus on: Ball trajectory and paddle position
├─ Attention pattern: Following ball movement toward right side
└─ Decision rationale: Agent preparing to intercept ball
```

## 🔧 Troubleshooting

### Common Issues

1. **"Could not find q_net in the model"**
   - Ensure you're using a DQN or QR-DQN model
   - Check that the model was trained with SB3

2. **"No convolutional layers found"**
   - Verify your model has CNN layers
   - Check model architecture matches expected DQN structure

3. **Import errors for pytorch-grad-cam**
   ```bash
   pip install grad-cam
   ```

4. **Model not found errors**
   ```bash
   # Check your model path
   ls rl-baselines3-zoo/rl-trained-agents/dqn/
   
   # Use specific experiment ID
   python record_video_with_gradcam.py --exp-id 1
   ```

### Performance Considerations

- **GPU Recommended**: Grad-CAM computation is faster on GPU
- **Memory Usage**: Grad-CAM requires additional memory for gradient computation
- **Recording Time**: Expect ~2-3x longer recording time due to Grad-CAM processing

## 🧠 Technical Details

### Architecture Compatibility

Currently supports:
- ✅ **DQN**: Deep Q-Network
- ✅ **QR-DQN**: Quantile Regression DQN
- ❌ **PPO/A2C**: Not supported (different network architecture)

### Grad-CAM Implementation

The system:
1. Extracts the Q-network from the SB3 model wrapper
2. Auto-detects the last 1-2 convolutional layers
3. Computes gradients with respect to the selected action
4. Generates activation maps showing decision importance
5. Overlays heatmaps on original game frames

### Reproducibility Guarantees

The implementation ensures reproducibility by:
- Using identical environment setup as `record_video.py`
- Preserving all random seeds and initialization
- Maintaining exact model inference pipeline
- Only adding visualization without affecting decision-making

## 📈 Example Use Cases

### 1. Research Analysis
```python
# Analyze agent decision patterns
python record_video_with_gradcam.py \
    --seed 42 --deterministic \
    --save-frames \
    -n 5000
```

### 2. Model Comparison
```bash
# Compare different training checkpoints
for checkpoint in 100000 500000 1000000; do
    python record_video_with_gradcam.py \
        --load-checkpoint $checkpoint \
        --seed 42 \
        -o "gradcam_checkpoint_$checkpoint"
done
```

### 3. Seed Analysis
```bash
# Test different random seeds
for seed in 0 42 123 999; do
    python record_video_with_gradcam.py \
        --seed $seed \
        --deterministic \
        -o "gradcam_seed_$seed"
done
```

## 🤝 Contributing

To extend this implementation:

1. **Add new algorithms**: Modify `DQNGradCAMVisualizer` class
2. **Improve visualizations**: Enhance `create_visualization_frame` method
3. **Add new CAM methods**: Integrate additional pytorch-grad-cam methods

## 📄 Citation

If you use this implementation in your research, please cite:

```bibtex
@misc{gradcam_rl_breakout,
  title={Grad-CAM Visualization for DQN Agents in Atari Breakout},
  author={Your Name},
  year={2024},
  howpublished={\url{https://github.com/your-repo/strategy2text}}
}
```

## 📚 References

1. **Grad-CAM Paper**: Selvaraju et al. "Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization" (ICCV 2017)
2. **pytorch-grad-cam**: https://github.com/jacobgil/pytorch-grad-cam
3. **Stable-Baselines3**: https://github.com/DLR-RM/stable-baselines3
4. **RL Baselines3 Zoo**: https://github.com/DLR-RM/rl-baselines3-zoo

---

## 🎬 Expected Output

After running the Grad-CAM visualization, you should see:

```
✅ pytorch-grad-cam imported successfully
Loading /path/to/model.zip
Using seed: 42
Using target layers: ['Conv2d(32, 64, kernel_size=(4, 4), stride=(2, 2))', ...]
Using deterministic=True, stochastic=False
Initializing Grad-CAM visualizer...
Recording original video...
Setting up Grad-CAM recording...
Starting recording for 1000 timesteps...
Environment shape: (1, 4, 84, 84)
Step 0/1000, Reward: 0.00, Total: 0.00
Step 100/1000, Reward: 0.00, Total: 0.00
...
Creating Grad-CAM video from 1000 frames...
Grad-CAM video saved to: /path/to/videos/gradcam/model_gradcam.mp4
Recording completed!
Total reward: 12.0
Original video folder: /path/to/videos/original
Grad-CAM video folder: /path/to/videos/gradcam
```

The agent behavior and total reward should be **identical** to running the original `record_video.py` with the same seed! 🎯 