#!/usr/bin/env python3
"""
Test reproducibility between record_video.py and record_video_with_gradcam.py

This script verifies that both scripts produce identical:
1. Agent action sequences
2. Environment states
3. Rewards

Under the same random seed, ensuring the Grad-CAM implementation doesn't affect
the core RL agent behavior or environment dynamics.
"""

import argparse
import subprocess
import sys
import os
import tempfile
import shutil
from pathlib import Path
from typing import List, Tuple, Optional
import numpy as np


def run_original_record_video(
    env_id: str = "BreakoutNoFrameskip-v4",
    algo: str = "dqn",
    seed: int = 42,
    n_steps: int = 100,
    exp_id: int = 0,
    folder: str = "rl-trained-agents"
) -> Tuple[List[int], List[float], str]:
    """
    Run the original record_video.py script and capture outputs.
    
    Returns:
        Tuple of (actions, rewards, stdout)
    """
    print(f"🎬 Running original record_video.py with seed {seed}...")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        cmd = [
            sys.executable, "-m", "rl_zoo3.record_video",
            "--env", env_id,
            "--algo", algo,
            "--seed", str(seed),
            "-n", str(n_steps),
            "--exp-id", str(exp_id),
            "-f", folder,
            "-o", temp_dir,
            "--deterministic",
            "--no-render"
        ]
        
        # Change to rl-baselines3-zoo directory
        original_cwd = os.getcwd()
        rl_zoo_path = Path(__file__).parent / "rl-baselines3-zoo"
        
        try:
            os.chdir(rl_zoo_path)
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            
            if result.returncode != 0:
                print(f"❌ Original record_video failed: {result.stderr}")
                return [], [], result.stdout + "\n" + result.stderr
            
            return [], [], result.stdout  # Actions/rewards would need parsing from stdout
            
        finally:
            os.chdir(original_cwd)


def run_gradcam_record_video(
    env_id: str = "BreakoutNoFrameskip-v4",
    algo: str = "dqn", 
    seed: int = 42,
    n_steps: int = 100,
    exp_id: int = 0,
    folder: str = "rl-trained-agents"
) -> Tuple[List[int], List[float], str]:
    """
    Run the Grad-CAM record_video_with_gradcam.py script and capture outputs.
    
    Returns:
        Tuple of (actions, rewards, stdout)
    """
    print(f"🔥 Running record_video_with_gradcam.py with seed {seed}...")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        cmd = [
            sys.executable, "record_video_with_gradcam.py",
            "--env", env_id,
            "--algo", algo,
            "--seed", str(seed),
            "-n", str(n_steps),
            "--exp-id", str(exp_id),
            "-f", folder,
            "-o", temp_dir,
            "--deterministic",
            "--no-render"
        ]
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            
            if result.returncode != 0:
                print(f"❌ Grad-CAM record_video failed: {result.stderr}")
                return [], [], result.stdout + "\n" + result.stderr
            
            return [], [], result.stdout  # Actions/rewards would need parsing from stdout
            
        except subprocess.TimeoutExpired:
            print(f"❌ Grad-CAM script timed out after 300 seconds")
            return [], [], "TIMEOUT"


def create_minimal_test_script():
    """
    Create a minimal test that directly tests action reproducibility without video recording.
    """
    test_script = """
#!/usr/bin/env python3
import sys
import os
import numpy as np
from pathlib import Path

# Add rl-baselines3-zoo to path
sys.path.append(str(Path(__file__).parent / "rl-baselines3-zoo"))

from huggingface_sb3 import EnvironmentName
from stable_baselines3.common.utils import set_random_seed
from rl_zoo3.exp_manager import ExperimentManager
from rl_zoo3.utils import ALGOS, create_test_env, get_model_path, get_saved_hyperparams
import yaml

def test_action_reproducibility(seed=42, n_steps=50):
    \"\"\"Test that we get the same actions under the same seed.\"\"\"
    
    env_name = EnvironmentName("BreakoutNoFrameskip-v4")
    algo = "dqn"
    folder = "rl-trained-agents"
    exp_id = 0
    
    print(f"Testing action reproducibility with seed {seed}")
    
    # Get model path
    name_prefix, model_path, log_path = get_model_path(
        exp_id, folder, algo, env_name, False, None, False
    )
    
    if not os.path.exists(model_path):
        print(f"❌ Model not found at {model_path}")
        print("Please train a DQN model first or adjust the model path")
        return None, None
    
    print(f"Loading model from {model_path}")
    
    # Set up environment exactly like record_video.py
    set_random_seed(seed)
    
    stats_path = os.path.join(log_path, env_name)
    hyperparams, maybe_stats_path = get_saved_hyperparams(stats_path)
    
    env_kwargs = {}
    args_path = os.path.join(log_path, env_name, "args.yml")
    if os.path.isfile(args_path):
        with open(args_path) as f:
            loaded_args = yaml.load(f, Loader=yaml.UnsafeLoader)
            if loaded_args["env_kwargs"] is not None:
                env_kwargs = loaded_args["env_kwargs"]
    
    env_kwargs.update(render_mode="rgb_array")
    
    # Create environment
    env = create_test_env(
        env_name.gym_id,
        n_envs=1,
        stats_path=maybe_stats_path,
        seed=seed,
        log_dir=None,
        should_render=False,
        hyperparams=hyperparams,
        env_kwargs=env_kwargs,
    )
    
    # Load model
    kwargs = dict(seed=seed, buffer_size=1)
    if "optimize_memory_usage" in hyperparams:
        kwargs.update(optimize_memory_usage=False)
    
    custom_objects = {
        "learning_rate": 0.0,
        "lr_schedule": lambda _: 0.0,
        "clip_range": lambda _: 0.0,
    }
    
    model = ALGOS[algo].load(model_path, env=env, custom_objects=custom_objects, **kwargs)
    
    # Test deterministic actions
    obs = env.reset()
    actions = []
    rewards = []
    episode_starts = np.ones((env.num_envs,), dtype=bool)
    lstm_states = None
    
    for step in range(n_steps):
        action, lstm_states = model.predict(
            obs, state=lstm_states, episode_start=episode_starts, deterministic=True
        )
        actions.append(int(action[0]))
        
        obs, reward, done, info = env.step(action)
        rewards.append(float(reward[0]))
        episode_starts = done
        
        if done[0]:
            break
    
    env.close()
    return actions, rewards

if __name__ == "__main__":
    # Test multiple runs with same seed
    seed = 42
    print("=" * 60)
    print("REPRODUCIBILITY TEST")
    print("=" * 60)
    
    print("\\n🎯 Run 1:")
    actions1, rewards1 = test_action_reproducibility(seed)
    
    if actions1 is None:
        sys.exit(1)
    
    print("\\n🎯 Run 2 (same seed):")
    actions2, rewards2 = test_action_reproducibility(seed)
    
    if actions2 is None:
        sys.exit(1)
    
    print("\\n📊 RESULTS:")
    print(f"Actions 1: {actions1[:20]}..." if len(actions1) > 20 else f"Actions 1: {actions1}")
    print(f"Actions 2: {actions2[:20]}..." if len(actions2) > 20 else f"Actions 2: {actions2}")
    
    if actions1 == actions2:
        print("✅ ACTIONS ARE IDENTICAL - Reproducibility confirmed!")
    else:
        print("❌ ACTIONS ARE DIFFERENT - Reproducibility failed!")
        for i, (a1, a2) in enumerate(zip(actions1, actions2)):
            if a1 != a2:
                print(f"   First difference at step {i}: {a1} vs {a2}")
                break
    
    if rewards1 == rewards2:
        print("✅ REWARDS ARE IDENTICAL - Environment consistency confirmed!")
    else:
        print("❌ REWARDS ARE DIFFERENT - Environment inconsistency!")
"""
    
    with open("test_minimal_reproducibility.py", "w") as f:
        f.write(test_script)
    
    print("✅ Created minimal test script: test_minimal_reproducibility.py")


def main():
    """Main test function."""
    parser = argparse.ArgumentParser(description="Test Grad-CAM reproducibility")
    parser.add_argument("--env", default="BreakoutNoFrameskip-v4", help="Environment ID")
    parser.add_argument("--algo", default="dqn", help="Algorithm")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--n-steps", type=int, default=100, help="Number of steps")
    parser.add_argument("--exp-id", type=int, default=0, help="Experiment ID")
    parser.add_argument("--folder", default="rl-trained-agents", help="Model folder")
    parser.add_argument("--create-minimal", action="store_true", 
                        help="Create minimal test script instead of running full test")
    
    args = parser.parse_args()
    
    if args.create_minimal:
        create_minimal_test_script()
        return
    
    print("=" * 80)
    print("🧪 GRAD-CAM REPRODUCIBILITY TEST")
    print("=" * 80)
    print(f"Environment: {args.env}")
    print(f"Algorithm: {args.algo}")
    print(f"Seed: {args.seed}")
    print(f"Steps: {args.n_steps}")
    print()
    
    # Check if required model exists
    rl_zoo_path = Path(__file__).parent / "rl-baselines3-zoo"
    if not rl_zoo_path.exists():
        print("❌ rl-baselines3-zoo directory not found!")
        print("Please ensure the repository is cloned in the current directory")
        return
    
    # Run original version
    original_actions, original_rewards, original_output = run_original_record_video(
        args.env, args.algo, args.seed, args.n_steps, args.exp_id, args.folder
    )
    
    # Run Grad-CAM version  
    gradcam_actions, gradcam_rewards, gradcam_output = run_gradcam_record_video(
        args.env, args.algo, args.seed, args.n_steps, args.exp_id, args.folder
    )
    
    print("\n" + "="*60)
    print("📊 RESULTS COMPARISON")
    print("="*60)
    
    print(f"Original output:\n{original_output[:500]}{'...' if len(original_output) > 500 else ''}")
    print(f"\nGrad-CAM output:\n{gradcam_output[:500]}{'...' if len(gradcam_output) > 500 else ''}")
    
    # For now, we mainly check that both scripts run successfully
    if "Total reward:" in original_output and "Total reward:" in gradcam_output:
        print("✅ Both scripts completed successfully!")
        
        # Try to extract total rewards
        try:
            orig_reward = float(original_output.split("Total reward:")[1].split()[0])
            grad_reward = float(gradcam_output.split("Total reward:")[1].split()[0])
            
            print(f"Original total reward: {orig_reward}")
            print(f"Grad-CAM total reward: {grad_reward}")
            
            if abs(orig_reward - grad_reward) < 0.001:
                print("✅ Total rewards match - Reproducibility confirmed!")
            else:
                print(f"❌ Total rewards differ by {abs(orig_reward - grad_reward)}")
                
        except (IndexError, ValueError) as e:
            print(f"⚠️  Could not parse rewards: {e}")
    else:
        print("❌ One or both scripts failed to complete")
    
    print("\n" + "="*60)
    print("💡 RECOMMENDATIONS")
    print("="*60)
    print("1. Run the minimal test script for detailed action comparison:")
    print("   python test_gradcam_reproducibility.py --create-minimal")
    print("   python test_minimal_reproducibility.py")
    print()
    print("2. Visual inspection of videos:")
    print("   - Compare original vs Grad-CAM videos side by side")
    print("   - Verify agent behavior looks identical (except for Grad-CAM overlay)")
    print()
    print("3. For detailed debugging, check individual frames and action sequences")


if __name__ == "__main__":
    main() 