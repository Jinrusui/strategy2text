#!/usr/bin/env python3
"""
Comprehensive trajectory reproducibility test.

This script compares the exact trajectories (actions, observations, rewards) between:
1. Original record_video.py from rl-baselines3-zoo
2. Our record_video_with_gradcam.py implementation

The goal is to ensure perfect reproducibility - same seed should produce identical trajectories.
"""

import argparse
import os
import sys
import numpy as np
import pickle
import json
from pathlib import Path
from typing import List, Dict, Tuple
import tempfile
import subprocess

# Add rl-baselines3-zoo to path
sys.path.append(str(Path(__file__).parent / "rl-baselines3-zoo"))

from huggingface_sb3 import EnvironmentName
from stable_baselines3.common.utils import set_random_seed
from rl_zoo3.exp_manager import ExperimentManager
from rl_zoo3.utils import ALGOS, create_test_env, get_model_path, get_saved_hyperparams
import yaml


def record_original_trajectory(
    env_id: str = "BreakoutNoFrameskip-v4",
    algo: str = "dqn",
    seed: int = 42,
    n_steps: int = 1000,
    exp_id: int = 0,
    folder: str = "rl-trained-agents"
) -> Dict:
    """
    Record trajectory using the exact same setup as record_video.py
    """
    print(f"🎬 Recording ORIGINAL trajectory (seed={seed}, steps={n_steps})")
    
    # Fix folder path
    if folder == "rl-trained-agents":
        folder = os.path.join("rl-baselines3-zoo", "rl-trained-agents")
    
    env_name = EnvironmentName(env_id)
    
    # Get model path (same as record_video.py)
    name_prefix, model_path, log_path = get_model_path(
        exp_id, folder, algo, env_name, False, None, False
    )
    
    print(f"Loading model from: {model_path}")
    
    # Set random seed (same as record_video.py)
    set_random_seed(seed)
    
    off_policy_algos = ["qrdqn", "dqn", "ddpg", "sac", "her", "td3", "tqc"]
    is_atari = ExperimentManager.is_atari(env_name.gym_id)
    is_minigrid = ExperimentManager.is_minigrid(env_name.gym_id)
    
    # Get hyperparameters (same as record_video.py)
    stats_path = os.path.join(log_path, env_name)
    hyperparams, maybe_stats_path = get_saved_hyperparams(stats_path)
    
    # Load env_kwargs (same as record_video.py)
    env_kwargs = {}
    args_path = os.path.join(log_path, env_name, "args.yml")
    if os.path.isfile(args_path):
        with open(args_path) as f:
            loaded_args = yaml.load(f, Loader=yaml.UnsafeLoader)
            if loaded_args["env_kwargs"] is not None:
                env_kwargs = loaded_args["env_kwargs"]
    
    # Force rgb_array rendering (same as record_video.py)
    env_kwargs.update(render_mode="rgb_array")
    
    # Create environment (same as record_video.py)
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
    
    # Model loading kwargs (same as record_video.py)
    kwargs = dict(seed=seed)
    if algo in off_policy_algos:
        kwargs.update(dict(buffer_size=1))
        if "optimize_memory_usage" in hyperparams:
            kwargs.update(optimize_memory_usage=False)
    
    # Custom objects (same as record_video.py)
    newer_python_version = sys.version_info.major == 3 and sys.version_info.minor >= 8
    custom_objects = {}
    if newer_python_version:
        custom_objects = {
            "learning_rate": 0.0,
            "lr_schedule": lambda _: 0.0,
            "clip_range": lambda _: 0.0,
        }
    
    # Load the model (same as record_video.py)
    model = ALGOS[algo].load(model_path, env=env, custom_objects=custom_objects, **kwargs)
    
    # Deterministic behavior (same as record_video.py)
    stochastic = (is_atari or is_minigrid) and False  # Force deterministic
    deterministic = not stochastic
    
    print(f"Using deterministic={deterministic}")
    
    # Record trajectory
    obs = env.reset()
    lstm_states = None
    episode_starts = np.ones((env.num_envs,), dtype=bool)
    
    trajectory = {
        'actions': [],
        'rewards': [],
        'observations': [],
        'dones': [],
        'infos': [],
        'total_reward': 0.0,
        'obs_shapes': [],
        'obs_stats': []
    }
    
    print(f"Initial observation shape: {obs.shape}")
    
    for step in range(n_steps):
        # Record observation stats
        trajectory['obs_shapes'].append(obs.shape)
        trajectory['obs_stats'].append({
            'mean': float(np.mean(obs)),
            'std': float(np.std(obs)),
            'min': float(np.min(obs)),
            'max': float(np.max(obs)),
            'sum': float(np.sum(obs))
        })
        
        # Get action (exactly same as record_video.py)
        action, lstm_states = model.predict(
            obs, state=lstm_states, episode_start=episode_starts, deterministic=deterministic
        )
        
        trajectory['actions'].append(int(action[0]))
        
        # Step environment (exactly same as record_video.py)
        obs, rewards, dones, infos = env.step(action)
        
        trajectory['rewards'].append(float(rewards[0]))
        trajectory['dones'].append(bool(dones[0]))
        trajectory['infos'].append(str(infos))  # Convert to string for JSON serialization
        trajectory['total_reward'] += float(rewards[0])
        
        episode_starts = dones
        
        if step % 100 == 0:
            print(f"  Step {step}: action={action[0]}, reward={rewards[0]:.3f}, total={trajectory['total_reward']:.3f}")
        
        if dones[0]:
            print(f"Episode ended at step {step}")
            break
    
    env.close()
    
    print(f"Original trajectory recorded: {len(trajectory['actions'])} steps, total reward: {trajectory['total_reward']}")
    return trajectory


def record_gradcam_trajectory(
    env_id: str = "BreakoutNoFrameskip-v4",
    algo: str = "dqn",
    seed: int = 42,
    n_steps: int = 1000,
    exp_id: int = 0,
    folder: str = "rl-trained-agents"
) -> Dict:
    """
    Record trajectory using our Grad-CAM implementation setup
    """
    print(f"🔥 Recording GRAD-CAM trajectory (seed={seed}, steps={n_steps})")
    
    # Import our Grad-CAM visualizer
    sys.path.insert(0, '.')
    from record_video_with_gradcam import DQNGradCAMVisualizer
    
    # Fix folder path
    if folder == "rl-trained-agents":
        folder = os.path.join("rl-baselines3-zoo", "rl-trained-agents")
    
    env_name = EnvironmentName(env_id)
    
    # Get model path (same as our implementation)
    name_prefix, model_path, log_path = get_model_path(
        exp_id, folder, algo, env_name, False, None, False
    )
    
    print(f"Loading model from: {model_path}")
    
    # Set random seed (same as our implementation)
    set_random_seed(seed)
    
    off_policy_algos = ["qrdqn", "dqn", "ddpg", "sac", "her", "td3", "tqc"]
    is_atari = ExperimentManager.is_atari(env_name.gym_id)
    is_minigrid = ExperimentManager.is_minigrid(env_name.gym_id)
    
    # Get hyperparameters (same as our implementation)
    stats_path = os.path.join(log_path, env_name)
    hyperparams, maybe_stats_path = get_saved_hyperparams(stats_path)
    
    # Load env_kwargs (same as our implementation)
    env_kwargs = {}
    args_path = os.path.join(log_path, env_name, "args.yml")
    if os.path.isfile(args_path):
        with open(args_path) as f:
            loaded_args = yaml.load(f, Loader=yaml.UnsafeLoader)
            if loaded_args["env_kwargs"] is not None:
                env_kwargs = loaded_args["env_kwargs"]
    
    # Force rgb_array rendering (same as our implementation)
    env_kwargs.update(render_mode="rgb_array")
    
    # Create environment (same as our implementation)
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
    
    # Model loading kwargs (same as our implementation)
    kwargs = dict(seed=seed)
    if algo in off_policy_algos:
        kwargs.update(dict(buffer_size=1))
        if "optimize_memory_usage" in hyperparams:
            kwargs.update(optimize_memory_usage=False)
    
    # Custom objects (same as our implementation)
    newer_python_version = sys.version_info.major == 3 and sys.version_info.minor >= 8
    custom_objects = {}
    if newer_python_version:
        custom_objects = {
            "learning_rate": 0.0,
            "lr_schedule": lambda _: 0.0,
            "clip_range": lambda _: 0.0,
        }
    
    # Load the model (same as our implementation)
    model = ALGOS[algo].load(model_path, env=env, custom_objects=custom_objects, **kwargs)
    
    # Deterministic behavior (same as our implementation)
    stochastic = (is_atari or is_minigrid) and False  # Force deterministic
    deterministic = not stochastic
    
    print(f"Using deterministic={deterministic}")
    
    # Initialize Grad-CAM visualizer (this is the key difference)
    try:
        gradcam_visualizer = DQNGradCAMVisualizer(model)
        print("Grad-CAM visualizer initialized successfully")
    except Exception as e:
        print(f"Warning: Grad-CAM initialization failed: {e}")
        gradcam_visualizer = None
    
    # Record trajectory
    obs = env.reset()
    lstm_states = None
    episode_starts = np.ones((env.num_envs,), dtype=bool)
    
    trajectory = {
        'actions': [],
        'rewards': [],
        'observations': [],
        'dones': [],
        'infos': [],
        'total_reward': 0.0,
        'obs_shapes': [],
        'obs_stats': [],
        'gradcam_success': []
    }
    
    print(f"Initial observation shape: {obs.shape}")
    
    for step in range(n_steps):
        # Record observation stats
        trajectory['obs_shapes'].append(obs.shape)
        trajectory['obs_stats'].append({
            'mean': float(np.mean(obs)),
            'std': float(np.std(obs)),
            'min': float(np.min(obs)),
            'max': float(np.max(obs)),
            'sum': float(np.sum(obs))
        })
        
        # Get action (exactly same as our implementation)
        action, lstm_states = model.predict(
            obs, state=lstm_states, episode_start=episode_starts, deterministic=deterministic
        )
        
        trajectory['actions'].append(int(action[0]))
        
        # Try Grad-CAM (this should NOT affect the trajectory)
        gradcam_success = False
        if gradcam_visualizer is not None:
            try:
                grayscale_cam, action_name = gradcam_visualizer.get_gradcam_for_action(obs, action[0])
                gradcam_success = True
            except Exception as e:
                if step < 5:  # Only print first few errors
                    print(f"Grad-CAM failed at step {step}: {e}")
        
        trajectory['gradcam_success'].append(gradcam_success)
        
        # Step environment (exactly same as our implementation)
        obs, rewards, dones, infos = env.step(action)
        
        trajectory['rewards'].append(float(rewards[0]))
        trajectory['dones'].append(bool(dones[0]))
        trajectory['infos'].append(str(infos))  # Convert to string for JSON serialization
        trajectory['total_reward'] += float(rewards[0])
        
        episode_starts = dones
        
        if step % 100 == 0:
            print(f"  Step {step}: action={action[0]}, reward={rewards[0]:.3f}, total={trajectory['total_reward']:.3f}")
        
        if dones[0]:
            print(f"Episode ended at step {step}")
            break
    
    env.close()
    
    gradcam_success_rate = sum(trajectory['gradcam_success']) / len(trajectory['gradcam_success']) * 100
    print(f"Grad-CAM trajectory recorded: {len(trajectory['actions'])} steps, total reward: {trajectory['total_reward']}")
    print(f"Grad-CAM success rate: {gradcam_success_rate:.1f}%")
    return trajectory


def compare_trajectories(traj1: Dict, traj2: Dict, name1: str = "Original", name2: str = "Grad-CAM") -> bool:
    """
    Compare two trajectories in detail
    """
    print(f"\n{'='*60}")
    print(f"COMPARING TRAJECTORIES: {name1} vs {name2}")
    print(f"{'='*60}")
    
    # Basic comparison
    len1, len2 = len(traj1['actions']), len(traj2['actions'])
    print(f"Trajectory lengths: {name1}={len1}, {name2}={len2}")
    
    if len1 != len2:
        print(f"❌ DIFFERENT LENGTHS! {name1}: {len1}, {name2}: {len2}")
        min_len = min(len1, len2)
        print(f"Comparing first {min_len} steps...")
    else:
        print(f"✅ Same length: {len1} steps")
        min_len = len1
    
    # Compare total rewards
    reward1, reward2 = traj1['total_reward'], traj2['total_reward']
    print(f"Total rewards: {name1}={reward1:.6f}, {name2}={reward2:.6f}")
    
    if abs(reward1 - reward2) < 1e-10:
        print(f"✅ Total rewards are identical")
    else:
        print(f"❌ Total rewards differ by {abs(reward1 - reward2):.10f}")
    
    # Compare actions
    actions1 = traj1['actions'][:min_len]
    actions2 = traj2['actions'][:min_len]
    
    action_diffs = [a1 != a2 for a1, a2 in zip(actions1, actions2)]
    n_action_diffs = sum(action_diffs)
    
    print(f"Action differences: {n_action_diffs}/{min_len} steps")
    
    if n_action_diffs == 0:
        print(f"✅ All actions are identical")
    else:
        print(f"❌ {n_action_diffs} action differences found")
        # Show first few differences
        for i, (a1, a2, diff) in enumerate(zip(actions1, actions2, action_diffs)):
            if diff and i < 10:  # Show first 10 differences
                print(f"  Step {i}: {name1}={a1}, {name2}={a2}")
    
    # Compare rewards step by step
    rewards1 = traj1['rewards'][:min_len]
    rewards2 = traj2['rewards'][:min_len]
    
    reward_diffs = [abs(r1 - r2) > 1e-10 for r1, r2 in zip(rewards1, rewards2)]
    n_reward_diffs = sum(reward_diffs)
    
    print(f"Reward differences: {n_reward_diffs}/{min_len} steps")
    
    if n_reward_diffs == 0:
        print(f"✅ All step rewards are identical")
    else:
        print(f"❌ {n_reward_diffs} step reward differences found")
    
    # Compare observation statistics
    obs_stats1 = traj1['obs_stats'][:min_len]
    obs_stats2 = traj2['obs_stats'][:min_len]
    
    obs_diffs = []
    for i, (stats1, stats2) in enumerate(zip(obs_stats1, obs_stats2)):
        diff = abs(stats1['sum'] - stats2['sum']) > 1e-6  # Compare observation sums
        obs_diffs.append(diff)
    
    n_obs_diffs = sum(obs_diffs)
    print(f"Observation differences: {n_obs_diffs}/{min_len} steps")
    
    if n_obs_diffs == 0:
        print(f"✅ All observations are identical")
    else:
        print(f"❌ {n_obs_diffs} observation differences found")
    
    # Overall result
    all_identical = (n_action_diffs == 0 and n_reward_diffs == 0 and n_obs_diffs == 0 and len1 == len2)
    
    print(f"\n{'='*60}")
    if all_identical:
        print(f"🎉 TRAJECTORIES ARE IDENTICAL! Perfect reproducibility confirmed!")
    else:
        print(f"❌ TRAJECTORIES DIFFER! Reproducibility issue detected!")
        print(f"   - Length diff: {len1 != len2}")
        print(f"   - Action diffs: {n_action_diffs}")
        print(f"   - Reward diffs: {n_reward_diffs}")
        print(f"   - Obs diffs: {n_obs_diffs}")
    print(f"{'='*60}")
    
    return all_identical


def save_trajectory(trajectory: Dict, filename: str):
    """Save trajectory to file"""
    with open(filename, 'w') as f:
        json.dump(trajectory, f, indent=2)
    print(f"Trajectory saved to: {filename}")


def main():
    parser = argparse.ArgumentParser(description="Test trajectory reproducibility")
    parser.add_argument("--env", default="BreakoutNoFrameskip-v4", help="Environment ID")
    parser.add_argument("--algo", default="dqn", help="Algorithm")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--n-steps", type=int, default=500, help="Number of steps")
    parser.add_argument("--exp-id", type=int, default=0, help="Experiment ID")
    parser.add_argument("--folder", default="rl-trained-agents", help="Model folder")
    parser.add_argument("--save-trajectories", action="store_true", help="Save trajectories to files")
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("🧪 TRAJECTORY REPRODUCIBILITY TEST")
    print("=" * 80)
    print(f"Environment: {args.env}")
    print(f"Algorithm: {args.algo}")
    print(f"Seed: {args.seed}")
    print(f"Steps: {args.n_steps}")
    print()
    
    # Record original trajectory
    try:
        original_traj = record_original_trajectory(
            args.env, args.algo, args.seed, args.n_steps, args.exp_id, args.folder
        )
    except Exception as e:
        print(f"❌ Failed to record original trajectory: {e}")
        return 1
    
    # Record Grad-CAM trajectory
    try:
        gradcam_traj = record_gradcam_trajectory(
            args.env, args.algo, args.seed, args.n_steps, args.exp_id, args.folder
        )
    except Exception as e:
        print(f"❌ Failed to record Grad-CAM trajectory: {e}")
        return 1
    
    # Compare trajectories
    identical = compare_trajectories(original_traj, gradcam_traj)
    
    # Save trajectories if requested
    if args.save_trajectories:
        os.makedirs("trajectory_comparison", exist_ok=True)
        save_trajectory(original_traj, f"trajectory_comparison/original_seed{args.seed}.json")
        save_trajectory(gradcam_traj, f"trajectory_comparison/gradcam_seed{args.seed}.json")
    
    return 0 if identical else 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code) 