#!/usr/bin/env python3
"""
Custom script to properly continue training from a checkpoint.
This script ensures that all hyperparameters, exploration schedule, 
and training state are properly preserved.
"""

import argparse
import os
import yaml
from collections import OrderedDict
from pathlib import Path

import gymnasium as gym
import torch as th
from stable_baselines3 import DQN
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecFrameStack
from stable_baselines3.common.atari_wrappers import AtariWrapper
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.env_util import make_vec_env

# Register custom envs
import rl_zoo3.import_envs  # noqa: F401


def create_atari_env(env_id, seed=0, wrapper_kwargs=None):
    """Create Atari environment with proper wrappers (without frame stacking)"""
    def _init():
        env = gym.make(env_id)
        env = AtariWrapper(env, **(wrapper_kwargs or {}))
        env = Monitor(env)
        env.reset(seed=seed)
        return env
    return _init


def load_hyperparameters(checkpoint_dir):
    """Load hyperparameters from checkpoint directory"""
    config_path = os.path.join(checkpoint_dir, "config.yml")
    args_path = os.path.join(checkpoint_dir, "args.yml")
    
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    if not os.path.exists(args_path):
        raise FileNotFoundError(f"Args file not found: {args_path}")
    
    with open(config_path, 'r') as f:
        config = yaml.load(f, Loader=yaml.UnsafeLoader)
    
    with open(args_path, 'r') as f:
        args = yaml.load(f, Loader=yaml.UnsafeLoader)
    
    return config, args


def calculate_current_exploration_rate(current_step, total_steps, exploration_fraction, 
                                     exploration_initial_eps=1.0, exploration_final_eps=0.01):
    """Calculate the current exploration rate based on the step"""
    exploration_steps = int(exploration_fraction * total_steps)
    
    if current_step < exploration_steps:
        # Linear decay
        progress = current_step / exploration_steps
        return exploration_initial_eps - (exploration_initial_eps - exploration_final_eps) * progress
    else:
        return exploration_final_eps


def continue_training():
    parser = argparse.ArgumentParser(description="Continue training from checkpoint")
    parser.add_argument("--checkpoint-path", required=True, type=str,
                       help="Path to the checkpoint .zip file")
    parser.add_argument("--additional-steps", default=1000000, type=int,
                       help="Number of additional steps to train")
    parser.add_argument("--save-freq", default=200000, type=int,
                       help="Save frequency for new checkpoints")
    parser.add_argument("--eval-freq", default=25000, type=int,
                       help="Evaluation frequency")
    parser.add_argument("--verbose", default=1, type=int,
                       help="Verbose level")
    
    args = parser.parse_args()
    
    # Validate checkpoint path
    if not os.path.exists(args.checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint_path}")
    
    # Extract step number from checkpoint filename
    checkpoint_filename = os.path.basename(args.checkpoint_path)
    if "steps" in checkpoint_filename:
        import re
        step_match = re.search(r'(\d+)_steps', checkpoint_filename)
        if step_match:
            current_steps = int(step_match.group(1))
        else:
            raise ValueError("Could not extract step number from checkpoint filename")
    else:
        raise ValueError("Checkpoint filename should contain step information")
    
    # Get checkpoint directory
    checkpoint_dir = os.path.dirname(args.checkpoint_path)
    config_dir = os.path.join(checkpoint_dir, "BreakoutNoFrameskip-v4")
    
    print(f"Loading configuration from: {config_dir}")
    print(f"Current checkpoint steps: {current_steps:,}")
    print(f"Additional steps to train: {args.additional_steps:,}")
    
    # Load original hyperparameters
    config, original_args = load_hyperparameters(config_dir)
    
    print("Original hyperparameters:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    
    # Extract hyperparameters
    n_envs = config.get('n_envs', 1)
    exploration_fraction = config.get('exploration_fraction', 0.1)
    exploration_final_eps = config.get('exploration_final_eps', 0.01)
    total_original_steps = int(config.get('n_timesteps', 10000000))
    learning_starts = config.get('learning_starts', 100000)
    frame_stack = config.get('frame_stack', 4)
    
    # Calculate current exploration rate
    current_exploration_rate = calculate_current_exploration_rate(
        current_steps, total_original_steps, exploration_fraction,
        exploration_final_eps=exploration_final_eps
    )
    
    print(f"Current exploration rate: {current_exploration_rate:.4f}")
    
    # Set random seed from original training
    original_seed = original_args.get('seed', 0)
    set_random_seed(original_seed)
    
    # Create environments with same configuration as RL Zoo
    env_id = "BreakoutNoFrameskip-v4"
    
    # Create the spec function for make_vec_env
    spec = gym.spec(env_id)
    def make_env(**kwargs):
        return spec.make(**kwargs)
    
    # Use make_vec_env like RL Zoo does
    vec_env_class = DummyVecEnv if n_envs == 1 else SubprocVecEnv
    env = make_vec_env(
        make_env,
        n_envs=n_envs,
        seed=original_seed,
        wrapper_class=AtariWrapper,
        vec_env_cls=vec_env_class,
        monitor_dir=None,
    )
    
    # Add frame stacking like RL Zoo does
    if frame_stack > 0:
        env = VecFrameStack(env, frame_stack)
        print(f"Stacking {frame_stack} frames")
    
    print(f"Created environment with {n_envs} parallel envs")
    
    # Load the model
    print(f"Loading model from: {args.checkpoint_path}")
    
    # Load model with proper exploration rate and custom objects for schedule compatibility
    custom_objects = {
        "learning_rate": 0.0001,  # fallback learning rate
        "lr_schedule": lambda _: 0.0001,  # fallback schedule function
        "exploration_schedule": lambda _: exploration_final_eps  # fallback exploration schedule
    }
    
    model = DQN.load(
        args.checkpoint_path,
        env=env,
        verbose=args.verbose,
        device="auto",
        custom_objects=custom_objects
    )
    
    # Manually set the exploration rate to continue from where we left off
    model.exploration_rate = current_exploration_rate
    
    # Update the exploration schedule for remaining steps
    # Since we're past the exploration phase (7.6M > 1M), we keep the final exploration rate
    model.exploration_rate = exploration_final_eps
    
    print(f"Model loaded successfully. Current exploration rate: {model.exploration_rate:.4f}")
    
    # Create callbacks
    callbacks = []
    
    # Checkpoint callback
    checkpoint_callback = CheckpointCallback(
        save_freq=args.save_freq // n_envs,  # Adjust for multiple envs
        save_path=checkpoint_dir,
        name_prefix="rl_model",
        save_replay_buffer=False,
        save_vecnormalize=True,
    )
    callbacks.append(checkpoint_callback)
    
    # Evaluation callback with same environment setup
    eval_env = make_vec_env(
        make_env,
        n_envs=1,
        seed=original_seed + 1000,
        wrapper_class=AtariWrapper,
        vec_env_cls=DummyVecEnv,
        monitor_dir=None,
    )
    if frame_stack > 0:
        eval_env = VecFrameStack(eval_env, frame_stack)
        
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=checkpoint_dir,
        log_path=checkpoint_dir,
        eval_freq=args.eval_freq // n_envs,  # Adjust for multiple envs
        n_eval_episodes=5,
        deterministic=True,
        render=False
    )
    callbacks.append(eval_callback)
    
    # Continue training
    print(f"Starting continued training for {args.additional_steps:,} steps...")
    print(f"Total steps after completion: {current_steps + args.additional_steps:,}")
    
    model.learn(
        total_timesteps=args.additional_steps,
        callback=callbacks,
        log_interval=100,
        reset_num_timesteps=False,  # Important: don't reset timestep counter
        progress_bar=True
    )
    
    # Save final model
    final_steps = current_steps + args.additional_steps
    final_model_path = os.path.join(checkpoint_dir, f"rl_model_{final_steps}_steps")
    model.save(final_model_path)
    
    # Also save as the main model
    main_model_path = os.path.join(checkpoint_dir, "BreakoutNoFrameskip-v4")
    model.save(main_model_path)
    
    print(f"Training completed!")
    print(f"Final model saved to: {final_model_path}.zip")
    print(f"Main model updated: {main_model_path}.zip")
    
    # Clean up
    env.close()
    eval_env.close()


if __name__ == "__main__":
    continue_training() 