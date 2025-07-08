#!/usr/bin/env python3
"""
Inference Utilities for RL Models
=================================

This module provides utilities for loading and running inference with trained RL models,
extracted from rl_zoo3 functionality to work standalone.

"""

import os
import sys
import yaml
import pickle
from typing import Dict, Any, Optional, Tuple
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
from stable_baselines3 import PPO, A2C, DQN, SAC, TD3, DDPG
from stable_baselines3.common.atari_wrappers import AtariWrapper
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack, VecTransposeImage, VecNormalize
from stable_baselines3.common.utils import set_random_seed

# Algorithm mapping
ALGOS = {
    'ppo': PPO,
    'a2c': A2C,
    'dqn': DQN,
    'sac': SAC,
    'td3': TD3,
    'ddpg': DDPG,
}

# Try to import additional algorithms from sb3-contrib
try:
    from sb3_contrib import ARS, QRDQN, TQC, TRPO, CrossQ, RecurrentPPO
    ALGOS.update({
        'ars': ARS,
        'qrdqn': QRDQN,
        'tqc': TQC,
        'trpo': TRPO,
        'crossq': CrossQ,
        'ppo_lstm': RecurrentPPO,
    })
except ImportError:
    print("Warning: sb3-contrib not available, some algorithms may not work")


class ModelLoader:
    """
    Utility class for loading trained RL models and setting up environments.
    """
    
    def __init__(self, model_path: str, env_id: str, algo: str = 'ppo'):
        """
        Initialize the model loader.
        
        Args:
            model_path: Path to the model file (.zip) or directory containing model
            env_id: Environment ID
            algo: Algorithm name
        """
        self.model_path = model_path
        self.env_id = env_id
        self.algo = algo.lower()
        
        # Try to determine model path if directory is provided
        if os.path.isdir(model_path):
            self.model_path = self._find_model_file(model_path)
        
        # Load hyperparameters if available
        self.hyperparams = self._load_hyperparams()
        
    def _find_model_file(self, model_dir: str) -> str:
        """Find the model file in the directory."""
        # Look for .zip files
        zip_files = list(Path(model_dir).glob("*.zip"))
        if zip_files:
            return str(zip_files[0])
        
        # Look for best_model.zip
        best_model = Path(model_dir) / "best_model.zip"
        if best_model.exists():
            return str(best_model)
            
        # Look for final_model.zip
        final_model = Path(model_dir) / "final_model.zip"
        if final_model.exists():
            return str(final_model)
            
        raise FileNotFoundError(f"No model file found in {model_dir}")
        
    def _load_hyperparams(self) -> Dict[str, Any]:
        """Load hyperparameters from config files."""
        hyperparams = {}
        
        # Try to load from config.yml in the same directory
        model_dir = os.path.dirname(self.model_path)
        config_path = os.path.join(model_dir, self.env_id, "config.yml")
        
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                hyperparams = yaml.safe_load(f)
                
        # Also try to load vecnormalize stats
        vecnormalize_path = os.path.join(model_dir, self.env_id, "vecnormalize.pkl")
        if os.path.exists(vecnormalize_path):
            hyperparams['vecnormalize_path'] = vecnormalize_path
            
        return hyperparams
        
    def load_model(self, device: str = 'auto') -> Any:
        """
        Load the trained model.
        
        Args:
            device: Device to load the model on
            
        Returns:
            Loaded model
        """
        if self.algo not in ALGOS:
            raise ValueError(f"Unsupported algorithm: {self.algo}")
            
        print(f"Loading {self.algo.upper()} model from: {self.model_path}")
        
        # Load the model
        model_class = ALGOS[self.algo]
        model = model_class.load(self.model_path, device=device)
        
        print(f"Model loaded successfully. Policy: {type(model.policy)}")
        return model
        
    def create_env(self, render_mode: Optional[str] = None) -> gym.Env:
        """
        Create the environment with appropriate wrappers.
        
        Args:
            render_mode: Rendering mode for the environment
            
        Returns:
            Wrapped environment
        """
        # Create base environment
        env = gym.make(self.env_id, render_mode=render_mode)
        
        # Apply Atari wrapper if it's an Atari environment
        if self._is_atari():
            env = AtariWrapper(env)
            
        return env
        
    def create_vec_env(self, n_envs: int = 1, render_mode: Optional[str] = None) -> Any:
        """
        Create vectorized environment with appropriate wrappers.
        
        Args:
            n_envs: Number of environments
            render_mode: Rendering mode
            
        Returns:
            Vectorized environment
        """
        # Create environment factory
        def make_env():
            return self.create_env(render_mode=render_mode)
            
        # Create vectorized environment
        env = DummyVecEnv([make_env for _ in range(n_envs)])
        
        # Apply frame stacking for Atari
        if self._is_atari():
            frame_stack = self.hyperparams.get('frame_stack', 4)
            env = VecFrameStack(env, n_stack=frame_stack)
            
        # Apply image transposition for CNN policies
        if self._uses_cnn_policy():
            env = VecTransposeImage(env)
            
        # Apply normalization if available
        if 'vecnormalize_path' in self.hyperparams:
            env = VecNormalize.load(self.hyperparams['vecnormalize_path'], env)
            env.training = False
            env.norm_reward = False
            
        return env
        
    def _is_atari(self) -> bool:
        """Check if the environment is an Atari game."""
        atari_envs = [
            'Alien', 'Amidar', 'Assault', 'Asterix', 'Asteroids',
            'Atlantis', 'BankHeist', 'BattleZone', 'BeamRider', 'Berzerk',
            'Bowling', 'Boxing', 'Breakout', 'Carnival', 'Centipede',
            'ChopperCommand', 'CrazyClimber', 'Defender', 'DemonAttack',
            'DoubleDunk', 'ElevatorAction', 'Enduro', 'FishingDerby',
            'Freeway', 'Frostbite', 'Gopher', 'Gravitar', 'Hero',
            'IceHockey', 'Jamesbond', 'JourneyEscape', 'Kangaroo',
            'Krull', 'KungFuMaster', 'MontezumaRevenge', 'MsPacman',
            'NameThisGame', 'Phoenix', 'Pitfall', 'Pong', 'Pooyan',
            'PrivateEye', 'Qbert', 'Riverraid', 'RoadRunner', 'Robotank',
            'Seaquest', 'Skiing', 'Solaris', 'SpaceInvaders', 'StarGunner',
            'Tennis', 'TimePilot', 'Tutankham', 'UpNDown', 'Venture',
            'VideOlympics', 'WizardOfWor', 'YarsRevenge', 'Zaxxon'
        ]
        
        return any(atari_env in self.env_id for atari_env in atari_envs)
        
    def _uses_cnn_policy(self) -> bool:
        """Check if the model uses a CNN policy."""
        policy_type = self.hyperparams.get('policy', '')
        return 'Cnn' in policy_type or self._is_atari()


class InferenceRunner:
    """
    Utility class for running inference with trained models.
    """
    
    def __init__(self, model_path: str, env_id: str, algo: str = 'ppo'):
        """
        Initialize the inference runner.
        
        Args:
            model_path: Path to the model file or directory
            env_id: Environment ID
            algo: Algorithm name
        """
        self.loader = ModelLoader(model_path, env_id, algo)
        self.model = None
        self.env = None
        
    def setup(self, device: str = 'auto', render_mode: Optional[str] = None):
        """
        Set up the model and environment.
        
        Args:
            device: Device to use for the model
            render_mode: Rendering mode for the environment
        """
        self.model = self.loader.load_model(device=device)
        self.env = self.loader.create_vec_env(n_envs=1, render_mode=render_mode)
        
    def run_episode(self, max_steps: int = 1000, deterministic: bool = True) -> Dict[str, Any]:
        """
        Run a single episode.
        
        Args:
            max_steps: Maximum number of steps
            deterministic: Whether to use deterministic actions
            
        Returns:
            Episode data including observations, actions, rewards, etc.
        """
        if self.model is None or self.env is None:
            raise RuntimeError("Must call setup() before running episodes")
            
        obs = self.env.reset()
        episode_data = {
            'observations': [],
            'actions': [],
            'rewards': [],
            'dones': [],
            'infos': [],
            'total_reward': 0.0,
            'episode_length': 0
        }
        
        for step in range(max_steps):
            # Store observation
            episode_data['observations'].append(obs[0].copy())
            
            # Predict action
            action, _ = self.model.predict(obs, deterministic=deterministic)
            episode_data['actions'].append(action[0])
            
            # Take step
            obs, reward, done, info = self.env.step(action)
            
            # Store step data
            episode_data['rewards'].append(reward[0])
            episode_data['dones'].append(done[0])
            episode_data['infos'].append(info[0])
            episode_data['total_reward'] += reward[0]
            episode_data['episode_length'] += 1
            
            if done[0]:
                break
                
        return episode_data
        
    def get_action_probabilities(self, obs: np.ndarray) -> np.ndarray:
        """
        Get action probabilities for a given observation.
        
        Args:
            obs: Observation
            
        Returns:
            Action probabilities
        """
        if self.model is None:
            raise RuntimeError("Must call setup() before getting action probabilities")
            
        # Convert to tensor and add batch dimension if needed
        if isinstance(obs, np.ndarray):
            if len(obs.shape) == 3:  # Single observation
                obs = obs[np.newaxis, ...]
            obs_tensor = torch.FloatTensor(obs).to(self.model.device)
        else:
            obs_tensor = obs
            
        # Get action distribution
        with torch.no_grad():
            if hasattr(self.model.policy, 'get_distribution'):
                # For on-policy algorithms
                distribution = self.model.policy.get_distribution(obs_tensor)
                probs = distribution.distribution.probs
            else:
                # For off-policy algorithms, use forward pass
                logits = self.model.policy(obs_tensor)
                probs = torch.softmax(logits, dim=-1)
                
        return probs.cpu().numpy()


def load_breakout_ppo_model(model_path: str = None) -> Tuple[Any, Any]:
    """
    Convenience function to load a Breakout PPO model.
    
    Args:
        model_path: Path to the model. If None, tries to find in rl-trained-agents
        
    Returns:
        Tuple of (model, environment)
    """
    if model_path is None:
        # Try to find the model in rl-trained-agents
        default_path = "rl-baselines3-zoo/rl-trained-agents/ppo/BreakoutNoFrameskip-v4_1/BreakoutNoFrameskip-v4.zip"
        if os.path.exists(default_path):
            model_path = default_path
        else:
            raise FileNotFoundError(
                "No model path provided and default Breakout PPO model not found. "
                "Please provide a model path or ensure the model exists at: " + default_path
            )
    
    runner = InferenceRunner(model_path, "BreakoutNoFrameskip-v4", "ppo")
    runner.setup()
    
    return runner.model, runner.env


def main():
    """Example usage of the inference utilities."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Run inference with trained RL models')
    parser.add_argument('--model-path', type=str, required=True,
                       help='Path to the model file or directory')
    parser.add_argument('--env', type=str, default='BreakoutNoFrameskip-v4',
                       help='Environment ID')
    parser.add_argument('--algo', type=str, default='ppo',
                       help='Algorithm name')
    parser.add_argument('--n-episodes', type=int, default=5,
                       help='Number of episodes to run')
    parser.add_argument('--max-steps', type=int, default=1000,
                       help='Maximum steps per episode')
    parser.add_argument('--render', action='store_true',
                       help='Render the environment')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    
    args = parser.parse_args()
    
    # Set random seed
    set_random_seed(args.seed)
    
    # Create inference runner
    runner = InferenceRunner(args.model_path, args.env, args.algo)
    runner.setup(render_mode='human' if args.render else None)
    
    # Run episodes
    total_rewards = []
    for episode in range(args.n_episodes):
        print(f"Running episode {episode + 1}/{args.n_episodes}")
        
        episode_data = runner.run_episode(max_steps=args.max_steps)
        total_rewards.append(episode_data['total_reward'])
        
        print(f"Episode {episode + 1}: Reward = {episode_data['total_reward']:.2f}, "
              f"Length = {episode_data['episode_length']}")
    
    print(f"\nAverage reward: {np.mean(total_rewards):.2f} ± {np.std(total_rewards):.2f}")
    print(f"Min reward: {np.min(total_rewards):.2f}")
    print(f"Max reward: {np.max(total_rewards):.2f}")


if __name__ == "__main__":
    main() 