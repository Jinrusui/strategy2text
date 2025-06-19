"""
Stable-Baselines3 agent wrapper for pre-trained agent demos.

Provides utilities for loading and running pre-trained SB3 agents.
"""

import os
import numpy as np
import gymnasium as gym
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path

try:
    from stable_baselines3 import PPO, A2C, DQN, SAC, TD3
    from stable_baselines3.common.base_class import BaseAlgorithm
    from stable_baselines3.common.atari_wrappers import AtariWrapper
except ImportError as e:
    raise ImportError(f"stable-baselines3 is required. Install with: pip install stable-baselines3 gymnasium ale-py. Error: {e}")

# Try to register ALE environments
try:
    import ale_py
    gym.register_envs(ale_py)
except ImportError:
    print("Warning: ale-py not found. Install with: pip install ale-py")
except Exception as e:
    print(f"Warning: Could not register ALE environments: {e}")

# Try to register shimmy for Atari compatibility
try:
    import shimmy
    # Ensure gym compatibility for older models
    import gym as old_gym
    print("✓ Gym compatibility enabled for older pre-trained models")
except ImportError:
    print("Warning: shimmy or gym not found. Install with: pip install gym 'shimmy[atari]'")

# Gym environment compatibility wrapper
def make_env_compatible(env_id, **kwargs):
    """Create environment with compatibility for both gym and gymnasium."""
    try:
        # Try gymnasium first (preferred)
        return gym.make(env_id, **kwargs)
    except Exception as gym_error:
        try:
            # Fallback to old gym
            import gym as old_gym
            return old_gym.make(env_id, **kwargs)
        except Exception as old_gym_error:
            raise RuntimeError(f"Failed to create environment with both gymnasium and gym: {gym_error}, {old_gym_error}")


class SB3Agent:
    """
    Wrapper for pre-trained Stable-Baselines3 agents.
    
    Supports loading trained agents and running them on environments.
    """
    
    # Mapping of algorithm names to classes
    ALGORITHM_MAP = {
        'ppo': PPO,
        'a2c': A2C,
        'dqn': DQN,
        'sac': SAC,
        'td3': TD3,
        'qr-dqn': DQN,  # QR-DQN uses the same class as DQN
    }
    
    def __init__(
        self,
        agent_path: str,
        algorithm: Optional[str] = None,
        env_id: str = "ALE/Breakout-v4",
    ):
        """
        Initialize SB3 agent wrapper.
        
        Args:
            agent_path: Path to saved agent model
            algorithm: Algorithm type (auto-detected if None)
            env_id: Environment ID
        """
        self.agent_path = Path(agent_path)
        self.algorithm = algorithm
        self.env_id = env_id
        
        self.agent: Optional[BaseAlgorithm] = None
        self.env: Optional[gym.Env] = None
        self._is_vec_env = False
        
        # Load the agent
        self.load_agent()
    
    def load_agent(self) -> None:
        """Load a trained agent from file."""
        if not self.agent_path.exists():
            raise FileNotFoundError(f"Agent file not found: {self.agent_path}")
        
        # Auto-detect algorithm if not specified
        if self.algorithm is None:
            self.algorithm = self._detect_algorithm()
        
        # Load the agent
        algorithm_class = self.ALGORITHM_MAP.get(self.algorithm.lower())
        if algorithm_class is None:
            raise ValueError(f"Unsupported algorithm: {self.algorithm}")
        
        try:
            self.agent = algorithm_class.load(str(self.agent_path))
            print(f"✓ Successfully loaded {self.algorithm.upper()} agent from {self.agent_path.name}")
            
        except Exception as e:
            raise RuntimeError(f"Failed to load agent: {e}")
    
    def _detect_algorithm(self) -> str:
        """Detect algorithm type from filename or path."""
        path_str = str(self.agent_path).lower()
        
        for alg_name in self.ALGORITHM_MAP.keys():
            if alg_name in path_str:
                return alg_name
        
        # Default to PPO if can't detect
        print("Warning: Could not detect algorithm from path, defaulting to PPO")
        return 'ppo'
    
    def create_environment(self, render_mode: Optional[str] = None) -> gym.Env:
        """
        Create the environment for the agent.
        
        Args:
            render_mode: Rendering mode ('human', 'rgb_array', etc.)
            
        Returns:
            Created environment
        """
        try:
            # For Atari environments (including NoFrameskip variants), always use proper wrappers
            if 'ALE/' in self.env_id or 'Atari' in self.env_id or 'Breakout' in self.env_id:
                # Use the standard SB3 Atari wrappers for compatibility
                from stable_baselines3.common.env_util import make_atari_env
                from stable_baselines3.common.vec_env import VecFrameStack
                
                # Create Atari environment with proper wrappers
                # Convert environment ID to the format expected by make_atari_env
                env_id = self.env_id
                if 'BreakoutNoFrameskip-v4' in env_id:
                    env_id = 'BreakoutNoFrameskip-v4'
                elif 'ALE/Breakout' in env_id:
                    env_id = self.env_id
                
                env = make_atari_env(env_id, n_envs=1, seed=0, wrapper_kwargs=dict(episodic_life=False),env_kwargs=dict(lives=3))
                env = VecFrameStack(env, n_stack=4)
                
                # For single environment usage, we need to handle VecEnv differently
                self._is_vec_env = True
            else:
                # Create environment with render mode for non-Atari games
                env_kwargs = {}
                if render_mode:
                    env_kwargs['render_mode'] = render_mode
                
                env = make_env_compatible(self.env_id, **env_kwargs)
                self._is_vec_env = False
            
            self.env = env
            return env
            
        except Exception as e:
            # Try fallback environments
            fallback_envs = ["BreakoutNoFrameskip-v4", "ALE/Breakout-v4", "CartPole-v1"]
            
            for fallback_env in fallback_envs:
                if fallback_env == self.env_id:
                    continue
                    
                try:
                    print(f"Trying fallback environment: {fallback_env}")
                    
                    if 'Breakout' in fallback_env:
                        # Use Atari wrappers for Breakout variants
                        from stable_baselines3.common.env_util import make_atari_env
                        from stable_baselines3.common.vec_env import VecFrameStack
                        
                        env = make_atari_env(fallback_env, n_envs=1, seed=0)
                        env = VecFrameStack(env, n_stack=4)
                        self._is_vec_env = True
                    else:
                        # Regular environment
                        env_kwargs = {}
                        if render_mode:
                            env_kwargs['render_mode'] = render_mode
                        
                        env = make_env_compatible(fallback_env, **env_kwargs)
                        self._is_vec_env = False
                    
                    self.env = env
                    self.env_id = fallback_env
                    print(f"✓ Using fallback environment: {fallback_env}")
                    return env
                    
                except Exception:
                    continue
            
            raise RuntimeError(f"Failed to create environment {self.env_id} or any fallback: {e}")
    
    def run_episode(
        self,
        max_steps: int = 10000,
        render: bool = False,
        record_frames: bool = True,
        seed: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Run a single episode with the agent.
        
        Args:
            max_steps: Maximum steps per episode
            render: Whether to render the environment
            record_frames: Whether to record frames for video
            seed: Random seed for reproducibility
            
        Returns:
            Episode data including frames, actions, rewards, etc.
        """
        if self.agent is None:
            raise RuntimeError("No agent loaded.")
        
        # Create environment
        render_mode = 'rgb_array' if record_frames else ('human' if render else None)
        env = self.create_environment(render_mode=render_mode)
        
        # Reset environment (handle both VecEnv and regular env)
        if hasattr(env, 'reset') and hasattr(env, 'num_envs'):
            # VecEnv
            obs = env.reset()
            if seed is not None:
                env.seed(seed)
        else:
            # Regular env
            if seed is not None:
                obs, info = env.reset(seed=seed)
            else:
                obs, info = env.reset()
        
        # Episode data
        episode_data = {
            'frames': [],
            'actions': [],
            'rewards': [],
            'observations': [],
            'dones': [],
            'total_reward': 0,
            'steps': 0
        }
        
        done = False
        truncated = False
        step = 0
        
        while not (done or truncated) and step < max_steps:
            # Get action from agent
            action, _states = self.agent.predict(obs, deterministic=True)
            
            # Take step in environment (handle both VecEnv and regular env)
            if hasattr(env, 'num_envs'):
                # VecEnv
                obs, reward, done, info = env.step(action)
                # VecEnv returns arrays, so extract first element
                reward = reward[0] if isinstance(reward, (list, np.ndarray)) else reward
                done = done[0] if isinstance(done, (list, np.ndarray)) else done
                truncated = False  # VecEnv doesn't return truncated in older versions
            else:
                # Regular env
                obs, reward, done, truncated, info = env.step(action)
            
            # Record data
            if record_frames and render_mode == 'rgb_array':
                try:
                    if hasattr(env, 'num_envs'):
                        # VecEnv rendering - different methods to try
                        frame = None
                        
                        # Method 1: Try direct render
                        try:
                            frame = env.render(mode='rgb_array')
                        except:
                            pass
                        
                        # Method 2: Try render without mode argument
                        if frame is None:
                            try:
                                frame = env.render()
                            except:
                                pass
                        
                        # Method 3: Try accessing the underlying environment
                        if frame is None:
                            try:
                                if hasattr(env, 'envs') and len(env.envs) > 0:
                                    frame = env.envs[0].render(mode='rgb_array')
                            except:
                                pass
                        
                        # Method 4: Try VecEnv get_images method
                        if frame is None:
                            try:
                                if hasattr(env, 'get_images'):
                                    images = env.get_images()
                                    if images and len(images) > 0:
                                        frame = images[0]
                            except:
                                pass
                        
                    else:
                        # Regular env rendering
                        frame = env.render()
                    
                    if frame is not None:
                        # Handle different frame formats
                        if isinstance(frame, list) and len(frame) > 0:
                            frame = frame[0]  # Take first frame if it's a list
                        
                        # Ensure frame is a numpy array
                        if not isinstance(frame, np.ndarray):
                            continue
                            
                        # Handle different dimensions
                        if len(frame.shape) == 4:
                            frame = frame[0]  # Remove batch dimension
                        
                        episode_data['frames'].append(frame)
                except Exception as e:
                    # Skip frame recording if rendering fails, but don't print every error
                    if step == 1:  # Only print error for first step
                        print(f"  Warning: Frame capture failed: {e}")
                    pass
            
            episode_data['actions'].append(action)
            episode_data['rewards'].append(reward)
            episode_data['observations'].append(obs)
            episode_data['dones'].append(done)
            episode_data['total_reward'] += reward
            
            step += 1
        
        episode_data['steps'] = step
        env.close()
        
        return episode_data
    
    def run_multiple_episodes(
        self,
        num_episodes: int = 5,
        max_steps_per_episode: int = 10000,
        seeds: Optional[List[int]] = None
    ) -> List[Dict[str, Any]]:
        """
        Run multiple episodes with the agent.
        
        Args:
            num_episodes: Number of episodes to run
            max_steps_per_episode: Maximum steps per episode
            seeds: Optional list of seeds for reproducibility
            
        Returns:
            List of episode data
        """
        episodes = []
        
        for i in range(num_episodes):
            seed = seeds[i] if seeds and i < len(seeds) else None
            
            episode_data = self.run_episode(
                max_steps=max_steps_per_episode,
                seed=seed
            )
            episodes.append(episode_data)
        
        return episodes
    
    def evaluate_agent(
        self,
        num_episodes: int = 10,
        max_steps_per_episode: int = 10000
    ) -> Dict[str, float]:
        """
        Evaluate agent performance over multiple episodes.
        
        Args:
            num_episodes: Number of episodes for evaluation
            max_steps_per_episode: Maximum steps per episode
            
        Returns:
            Performance metrics
        """
        episodes = self.run_multiple_episodes(num_episodes, max_steps_per_episode)
        
        rewards = [ep['total_reward'] for ep in episodes]
        lengths = [ep['steps'] for ep in episodes]
        
        return {
            'mean_reward': np.mean(rewards),
            'std_reward': np.std(rewards),
            'min_reward': np.min(rewards),
            'max_reward': np.max(rewards),
            'mean_episode_length': np.mean(lengths),
            'total_episodes': len(episodes)
        }
    
    def get_agent_info(self) -> Dict[str, Any]:
        """Get information about the loaded agent."""
        return {
            'algorithm': self.algorithm,
            'env_id': self.env_id,
            'model_path': str(self.agent_path),
            'model_size_mb': self.agent_path.stat().st_size / (1024 * 1024) if self.agent_path.exists() else 0
        } 