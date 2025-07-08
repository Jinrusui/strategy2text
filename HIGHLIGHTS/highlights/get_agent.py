"""
Agent and environment loading for rl-baselines3-zoo integration
"""

import os
import sys
import yaml
import numpy as np
from huggingface_sb3 import EnvironmentName
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import VecVideoRecorder, VecEnv

from rl_zoo3.exp_manager import ExperimentManager
from rl_zoo3.utils import ALGOS, create_test_env, get_model_path, get_saved_hyperparams
from highlights.value_extractor import is_value_based_model


def get_agent(args):
    """
    Load agent and environment from rl-baselines3-zoo format
    
    Args:
        args: Arguments containing env, algo, folder, etc.
    
    Returns:
        tuple: (environment, model)
    """
    
    # Handle environment name
    if hasattr(args, 'env') and isinstance(args.env, str):
        env_name = EnvironmentName(args.env)
    elif hasattr(args, 'env'):
        env_name = args.env
    else:
        raise ValueError("Environment name not specified in args.env")
    
    # Get algorithm
    algo = getattr(args, 'algo', 'dqn')
    
    # Get folder path
    folder = getattr(args, 'folder', 'rl-trained-agents')
    
    # Get model path components
    exp_id = getattr(args, 'exp_id', 0)
    load_best = getattr(args, 'load_best', False)
    load_checkpoint = getattr(args, 'load_checkpoint', None)
    load_last_checkpoint = getattr(args, 'load_last_checkpoint', False)
    
    # Get model path
    name_prefix, model_path, log_path = get_model_path(
        exp_id,
        folder,
        algo,
        env_name,
        load_best,
        load_checkpoint,
        load_last_checkpoint,
    )
    
    print(f"Loading {model_path}")
    
    # Check if algorithm is value-based
    if algo.lower() not in ['dqn', 'qrdqn', 'ddqn']:
        print(f"Warning: Algorithm {algo} may not be value-based. Q-value extraction might not work.")
    
    # Set random seed
    seed = getattr(args, 'seed', 0)
    set_random_seed(seed)
    
    # Check environment type
    is_atari = ExperimentManager.is_atari(env_name.gym_id)
    is_minigrid = ExperimentManager.is_minigrid(env_name.gym_id)
    
    # Get hyperparameters and stats
    stats_path = os.path.join(log_path, env_name)
    hyperparams, maybe_stats_path = get_saved_hyperparams(stats_path)
    
    # Load environment kwargs
    env_kwargs = {}
    args_path = os.path.join(log_path, env_name, "args.yml")
    if os.path.isfile(args_path):
        with open(args_path) as f:
            loaded_args = yaml.load(f, Loader=yaml.UnsafeLoader)
            if loaded_args.get("env_kwargs") is not None:
                env_kwargs = loaded_args["env_kwargs"]
    
    # Override with command line arguments
    if hasattr(args, 'env_kwargs') and args.env_kwargs is not None:
        env_kwargs.update(args.env_kwargs)
    
    # Force rgb_array rendering for video generation
    env_kwargs.update(render_mode="rgb_array")
    
    # Create environment
    n_envs = getattr(args, 'n_envs', 1)
    env = create_test_env(
        env_name.gym_id,
        n_envs=n_envs,
        stats_path=maybe_stats_path,
        seed=seed,
        log_dir=None,
        should_render=True,  # Enable rendering for state images
        hyperparams=hyperparams,
        env_kwargs=env_kwargs,
    )
    
    # Prepare model loading kwargs
    kwargs = dict(seed=seed)
    
    # Special handling for off-policy algorithms
    off_policy_algos = ["qrdqn", "dqn", "ddqn", "sac", "her", "td3", "tqc"]
    if algo in off_policy_algos:
        # Dummy buffer size as we don't need memory for inference
        kwargs.update(dict(buffer_size=1))
        # Handle timeout termination vs optimize_memory_usage conflict
        if "optimize_memory_usage" in hyperparams:
            kwargs.update(optimize_memory_usage=False)
    
    # Custom objects for compatibility
    newer_python_version = sys.version_info.major == 3 and sys.version_info.minor >= 8
    custom_objects = {}
    if newer_python_version or getattr(args, 'custom_objects', False):
        custom_objects = {
            "learning_rate": 0.0,
            "lr_schedule": lambda _: 0.0,
            "clip_range": lambda _: 0.0,
        }
    
    # Load model
    print(f"Loading model: {model_path}")
    model = ALGOS[algo].load(model_path, env=env, custom_objects=custom_objects, **kwargs)
    
    # Verify model is value-based
    if not is_value_based_model(model):
        raise ValueError(f"Model {algo} is not value-based. HIGHLIGHTS requires Q-value access.")
    
    # Store additional information in args for later use
    args.model_path = model_path
    args.env_name = env_name
    args.log_path = log_path
    args.name_prefix = name_prefix
    args.is_atari = is_atari
    args.is_minigrid = is_minigrid
    
    return env, model
