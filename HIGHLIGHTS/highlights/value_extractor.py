"""
Utility functions to extract Q-values from different value-based RL algorithms
supported by stable-baselines3
"""

import numpy as np
import torch
from stable_baselines3 import DQN
from stable_baselines3.common.preprocessing import preprocess_obs


def extract_q_values(model, observation, deterministic=True):
    """
    Extract Q-values from a value-based RL model for a given observation.
    
    Args:
        model: Trained stable-baselines3 model
        observation: Environment observation
        deterministic: Whether to use deterministic policy (not used for Q-values)
    
    Returns:
        np.ndarray: Q-values for all actions
    """
    # Check if model is DQN-based
    if hasattr(model, 'q_net') or hasattr(model, 'q_net_target'):
        return _extract_dqn_q_values(model, observation)
    else:
        raise ValueError(f"Q-value extraction not supported for model type: {type(model)}")


def _extract_dqn_q_values(model, observation):
    """
    Extract Q-values from DQN-based models (DQN, DDQN, etc.)
    
    Args:
        model: DQN model
        observation: Environment observation
    
    Returns:
        np.ndarray: Q-values for all actions
    """
    # Preprocess observation using the model's preprocessing
    try:
        # Use the model's preprocessing function if available
        if hasattr(model.policy, 'obs_to_tensor'):
            obs_tensor = model.policy.obs_to_tensor(observation)[0]
        else:
            # Manual preprocessing
            if isinstance(observation, dict):
                # Handle dict observations
                obs_tensor = {}
                for key, value in observation.items():
                    if isinstance(value, np.ndarray):
                        obs_tensor[key] = torch.as_tensor(value, dtype=torch.float32)
                    else:
                        obs_tensor[key] = torch.as_tensor([value], dtype=torch.float32)
            else:
                # Handle array observations
                obs_tensor = torch.as_tensor(observation, dtype=torch.float32)
                
                # Handle different observation shapes
                if obs_tensor.dim() == len(model.observation_space.shape):
                    obs_tensor = obs_tensor.unsqueeze(0)
                
                # For Atari environments, ensure correct channel order
                if len(obs_tensor.shape) == 4 and obs_tensor.shape[-1] in [1, 3, 4]:
                    # Convert from (batch, height, width, channels) to (batch, channels, height, width)
                    obs_tensor = obs_tensor.permute(0, 3, 1, 2)
                elif len(obs_tensor.shape) == 3 and obs_tensor.shape[-1] in [1, 3, 4]:
                    # Convert from (height, width, channels) to (batch, channels, height, width)
                    obs_tensor = obs_tensor.permute(2, 0, 1).unsqueeze(0)
                
                # Normalize to [0, 1] if values are in [0, 255] range
                if obs_tensor.max() > 1.0:
                    obs_tensor = obs_tensor / 255.0
        
        # Set model to evaluation mode
        model.q_net.eval()
        
        # Get Q-values
        with torch.no_grad():
            q_values = model.q_net(obs_tensor)
            
        # Convert to numpy and squeeze batch dimension
        if isinstance(q_values, torch.Tensor):
            q_values_np = q_values.cpu().numpy().squeeze(0)
        else:
            # Handle multiple outputs (e.g., from dueling networks)
            q_values_np = q_values[0].cpu().numpy().squeeze(0)
        
        return q_values_np
        
    except Exception as e:
        # If preprocessing fails, return zeros
        print(f"Warning: Q-value extraction failed: {e}")
        n_actions = model.action_space.n if hasattr(model.action_space, 'n') else 1
        return np.zeros(n_actions)


def get_best_action(q_values):
    """
    Get the best action from Q-values
    
    Args:
        q_values: Q-values array
    
    Returns:
        int: Best action index
    """
    return np.argmax(q_values)


def get_action_values_stats(q_values):
    """
    Get statistics about Q-values
    
    Args:
        q_values: Q-values array
    
    Returns:
        dict: Statistics including max, min, mean, std, etc.
    """
    return {
        'max': np.max(q_values),
        'min': np.min(q_values),
        'mean': np.mean(q_values),
        'std': np.std(q_values),
        'range': np.max(q_values) - np.min(q_values),
        'best_action': get_best_action(q_values),
        'q_values': q_values.copy()
    }


def compute_state_importance(q_values, method='second_best'):
    """
    Compute state importance based on Q-values
    
    Args:
        q_values: Q-values array
        method: Method to compute importance ('worst', 'second_best', 'variance')
    
    Returns:
        float: State importance score
    """
    if method == 'worst':
        # Difference between best and worst action
        return np.max(q_values) - np.min(q_values)
    elif method == 'second_best' or method == 'second':
        # Difference between best and second-best action
        if len(q_values) < 2:
            return 0.0
        sorted_q = np.sort(q_values.flatten())
        return sorted_q[-1] - sorted_q[-2]
    elif method == 'variance':
        # Variance of Q-values
        return np.var(q_values)
    else:
        raise ValueError(f"Unknown importance method: {method}")


def is_value_based_model(model):
    """
    Check if a model is value-based (supports Q-value extraction)
    
    Args:
        model: Stable-baselines3 model
    
    Returns:
        bool: True if model is value-based
    """
    return hasattr(model, 'q_net') or hasattr(model, 'q_net_target') 