#!/usr/bin/env python3
"""
GradCAM Visualization for RL Agents
====================================

This script provides GradCAM visualization for reinforcement learning agents,
specifically designed for Atari environments like Breakout using PPO agents.

Requirements:
- pytorch-grad-cam
- stable-baselines3
- gymnasium[atari]
- opencv-python
- matplotlib
- numpy

Usage:
    python gradcam_visualizer.py --env BreakoutNoFrameskip-v4 --algo ppo --model-path path/to/model.zip

"""

import argparse
import os
import sys
import warnings
from typing import Optional, Tuple, List, Dict, Any

import cv2
import gymnasium as gym
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from pytorch_grad_cam import GradCAM, HiResCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from stable_baselines3 import PPO
from stable_baselines3.common.atari_wrappers import AtariWrapper
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack, VecTransposeImage
from stable_baselines3.common.utils import set_random_seed

# Add rl_zoo3 to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'rl-baselines3-zoo'))

try:
    from rl_zoo3 import ALGOS, create_test_env, get_saved_hyperparams
    from rl_zoo3.exp_manager import ExperimentManager
    from rl_zoo3.utils import get_model_path
except ImportError as e:
    print(f"Warning: Could not import rl_zoo3 modules: {e}")
    print("Some functionality may be limited.")
    ALGOS = {'ppo': PPO}


class PPOGradCAMWrapper(nn.Module):
    """
    Wrapper for PPO policy network to enable GradCAM visualization.
    """
    
    def __init__(self, policy_net):
        super().__init__()
        self.policy_net = policy_net
        self.features_extractor = policy_net.features_extractor
        self.mlp_extractor = policy_net.mlp_extractor
        self.action_net = policy_net.action_net
        self.value_net = policy_net.value_net
        
    def forward(self, x):
        """
        Forward pass through the policy network.
        Returns the action logits for GradCAM visualization.
        """
        # Extract features using CNN
        features = self.features_extractor(x)
        
        # Pass through MLP extractor
        latent_pi, latent_vf = self.mlp_extractor(features)
        
        # Get action logits
        action_logits = self.action_net(latent_pi)
        
        return action_logits


class DQNGradCAMWrapper(nn.Module):
    """
    Wrapper for DQN Q-network to enable GradCAM visualization.
    """
    
    def __init__(self, q_net):
        super().__init__()
        self.q_net = q_net
        
    def forward(self, x):
        """
        Forward pass through the Q-network.
        Returns the Q-values for GradCAM visualization.
        """
        return self.q_net(x)


class AtariGradCAMVisualizer:
    """
    GradCAM visualizer for Atari RL agents.
    """
    
    def __init__(self, 
                 model_path: str,
                 env_id: str = "BreakoutNoFrameskip-v4",
                 algo: str = "ppo",
                 device: str = "auto"):
        """
        Initialize the GradCAM visualizer.
        
        Args:
            model_path: Path to the trained model (.zip file)
            env_id: Environment ID (default: BreakoutNoFrameskip-v4)
            algo: Algorithm name (default: ppo)
            device: Device to use (auto, cpu, cuda)
        """
        self.model_path = model_path
        self.env_id = env_id
        self.algo = algo
        self.device = device
        
        # Load model and environment
        self.model = None
        self.env = None
        self.wrapped_model = None
        self.gradcam_methods = {
            'gradcam': GradCAM,
            'hirescam': HiResCAM,
            'scorecam': ScoreCAM,
            'gradcam++': GradCAMPlusPlus,
            'ablationcam': AblationCAM,
            'xgradcam': XGradCAM,
            'eigencam': EigenCAM,
        }
        
        self._load_model_and_env()
        
    def _load_model_and_env(self):
        """Load the trained model and create environment."""
        print(f"Loading model from: {self.model_path}")
        
        # Load the model with proper settings for different algorithms
        if self.algo.lower() == 'ppo':
            self.model = PPO.load(self.model_path, device=self.device)
        elif self.algo.lower() in ['dqn', 'ddpg', 'sac', 'td3', 'qrdqn', 'tqc']:
            # Off-policy algorithms need special handling
            kwargs = {
                'device': self.device,
                'buffer_size': 1,  # Dummy buffer size for inference
            }
            # Handle DQN specific issues
            if self.algo.lower() in ['dqn', 'qrdqn']:
                kwargs['optimize_memory_usage'] = False
            
            self.model = ALGOS[self.algo].load(self.model_path, **kwargs)
        else:
            if self.algo in ALGOS:
                self.model = ALGOS[self.algo].load(self.model_path, device=self.device)
            else:
                raise ValueError(f"Unsupported algorithm: {self.algo}")
        
        # Create environment
        self.env = self._create_env()
        
        # Wrap the policy/Q-network for GradCAM based on algorithm
        if self.algo.lower() in ['dqn', 'qrdqn']:
            self.wrapped_model = DQNGradCAMWrapper(self.model.q_net)
        else:
            # PPO and other policy-based algorithms
            self.wrapped_model = PPOGradCAMWrapper(self.model.policy)
        self.wrapped_model.eval()
        
        print(f"Model loaded successfully. Policy type: {type(self.model.policy)}")
        print(f"Observation space: {self.env.observation_space}")
        print(f"Action space: {self.env.action_space}")
        
    def _create_env(self):
        """Create the environment with proper wrappers."""
        # Create base environment
        env = gym.make(self.env_id)
        
        # Apply Atari wrapper
        env = AtariWrapper(env)
        
        # Vectorize environment
        env = DummyVecEnv([lambda: env])
        
        # Add frame stacking
        env = VecFrameStack(env, n_stack=4)
        
        # Transpose for PyTorch (channel first)
        env = VecTransposeImage(env)
        
        return env
        
    def _get_target_layers(self) -> List[nn.Module]:
        """
        Get the target layers for GradCAM visualization.
        For CNN policies, we typically want the last convolutional layer.
        """
        target_layers = []
        
        # Get target layers based on algorithm type
        if self.algo.lower() in ['dqn', 'qrdqn']:
            # For DQN, look in the Q-network
            if hasattr(self.wrapped_model.q_net, 'features_extractor'):
                features_extractor = self.wrapped_model.q_net.features_extractor
                if hasattr(features_extractor, 'cnn'):
                    for name, module in features_extractor.cnn.named_modules():
                        if isinstance(module, nn.Conv2d):
                            target_layers = [module]  # Keep only the last one
            else:
                # Direct search in q_net
                for name, module in self.wrapped_model.q_net.named_modules():
                    if isinstance(module, nn.Conv2d):
                        target_layers = [module]
        else:
            # For PPO with CNN policy, get the last conv layer from features extractor
            if hasattr(self.wrapped_model, 'features_extractor') and hasattr(self.wrapped_model.features_extractor, 'cnn'):
                # Find the last convolutional layer
                for name, module in self.wrapped_model.features_extractor.cnn.named_modules():
                    if isinstance(module, nn.Conv2d):
                        target_layers = [module]  # Keep only the last one
                        
        if not target_layers:
            # Fallback: try to find any conv layer
            for name, module in self.wrapped_model.named_modules():
                if isinstance(module, nn.Conv2d):
                    target_layers = [module]
                    
        if not target_layers:
            raise ValueError("No convolutional layers found in the model for GradCAM visualization")
            
        print(f"Using target layer: {target_layers[-1]}")
        return target_layers
        
    def generate_gradcam(self, 
                        obs: np.ndarray, 
                        action_idx: Optional[int] = None,
                        method: str = 'gradcam') -> Tuple[np.ndarray, np.ndarray, int]:
        """
        Generate GradCAM visualization for a given observation.
        
        Args:
            obs: Observation from the environment
            action_idx: Target action index (if None, use the predicted action)
            method: GradCAM method to use
            
        Returns:
            tuple: (gradcam_image, original_image, predicted_action)
        """
        if method not in self.gradcam_methods:
            raise ValueError(f"Unsupported GradCAM method: {method}")
            
        # Convert observation to tensor and move to model device
        if isinstance(obs, np.ndarray):
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.model.device)
        else:
            obs_tensor = obs.to(self.model.device)
            
        # Get prediction
        with torch.no_grad():
            action_logits = self.wrapped_model(obs_tensor)
            predicted_action = torch.argmax(action_logits, dim=1).item()
            
        # Use predicted action if no target action specified
        if action_idx is None:
            action_idx = predicted_action
            
        # Get target layers
        target_layers = self._get_target_layers()
        
        # Create GradCAM object
        gradcam_class = self.gradcam_methods[method]
        cam = gradcam_class(model=self.wrapped_model, target_layers=target_layers)
        
        # Generate targets
        targets = [ClassifierOutputTarget(action_idx)]
        
        # Generate GradCAM
        grayscale_cam = cam(input_tensor=obs_tensor, targets=targets)
        
        # Convert observation for visualization (take last frame from stack)
        if obs.shape[0] == 4:  # Frame stack
            vis_obs = obs[-1]  # Last frame
        else:
            vis_obs = obs[0] if len(obs.shape) == 4 else obs
            
        # Normalize observation for visualization
        vis_obs = (vis_obs - vis_obs.min()) / (vis_obs.max() - vis_obs.min())
        
        # Convert to RGB if grayscale
        if len(vis_obs.shape) == 2:
            vis_obs = np.stack([vis_obs] * 3, axis=-1)
        elif vis_obs.shape[0] == 1:  # Channel first grayscale
            vis_obs = np.stack([vis_obs[0]] * 3, axis=-1)
        elif vis_obs.shape[0] == 3:  # Channel first RGB
            vis_obs = np.transpose(vis_obs, (1, 2, 0))
            
        # Create visualization
        gradcam_image = show_cam_on_image(vis_obs, grayscale_cam[0], use_rgb=True)
        
        return gradcam_image, vis_obs, predicted_action
        
    def get_action_probabilities(self, obs: np.ndarray) -> np.ndarray:
        """
        Get action probabilities for a given observation.
        
        Args:
            obs: Observation
            
        Returns:
            Action probabilities
        """
        # Convert observation to tensor and move to model device
        if isinstance(obs, np.ndarray):
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.model.device)
        else:
            obs_tensor = obs.to(self.model.device)
            
        # Get action distribution
        with torch.no_grad():
            if self.algo.lower() in ['dqn', 'qrdqn']:
                # For DQN, get Q-values and convert to probabilities
                q_values = self.model.q_net(obs_tensor)
                # Convert Q-values to probabilities using softmax
                action_probs = torch.softmax(q_values, dim=1).cpu().numpy()[0]
            else:
                # For policy-based algorithms
                action_logits = self.wrapped_model(obs_tensor)
                action_probs = torch.softmax(action_logits, dim=1).cpu().numpy()[0]
                
        return action_probs
        
    def visualize_episode(self, 
                         n_steps: int = 100,
                         save_dir: str = "gradcam_output",
                         method: str = 'gradcam',
                         save_video: bool = True) -> List[Dict[str, Any]]:
        """
        Visualize an entire episode with GradCAM.
        
        Args:
            n_steps: Number of steps to visualize
            save_dir: Directory to save visualizations
            method: GradCAM method to use
            save_video: Whether to save as video
            
        Returns:
            List of visualization data for each step
        """
        os.makedirs(save_dir, exist_ok=True)
        
        # Reset environment
        obs = self.env.reset()
        episode_data = []
        
        # For video creation
        if save_video:
            fourcc = cv2.VideoWriter_fourcc(*'h264')
            video_writer = cv2.VideoWriter(
                os.path.join(save_dir, f'gradcam_episode_{method}.mp4'),
                fourcc, 10.0, (640, 480)
            )
        
        for step in range(n_steps):
            # Generate GradCAM
            gradcam_img, original_img, predicted_action = self.generate_gradcam(
                obs[0], method=method
            )
            
            # Get action probabilities
            action_probs = self.get_action_probabilities(obs[0])
            
            # Store step data
            step_data = {
                'step': step,
                'gradcam_image': gradcam_img,
                'original_image': original_img,
                'predicted_action': predicted_action,
                'action_probs': action_probs,
                'observation': obs[0].copy()
            }
            episode_data.append(step_data)
            
            # Create visualization
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            
            # Original image
            axes[0].imshow(original_img)
            axes[0].set_title('Original Frame')
            axes[0].axis('off')
            
            # GradCAM visualization
            axes[1].imshow(gradcam_img)
            axes[1].set_title(f'GradCAM ({method})\nAction: {predicted_action}')
            axes[1].axis('off')
            
            # Action probabilities
            axes[2].bar(range(len(action_probs)), action_probs)
            axes[2].set_title('Action Probabilities')
            axes[2].set_xlabel('Action')
            axes[2].set_ylabel('Probability')
            
            plt.tight_layout()
            
            # Save frame
            frame_path = os.path.join(save_dir, f'step_{step:04d}.png')
            plt.savefig(frame_path, dpi=100, bbox_inches='tight')
            
            # Add to video
            if save_video:
                # Convert matplotlib figure to image
                fig.canvas.draw()
                buf = fig.canvas.buffer_rgba()
                img = np.asarray(buf)
                # Convert RGBA to RGB
                img = img[:, :, :3]
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                img = cv2.resize(img, (640, 480))
                video_writer.write(img)
            
            plt.close(fig)
            
            # Take action in environment
            action, _ = self.model.predict(obs, deterministic=True)
            obs, reward, done, info = self.env.step(action)
            
            if done:
                obs = self.env.reset()
                print(f"Episode finished at step {step}")
                
            if step % 10 == 0:
                print(f"Processed step {step}/{n_steps}")
                
        if save_video:
            video_writer.release()
            print(f"Video saved to: {os.path.join(save_dir, f'gradcam_episode_{method}.mp4')}")
            
        return episode_data
        
    def compare_methods(self, 
                       obs: np.ndarray, 
                       save_path: str = "gradcam_comparison.png") -> None:
        """
        Compare different GradCAM methods on the same observation.
        
        Args:
            obs: Observation to analyze
            save_path: Path to save the comparison image
        """
        methods = ['gradcam', 'gradcam++', 'scorecam', 'eigencam']
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()
        
        # Original image
        _, original_img, predicted_action = self.generate_gradcam(obs, method='gradcam')
        axes[0].imshow(original_img)
        axes[0].set_title('Original Frame')
        axes[0].axis('off')
        
        # Generate GradCAM with different methods
        for i, method in enumerate(methods):
            try:
                gradcam_img, _, _ = self.generate_gradcam(obs, method=method)
                axes[i+1].imshow(gradcam_img)
                axes[i+1].set_title(f'{method.upper()}\nPredicted Action: {predicted_action}')
                axes[i+1].axis('off')
            except Exception as e:
                print(f"Error with method {method}: {e}")
                axes[i+1].text(0.5, 0.5, f'Error: {method}', 
                              ha='center', va='center', transform=axes[i+1].transAxes)
                axes[i+1].axis('off')
        
        # Hide last subplot if not needed
        if len(methods) < 5:
            axes[-1].axis('off')
            
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Comparison saved to: {save_path}")


def main():
    """Main function to run GradCAM visualization."""
    parser = argparse.ArgumentParser(description='GradCAM Visualization for RL Agents')
    parser.add_argument('--model-path', type=str, required=True,
                       help='Path to the trained model (.zip file)')
    parser.add_argument('--env', type=str, default='BreakoutNoFrameskip-v4',
                       help='Environment ID')
    parser.add_argument('--algo', type=str, default='ppo',
                       help='Algorithm name')
    parser.add_argument('--device', type=str, default='auto',
                       help='Device to use (auto, cpu, cuda)')
    parser.add_argument('--n-steps', type=int, default=4500,
                       help='Number of steps to visualize')
    parser.add_argument('--output-dir', type=str, default='gradcam_output',
                       help='Output directory for visualizations')
    parser.add_argument('--method', type=str, default='gradcam',
                       choices=['gradcam', 'hirescam', 'scorecam', 'gradcam++', 
                               'ablationcam', 'xgradcam', 'eigencam'],
                       help='GradCAM method to use')
    parser.add_argument('--compare-methods', action='store_true',
                       help='Compare different GradCAM methods')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    parser.add_argument('--no-video', action='store_true',
                       help='Disable video generation')
    
    args = parser.parse_args()
    
    # Set random seed
    set_random_seed(args.seed)
    
    # Check if model file exists
    if not os.path.exists(args.model_path):
        print(f"Error: Model file not found: {args.model_path}")
        sys.exit(1)
    
    try:
        # Initialize visualizer
        print("Initializing GradCAM visualizer...")
        visualizer = AtariGradCAMVisualizer(
            model_path=args.model_path,
            env_id=args.env,
            algo=args.algo,
            device=args.device
        )
        
        # Create output directory
        os.makedirs(args.output_dir, exist_ok=True)
        
        if args.compare_methods:
            # Compare different methods on a single observation
            print("Comparing different GradCAM methods...")
            obs = visualizer.env.reset()
            comparison_path = os.path.join(args.output_dir, 'gradcam_comparison.png')
            visualizer.compare_methods(obs[0], comparison_path)
            
        else:
            # Visualize episode
            print(f"Visualizing episode with {args.method} method...")
            episode_data = visualizer.visualize_episode(
                n_steps=args.n_steps,
                save_dir=args.output_dir,
                method=args.method,
                save_video=not args.no_video
            )
            
            print(f"Visualization complete! {len(episode_data)} steps processed.")
            print(f"Results saved to: {args.output_dir}")
            
    except Exception as e:
        print(f"Error during visualization: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main() 