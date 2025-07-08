"""
Grad-CAM visualization for Stable Baselines3 RL agents playing Breakout.

This module provides comprehensive Grad-CAM visualization capabilities for understanding
how SB3 RL agents make decisions in the Breakout environment.
"""

import os
import sys
import numpy as np
import torch
import torch.nn as nn
import cv2
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import argparse
from datetime import datetime
import json

# Add the parent directory to sys.path to import from rl-baselines3-zoo
sys.path.append(str(Path(__file__).parent.parent.parent / "rl-baselines3-zoo"))

import gymnasium as gym
from stable_baselines3 import PPO, A2C, DQN
from stable_baselines3.common.vec_env import VecFrameStack, DummyVecEnv
from stable_baselines3.common.env_util import make_vec_env
from sb3_contrib import QRDQN

# Grad-CAM imports
from pytorch_grad_cam import GradCAM, HiResCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM, LayerCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image, preprocess_image
from pytorch_grad_cam.utils.reshape_transforms import vit_reshape_transform


class SB3GradCAMAnalyzer:
    """
    Grad-CAM analyzer for Stable Baselines3 RL agents.
    
    Supports multiple algorithms (PPO, A2C, DQN, QR-DQN) and various CAM methods
    for visualizing decision-making in Breakout.
    """
    
    def __init__(
        self,
        model_path: str,
        algorithm: str = "ppo",
        device: str = "auto",
        cam_method: str = "gradcam"
    ):
        """
        Initialize the Grad-CAM analyzer.
        
        Args:
            model_path: Path to the trained SB3 model
            algorithm: Algorithm used (ppo, a2c, dqn, qrdqn)
            device: Device to run on ('auto', 'cpu', 'cuda')
            cam_method: CAM method to use ('gradcam', 'hirescam', 'scorecam', etc.)
        """
        self.model_path = model_path
        self.algorithm = algorithm.lower()
        self.device = self._setup_device(device)
        self.cam_method = cam_method.lower()
        
        # Load the model
        self.model = self._load_model()
        
        # Create environment
        self.env = self._create_env()
        
        # Setup Grad-CAM
        self.cam_extractor = None
        self.target_layers = None
        self._setup_grad_cam()
        
        # Action names for Breakout
        self.action_names = {
            0: "NOOP",
            1: "FIRE", 
            2: "RIGHT",
            3: "LEFT"
        }
    
    def _setup_device(self, device: str) -> torch.device:
        """Setup computation device."""
        if device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            return torch.device(device)
    
    def _load_model(self):
        """Load the SB3 model."""
        model_classes = {
            "ppo": PPO,
            "a2c": A2C,
            "dqn": DQN,
            "qrdqn": QRDQN
        }
        
        if self.algorithm not in model_classes:
            raise ValueError(f"Unsupported algorithm: {self.algorithm}")
        
        model_class = model_classes[self.algorithm]
        model = model_class.load(self.model_path, device=self.device)
        model.policy.eval()
        
        return model
    
    def _create_env(self):
        """Create the Breakout environment using rl_zoo3 utilities."""
        try:
            # Import rl_zoo3 modules (this also registers environments)
            import sys
            sys.path.append(str(Path(__file__).parent.parent.parent / "rl-baselines3-zoo"))
            
            # Import to register Atari environments
            import rl_zoo3.import_envs
            
            # Register ALE environments
            try:
                import ale_py
                gym.register_envs(ale_py)
            except ImportError:
                pass
            
            # Create environment with proper frame stacking
            def make_env():
                env = gym.make("BreakoutNoFrameskip-v4")
                # Use gymnasium's built-in Atari preprocessing
                env = gym.wrappers.AtariPreprocessing(
                    env, 
                    noop_max=30,
                    frame_skip=4,
                    screen_size=84,
                    terminal_on_life_loss=True,
                    grayscale_obs=True,
                    grayscale_newaxis=False,
                    scale_obs=False
                )
                return env
            
            env = make_vec_env(make_env, n_envs=1)
            env = VecFrameStack(env, n_stack=4)
            
            return env
            
        except Exception as e:
            print(f"Error creating environment with rl_zoo3: {e}")
            print("Falling back to basic environment creation...")
            
            # Fallback: try to register Atari environments manually
            try:
                import ale_py
                gym.register_envs(ale_py)
            except ImportError:
                print("ale_py not available. Please install it with: pip install ale-py")
                raise
            
            # Create environment with basic setup
            def make_env():
                env = gym.make("BreakoutNoFrameskip-v4")
                # Apply basic Atari preprocessing
                env = gym.wrappers.AtariPreprocessing(
                    env, 
                    noop_max=30,
                    frame_skip=4,
                    screen_size=84,
                    terminal_on_life_loss=True,
                    grayscale_obs=True,
                    grayscale_newaxis=False,
                    scale_obs=False
                )
                return env
            
            env = make_vec_env(make_env, n_envs=1)
            env = VecFrameStack(env, n_stack=4)
            
            return env
    
    def _setup_grad_cam(self):
        """Setup Grad-CAM with appropriate target layers."""
        # Get the CNN feature extractor
        if hasattr(self.model.policy, 'features_extractor'):
            cnn_model = self.model.policy.features_extractor.cnn
        elif hasattr(self.model.policy, 'q_net'):
            # For DQN-based algorithms
            cnn_model = self.model.policy.q_net.features_extractor.cnn
        else:
            raise ValueError("Cannot find CNN feature extractor in the model")
        
        # Find target layers (typically the last convolutional layer)
        self.target_layers = []
        for name, module in cnn_model.named_modules():
            if isinstance(module, nn.Conv2d):
                self.target_layers.append(module)
        
        # Use the last convolutional layer
        if self.target_layers:
            self.target_layers = [self.target_layers[-1]]
        else:
            raise ValueError("No convolutional layers found in the model")
        
        # Setup CAM method
        cam_methods = {
            "gradcam": GradCAM,
            "hirescam": HiResCAM,
            "scorecam": ScoreCAM,
            "gradcam++": GradCAMPlusPlus,
            "ablationcam": AblationCAM,
            "xgradcam": XGradCAM,
            "eigencam": EigenCAM,
            "layercam": LayerCAM
        }
        
        if self.cam_method not in cam_methods:
            raise ValueError(f"Unsupported CAM method: {self.cam_method}")
        
        cam_class = cam_methods[self.cam_method]
        
        # Create CAM extractor
        if hasattr(self.model.policy, 'features_extractor'):
            target_model = self.model.policy.features_extractor.cnn
        else:
            target_model = self.model.policy.q_net.features_extractor.cnn
            
        self.cam_extractor = cam_class(
            model=target_model,
            target_layers=self.target_layers
        )
    
    def _preprocess_observation(self, obs: np.ndarray) -> torch.Tensor:
        """Preprocess observation for the model."""
        # obs shape: (1, 4, 84, 84) for Atari
        if isinstance(obs, np.ndarray):
            obs = torch.FloatTensor(obs).to(self.device)
        
        # Ensure correct shape and normalization
        if obs.max() > 1.0:
            obs = obs / 255.0
            
        return obs
    
    def _get_model_prediction(self, obs: torch.Tensor) -> Tuple[int, torch.Tensor]:
        """Get model prediction and logits."""
        with torch.no_grad():
            if hasattr(self.model.policy, 'features_extractor'):
                # For PPO/A2C
                features = self.model.policy.features_extractor(obs)
                if hasattr(self.model.policy, 'action_net'):
                    logits = self.model.policy.action_net(features)
                else:
                    logits = self.model.policy.mlp_extractor.policy_net(features)
            else:
                # For DQN
                logits = self.model.policy.q_net(obs)
        
        predicted_action = torch.argmax(logits, dim=1).item()
        return predicted_action, logits
    
    def generate_cam(
        self,
        obs: np.ndarray,
        target_action: Optional[int] = None,
        use_predicted_action: bool = True
    ) -> Tuple[np.ndarray, int, Dict]:
        """
        Generate Grad-CAM visualization for a given observation.
        
        Args:
            obs: Input observation (4, 84, 84)
            target_action: Specific action to analyze (if None, uses predicted action)
            use_predicted_action: Whether to use the model's predicted action
            
        Returns:
            Tuple of (cam_visualization, predicted_action, metadata)
        """
        # Preprocess observation
        obs_tensor = self._preprocess_observation(obs)
        
        # Get model prediction
        predicted_action, logits = self._get_model_prediction(obs_tensor)
        
        # Determine target action
        if target_action is None and use_predicted_action:
            target_action = predicted_action
        elif target_action is None:
            target_action = predicted_action
        
        # Create target for CAM
        targets = [ClassifierOutputTarget(target_action)]
        
        # Generate CAM
        with self.cam_extractor:
            cam = self.cam_extractor(
                input_tensor=obs_tensor,
                targets=targets,
                aug_smooth=True,
                eigen_smooth=True
            )
        
        # Process CAM output
        cam_image = cam[0, :]  # Remove batch dimension
        
        # Create visualization
        # Use the last frame for visualization (most recent frame)
        last_frame = obs[0, -1, :, :] if obs.ndim == 4 else obs[-1, :, :]
        
        # Normalize frame for visualization
        if last_frame.max() > 1.0:
            last_frame = last_frame / 255.0
        
        # Convert to RGB if needed
        if last_frame.ndim == 2:
            rgb_img = np.stack([last_frame] * 3, axis=-1)
        else:
            rgb_img = last_frame
        
        # Overlay CAM on image
        cam_visualization = show_cam_on_image(
            rgb_img,
            cam_image,
            use_rgb=True,
            colormap=cv2.COLORMAP_JET
        )
        
        # Prepare metadata
        metadata = {
            "predicted_action": predicted_action,
            "predicted_action_name": self.action_names.get(predicted_action, "UNKNOWN"),
            "target_action": target_action,
            "target_action_name": self.action_names.get(target_action, "UNKNOWN"),
            "action_probabilities": torch.softmax(logits, dim=1).cpu().numpy().tolist(),
            "cam_method": self.cam_method,
            "algorithm": self.algorithm
        }
        
        return cam_visualization, predicted_action, metadata
    
    def analyze_episode(
        self,
        n_steps: int = 100,
        save_frames: bool = True,
        output_dir: str = "grad_cam_analysis"
    ) -> List[Dict]:
        """
        Analyze an entire episode with Grad-CAM visualizations.
        
        Args:
            n_steps: Number of steps to analyze
            save_frames: Whether to save individual frames
            output_dir: Directory to save results
            
        Returns:
            List of analysis results for each step
        """
        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize environment
        obs = self.env.reset()
        
        results = []
        total_reward = 0
        
        for step in range(n_steps):
            # Generate CAM visualization
            cam_viz, predicted_action, metadata = self.generate_cam(obs)
            
            # Take action in environment
            action, _ = self.model.predict(obs, deterministic=True)
            new_obs, reward, done, info = self.env.step(action)
            
            total_reward += reward[0]
            
            # Store results
            step_result = {
                "step": step,
                "action": action[0],
                "reward": reward[0],
                "total_reward": total_reward,
                "done": done[0],
                **metadata
            }
            results.append(step_result)
            
            # Save visualization if requested
            if save_frames:
                frame_path = output_path / f"step_{step:04d}_action_{action[0]}.png"
                
                # Create combined visualization
                fig, axes = plt.subplots(1, 3, figsize=(15, 5))
                
                # Original frame
                last_frame = obs[0, -1, :, :] if obs.ndim == 4 else obs[-1, :, :]
                axes[0].imshow(last_frame, cmap='gray')
                axes[0].set_title(f"Original Frame (Step {step})")
                axes[0].axis('off')
                
                # CAM visualization
                axes[1].imshow(cam_viz)
                axes[1].set_title(f"Grad-CAM ({self.cam_method})")
                axes[1].axis('off')
                
                # Action probabilities
                probs = metadata["action_probabilities"][0]
                action_names = [self.action_names.get(i, f"Action {i}") for i in range(len(probs))]
                bars = axes[2].bar(action_names, probs)
                bars[predicted_action].set_color('red')
                axes[2].set_title(f"Action Probabilities\nPredicted: {metadata['predicted_action_name']}")
                axes[2].set_ylabel("Probability")
                plt.setp(axes[2].get_xticklabels(), rotation=45, ha='right')
                
                plt.tight_layout()
                plt.savefig(frame_path, dpi=150, bbox_inches='tight')
                plt.close()
            
            # Update observation
            obs = new_obs
            
            # Break if episode is done
            if done[0]:
                break
        
        # Save summary
        summary_path = output_path / "episode_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        return results
    
    def compare_actions(
        self,
        obs: np.ndarray,
        actions_to_compare: List[int] = None,
        save_path: str = None
    ) -> Dict:
        """
        Compare Grad-CAM visualizations for different actions.
        
        Args:
            obs: Input observation
            actions_to_compare: List of actions to compare (if None, uses all actions)
            save_path: Path to save comparison visualization
            
        Returns:
            Dictionary with CAM visualizations for each action
        """
        if actions_to_compare is None:
            actions_to_compare = list(range(4))  # Breakout has 4 actions
        
        results = {}
        
        # Generate CAM for each action
        for action in actions_to_compare:
            cam_viz, _, metadata = self.generate_cam(obs, target_action=action)
            results[action] = {
                "visualization": cam_viz,
                "metadata": metadata
            }
        
        # Create comparison visualization
        if save_path:
            fig, axes = plt.subplots(2, len(actions_to_compare), figsize=(4*len(actions_to_compare), 8))
            
            if len(actions_to_compare) == 1:
                axes = axes.reshape(2, 1)
            
            # Original frame
            last_frame = obs[0, -1, :, :] if obs.ndim == 4 else obs[-1, :, :]
            
            for i, action in enumerate(actions_to_compare):
                # Show original frame
                axes[0, i].imshow(last_frame, cmap='gray')
                axes[0, i].set_title(f"Original Frame")
                axes[0, i].axis('off')
                
                # Show CAM visualization
                axes[1, i].imshow(results[action]["visualization"])
                action_name = self.action_names.get(action, f"Action {action}")
                axes[1, i].set_title(f"CAM for {action_name}")
                axes[1, i].axis('off')
            
            plt.tight_layout()
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()
        
        return results
    
    def cleanup(self):
        """Clean up resources."""
        if self.cam_extractor:
            self.cam_extractor.__exit__(None, None, None)
        if self.env:
            self.env.close()


def main():
    """Main function for command-line usage."""
    parser = argparse.ArgumentParser(description="Grad-CAM analysis for SB3 RL agents")
    parser.add_argument("--model-path", type=str, required=True,
                        help="Path to the trained SB3 model")
    parser.add_argument("--algorithm", type=str, default="ppo",
                        choices=["ppo", "a2c", "dqn", "qrdqn"],
                        help="Algorithm used for training")
    parser.add_argument("--cam-method", type=str, default="gradcam",
                        choices=["gradcam", "hirescam", "scorecam", "gradcam++", 
                                "ablationcam", "xgradcam", "eigencam", "layercam"],
                        help="Grad-CAM method to use")
    parser.add_argument("--n-steps", type=int, default=100,
                        help="Number of steps to analyze")
    parser.add_argument("--output-dir", type=str, default="grad_cam_analysis",
                        help="Output directory for results")
    parser.add_argument("--device", type=str, default="auto",
                        choices=["auto", "cpu", "cuda"],
                        help="Device to use for computation")
    parser.add_argument("--compare-actions", action="store_true",
                        help="Generate action comparison visualization")
    
    args = parser.parse_args()
    
    # Create analyzer
    analyzer = SB3GradCAMAnalyzer(
        model_path=args.model_path,
        algorithm=args.algorithm,
        device=args.device,
        cam_method=args.cam_method
    )
    
    # Create output directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"{args.output_dir}_{args.algorithm}_{args.cam_method}_{timestamp}"
    
    try:
        # Analyze episode
        print(f"Starting Grad-CAM analysis for {args.n_steps} steps...")
        results = analyzer.analyze_episode(
            n_steps=args.n_steps,
            save_frames=True,
            output_dir=output_dir
        )
        
        # Generate action comparison if requested
        if args.compare_actions:
            print("Generating action comparison...")
            obs = analyzer.env.reset()
            comparison_path = Path(output_dir) / "action_comparison.png"
            analyzer.compare_actions(obs, save_path=str(comparison_path))
        
        print(f"Analysis complete! Results saved to: {output_dir}")
        print(f"Total steps analyzed: {len(results)}")
        print(f"Total reward: {results[-1]['total_reward'] if results else 0}")
        
    finally:
        analyzer.cleanup()


if __name__ == "__main__":
    main() 