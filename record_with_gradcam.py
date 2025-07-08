#!/usr/bin/env python3
"""
Record Video with GradCAM Overlay
=================================

This script combines the functionality of record_video.py with GradCAM visualization,
creating videos that show both the agent's gameplay and the attention heatmaps.

Usage:
    python record_with_gradcam.py --env BreakoutNoFrameskip-v4 --algo ppo
"""

import argparse
import os
import sys
import numpy as np
import cv2
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from pathlib import Path

# Add rl_zoo3 to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'rl-baselines3-zoo'))

from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import VecVideoRecorder
from huggingface_sb3 import EnvironmentName

try:
    from rl_zoo3.exp_manager import ExperimentManager
    from rl_zoo3.utils import ALGOS, StoreDict, create_test_env, get_model_path, get_saved_hyperparams
    RL_ZOO3_AVAILABLE = True
except ImportError:
    print("Warning: rl_zoo3 not available, using fallback imports")
    RL_ZOO3_AVAILABLE = False
    from inference_utils import ALGOS, ModelLoader, InferenceRunner

from gradcam_visualizer import AtariGradCAMVisualizer


class GradCAMVideoRecorder:
    """
    Video recorder that combines gameplay with GradCAM visualization.
    """
    
    def __init__(self, model_path, env_id, algo, output_folder, video_length=1000):
        """
        Initialize the GradCAM video recorder.
        
        Args:
            model_path: Path to the trained model
            env_id: Environment ID
            algo: Algorithm name
            output_folder: Output folder for videos
            video_length: Length of video in timesteps
        """
        self.model_path = model_path
        self.env_id = env_id
        self.algo = algo
        self.output_folder = output_folder
        self.video_length = video_length
        
        # Create output directory
        os.makedirs(output_folder, exist_ok=True)
        
        # Initialize GradCAM visualizer
        self.gradcam_visualizer = AtariGradCAMVisualizer(
            model_path=model_path,
            env_id=env_id,
            algo=algo,
            device="auto"
        )
        
        # Setup video writer
        self.video_writer = None
        self.frame_count = 0
        
    def setup_video_writer(self, fps=10):
        """Setup the video writer."""
        video_path = os.path.join(self.output_folder, f"gradcam_{self.env_id}_{self.algo}.mp4")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.video_writer = cv2.VideoWriter(video_path, fourcc, fps, (1280, 480))  # Wide format for side-by-side
        print(f"Recording video to: {video_path}")
        
    def create_combined_frame(self, obs, action, action_probs):
        """
        Create a combined frame with original gameplay and GradCAM overlay.
        
        Args:
            obs: Current observation
            action: Predicted action
            action_probs: Action probabilities
            
        Returns:
            Combined frame as numpy array
        """
        # Generate GradCAM
        gradcam_img, original_img, predicted_action = self.gradcam_visualizer.generate_gradcam(obs)
        
        # Create matplotlib figure
        fig, axes = plt.subplots(1, 3, figsize=(16, 6))
        
        # Original gameplay
        axes[0].imshow(original_img)
        axes[0].set_title('Original Gameplay', fontsize=14)
        axes[0].axis('off')
        
        # GradCAM overlay
        axes[1].imshow(gradcam_img)
        axes[1].set_title(f'GradCAM Attention\nAction: {self._get_action_name(predicted_action)}', fontsize=14)
        axes[1].axis('off')
        
        # Action probabilities
        action_names = ['NOOP', 'FIRE', 'RIGHT', 'LEFT']
        bars = axes[2].bar(range(len(action_probs)), action_probs, 
                          color=['red' if i == predicted_action else 'blue' for i in range(len(action_probs))])
        axes[2].set_title('Action Probabilities', fontsize=14)
        axes[2].set_xlabel('Actions')
        axes[2].set_ylabel('Probability')
        axes[2].set_xticks(range(len(action_names)))
        axes[2].set_xticklabels(action_names, rotation=45)
        axes[2].set_ylim(0, 1)
        
        # Add value labels on bars
        for i, (bar, prob) in enumerate(zip(bars, action_probs)):
            axes[2].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                        f'{prob:.3f}', ha='center', va='bottom', fontsize=10)
        
        plt.tight_layout()
        
        # Convert matplotlib figure to numpy array
        fig.canvas.draw()
        buf = fig.canvas.buffer_rgba()
        img = np.asarray(buf)
        # Convert RGBA to RGB
        img = img[:, :, :3]
        
        # Convert RGB to BGR for OpenCV
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        
        # Resize to standard video size
        img = cv2.resize(img, (1280, 480))
        
        plt.close(fig)
        
        return img
        
    def _get_action_name(self, action):
        """Get human-readable action name."""
        action_names = {0: 'NOOP', 1: 'FIRE', 2: 'RIGHT', 3: 'LEFT'}
        return action_names.get(action, f'Action_{action}')
        
    def record_episode(self, deterministic=True):
        """
        Record an episode with GradCAM visualization.
        
        Args:
            deterministic: Whether to use deterministic actions
        """
        print(f"Recording {self.video_length} timesteps...")
        
        # Setup video writer
        self.setup_video_writer()
        
        # Reset environment
        obs = self.gradcam_visualizer.env.reset()
        
        for step in range(self.video_length):
            # Get action and probabilities
            action, _ = self.gradcam_visualizer.model.predict(obs, deterministic=deterministic)
            
            # Get action probabilities for visualization
            action_probs = self.gradcam_visualizer.get_action_probabilities(obs[0])
            
            # Create combined frame
            combined_frame = self.create_combined_frame(obs[0], action[0], action_probs)
            
            # Write frame to video
            self.video_writer.write(combined_frame)
            
            # Take step in environment
            obs, reward, done, info = self.gradcam_visualizer.env.step(action)
            
            # Progress update
            if step % 50 == 0:
                print(f"Recorded {step}/{self.video_length} frames...")
            
            # Handle episode end
            if done[0]:
                print(f"Episode ended at step {step}")
                obs = self.gradcam_visualizer.env.reset()
        
        # Clean up
        if self.video_writer:
            self.video_writer.release()
            
        print(f"Video recording complete! Saved to: {self.output_folder}")
        
    def get_action_probabilities(self, obs):
        """Get action probabilities from the model."""
        import torch
        
        # Convert to tensor
        obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.gradcam_visualizer.model.device)
        
        # Get action distribution
        with torch.no_grad():
            if self.gradcam_visualizer.algo.lower() in ['dqn', 'qrdqn']:
                # For DQN, get Q-values and convert to probabilities
                q_values = self.gradcam_visualizer.model.q_net(obs_tensor)
                # Convert Q-values to probabilities using softmax
                action_probs = torch.softmax(q_values, dim=1).cpu().numpy()[0]
            else:
                # For policy-based algorithms
                action_logits = self.gradcam_visualizer.wrapped_model(obs_tensor)
                action_probs = torch.softmax(action_logits, dim=1).cpu().numpy()[0]
            
        return action_probs


def main():
    """Main function for recording video with GradCAM."""
    parser = argparse.ArgumentParser(description='Record video with GradCAM overlay')
    parser.add_argument("--env", help="Environment ID", type=str, default="BreakoutNoFrameskip-v4")
    parser.add_argument("--algo", help="RL Algorithm", default="ppo", type=str)
    parser.add_argument("--model-path", help="Path to model file", type=str)
    parser.add_argument("-f", "--folder", help="Log folder", type=str, default="rl-trained-agents")
    parser.add_argument("-o", "--output-folder", help="Output folder", type=str, default="gradcam_videos")
    parser.add_argument("-n", "--n-timesteps", help="Number of timesteps", default=500, type=int)
    parser.add_argument("--exp-id", help="Experiment ID", default=0, type=int)
    parser.add_argument("--deterministic", action="store_true", default=True, help="Use deterministic actions")
    parser.add_argument("--seed", help="Random generator seed", type=int, default=42)
    parser.add_argument("--load-best", action="store_true", default=False, help="Load best model")
    
    args = parser.parse_args()
    
    # Set random seed
    set_random_seed(args.seed)
    
    # Determine model path
    if args.model_path:
        model_path = args.model_path
    else:
        # Try to find model using rl_zoo3 method
        if RL_ZOO3_AVAILABLE:
            try:
                env_name = EnvironmentName(args.env)
                _, model_path, _ = get_model_path(
                    args.exp_id,
                    args.folder,
                    args.algo,
                    env_name,
                    args.load_best,
                    None,
                    False,
                )
            except Exception as e:
                print(f"Error finding model: {e}")
                # Fallback to default path
                model_path = f"rl-baselines3-zoo/rl-trained-agents/{args.algo}/{args.env}_1/{args.env}.zip"
        else:
            model_path = f"rl-baselines3-zoo/rl-trained-agents/{args.algo}/{args.env}_1/{args.env}.zip"
    
    # Check if model exists
    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}")
        print("Please specify a valid model path with --model-path")
        return 1
    
    print(f"Using model: {model_path}")
    
    try:
        # Create recorder
        recorder = GradCAMVideoRecorder(
            model_path=model_path,
            env_id=args.env,
            algo=args.algo,
            output_folder=args.output_folder,
            video_length=args.n_timesteps
        )
        
        # Record episode
        recorder.record_episode(deterministic=args.deterministic)
        
        print("🎉 Recording complete!")
        
    except Exception as e:
        print(f"Error during recording: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code) 