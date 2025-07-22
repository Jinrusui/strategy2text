#!/usr/bin/env python3
"""
Record video with Grad-CAM visualization for RL agents playing Breakout.

This script extends the record_video.py functionality to include Grad-CAM visualizations
while maintaining exact reproducibility of agent behavior and environment state.
"""

import argparse
import os
import sys
import numpy as np
import yaml
import torch
import torch.nn.functional as F
import cv2
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import matplotlib.pyplot as plt
from matplotlib import cm

# Add rl-baselines3-zoo to path
sys.path.append(str(Path(__file__).parent / "rl-baselines3-zoo"))

from huggingface_sb3 import EnvironmentName
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import VecVideoRecorder
from stable_baselines3 import DQN
from sb3_contrib import QRDQN

# Import from rl-zoo3
from rl_zoo3.exp_manager import ExperimentManager
from rl_zoo3.utils import ALGOS, StoreDict, create_test_env, get_model_path, get_saved_hyperparams

# Grad-CAM imports
try:
    from pytorch_grad_cam import GradCAM
    from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
    from pytorch_grad_cam.utils.image import show_cam_on_image
    print("✅ pytorch-grad-cam imported successfully")
except ImportError as e:
    print(f"❌ Failed to import pytorch-grad-cam: {e}")
    print("Please install with: pip install grad-cam")
    sys.exit(1)


class DQNGradCAMVisualizer:
    """
    Grad-CAM visualizer for DQN agents playing Atari games.
    
    This class provides Grad-CAM visualization while ensuring the exact same
    agent behavior and environment state as the original record_video.py script.
    """
    
    def __init__(self, model, target_layers=None, device="auto"):
        """
        Initialize the Grad-CAM visualizer.
        
        Args:
            model: The trained DQN model
            target_layers: List of target layers for Grad-CAM (auto-detected if None)
            device: Device to run on ('auto', 'cpu', 'cuda')
        """
        self.model = model
        self.device = self._setup_device(device)
        
        # Get the PyTorch model from SB3 wrapper
        if hasattr(model, 'q_net'):
            self.pytorch_model = model.q_net
        elif hasattr(model, 'policy') and hasattr(model.policy, 'q_net'):
            self.pytorch_model = model.policy.q_net
        else:
            raise ValueError("Could not find q_net in the model")
            
        # Auto-detect target layers if not provided
        if target_layers is None:
            self.target_layers = self._auto_detect_target_layers()
        else:
            self.target_layers = target_layers
            
        print(f"Using target layers: {[str(layer) for layer in self.target_layers]}")
        
        # Initialize Grad-CAM
        self.cam_extractor = GradCAM(
            model=self.pytorch_model,
            target_layers=self.target_layers
        )
        
        # Breakout action names
        self.action_names = {
            0: "NOOP",
            1: "FIRE", 
            2: "RIGHT",
            3: "LEFT"
        }
        
    def _setup_device(self, device):
        """Setup computation device."""
        if device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            return torch.device(device)
            
    def _auto_detect_target_layers(self):
        """Auto-detect suitable target layers for Grad-CAM."""
        target_layers = []
        
        # Look for the last convolutional layers
        for name, module in self.pytorch_model.named_modules():
            if isinstance(module, (torch.nn.Conv2d, torch.nn.ConvTranspose2d)):
                target_layers.append(module)
                
        # Use the last few conv layers
        if len(target_layers) >= 2:
            return target_layers[-2:]  # Use last 2 conv layers
        elif len(target_layers) >= 1:
            return target_layers[-1:]  # Use last conv layer
        else:
            raise ValueError("No convolutional layers found in the model")
    
    def get_gradcam_for_action(self, obs, action):
        """
        Generate Grad-CAM visualization for a specific action.
        
        Args:
            obs: Observation from environment (numpy array)
            action: The action taken by the agent
            
        Returns:
            tuple: (gradcam_heatmap, action_name)
        """
        # Convert observation to tensor with correct shape
        if isinstance(obs, np.ndarray):
            # Handle different observation formats
            if obs.ndim == 4:  # (batch, channels, height, width)
                obs_tensor = torch.FloatTensor(obs).to(self.device)
            elif obs.ndim == 3:  # (channels, height, width) - add batch dimension
                obs_tensor = torch.FloatTensor(obs[np.newaxis, ...]).to(self.device)
            else:
                raise ValueError(f"Unexpected observation shape: {obs.shape}")
        else:
            obs_tensor = obs.to(self.device)
        
        # Ensure correct tensor shape: (batch, channels, height, width)
        if obs_tensor.dim() == 4:
            # Check if channels and spatial dims are swapped
            batch, c, h, w = obs_tensor.shape
            if c > 16:  # Likely spatial dimension, not channels
                # Reshape from (batch, height, width, channels) to (batch, channels, height, width)
                obs_tensor = obs_tensor.permute(0, 3, 1, 2)
        
        # Debug: obs_tensor shape for Grad-CAM: {obs_tensor.shape}
        
        # Create target for the selected action
        targets = [ClassifierOutputTarget(int(action))]
        
        try:
            # Generate Grad-CAM
            grayscale_cam = self.cam_extractor(input_tensor=obs_tensor, targets=targets)
            
            # Process the heatmap
            if grayscale_cam.ndim > 2:
                grayscale_cam = grayscale_cam[0]  # Remove batch dimension
                
            # Resize to match original frame size (84x84 for Atari)
            target_size = (84, 84)
            if grayscale_cam.shape != target_size:
                grayscale_cam = cv2.resize(grayscale_cam, target_size)
                
        except Exception as e:
            print(f"Grad-CAM generation failed: {e}")
            # Return a blank heatmap as fallback
            grayscale_cam = np.zeros((84, 84), dtype=np.float32)
            
        action_name = self.action_names.get(int(action), f"ACTION_{action}")
        
        return grayscale_cam, action_name
    
    def create_visualization_frame(self, obs, action, grayscale_cam, action_name):
        """
        Create a visualization frame combining the original observation with Grad-CAM overlay.
        
        Args:
            obs: Original observation
            action: Action taken
            grayscale_cam: Grad-CAM heatmap
            action_name: Name of the action
            
        Returns:
            Visualization frame as numpy array
        """
        # Handle different observation formats
        if obs.ndim == 4:  # Batch dimension
            frame = obs[0]
        else:
            frame = obs
            
        # Convert to proper format for visualization
        if frame.shape[0] == 4:  # Stacked frames (C, H, W) - use most recent frame
            # Take the last frame and convert to grayscale
            frame = frame[-1]  # Shape: (H, W)
            if frame.ndim == 2:
                frame = np.stack([frame, frame, frame], axis=-1)  # Convert to RGB
        elif frame.shape[0] == 1:  # Single grayscale frame
            frame = frame[0]  # Remove channel dimension
            frame = np.stack([frame, frame, frame], axis=-1)  # Convert to RGB
        elif frame.shape[0] == 3:  # RGB frame
            frame = np.transpose(frame, (1, 2, 0))  # CHW to HWC
        else:
            # Handle other cases - convert from CHW to HWC if needed
            if frame.shape[0] <= 4:  # Channel first
                frame = np.transpose(frame, (1, 2, 0))
                if frame.shape[-1] == 1:
                    frame = np.repeat(frame, 3, axis=-1)
                elif frame.shape[-1] > 3:
                    frame = frame[:, :, :3]
            
        # Normalize to [0, 1]
        if frame.max() > 1.0:
            frame = frame.astype(np.float32) / 255.0
        else:
            frame = frame.astype(np.float32)
            
        # Ensure frame is the right size (84, 84, 3)
        if frame.shape[:2] != (84, 84):
            frame = cv2.resize(frame, (84, 84))
        if frame.shape[-1] != 3:
            if frame.ndim == 2:
                frame = np.stack([frame, frame, frame], axis=-1)
            else:
                frame = frame[:, :, :3]
        
        # Normalize Grad-CAM heatmap to [0, 1]
        if grayscale_cam.max() > 0:
            grayscale_cam = (grayscale_cam - grayscale_cam.min()) / (grayscale_cam.max() - grayscale_cam.min())
        
        # Create vibrant Grad-CAM overlay using jet colormap
        heatmap = cm.jet(grayscale_cam)[:, :, :3]  # Remove alpha channel
        
        # Apply stronger blending for more visible heatmap
        alpha = 0.6  # Increased transparency of heatmap
        blended = (1 - alpha) * frame + alpha * heatmap
        blended = np.clip(blended, 0, 1)
        
        # Enhance contrast
        blended = np.power(blended, 0.8)  # Gamma correction for better visibility
        
        # Convert back to uint8
        blended = (blended * 255).astype(np.uint8)
        
        # Resize to higher resolution for better visibility
        blended = cv2.resize(blended, (640, 640), interpolation=cv2.INTER_CUBIC)
        
        # Add text overlay with action information
        blended = self._add_text_overlay(blended, action_name, int(action))
        
        return blended
    
    def _add_text_overlay(self, frame, action_name, action_value):
        """Add text overlay showing the current action."""
        # Scale text size based on frame size - smaller legend
        height, width = frame.shape[:2]
        font_scale = width / 640.0  # Base scale on 640px width
        thickness = max(1, int(2 * font_scale))
        
        # Add smaller text background
        bg_width = int(120 * font_scale)
        bg_height = int(35 * font_scale)
        margin = int(8 * font_scale)
        
        cv2.rectangle(frame, (margin, margin), (margin + bg_width, margin + bg_height), (0, 0, 0, 180), -1)
        cv2.rectangle(frame, (margin, margin), (margin + bg_width, margin + bg_height), (255, 255, 255), 1)
        
        # Add smaller text
        text_y1 = int(18 * font_scale)
        text_y2 = int(30 * font_scale)
        font_size = 0.4 * font_scale
        
        cv2.putText(frame, f"Action: {action_name}", (margin + 4, text_y1), 
                    cv2.FONT_HERSHEY_SIMPLEX, font_size, (255, 255, 255), thickness)
        cv2.putText(frame, f"Value: {action_value}", (margin + 4, text_y2), 
                    cv2.FONT_HERSHEY_SIMPLEX, font_size, (255, 255, 255), thickness)
        
        return frame


def main():
    """Main function that replicates record_video.py with Grad-CAM visualization."""
    parser = argparse.ArgumentParser(description="Record video with Grad-CAM visualization")
    parser.add_argument("--env", help="Environment ID", type=EnvironmentName, default="BreakoutNoFrameskip-v4")
    parser.add_argument("-f", "--folder", help="Log folder", type=str, default="rl-trained-agents")
    parser.add_argument("-o", "--output-folder", help="Output folder", type=str)
    parser.add_argument("--algo", help="RL Algorithm", default="dqn", type=str, required=False, 
                        choices=["dqn", "qrdqn"])
    parser.add_argument("-n", "--n-timesteps", help="Number of timesteps", default=1000, type=int)
    parser.add_argument("--n-envs", help="Number of environments", default=1, type=int)
    parser.add_argument("--deterministic", action="store_true", default=False, help="Use deterministic actions")
    parser.add_argument("--stochastic", action="store_true", default=False, help="Use stochastic actions")
    parser.add_argument("--seed", help="Random generator seed", type=int, default=0)
    parser.add_argument("--no-render", action="store_true", default=False, 
                        help="Do not render the environment (useful for tests)")
    parser.add_argument("--exp-id", help="Experiment ID (default: 0: latest, -1: no exp folder)", 
                        default=0, type=int)
    parser.add_argument("--load-best", action="store_true", default=False, 
                        help="Load best model instead of last model if available")
    parser.add_argument("--load-checkpoint", type=int, 
                        help="Load checkpoint instead of last model if available")
    parser.add_argument("--load-last-checkpoint", action="store_true", default=False, 
                        help="Load last checkpoint instead of last model if available")
    parser.add_argument("--env-kwargs", type=str, nargs="+", action=StoreDict, 
                        help="Optional keyword argument to pass to the env constructor")
    parser.add_argument("--custom-objects", action="store_true", default=False, 
                        help="Use custom objects to solve loading issues")
    parser.add_argument("--save-frames", action="store_true", default=False,
                        help="Save individual Grad-CAM frames as images")
    parser.add_argument("--gradcam-method", type=str, default="gradcam", 
                        choices=["gradcam"], help="Grad-CAM method to use")

    args = parser.parse_args()

    # Extract arguments (same as record_video.py)
    env_name: EnvironmentName = args.env
    algo = args.algo
    folder = args.folder
    video_folder = args.output_folder
    seed = args.seed
    video_length = args.n_timesteps
    n_envs = args.n_envs

    # Fix folder path to use the correct rl-baselines3-zoo location
    if folder == "rl-trained-agents":
        folder = os.path.join("rl-baselines3-zoo", "rl-trained-agents")

    # Get model path (same as record_video.py)
    name_prefix, model_path, log_path = get_model_path(
        args.exp_id,
        folder,
        algo,
        env_name,
        args.load_best,
        args.load_checkpoint,
        args.load_last_checkpoint,
    )

    print(f"Loading {model_path}")
    print(f"Using seed: {seed}")
    
    # Set random seed (same as record_video.py)
    set_random_seed(args.seed)

    # Check algorithm compatibility
    if algo not in ["dqn", "qrdqn"]:
        raise ValueError(f"Grad-CAM visualization is currently only supported for DQN and QR-DQN, got {algo}")

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
    
    if args.env_kwargs is not None:
        env_kwargs.update(args.env_kwargs)

    # Force rgb_array rendering (same as record_video.py)
    env_kwargs.update(render_mode="rgb_array")

    # Create environment (same as record_video.py)
    env = create_test_env(
        env_name.gym_id,
        n_envs=n_envs,
        stats_path=maybe_stats_path,
        seed=seed,
        log_dir=None,
        should_render=not args.no_render,
        hyperparams=hyperparams,
        env_kwargs=env_kwargs,
    )

    # Model loading kwargs (same as record_video.py)
    kwargs = dict(seed=args.seed)
    if algo in off_policy_algos:
        kwargs.update(dict(buffer_size=1))
        if "optimize_memory_usage" in hyperparams:
            kwargs.update(optimize_memory_usage=False)

    # Custom objects (same as record_video.py)
    newer_python_version = sys.version_info.major == 3 and sys.version_info.minor >= 8
    custom_objects = {}
    if newer_python_version or args.custom_objects:
        custom_objects = {
            "learning_rate": 0.0,
            "lr_schedule": lambda _: 0.0,
            "clip_range": lambda _: 0.0,
        }

    print(f"Loading {model_path}")

    # Load the model (same as record_video.py)
    model = ALGOS[algo].load(model_path, env=env, custom_objects=custom_objects, **kwargs)

    # Deterministic behavior (same as record_video.py)
    stochastic = args.stochastic or ((is_atari or is_minigrid) and not args.deterministic)
    deterministic = not stochastic

    print(f"Using deterministic={deterministic}, stochastic={stochastic}")

    # Set output folder - save under main project directory
    if video_folder is None:
        video_folder = os.path.join(".", "gradcam_videos", f"{env_name}_{algo}_seed{seed}")
    os.makedirs(video_folder, exist_ok=True)

    # Initialize Grad-CAM visualizer
    print("Initializing Grad-CAM visualizer...")
    gradcam_visualizer = DQNGradCAMVisualizer(model)

    # Setup video recording for both original and Grad-CAM versions
    original_video_folder = os.path.join(video_folder, "original")
    gradcam_video_folder = os.path.join(video_folder, "gradcam")
    os.makedirs(original_video_folder, exist_ok=True)
    os.makedirs(gradcam_video_folder, exist_ok=True)

    # Record original video (same as record_video.py)
    print("Recording original video...")
    env_original = VecVideoRecorder(
        env,
        original_video_folder,
        record_video_trigger=lambda x: x == 0,
        video_length=video_length,
        name_prefix=f"{name_prefix}_original",
    )

    # Create a separate environment for Grad-CAM recording
    print("Setting up Grad-CAM recording...")
    
    # Create frames storage for Grad-CAM video
    gradcam_frames = []
    frame_save_dir = None
    if args.save_frames:
        frame_save_dir = os.path.join(gradcam_video_folder, "frames")
        os.makedirs(frame_save_dir, exist_ok=True)

    # Reset environment and start recording
    obs = env_original.reset()
    lstm_states = None
    episode_starts = np.ones((env_original.num_envs,), dtype=bool)
    
    print(f"Starting recording for {video_length} timesteps...")
    print(f"Environment shape: {obs.shape}")
    
    try:
        total_reward = 0
        for step in range(video_length):
            # Get action from model (exactly same as record_video.py)
            action, lstm_states = model.predict(
                obs,
                state=lstm_states,
                episode_start=episode_starts,
                deterministic=deterministic,
            )
            
            # Generate Grad-CAM for this step
            try:
                grayscale_cam, action_name = gradcam_visualizer.get_gradcam_for_action(obs, action[0])
                
                # Create visualization frame
                viz_frame = gradcam_visualizer.create_visualization_frame(
                    obs, action[0], grayscale_cam, action_name
                )
                gradcam_frames.append(viz_frame)
                
                # Save individual frame if requested
                if frame_save_dir:
                    frame_path = os.path.join(frame_save_dir, f"frame_{step:06d}.png")
                    cv2.imwrite(frame_path, cv2.cvtColor(viz_frame, cv2.COLOR_RGB2BGR))
                    
            except Exception as e:
                print(f"Warning: Grad-CAM failed for step {step}: {e}")
                # Create a fallback frame
                if obs.ndim == 4:
                    fallback_frame = obs[0]
                else:
                    fallback_frame = obs
                if fallback_frame.shape[0] <= 4:  # Channel first
                    fallback_frame = np.transpose(fallback_frame, (1, 2, 0))
                if fallback_frame.shape[-1] == 1:
                    fallback_frame = np.repeat(fallback_frame, 3, axis=-1)
                if fallback_frame.max() > 1.0:
                    fallback_frame = (fallback_frame / 255.0 * 255).astype(np.uint8)
                gradcam_frames.append(fallback_frame)

            # Render original environment
            if not args.no_render:
                env_original.render()

            # Step environment (exactly same as record_video.py)
            obs, rewards, dones, infos = env_original.step(action)
            episode_starts = dones
            total_reward += rewards[0]

            if step % 100 == 0:
                print(f"Step {step}/{video_length}, Reward: {rewards[0]:.2f}, Total: {total_reward:.2f}")

    except KeyboardInterrupt:
        print("Recording interrupted by user")

    # Close original environment
    env_original.close()

    # Create Grad-CAM video
    if gradcam_frames:
        print(f"Creating Grad-CAM video from {len(gradcam_frames)} frames...")
        
        # Setup video writer with better encoding
        height, width = gradcam_frames[0].shape[:2]
        gradcam_video_path = os.path.join(gradcam_video_folder, f"{name_prefix}_gradcam.mp4")
        
        # Try different codecs for better compatibility
        codecs = [
            cv2.VideoWriter_fourcc(*'H264'),  # H.264 codec
            cv2.VideoWriter_fourcc(*'XVID'),  # XVID codec
            cv2.VideoWriter_fourcc(*'MJPG'),  # Motion JPEG
            cv2.VideoWriter_fourcc(*'mp4v'),  # MP4 fallback
        ]
        
        out = None
        for fourcc in codecs:
            out = cv2.VideoWriter(gradcam_video_path, fourcc, 30.0, (width, height))
            if out.isOpened():
                print(f"Using codec: {fourcc}")
                break
            out.release()
        
        if out is None or not out.isOpened():
            print("Warning: Could not initialize video writer, trying fallback...")
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(gradcam_video_path, fourcc, 30.0, (width, height))
        
        # Write frames
        for i, frame in enumerate(gradcam_frames):
            # Convert RGB to BGR for OpenCV
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            out.write(frame_bgr)
            
            if i % 50 == 0:
                print(f"  Written {i}/{len(gradcam_frames)} frames...")
        
        out.release()
        print(f"Grad-CAM video saved to: {gradcam_video_path}")
        
        # Also save a high-quality version using ffmpeg if available
        try:
            import subprocess
            hq_video_path = os.path.join(gradcam_video_folder, f"{name_prefix}_gradcam_hq.mp4")
            cmd = [
                'ffmpeg', '-y', '-r', '30',
                '-i', gradcam_video_path,
                '-c:v', 'libx264', '-preset', 'medium', '-crf', '18',
                '-pix_fmt', 'yuv420p',
                hq_video_path
            ]
            subprocess.run(cmd, check=True, capture_output=True)
            print(f"High-quality video saved to: {hq_video_path}")
        except (ImportError, subprocess.CalledProcessError, FileNotFoundError):
            print("ffmpeg not available, using OpenCV encoding only")
    
    print(f"Recording completed!")
    print(f"Total reward: {total_reward}")
    print(f"Original video folder: {original_video_folder}")
    print(f"Grad-CAM video folder: {gradcam_video_folder}")
    
    if frame_save_dir:
        print(f"Individual frames saved to: {frame_save_dir}")


if __name__ == "__main__":
    main() 