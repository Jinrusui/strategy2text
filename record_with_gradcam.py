#!/usr/bin/env python3
"""
Record Video with GradCAM Overlay
=================================

This script combines the functionality of record_video.py with GradCAM visualization,
creating videos that show both the agent's gameplay and the attention heatmaps.

Video Format Alignment:
- FPS: Auto-detects from environment metadata (typically 30 FPS for Atari) or uses --fps argument
- Codec: H.264 by default (same as VecVideoRecorder via ffmpeg) with fallback options
- Format: MP4 container
- Resolution: 1280x480 (wide format for side-by-side GradCAM visualization)

Usage:
    python record_with_gradcam.py --env BreakoutNoFrameskip-v4 --algo ppo
    python record_with_gradcam.py --env BreakoutNoFrameskip-v4 --algo ppo --fps 30 --video-codec H264
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
import subprocess
import tempfile
import gc
import psutil

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


class FFmpegVideoWriter:
    """
    Video writer using ffmpeg directly to ensure H.264 encoding compatibility.
    """
    
    def __init__(self, filename, fps, width, height, codec='libx264'):
        self.filename = filename
        self.fps = fps
        self.width = width
        self.height = height
        self.codec = codec
        self.temp_dir = tempfile.mkdtemp()
        self.frame_count = 0
        self.process = None
        
        # Start ffmpeg process
        self._start_ffmpeg()
    
    def _start_ffmpeg(self):
        """Start the ffmpeg process."""
        cmd = [
            'ffmpeg',
            '-y',  # Overwrite output file
            '-f', 'rawvideo',
            '-vcodec', 'rawvideo',
            '-s', f'{self.width}x{self.height}',
            '-pix_fmt', 'bgr24',
            '-r', str(self.fps),
            '-i', '-',  # Read from stdin
            '-c:v', self.codec,
            '-pix_fmt', 'yuv420p',
            '-preset', 'medium',
            '-crf', '23',
            self.filename
        ]
        
        try:
            self.process = subprocess.Popen(cmd, stdin=subprocess.PIPE, 
                                          stderr=subprocess.PIPE, 
                                          stdout=subprocess.PIPE)
            return True
        except FileNotFoundError:
            print("Warning: ffmpeg not found. Falling back to OpenCV.")
            return False
    
    def write(self, frame):
        """Write a frame to the video."""
        if self.process is None:
            return False
            
        try:
            # Ensure frame is the correct size and format
            if frame.shape[:2] != (self.height, self.width):
                frame = cv2.resize(frame, (self.width, self.height))
            
            # Write frame to ffmpeg stdin with error handling
            frame_bytes = frame.tobytes()
            self.process.stdin.write(frame_bytes)
            self.process.stdin.flush()  # Force flush to prevent buffering issues
            
            self.frame_count += 1
            
            # Check if ffmpeg process is still alive every 100 frames
            if self.frame_count % 100 == 0:
                if self.process.poll() is not None:
                    print(f"Warning: ffmpeg process ended unexpectedly at frame {self.frame_count}")
                    return False
            
            return True
        except BrokenPipeError:
            print(f"Broken pipe error at frame {self.frame_count} - ffmpeg may have crashed")
            return False
        except Exception as e:
            print(f"Error writing frame {self.frame_count}: {e}")
            return False
    
    def release(self):
        """Close the video writer."""
        if self.process is not None:
            self.process.stdin.close()
            self.process.wait()
            self.process = None
    
    def isOpened(self):
        """Check if the video writer is opened."""
        return self.process is not None
    
    def __del__(self):
        self.release()


class GradCAMVideoRecorder:
    """
    Video recorder that combines gameplay with GradCAM visualization.
    """
    
    def __init__(self, model_path, env_id, algo, output_folder, video_length=1000, fps=None, codec="H264", show_matplotlib=False):
        """
        Initialize the GradCAM video recorder.
        
        Args:
            model_path: Path to the trained model
            env_id: Environment ID
            algo: Algorithm name
            output_folder: Output folder for videos
            video_length: Length of video in timesteps
            fps: Frames per second (None for auto-detect)
            codec: Video codec to use
            show_matplotlib: Whether to show matplotlib visualization (default: False to save memory)
        """
        self.model_path = model_path
        self.env_id = env_id
        self.algo = algo
        self.output_folder = output_folder
        self.video_length = video_length
        self.fps = fps
        self.codec = codec
        self.show_matplotlib = show_matplotlib
        
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
        
    def setup_video_writer(self, fps=None):
        """Setup the video writer with settings aligned to record_video.py."""
        # Use instance fps if not provided
        if fps is None:
            fps = self.fps
            
        # Get FPS from environment metadata if still not specified
        if fps is None:
            try:
                # Try to get FPS from environment metadata (same as VecVideoRecorder)
                env_metadata = self.gradcam_visualizer.env.get_attr('metadata')[0]
                fps = env_metadata.get('render_fps', env_metadata.get('video.frames_per_second', 30))
            except (AttributeError, KeyError, IndexError):
                # Default FPS based on environment type
                if 'Atari' in self.env_id or 'NoFrameskip' in self.env_id:
                    fps = 30  # Standard Atari FPS
                else:
                    fps = 30  # General default
        
        video_path = os.path.join(self.output_folder, f"gradcam_{self.env_id}_{self.algo}.mp4")
        
        # Try to use ffmpeg first for proper H.264 encoding
        if self.codec == "H264":
            print("Attempting to use ffmpeg for H.264 encoding...")
            self.video_writer = FFmpegVideoWriter(video_path, fps, 1280, 480, 'libx264')
            
            if self.video_writer.isOpened():
                print(f"✓ Using ffmpeg for H.264 encoding at {fps} FPS")
                return
            else:
                print("✗ ffmpeg failed, falling back to OpenCV...")
                self.video_writer = None
        
        # Fallback to OpenCV with codec fallback chain
        codec_fallbacks = {
            "H264": ["mp4v"],  # Since H264 doesn't work in OpenCV, use mp4v
            "avc1": ["avc1", "mp4v"], 
            "mp4v": ["mp4v"]
        }
        
        fourcc = None
        used_codec = None
        for codec_option in codec_fallbacks.get(self.codec, ["mp4v"]):
            try:
                fourcc = cv2.VideoWriter_fourcc(*codec_option)
                self.video_writer = cv2.VideoWriter(video_path, fourcc, fps, (1280, 480))
                if self.video_writer.isOpened():
                    used_codec = codec_option
                    break
            except Exception as e:
                print(f"Failed to use codec {codec_option}: {e}")
                continue
        
        if fourcc is None or not self.video_writer.isOpened():
            # Final fallback
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            self.video_writer = cv2.VideoWriter(video_path, fourcc, fps, (1280, 480))
            used_codec = "mp4v"
        
        if self.video_writer.isOpened():
            print(f"✓ Using OpenCV with {used_codec} codec at {fps} FPS")
            print(f"⚠️  Note: {used_codec} codec used instead of H.264 due to OpenCV limitations")
        else:
            raise RuntimeError("Failed to initialize video writer with any codec")
        
        print(f"Video settings: {fps} FPS, {used_codec} codec, 1280x480 resolution")
        print(f"Original record_video.py uses: Environment FPS (~30), H.264 codec, Native resolution")
        
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
        
        # Debug: Print image info for first few frames
        if hasattr(self, 'debug_frame_count'):
            self.debug_frame_count += 1
        else:
            self.debug_frame_count = 1
            
        if self.debug_frame_count <= 3:
            print(f"Frame {self.debug_frame_count} - Original img: shape={original_img.shape}, dtype={original_img.dtype}, range=[{original_img.min():.3f}, {original_img.max():.3f}]")
            print(f"Frame {self.debug_frame_count} - GradCAM img: shape={gradcam_img.shape}, dtype={gradcam_img.dtype}, range=[{gradcam_img.min():.3f}, {gradcam_img.max():.3f}]")
        
        # If matplotlib is disabled, create a simple combined frame to save memory
        if not self.show_matplotlib:
            # Simple side-by-side layout without matplotlib
            # Ensure images are in the correct format before processing
            def normalize_image(img):
                """Normalize image to uint8 format for OpenCV processing."""
                if img.dtype == np.float64 or img.dtype == np.float32:
                    # Convert float images to uint8
                    if img.max() <= 1.0:
                        img = (img * 255).astype(np.uint8)
                    else:
                        img = img.astype(np.uint8)
                elif img.dtype != np.uint8:
                    img = img.astype(np.uint8)
                return img
            
            # Normalize and resize images
            original_normalized = normalize_image(original_img)
            gradcam_normalized = normalize_image(gradcam_img)
            
            original_resized = cv2.resize(original_normalized, (640, 480))
            gradcam_resized = cv2.resize(gradcam_normalized, (640, 480))
            
            # Convert to BGR if needed (only for RGB images)
            if len(original_resized.shape) == 3 and original_resized.shape[2] == 3:
                try:
                    original_resized = cv2.cvtColor(original_resized, cv2.COLOR_RGB2BGR)
                except cv2.error as e:
                    print(f"Warning: Could not convert original image to BGR: {e}")
                    # If conversion fails, use image as is
                    pass
            
            if len(gradcam_resized.shape) == 3 and gradcam_resized.shape[2] == 3:
                try:
                    gradcam_resized = cv2.cvtColor(gradcam_resized, cv2.COLOR_RGB2BGR)
                except cv2.error as e:
                    print(f"Warning: Could not convert gradcam image to BGR: {e}")
                    # If conversion fails, use image as is
                    pass
            
            # Ensure both images have the same number of channels
            if len(original_resized.shape) == 2:
                original_resized = cv2.cvtColor(original_resized, cv2.COLOR_GRAY2BGR)
            if len(gradcam_resized.shape) == 2:
                gradcam_resized = cv2.cvtColor(gradcam_resized, cv2.COLOR_GRAY2BGR)
            
            # Create combined frame
            combined_frame = np.hstack([original_resized, gradcam_resized])
            
            # Add simple text overlay for action info
            action_text = f"Action: {self._get_action_name(predicted_action)}"
            cv2.putText(combined_frame, action_text, (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            
            # Add probability text
            prob_text = f"Prob: {action_probs[predicted_action]:.3f}"
            cv2.putText(combined_frame, prob_text, (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            return combined_frame
        
        # Full matplotlib visualization (memory intensive)
        # Create matplotlib figure with explicit cleanup
        fig, axes = plt.subplots(1, 3, figsize=(16, 6), dpi=80)  # Lower DPI to save memory
        
        try:
            # Normalize images for matplotlib display
            def normalize_for_matplotlib(img):
                """Normalize image for matplotlib display."""
                if img.dtype == np.float64 or img.dtype == np.float32:
                    # Ensure values are in [0, 1] range for matplotlib
                    if img.max() > 1.0:
                        img = img / 255.0
                    return np.clip(img, 0, 1)
                elif img.dtype == np.uint8:
                    return img
                else:
                    return img.astype(np.float32)
            
            original_display = normalize_for_matplotlib(original_img)
            gradcam_display = normalize_for_matplotlib(gradcam_img)
            
            # Original gameplay
            axes[0].imshow(original_display)
            axes[0].set_title('Original Gameplay', fontsize=14)
            axes[0].axis('off')
            
            # GradCAM overlay
            axes[1].imshow(gradcam_display)
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
            
        finally:
            # Explicit cleanup to prevent memory leaks
            plt.close(fig)
            plt.clf()  # Clear the current figure
            plt.cla()  # Clear the current axes
            
            # Force garbage collection for large objects
            del fig, axes
            if 'buf' in locals():
                del buf
                
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
        
        # Get initial memory usage
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        print(f"Initial memory usage: {initial_memory:.1f} MB")
        
        # Setup video writer
        self.setup_video_writer()
        
        # Reset environment
        obs = self.gradcam_visualizer.env.reset()
        
        for step in range(self.video_length):
            try:
                # Get action and probabilities
                action, _ = self.gradcam_visualizer.model.predict(obs, deterministic=deterministic)
                
                # Get action probabilities for visualization
                action_probs = self.gradcam_visualizer.get_action_probabilities(obs[0])
                
                # Create combined frame
                try:
                    combined_frame = self.create_combined_frame(obs[0], action[0], action_probs)
                    
                    # Write frame to video
                    self.video_writer.write(combined_frame)
                except Exception as frame_error:
                    print(f"Error creating/writing frame at step {step}: {frame_error}")
                    # Skip this frame and continue
                    continue
                
                # Take step in environment
                obs, reward, done, info = self.gradcam_visualizer.env.step(action)
                
                # Memory management and monitoring
                if step % 50 == 0:
                    current_memory = process.memory_info().rss / 1024 / 1024  # MB
                    print(f"Recorded {step}/{self.video_length} frames... Memory: {current_memory:.1f} MB (+{current_memory-initial_memory:.1f} MB)")
                
                # Periodic garbage collection to prevent memory buildup
                if step % 50 == 0 and step > 0:  # More frequent cleanup
                    gc.collect()  # Force garbage collection
                    
                    # Additional memory cleanup for matplotlib (only if enabled)
                    if self.show_matplotlib:
                        plt.close('all')  # Close any lingering matplotlib figures
                    
                    # Check memory usage and warn if getting too high
                    current_memory = process.memory_info().rss / 1024 / 1024  # MB
                    if current_memory > initial_memory + 2000:  # If memory increased by more than 2GB
                        print(f"⚠️  Warning: High memory usage detected: {current_memory:.1f} MB")
                        print("   Forcing additional cleanup...")
                        
                        # Force more aggressive cleanup
                        import torch
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()  # Clear GPU cache if using CUDA
                        gc.collect()
                
                # Handle episode end
                if done[0]:
                    print(f"Episode ended at step {step}")
                    obs = self.gradcam_visualizer.env.reset()
                    
            except Exception as e:
                print(f"Error at step {step}: {e}")
                # Try to continue with next frame
                continue
        
        # Final cleanup
        if self.video_writer:
            self.video_writer.release()
            
        # Final memory report
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        print(f"Final memory usage: {final_memory:.1f} MB (+{final_memory-initial_memory:.1f} MB)")
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
    parser = argparse.ArgumentParser(description='Record video with GradCAM overlay (aligned with record_video.py format)')
    parser.add_argument("--env", help="Environment ID", type=str, default="BreakoutNoFrameskip-v4")
    parser.add_argument("--algo", help="RL Algorithm", default="ppo", type=str)
    parser.add_argument("--model-path", help="Path to model file", type=str)
    parser.add_argument("-f", "--folder", help="Log folder", type=str, default="rl-trained-agents")
    parser.add_argument("-o", "--output-folder", help="Output folder", type=str, default="gradcam_videos")
    parser.add_argument("-n", "--n-timesteps", help="Number of timesteps", default=4000, type=int)
    parser.add_argument("--exp-id", help="Experiment ID", default=0, type=int)
    parser.add_argument("--deterministic", action="store_true", default=True, help="Use deterministic actions")
    parser.add_argument("--seed", help="Random generator seed", type=int, default=42)
    parser.add_argument("--load-best", action="store_true", default=False, help="Load best model")
    parser.add_argument("--fps", help="Video frames per second (default: auto-detect from env)", type=int, default=None)
    parser.add_argument("--video-codec", help="Video codec (H264, avc1, mp4v)", type=str, default="H264", choices=["H264", "avc1", "mp4v"])
    parser.add_argument("--show-matplotlib", action="store_true", default=False, help="Show matplotlib visualization (memory intensive, default: False)")
    parser.add_argument("--max-memory", help="Maximum memory usage in MB before warning", type=int, default=8192)
    parser.add_argument("--gc-frequency", help="Garbage collection frequency (every N frames)", type=int, default=100)
    
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
    
    # Memory usage information
    if args.show_matplotlib:
        print("🎨 Matplotlib visualization enabled (memory intensive)")
    else:
        print("💾 Matplotlib visualization disabled (memory optimized)")
        print("   Use --show-matplotlib to enable full visualization")
    
    try:
        # Create recorder
        recorder = GradCAMVideoRecorder(
            model_path=model_path,
            env_id=args.env,
            algo=args.algo,
            output_folder=args.output_folder,
            video_length=args.n_timesteps,
            fps=args.fps,
            codec=args.video_codec,
            show_matplotlib=args.show_matplotlib
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