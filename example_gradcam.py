#!/usr/bin/env python3
"""
Example: GradCAM Visualization for Breakout PPO Agent
=====================================================

This script demonstrates how to use the GradCAM visualizer with a trained
Breakout PPO agent from the rl-baselines3-zoo.

Usage:
    python example_gradcam.py

Requirements:
    - Install dependencies: pip install -r requirements.txt
    - Ensure you have the trained Breakout PPO model in the expected location
"""

import os
import sys
from pathlib import Path

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from gradcam_visualizer import AtariGradCAMVisualizer
from inference_utils import load_breakout_ppo_model


def main():
    """Main function to demonstrate GradCAM visualization."""
    
    # Model path - adjust this to your model location
    model_path = "rl-baselines3-zoo/rl-trained-agents/ppo/BreakoutNoFrameskip-v4_1/BreakoutNoFrameskip-v4.zip"
    
    # Check if model exists
    if not os.path.exists(model_path):
        print(f"Model not found at: {model_path}")
        print("Please ensure you have the trained Breakout PPO model.")
        print("You can download it from the rl-baselines3-zoo repository or train your own.")
        return
    
    print("🎮 GradCAM Visualization for Breakout PPO Agent")
    print("=" * 50)
    
    try:
        # Initialize the visualizer
        print("📥 Loading model and setting up environment...")
        visualizer = AtariGradCAMVisualizer(
            model_path=model_path,
            env_id="BreakoutNoFrameskip-v4",
            algo="ppo",
            device="auto"
        )
        
        # Create output directory
        output_dir = "gradcam_results"
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"📁 Output directory: {output_dir}")
        
        # Example 1: Compare different GradCAM methods
        print("\n🔍 Comparing different GradCAM methods...")
        obs = visualizer.env.reset()
        comparison_path = os.path.join(output_dir, "gradcam_comparison.png")
        visualizer.compare_methods(obs[0], comparison_path)
        
        # Example 2: Visualize a short episode
        print("\n🎬 Visualizing episode with GradCAM...")
        episode_data = visualizer.visualize_episode(
            n_steps=50,  # Short episode for demo
            save_dir=output_dir,
            method='gradcam',
            save_video=True
        )
        
        print(f"\n✅ Visualization complete!")
        print(f"📊 Processed {len(episode_data)} steps")
        print(f"📂 Results saved to: {output_dir}")
        print(f"🎥 Video saved as: gradcam_episode_gradcam.mp4")
        
        # Example 3: Analyze a specific frame
        print("\n🔬 Analyzing a specific frame...")
        if episode_data:
            # Get a frame from the middle of the episode
            mid_frame = len(episode_data) // 2
            frame_data = episode_data[mid_frame]
            
            print(f"Frame {mid_frame}:")
            print(f"  Predicted action: {frame_data['predicted_action']}")
            print(f"  Action probabilities: {frame_data['action_probs']}")
            print(f"  Max probability: {max(frame_data['action_probs']):.3f}")
            
            # Show action meanings for Breakout
            action_meanings = {
                0: "NOOP",
                1: "FIRE", 
                2: "RIGHT",
                3: "LEFT"
            }
            
            print(f"  Action meaning: {action_meanings.get(frame_data['predicted_action'], 'Unknown')}")
        
        print("\n🎯 Key files generated:")
        print(f"  - {comparison_path}")
        print(f"  - {output_dir}/gradcam_episode_gradcam.mp4")
        print(f"  - {output_dir}/step_*.png (individual frames)")
        
    except Exception as e:
        print(f"❌ Error during visualization: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


def demo_inference_only():
    """Demonstrate inference without GradCAM (simpler example)."""
    
    model_path = "rl-baselines3-zoo/rl-trained-agents/ppo/BreakoutNoFrameskip-v4_1/BreakoutNoFrameskip-v4.zip"
    
    if not os.path.exists(model_path):
        print("Model not found. Please ensure you have the trained model.")
        return
    
    try:
        # Load model and environment
        print("Loading Breakout PPO model...")
        model, env = load_breakout_ppo_model(model_path)
        
        # Run a few steps
        obs = env.reset()
        total_reward = 0
        
        for step in range(100):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            total_reward += reward[0]
            
            if step % 20 == 0:
                print(f"Step {step}: Action = {action[0]}, Reward = {reward[0]}")
            
            if done[0]:
                print(f"Episode finished at step {step}")
                break
        
        print(f"Total reward: {total_reward}")
        
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    # Check if we want to run the simple inference demo
    if len(sys.argv) > 1 and sys.argv[1] == "--inference-only":
        demo_inference_only()
    else:
        exit_code = main()
        sys.exit(exit_code) 