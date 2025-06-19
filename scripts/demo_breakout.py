#!/usr/bin/env python3
"""
Demo script for running pre-trained Breakout agents.

This script loads and runs pre-trained PPO or DQN agents on the Breakout game,
displaying the gameplay and optionally saving videos.
"""

import os
import sys
import argparse
import time
from pathlib import Path
from typing import Optional, List, Tuple
import numpy as np
import cv2


# Add the src directory to the path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

try:
    from rl_agents.sb3_agent import SB3Agent
    import gymnasium as gym
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation
except ImportError as e:
    print(f"Import error: {e}")
    print("Please install required dependencies:")
    print("  pip install -r requirements.txt")
    sys.exit(1)


def find_model_path(algorithm: str, models_dir: str = "rl-trained-agents") -> Path:
    """Find the model file for the specified algorithm."""
    models_path = Path(models_dir)
    
    # Look for the standard RL Zoo structure
    model_path = models_path / algorithm / "BreakoutNoFrameskip-v4_1" / "BreakoutNoFrameskip-v4.zip"
    
    if model_path.exists():
        return model_path
    
    # Alternative search patterns
    for pattern in [
        f"{algorithm}*breakout*.zip",
        f"*{algorithm}*Breakout*.zip",
        f"breakout*{algorithm}*.zip"
    ]:
        matches = list(models_path.rglob(pattern))
        if matches:
            return matches[0]
    
    raise FileNotFoundError(f"Could not find {algorithm} model for Breakout in {models_path}")


def run_breakout_demo(
    algorithm: str = "ppo",
    num_episodes: int = 3,
    max_steps: int = 10000,
    render: bool = True,
    save_video: bool = False,
    models_dir: str = "rl-trained-agents"
):
    """
    Run a demo of the pre-trained Breakout agent.
    
    Args:
        algorithm: Algorithm to use ('ppo' or 'dqn')
        num_episodes: Number of episodes to run
        max_steps: Maximum steps per episode
        render: Whether to display the game
        save_video: Whether to save video files
        models_dir: Directory containing the models
    """
    print(f"Loading pre-trained {algorithm.upper()} agent for Breakout...")
    
    try:
        # Find and load the model
        model_path = find_model_path(algorithm, models_dir)
        print(f"Found model: {model_path}")
        
        # Create agent
        agent = SB3Agent(
            agent_path=str(model_path),
            algorithm=algorithm,
            env_id="ALE/Breakout-v4"  # Use v4 to match training environment
        )
        
        print(f"✓ Successfully loaded {algorithm.upper()} agent")
        print(f"Agent info: {agent.get_agent_info()}")
        
    except Exception as e:
        print(f"✗ Failed to load agent: {e}")
        print("\nTry downloading the model first:")
        print(f"  python scripts/download_rl_zoo_models.py --algorithm {algorithm}")
        return
    
    # Run episodes
    print(f"\nRunning {num_episodes} episodes...")
    print("=" * 50)
    
    total_rewards = []
    episode_lengths = []
    
    for episode in range(num_episodes):
        print(f"\nEpisode {episode + 1}/{num_episodes}")
        print("-" * 30)
        
        try:
            # Run episode
            episode_data = agent.run_episode(
                max_steps=max_steps,
                render=render,
                record_frames=save_video,
                seed=episode * 42  # Reproducible seeds
            )
            
            total_reward = sum(episode_data['rewards'])
            episode_length = len(episode_data['rewards'])
            
            total_rewards.append(total_reward)
            episode_lengths.append(episode_length)
            
            print(f"  Total Reward: {total_reward}")
            print(f"  Steps: {episode_length}")
            print(f"  Average Reward: {total_reward/episode_length:.3f}")
            
            # Save video if requested
            if save_video and episode_data['frames']:
                video_path = f"breakout_{algorithm}_episode_{episode + 1}.mp4"
                save_episode_video(episode_data['frames'], video_path)
                print(f"  Video saved: {video_path}")
            
            # Small delay between episodes if rendering
            if render and episode < num_episodes - 1:
                time.sleep(2)
                
        except Exception as e:
            print(f"  ✗ Episode failed: {e}")
            continue
    
    # Print summary
    if total_rewards:
        print("\n" + "=" * 50)
        print("SUMMARY")
        print("=" * 50)
        print(f"Algorithm: {algorithm.upper()}")
        print(f"Episodes completed: {len(total_rewards)}")
        print(f"Average reward: {np.mean(total_rewards):.2f} ± {np.std(total_rewards):.2f}")
        print(f"Best reward: {max(total_rewards)}")
        print(f"Average episode length: {np.mean(episode_lengths):.1f} steps")
        
        # Simple performance assessment
        avg_reward = np.mean(total_rewards)
        if avg_reward > 400:
            print("🎉 Excellent performance!")
        elif avg_reward > 200:
            print("👍 Good performance!")
        elif avg_reward > 50:
            print("👌 Decent performance")
        else:
            print("🤔 Could be better - agent might need more training")




# ==================== 从这里开始复制 ====================

def save_episode_video(
    frames: List[Optional[np.ndarray]],
    output_path: str,
    fps: int = 30
):
    """
    将一系列的帧保存为视频文件，并以强大的兼容性处理编码器。
    (最终修正版：移除了 getBackendName 调用以避免特定环境下的OpenCV错误)
    """
    # 1. --- 输入验证 ---
    if not frames:
        print("Warning: 传入的帧列表为空，无法保存视频。")
        return

    # 2. --- 从有效的帧中获取视频尺寸 ---
    try:
        valid_frame = next(f for f in frames if f is not None)
    except StopIteration:
        print("Warning: 传入的帧列表不包含任何有效的帧，无法保存视频。")
        return

    if valid_frame.ndim == 4 and valid_frame.shape[0] == 1:
        valid_frame = valid_frame[0]
    
    height, width, _ = valid_frame.shape
    size = (width, height)

    # 3. --- 定义并选择最合适的编码器 ---
    output_path_obj = Path(output_path)
    output_path_obj.parent.mkdir(parents=True, exist_ok=True)
    
    codecs = [
        ('avc1', cv2.VideoWriter_fourcc(*'avc1')),
        ('mp4v', cv2.VideoWriter_fourcc(*'mp4v')),
        ('H264', cv2.VideoWriter_fourcc(*'H264')),
        ('X264', cv2.VideoWriter_fourcc(*'X264')),
    ]
    if output_path.lower().endswith('.avi'):
        codecs.append(('mjpeg', cv2.VideoWriter_fourcc(*'mjpeg')))

    writer = None
    used_codec_name = "N/A"
    print(f"准备保存视频至: {output_path}")
    for codec_name, fourcc in codecs:
        try:
            print(f"--> 正在尝试使用编码器: '{codec_name}'")
            writer = cv2.VideoWriter(str(output_path_obj), fourcc, float(fps), size)
            if writer.isOpened():
                print(f"    成功! 视频写入器已使用 '{codec_name}' 编码器打开。")
                used_codec_name = codec_name
                break
            else:
                writer = None
        except Exception as e:
            print(f"    编码器 '{codec_name}' 初始化失败，错误: {e}")
            writer = None
    
    if not writer or not writer.isOpened():
        print("错误: 无法使用任何合适的编码器打开视频写入器。")
        return

    # 4. --- 循环写入所有帧 ---
    frames_written = 0
    for frame in frames:
        if frame is None:
            continue
        if frame.ndim == 4 and frame.shape[0] == 1:
            frame = frame[0]
        writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        frames_written += 1
    
    # 5. --- 释放资源并完成写入 ---
    writer.release()
    
    # --- 【核心修改】替换了下面这行日志 ---
    if frames_written > 0:
        print(f"✓ 视频已使用 '{used_codec_name}' 编码器成功写入并关闭。")
    else:
        print("Warning: 视频文件已创建，但没有帧被写入。")

# ==================== 到这里结束复制 ====================


def check_model_availability(models_dir: str = "rl-trained-agents"):
    """Check which models are available."""
    models_path = Path(models_dir)
    
    if not models_path.exists():
        print(f"Models directory not found: {models_path}")
        return []
    
    available = []
    algorithms = ['ppo', 'dqn', 'a2c', 'qr-dqn']
    
    for alg in algorithms:
        try:
            model_path = find_model_path(alg, models_dir)
            size_mb = model_path.stat().st_size / (1024 * 1024)
            available.append((alg, model_path, size_mb))
        except FileNotFoundError:
            continue
    
    return available


def main():
    parser = argparse.ArgumentParser(
        description="Demo pre-trained Breakout agents",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run PPO agent for 5 episodes
  python scripts/demo_breakout.py --algorithm ppo --episodes 5
  
  # Run DQN agent and save videos
  python scripts/demo_breakout.py --algorithm dqn --save-video
  
  # Run without rendering (faster)
  python scripts/demo_breakout.py --algorithm ppo --no-render
  
  # Check available models
  python scripts/demo_breakout.py --list
        """
    )
    
    parser.add_argument("--algorithm", "-a", default="ppo", 
                       choices=['ppo', 'dqn', 'a2c', 'qr-dqn'],
                       help="Algorithm to use (default: ppo)")
    parser.add_argument("--episodes", "-e", type=int, default=1,
                       help="Number of episodes to run (default: 1)")
    parser.add_argument("--max-steps", type=int, default=10000,
                       help="Maximum steps per episode (default: 10000)")
    parser.add_argument("--no-render", action="store_true",
                       help="Don't render the game (faster)")
    parser.add_argument("--save-video", action="store_true",
                       help="Save episode videos as MP4 files")
    parser.add_argument("--models-dir", default="rl-trained-agents",
                       help="Directory containing models (default: rl-trained-agents)")
    parser.add_argument("--list", action="store_true",
                       help="List available models and exit")
    
    args = parser.parse_args()
    
    if args.list:
        print("Checking available models...")
        available = check_model_availability(args.models_dir)
        
        if available:
            print(f"\nAvailable models in {args.models_dir}:")
            for alg, path, size_mb in available:
                print(f"  ✓ {alg.upper()}: {path.name} ({size_mb:.1f} MB)")
            
            print(f"\nTo run a demo:")
            print(f"  python scripts/demo_breakout.py --algorithm {available[0][0]}")
        else:
            print(f"\nNo models found in {args.models_dir}")
            print("Download models first:")
            print("  python scripts/download_rl_zoo_models.py")
        
        return
    
    # Run the demo
    run_breakout_demo(
        algorithm=args.algorithm,
        num_episodes=args.episodes,
        max_steps=args.max_steps,
        render=not args.no_render,
        save_video=args.save_video,
        models_dir=args.models_dir
    )


if __name__ == "__main__":
    main() 