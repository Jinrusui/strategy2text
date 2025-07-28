import os
import random
import sys
import time
from pathlib import Path
import imageio
import numpy as np
import gymnasium as gym
import ale_py # Import to register ALE environments

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

OUTPUT_DIR = Path("causality_hallucination_test")
NUM_SAMPLES = 10
# PADDLE_POSITIONS will be set dynamically after paddle limits are detected
FRAME_SKIP = 1
# Removed NUM_RECORDING_FRAMES limit - now records until ball is lost or game ends
NUM_BALL_LAUNCH_VALIDATION_FRAMES = 50 # Increased to allow paddle to settle
MIN_VIDEO_FRAMES = 10 # Minimum frames required for a valid video
PADDLE_TOLERANCE = 5 # Tolerance for paddle positioning
MAX_PADDLE_MOVE_FRAMES = 50 # Max frames for positioning to prevent infinite loops
VIDEO_CLIP_FRAMES = 50 # Number of frames to remove from end to hide final position

# Breakout v4 RAM addresses
PADDLE_X_RAM_ADDRESS = 72
BALL_X_RAM_ADDRESS = 99
BALL_Y_RAM_ADDRESS = 101
LIVES_RAM_ADDRESS = 57

import cv2

def test_paddle_limits(env, obs):
    """Test the minimum and maximum paddle positions."""
    print("Testing paddle position limits...")
    # 激活 paddle：先 FIRE，再多次 NOOP
    obs, _ = env.reset()
    obs, _, terminated, truncated, info = env.step(1)  # FIRE
    for _ in range(10):
        obs, _, terminated, truncated, info = env.step(0)  # NOOP
        if terminated or truncated:
            obs, _ = env.reset()
            obs, _, terminated, truncated, info = env.step(1)
    # Test moving left as far as possible
    current_x = get_paddle_x_from_ram(obs)
    print(f"Initial paddle position: {current_x}")
    left_limit = current_x
    stuck_count = 0
    for i in range(50):
        obs, _, terminated, truncated, info = env.step(3)  # Left action
        if terminated or truncated:
            break
        new_x = get_paddle_x_from_ram(obs)
        if new_x == left_limit:
            stuck_count += 1
            if stuck_count >= 3:  # 连续3步不变才停止
                break
        else:
            stuck_count = 0
            left_limit = new_x
        print(f"  Left step {i+1}: {new_x}")
    # Reset and activate paddle again for right test
    obs, _ = env.reset()
    obs, _, terminated, truncated, info = env.step(1)  # FIRE
    for _ in range(10):
        obs, _, terminated, truncated, info = env.step(0)  # NOOP
        if terminated or truncated:
            obs, _ = env.reset()
            obs, _, terminated, truncated, info = env.step(1)
    current_x = get_paddle_x_from_ram(obs)
    right_limit = current_x
    stuck_count = 0
    for i in range(50):
        obs, _, terminated, truncated, info = env.step(2)  # Right action
        if terminated or truncated:
            break
        new_x = get_paddle_x_from_ram(obs)
        if new_x == right_limit:
            stuck_count += 1
            if stuck_count >= 3:  # 连续3步不变才停止
                break
        else:
            stuck_count = 0
            right_limit = new_x
        print(f"  Right step {i+1}: {new_x}")
    print(f"Paddle limits - Left: {left_limit}, Right: {right_limit}")
    return left_limit, right_limit

def debug_ram_structure(obs):
    """Debug function to understand RAM structure."""
    print(f"RAM observation type: {type(obs)}")
    print(f"RAM observation length: {len(obs)}")
    
    # Extract the actual RAM array from the tuple
    ram_array = obs[0] if isinstance(obs, tuple) else obs
    print(f"RAM array type: {type(ram_array)}")
    print(f"RAM array shape: {ram_array.shape}")
    print(f"RAM array length: {len(ram_array)}")
    print(f"First 20 RAM values: {ram_array[:20]}")
    print(f"RAM values around paddle area (60-80): {ram_array[60:80]}")
    print(f"RAM values around ball area (95-105): {ram_array[95:105]}")
    
    # Check specific addresses
    print(f"Paddle X at address {PADDLE_X_RAM_ADDRESS}: {ram_array[PADDLE_X_RAM_ADDRESS]}")
    print(f"Ball X at address {BALL_X_RAM_ADDRESS}: {ram_array[BALL_X_RAM_ADDRESS]}")
    print(f"Ball Y at address {BALL_Y_RAM_ADDRESS}: {ram_array[BALL_Y_RAM_ADDRESS]}")
    print(f"Lives at address {LIVES_RAM_ADDRESS}: {ram_array[LIVES_RAM_ADDRESS]}")

def get_ball_coords_from_ram(obs):
    """Gets ball coordinates from RAM observation."""
    ram_array = obs[0] if isinstance(obs, tuple) else obs
    ball_x = ram_array[BALL_X_RAM_ADDRESS]
    ball_y = ram_array[BALL_Y_RAM_ADDRESS]
    return ball_x, ball_y

def get_paddle_x_from_ram(obs):
    """Gets paddle X position from RAM observation."""
    ram_array = obs[0] if isinstance(obs, tuple) else obs
    return ram_array[PADDLE_X_RAM_ADDRESS]

def get_lives_from_ram(obs):
    """Gets remaining lives from RAM observation."""
    ram_array = obs[0] if isinstance(obs, tuple) else obs
    return ram_array[LIVES_RAM_ADDRESS]

# --- MODIFIED FUNCTION ---
def move_paddle_to_position(env, obs, target_x, move_frames=None):
    """Move paddle to target position by checking its location after each step. Optionally record each move frame."""
    print(f"    Paddle moving to {target_x}...")
    initial_x = get_paddle_x_from_ram(obs)
    print(f"      Start at X={initial_x}")

    for i in range(MAX_PADDLE_MOVE_FRAMES):
        current_x = get_paddle_x_from_ram(obs)

        # Check if we are within the target tolerance
        if abs(current_x - target_x) <= PADDLE_TOLERANCE:
            print(f"    Paddle successfully positioned at {current_x} (Target was: {target_x})")
            return True, obs

        # Decide which way to move: 3 for Left, 2 for Right
        action = 3 if current_x > target_x else 2

        obs, _, terminated, truncated, _ = env.step(action)
        new_x = get_paddle_x_from_ram(obs)
        print(f"      Step {i+1}: {'Left' if action == 3 else 'Right'} → X={new_x}")

        # 记录移动过程帧
        if move_frames is not None:
            move_frames.append(env.render())

        if terminated or truncated:
            print("      [TERMINATED] during paddle movement.")
            return False, obs

    final_x = get_paddle_x_from_ram(obs)
    print(f"      [FAILED] Could not reach {target_x} in {MAX_PADDLE_MOVE_FRAMES} steps. Final X={final_x}")
    return False, obs



def run_experiment(condition, num_samples, paddle_limits):
    """Generates video data for a given experimental condition."""
    env = gym.make("Breakout-v4", render_mode="rgb_array", obs_type="ram", frameskip=1)

    left_limit, right_limit = paddle_limits
    # 计算 paddle 位置：在中点和极限之间，不太靠近边界
    range_quarter = (right_limit - left_limit) // 4
    mid_limit = left_limit + (right_limit - left_limit) // 2
    
    PADDLE_POSITIONS = {
        "condition_A": left_limit + range_quarter,      # 左侧1/4位置
        "condition_B": mid_limit,                       # 中间位置
        "condition_C": right_limit - range_quarter,     # 右侧3/4位置
    }

    condition_dir = OUTPUT_DIR / condition
    condition_dir.mkdir(exist_ok=True)

    samples_collected = 0
    while samples_collected < num_samples:
        print(f"Attempting to generate sample {samples_collected + 1}/{num_samples} for {condition}...")

        obs, _ = env.reset()

        video_frames = []

        # --- Paddle Positioning Phase ---
        target_paddle_x = PADDLE_POSITIONS[condition]
        # 增加随机性：在目标位置附近随机偏移
        random_offset = random.randint(-10, 10)
        final_target_x = int(max(left_limit, min(right_limit, int(target_paddle_x) + random_offset)))
        
        # 先移动到一个随机位置发球（增加发球方向的随机性）
        random_launch_x = random.randint(left_limit + 20, right_limit - 20)  # 避免太靠近边界
        print(f"\n=== Sample {samples_collected+1}/{num_samples} ===")
        print(f"  [LAUNCH] Paddle launch position: {random_launch_x}")
        paddle_moved, obs = move_paddle_to_position(env, obs, random_launch_x, move_frames=video_frames)
        
        if not paddle_moved:
            print("  Could not position paddle at launch position. Retrying...")
            continue

        # --- Fire Ball from Random Position ---
        # 增加随机性：随机等待几帧再发球
        for _ in range(random.randint(0, 5)):
            obs, _, terminated, truncated, info = env.step(0)  # NOOP
            if terminated or truncated:
                break
        
        obs, _, terminated, truncated, info = env.step(1)  # Fire
        video_frames.append(env.render())
        
        # 发球后立即移动到目标位置
        print(f"  [TARGET] Paddle target position: {final_target_x}")
        paddle_moved, obs = move_paddle_to_position(env, obs, final_target_x, move_frames=video_frames)

        # 记录实际 paddle X（可用于调试）
        actual_paddle_x = get_paddle_x_from_ram(obs)

        if not paddle_moved:
            print("  Could not position paddle at target position after launch. Retrying...")
            continue

        # --- Ball Launch Validation Phase ---
        ball_launch_valid = True
        for frame_num in range(NUM_BALL_LAUNCH_VALIDATION_FRAMES):
            obs, _, terminated, truncated, info = env.step(0)  # NOOP
            ball_x, ball_y = get_ball_coords_from_ram(obs)

            if ball_x == 0 or terminated or truncated:
                ball_launch_valid = False
                break
            video_frames.append(env.render())

        if not ball_launch_valid:
            print("  Ball lost or episode terminated during launch validation. Retrying...")
            continue
        else:
            print(f"  [BALL] Launch validated. Paddle X after move: {get_paddle_x_from_ram(obs)}. Start recording...")

        # --- Main Recording Phase ---
        last_known_ball_x = -1
        initial_lives = get_lives_from_ram(obs)
        recording_frames = []

        # Record until ball is lost, lives decrease, or game ends (no frame limit)
        frame_count = 0
        consecutive_ball_missing = 0  # Count consecutive frames where ball_x == 0
        BALL_MISSING_THRESHOLD = 3    # Consider ball lost after 3 consecutive frames with ball_x == 0
        
        while True:
            obs, _, terminated, truncated, info = env.step(0)  # NOOP
            ball_x, ball_y = get_ball_coords_from_ram(obs)
            current_lives = get_lives_from_ram(obs)
            frame_count += 1

            if ball_x != 0:
                last_known_ball_x = ball_x
                consecutive_ball_missing = 0  # Reset counter when ball is visible
            else:
                consecutive_ball_missing += 1

            # End recording if:
            # 1. Lives decreased (ball truly lost)
            # 2. Game terminated/truncated
            # 3. Ball missing for multiple consecutive frames (likely truly lost)
            if (current_lives < initial_lives or 
                terminated or truncated or 
                consecutive_ball_missing >= BALL_MISSING_THRESHOLD):
                
                reason = []
                if current_lives < initial_lives:
                    reason.append("life_lost")
                if terminated:
                    reason.append("terminated")
                if truncated:
                    reason.append("truncated")
                if consecutive_ball_missing >= BALL_MISSING_THRESHOLD:
                    reason.append(f"ball_missing_{consecutive_ball_missing}_frames")
                
                print(f"  Recording ended after {frame_count} frames")
                print(f"  End reason: {', '.join(reason)}")
                print(f"  Final state: ball_x={ball_x}, lives={current_lives}/{initial_lives}")
                break

            recording_frames.append(env.render())

        # --- Ground Truth Collection Phase ---
        # 记录当前 RAM 状态和 ball_x
        prev_obs = obs
        prev_ball_x = last_known_ball_x
        found_precise = False
        while not (terminated or truncated):
            obs, _, terminated, truncated, info = env.step(0)  # NOOP
            ball_x, _ = get_ball_coords_from_ram(obs)
            if ball_x != 0:
                prev_obs = obs
                prev_ball_x = ball_x
            else:
                # ball_x变为0，回退到前一帧，精确 replay
                found_precise = True
                break

        if found_precise:
            # 用 frame_skip=1 环境 replay
            precise_env = gym.make("Breakout-v4", render_mode="rgb_array", obs_type="ram", frameskip=1)
            # 恢复 RAM 状态
            precise_env.reset()
            precise_env.unwrapped.ale.setRAM(prev_obs)
            precise_ball_x = prev_ball_x
            for _ in range(10):  # 最多 replay 10 帧
                precise_obs, _, precise_terminated, precise_truncated, _ = precise_env.step(0)
                bx, _ = get_ball_coords_from_ram(precise_obs)
                if bx != 0:
                    precise_ball_x = bx
                else:
                    break
            ground_truth_x = precise_ball_x
            precise_env.close()
        else:
            ground_truth_x = prev_ball_x  # fallback

        # --- Video Saving ---
        all_frames = video_frames + recording_frames
        print(f"  Total frames recorded: {len(all_frames)}")
        print(f"  Video frames: {len(video_frames)}, Recording frames: {len(recording_frames)}")

        original_video_path = condition_dir / f"sample_{samples_collected:03d}_original.mp4"
        imageio.mimsave(original_video_path, all_frames, fps=30 // FRAME_SKIP, macro_block_size=1)

        # Apply frame clipping logic with detailed logging
        if len(all_frames) > VIDEO_CLIP_FRAMES:
            frames_to_remove = VIDEO_CLIP_FRAMES
            clipped_frames = all_frames[:-VIDEO_CLIP_FRAMES]
            print(f"  Clipping applied: Removed {frames_to_remove} frames from end")
            print(f"  Original length: {len(all_frames)} → Clipped length: {len(clipped_frames)}")
        else:
            clipped_frames = all_frames
            print(f"  No clipping needed: Video has {len(all_frames)} frames (≤ {VIDEO_CLIP_FRAMES} threshold)")

        if ground_truth_x != -1 and len(clipped_frames) >= MIN_VIDEO_FRAMES:
            print(f"  Successfully recorded sample {samples_collected:03d}. Ground truth X: {ground_truth_x}")
            video_path = condition_dir / f"sample_{samples_collected:03d}.mp4"
            imageio.mimsave(video_path, clipped_frames, fps=30 // FRAME_SKIP, macro_block_size=1)

            with open(condition_dir / f"sample_{samples_collected:03d}_ground_truth.txt", "w") as f:
                f.write(str(ground_truth_x))

            samples_collected += 1
        else:
            print(f"  Discarding invalid sample. Retrying...")

    env.close()

if __name__ == "__main__":
    OUTPUT_DIR.mkdir(exist_ok=True)
    # 用一个环境检测 paddle 极限
    env = gym.make("Breakout-v4", render_mode="rgb_array", obs_type="ram", frameskip=1)
    obs, _ = env.reset()
    print("Debugging RAM structure on first run...")
    debug_ram_structure(obs)
    left_limit, right_limit = test_paddle_limits(env, obs)
    print(f"Paddle position range: {left_limit} to {right_limit}")
    env.close()

    paddle_limits = (left_limit, right_limit)
    for condition in ["condition_A", "condition_B", "condition_C"]:
        run_experiment(condition, NUM_SAMPLES, paddle_limits)