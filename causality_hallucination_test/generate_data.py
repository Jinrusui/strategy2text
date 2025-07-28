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

# Constants
OUTPUT_DIR = Path("causality_hallucination_test")
NUM_SAMPLES = 100
PADDLE_POSITIONS = {
    "condition_A": 80,  # Left (adjusted to be achievable - between 55-191)
    "condition_B": 135, # Middle (adjusted to be achievable)
    "condition_C": 160, # Right (adjusted to be achievable)
}
FRAME_SKIP = 4
NUM_RECORDING_FRAMES = 40
NUM_BALL_LAUNCH_VALIDATION_FRAMES = 50 # Increased to allow paddle to settle
MIN_VIDEO_FRAMES = 10 # Minimum frames required for a valid video
PADDLE_TOLERANCE = 5 # Tolerance for paddle positioning
MAX_PADDLE_MOVE_FRAMES = 50 # Max frames for positioning to prevent infinite loops
VIDEO_CLIP_FRAMES = 75 # Number of frames to remove from end to hide final position

# Breakout v4 RAM addresses
PADDLE_X_RAM_ADDRESS = 72
BALL_X_RAM_ADDRESS = 99
BALL_Y_RAM_ADDRESS = 101
LIVES_RAM_ADDRESS = 57

import cv2

def test_paddle_limits(env, obs):
    """Test the minimum and maximum paddle positions."""
    print("Testing paddle position limits...")
    
    # Test moving left as far as possible
    current_x = get_paddle_x_from_ram(obs)
    print(f"Initial paddle position: {current_x}")
    
    # Move left until it stops changing
    left_limit = current_x
    for i in range(50):
        obs, _, terminated, truncated, info = env.step(3)  # Left action
        if terminated or truncated:
            break
        new_x = get_paddle_x_from_ram(obs)
        if new_x == left_limit:
            break
        left_limit = new_x
        print(f"  Left step {i+1}: {new_x}")
    
    # Reset and test moving right
    obs, _ = env.reset()
    current_x = get_paddle_x_from_ram(obs)
    
    # Move right until it stops changing
    right_limit = current_x
    for i in range(50):
        obs, _, terminated, truncated, info = env.step(2)  # Right action
        if terminated or truncated:
            break
        new_x = get_paddle_x_from_ram(obs)
        if new_x == right_limit:
            break
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
def move_paddle_to_position(env, obs, target_x):
    """Move paddle to target position by checking its location after each step."""
    print(f"  Moving paddle to target X: {target_x}")
    initial_x = get_paddle_x_from_ram(obs)
    print(f"    Initial paddle X: {initial_x}")

    # Use a loop with a safety break to prevent getting stuck.
    for i in range(MAX_PADDLE_MOVE_FRAMES):
        current_x = get_paddle_x_from_ram(obs)

        # Check if we are within the target tolerance
        if abs(current_x - target_x) <= PADDLE_TOLERANCE:
            print(f"    Paddle successfully positioned at {current_x} (Target was: {target_x})")
            return True, obs

        # Decide which way to move: 3 for Left, 2 for Right
        action = 3 if current_x > target_x else 2
        
        # Take the step
        obs, _, terminated, truncated, _ = env.step(action)
        
        # Log the new position
        new_x = get_paddle_x_from_ram(obs)
        print(f"    Step {i+1}: Moved {'Left' if action == 3 else 'Right'}. Paddle X = {new_x}")


        # Stop if the episode ends unexpectedly
        if terminated or truncated:
            print("    Episode terminated during paddle movement.")
            return False, obs
            
    # If the loop finishes, we failed to reach the target in time
    final_x = get_paddle_x_from_ram(obs)
    print(f"    Failed to position paddle at {target_x} within {MAX_PADDLE_MOVE_FRAMES} frames. Final position: {final_x}")
    return False, obs


def run_experiment(condition, num_samples):
    """Generates video data for a given experimental condition."""
    env = gym.make("Breakout-v4", render_mode="rgb_array", obs_type="ram")

    condition_dir = OUTPUT_DIR / condition
    condition_dir.mkdir(exist_ok=True)

    samples_collected = 0
    while samples_collected < num_samples:
        print(f"Attempting to generate sample {samples_collected + 1}/{num_samples} for {condition}...")
        
        obs, _ = env.reset()
        
        if samples_collected == 0 and condition == list(PADDLE_POSITIONS.keys())[0]:
            print("Debugging RAM structure on first run...")
            debug_ram_structure(obs)
            left_limit, right_limit = test_paddle_limits(env, obs)
            print(f"Paddle position range: {left_limit} to {right_limit}")
            obs, _ = env.reset() # Reset after testing limits
        
        video_frames = []
        
        # --- Paddle Positioning Phase ---
        target_paddle_x = PADDLE_POSITIONS[condition]
        paddle_moved, obs = move_paddle_to_position(env, obs, target_paddle_x)
        
        if not paddle_moved:
            print("  Could not position paddle correctly. Retrying...")
            continue

        # --- Fire Ball and Initial Trajectory Validation ---
        obs, _, terminated, truncated, info = env.step(1)  # Fire
        video_frames.append(env.render())
        
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
            print("  Ball launch validated. Proceeding with recording.")

        # --- Main Recording Phase ---
        ground_truth_x = -1
        last_known_ball_x = -1
        initial_lives = get_lives_from_ram(obs)
        recording_frames = []

        for _ in range(NUM_RECORDING_FRAMES):
            obs, _, terminated, truncated, info = env.step(0)  # NOOP
            ball_x, ball_y = get_ball_coords_from_ram(obs)
            current_lives = get_lives_from_ram(obs)

            if ball_x != 0:
                last_known_ball_x = ball_x
            
            if ball_x == 0 or current_lives < initial_lives or terminated or truncated:
                break

            recording_frames.append(env.render())
            
        # --- Ground Truth Collection Phase ---
        while not (terminated or truncated):
            obs, _, terminated, truncated, info = env.step(0)  # NOOP
            ball_x, _ = get_ball_coords_from_ram(obs)
            if ball_x != 0:
                last_known_ball_x = ball_x

        ground_truth_x = last_known_ball_x

        # --- Video Saving ---
        all_frames = video_frames + recording_frames
        print(f"  Total frames recorded: {len(all_frames)}")
        
        original_video_path = condition_dir / f"sample_{samples_collected:03d}_original.mp4"
        imageio.mimsave(original_video_path, all_frames, fps=60 // FRAME_SKIP, macro_block_size=1)
        
        if len(all_frames) > VIDEO_CLIP_FRAMES:
            clipped_frames = all_frames[:-VIDEO_CLIP_FRAMES]
        else:
            clipped_frames = all_frames

        if ground_truth_x != -1 and len(clipped_frames) >= MIN_VIDEO_FRAMES:
            print(f"  Successfully recorded sample {samples_collected:03d}. Ground truth X: {ground_truth_x}")
            video_path = condition_dir / f"sample_{samples_collected:03d}.mp4"
            imageio.mimsave(video_path, clipped_frames, fps=60 // FRAME_SKIP, macro_block_size=1)

            with open(condition_dir / f"sample_{samples_collected:03d}_ground_truth.txt", "w") as f:
                f.write(str(ground_truth_x))
            
            samples_collected += 1
        else:
            print(f"  Discarding invalid sample. Retrying...")

    env.close()

if __name__ == "__main__":
    OUTPUT_DIR.mkdir(exist_ok=True)
    for condition in PADDLE_POSITIONS.keys():
        run_experiment(condition, NUM_SAMPLES)