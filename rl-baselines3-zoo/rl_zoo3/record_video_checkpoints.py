import argparse
import os
import re
import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np
import yaml
from huggingface_sb3 import EnvironmentName
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import VecVideoRecorder

from rl_zoo3.exp_manager import ExperimentManager
from rl_zoo3.utils import ALGOS, StoreDict, create_test_env, get_saved_hyperparams


def find_checkpoint_files(logs_dir: str, algo: str) -> List[Tuple[str, int, str]]:
    """
    Find all checkpoint files for a given algorithm in the logs directory.
    
    Args:
        logs_dir: Path to the logs directory
        algo: Algorithm name (e.g., 'dqn')
    
    Returns:
        List of tuples (checkpoint_path, steps, model_type) sorted by steps
        model_type is either 'checkpoint' or 'best'
    """
    algo_dir = os.path.join(logs_dir, algo)
    if not os.path.exists(algo_dir):
        raise FileNotFoundError(f"Algorithm directory not found: {algo_dir}")
    
    checkpoints = []
    
    # Look for experiment directories
    for exp_dir in os.listdir(algo_dir):
        exp_path = os.path.join(algo_dir, exp_dir)
        if not os.path.isdir(exp_path):
            continue
            
        # Look for checkpoint files in this experiment directory
        for file in os.listdir(exp_path):
            if file.startswith("rl_model_") and file.endswith("_steps.zip"):
                # Extract steps from filename
                match = re.search(r"rl_model_(\d+)_steps\.zip", file)
                if match:
                    steps = int(match.group(1))
                    checkpoint_path = os.path.join(exp_path, file)
                    checkpoints.append((checkpoint_path, steps, 'checkpoint'))
            elif file == "best_model.zip":
                # Add best model with a very high step number to ensure it's last
                checkpoint_path = os.path.join(exp_path, file)
                checkpoints.append((checkpoint_path, float('inf'), 'best'))
    
    # Sort by steps (best_model will be last due to inf steps)
    checkpoints.sort(key=lambda x: x[1])
    return checkpoints


def select_checkpoints_by_interval(checkpoints: List[Tuple[str, int, str]], interval: int) -> List[Tuple[str, int, str]]:
    """
    Select checkpoints based on interval, always including best_model.zip at the end.
    Only considers rl_model_*_steps.zip files for interval calculation.
    
    Args:
        checkpoints: List of (checkpoint_path, steps, model_type) tuples
        interval: Interval for selection (every nth checkpoint)
    
    Returns:
        Selected checkpoints including best_model.zip at the end
    """
    if interval <= 0:
        raise ValueError("Interval must be positive")
    
    # Separate regular checkpoints and best model
    rl_model_checkpoints = []
    best_model = None
    
    for checkpoint_path, steps, model_type in checkpoints:
        if model_type == 'best':
            best_model = (checkpoint_path, steps, model_type)
        elif model_type == 'checkpoint':
            rl_model_checkpoints.append((checkpoint_path, steps, model_type))
    
    # Select regular checkpoints by interval
    selected = []
    if interval == 1:
        selected = rl_model_checkpoints
    else:
        for i in range(0, len(rl_model_checkpoints), interval):
            selected.append(rl_model_checkpoints[i])
        
        # Always include the last regular checkpoint if not already included
        if len(rl_model_checkpoints) > 0 and rl_model_checkpoints[-1] not in selected:
            selected.append(rl_model_checkpoints[-1])
    
    # Always add best_model at the end if it exists
    if best_model is not None:
        selected.append(best_model)
    
    return selected


def get_env_name_from_checkpoint_path(checkpoint_path: str) -> str:
    """
    Extract environment name from checkpoint path.
    
    Args:
        checkpoint_path: Path to checkpoint file
    
    Returns:
        Environment name
    """
    # Extract from path structure: logs/algo/env_name_exp/checkpoint.zip
    path_parts = Path(checkpoint_path).parts
    for part in path_parts:
        if "_" in part and not part.startswith("rl_model_"):
            # Remove experiment number suffix (e.g., BreakoutNoFrameskip-v4_1 -> BreakoutNoFrameskip-v4)
            env_name = re.sub(r"_\d+$", "", part)
            return env_name
    
    raise ValueError(f"Could not extract environment name from path: {checkpoint_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Record videos from multiple checkpoints")
    parser.add_argument("--env", help="Environment ID (optional, will auto-detect from checkpoints if not provided)", 
                        type=EnvironmentName, default=EnvironmentName("BreakoutNoFrameskip-v4"))
    parser.add_argument("--algo", help="RL Algorithm", type=str, choices=list(ALGOS.keys()), default="dqn")
    parser.add_argument("--logs-dir", help="Logs directory", type=str, default="logs")
    parser.add_argument("-o", "--output-folder", help="Output folder", type=str, default="checkpoint_videos2")
    parser.add_argument("--num-videos-per-cpt", help="Number of videos per checkpoint (corresponds to different seeds)", 
                        default=5, type=int)
    parser.add_argument("--checkpoint-interval", help="Interval for selecting checkpoints (every nth checkpoint)", 
                        default=20, type=int)
    parser.add_argument("-n", "--n-timesteps", help="Number of timesteps per video", default=2000, type=int)
    parser.add_argument("--n-envs", help="Number of environments", default=1, type=int)
    parser.add_argument("--deterministic", action="store_true", default=False, help="Use deterministic actions")
    parser.add_argument("--stochastic", action="store_true", default=False, help="Use stochastic actions")
    parser.add_argument(
        "--no-render", action="store_true", default=False, help="Do not render the environment (useful for tests)"
    )
    parser.add_argument(
        "--env-kwargs", type=str, nargs="+", action=StoreDict, help="Optional keyword argument to pass to the env constructor"
    )
    parser.add_argument(
        "--custom-objects", action="store_true", default=False, help="Use custom objects to solve loading issues"
    )

    args = parser.parse_args()

    algo = args.algo
    logs_dir = args.logs_dir
    output_folder = args.output_folder
    num_videos_per_cpt = args.num_videos_per_cpt
    checkpoint_interval = args.checkpoint_interval
    video_length = args.n_timesteps
    n_envs = args.n_envs
    specified_env = args.env

    print(f"Looking for {algo} checkpoints in {logs_dir}")
    
    # Find all checkpoints for the algorithm
    try:
        all_checkpoints = find_checkpoint_files(logs_dir, algo)
        if not all_checkpoints:
            print(f"No checkpoints found for algorithm '{algo}' in '{logs_dir}'")
            sys.exit(1)
        
        print(f"Found {len(all_checkpoints)} total checkpoints")
        
        # Select checkpoints based on interval
        selected_checkpoints = select_checkpoints_by_interval(all_checkpoints, checkpoint_interval)
        print(f"Selected {len(selected_checkpoints)} checkpoints with interval {checkpoint_interval}")
        
        for checkpoint_path, steps, model_type in selected_checkpoints:
            if steps == float('inf'):
                print(f"  - BEST MODEL: {checkpoint_path}")
            else:
                print(f"  - {steps:,} steps: {checkpoint_path}")
        
    except Exception as e:
        print(f"Error finding checkpoints: {e}")
        sys.exit(1)

    # Get environment name
    if specified_env is not None:
        env_name = specified_env
        env_name_str = env_name.gym_id
        print(f"\nUsing specified environment: {env_name}")
    else:
        # Auto-detect environment from checkpoint path
        env_name_str = get_env_name_from_checkpoint_path(selected_checkpoints[0][0])
        env_name = EnvironmentName(env_name_str)
        print(f"\nAuto-detected environment: {env_name}")
    
    print(f"Algorithm: {algo}")
    print(f"Videos per checkpoint: {num_videos_per_cpt}")
    print(f"Video length: {video_length} timesteps")

    # Create output directory
    os.makedirs(output_folder, exist_ok=True)

    # Process each selected checkpoint
    for checkpoint_idx, (checkpoint_path, steps, model_type) in enumerate(selected_checkpoints):
        print(f"\n{'='*60}")
        print(f"Processing checkpoint {checkpoint_idx + 1}/{len(selected_checkpoints)}")
        if model_type == 'best':
            print(f"Model: BEST MODEL")
            steps_display = "BEST"
        else:
            print(f"Steps: {steps:,}")
            steps_display = f"{steps:08d}"
        print(f"Path: {checkpoint_path}")
        print(f"{'='*60}")

        # Get log path for this checkpoint
        log_path = os.path.dirname(checkpoint_path)
        
        # Load hyperparameters
        stats_path = os.path.join(log_path, env_name_str)
        hyperparams, maybe_stats_path = get_saved_hyperparams(stats_path)

        # Load env_kwargs if existing
        env_kwargs = {}
        args_path = os.path.join(log_path, env_name_str, "args.yml")
        if os.path.isfile(args_path):
            with open(args_path) as f:
                loaded_args = yaml.load(f, Loader=yaml.UnsafeLoader)
                if loaded_args["env_kwargs"] is not None:
                    env_kwargs = loaded_args["env_kwargs"]
        
        # Overwrite with command line arguments
        if args.env_kwargs is not None:
            env_kwargs.update(args.env_kwargs)

        # Force rgb_array rendering (gym 0.26+)
        env_kwargs.update(render_mode="rgb_array")

        is_atari = ExperimentManager.is_atari(env_name.gym_id)
        is_minigrid = ExperimentManager.is_minigrid(env_name.gym_id)

        off_policy_algos = ["qrdqn", "dqn", "ddpg", "sac", "her", "td3", "tqc"]

        # Determine stochastic behavior
        stochastic = args.stochastic or ((is_atari or is_minigrid) and not args.deterministic)
        deterministic = not stochastic

        # Prepare model loading kwargs and custom objects once per checkpoint
        off_policy_kwargs = dict(buffer_size=1)
        if "optimize_memory_usage" in hyperparams:
            off_policy_kwargs["optimize_memory_usage"] = False

        # Custom objects for compatibility
        newer_python_version = sys.version_info.major == 3 and sys.version_info.minor >= 8
        custom_objects = {}
        if newer_python_version or args.custom_objects:
            custom_objects = {
                "learning_rate": 0.0,
                "lr_schedule": lambda _: 0.0,
                "clip_range": lambda _: 0.0,
            }
        
        # Add LinearSchedule for DQN compatibility
        try:
            from stable_baselines3.common.utils import LinearSchedule
            custom_objects["LinearSchedule"] = LinearSchedule
        except ImportError:
            try:
                from stable_baselines3.dqn.policies import LinearSchedule
                custom_objects["LinearSchedule"] = LinearSchedule
            except ImportError:
                # Create a dummy LinearSchedule if not available
                class DummyLinearSchedule:
                    def __init__(self, *args, **kwargs):
                        pass
                    def __call__(self, *args, **kwargs):
                        return 0.0
                custom_objects["LinearSchedule"] = DummyLinearSchedule

        # Track scores for this checkpoint
        checkpoint_scores = []
        
        # Generate videos with different seeds
        for seed_idx in range(num_videos_per_cpt):
            seed = seed_idx
            print(f"\n--- Recording video {seed_idx + 1}/{num_videos_per_cpt} (seed={seed}) ---")
            
            set_random_seed(seed)

            # Reload hyperparameters for each video to ensure freshness
            video_stats_path = os.path.join(log_path, env_name_str)
            video_hyperparams, video_maybe_stats_path = get_saved_hyperparams(video_stats_path)

            # Reload env_kwargs for each video
            video_env_kwargs = {}
            video_args_path = os.path.join(log_path, env_name_str, "args.yml")
            if os.path.isfile(video_args_path):
                with open(video_args_path) as f:
                    loaded_args = yaml.load(f, Loader=yaml.UnsafeLoader)
                    if loaded_args["env_kwargs"] is not None:
                        video_env_kwargs = loaded_args["env_kwargs"]
            
            # Overwrite with command line arguments
            if args.env_kwargs is not None:
                video_env_kwargs.update(args.env_kwargs)

            # Force rgb_array rendering (gym 0.26+)
            video_env_kwargs.update(render_mode="rgb_array")

            # Create a completely fresh environment for each video
            env = create_test_env(
                env_name.gym_id,
                n_envs=n_envs,
                stats_path=video_maybe_stats_path,
                seed=seed,
                log_dir=None,
                should_render=not args.no_render,
                hyperparams=video_hyperparams,
                env_kwargs=video_env_kwargs,
            )

            # Prepare model loading kwargs for this specific seed
            kwargs = dict(seed=seed)
            if algo in off_policy_algos:
                kwargs.update(off_policy_kwargs)

            # Load model
            print(f"Loading checkpoint: {checkpoint_path}")
            model = ALGOS[algo].load(checkpoint_path, env=env, custom_objects=custom_objects, **kwargs)

            # Setup video recording
            video_folder = os.path.join(output_folder, f"steps_{steps_display}")
            os.makedirs(video_folder, exist_ok=True)
            
            if model_type == 'best':
                name_prefix = f"{algo}_{env_name_str}_best_model_seed_{seed}"
            else:
                name_prefix = f"{algo}_{env_name_str}_steps_{steps}_seed_{seed}"

            env = VecVideoRecorder(
                env,
                video_folder,
                record_video_trigger=lambda x: x == 0,
                video_length=video_length,
                name_prefix=name_prefix,
            )

            # Record video and track score
            obs = env.reset()
            lstm_states = None
            episode_starts = np.ones((env.num_envs,), dtype=bool)
            
            # For Atari games, add some deterministic no-op actions based on seed for reproducible variation
            if "NoFrameskip" in env_name_str:
                # Use seed to determine number of no-ops for reproducibility
                rng = np.random.RandomState(seed)  # Create separate RNG with the same seed
                n_noops = rng.randint(0, 31)
                print(f"    Adding {n_noops} no-op actions for variation (seed={seed})")
                for _ in range(n_noops):
                    obs, _, done, _ = env.step(np.array([0]))  # 0 is typically the no-op action in Atari
                    if done[0]:
                        obs = env.reset()
                        break
            
            try:
                # Track the true game score like in record_video_score.py
                true_game_score = 0
                
                for step_idx in range(video_length):
                    action, lstm_states = model.predict(
                        obs,
                        state=lstm_states,
                        episode_start=episode_starts,
                        deterministic=deterministic,
                    )
                    
                    if not args.no_render:
                        env.render()
                    
                    obs, rewards, dones, infos = env.step(action)
                    # The reward is the un-clipped, real reward from the game
                    true_game_score += float(rewards[0])
                    
                    if dones[0]:
                        # Check if this is a true game over (all lives lost) or just a life lost
                        if len(infos) > 0 and 'lives' in infos[0]:
                            lives_remaining = infos[0]['lives']
                            print(f"    Step {step_idx}: Lives remaining: {lives_remaining}, Current total score: {true_game_score:.2f}")
                            
                            # Only break if no lives remaining (true game over)
                            if lives_remaining == 0:
                                print(f"    TRUE GAME OVER. Final game score: {true_game_score:.2f}")
                                break
                            else:
                                print(f"    Life lost, continuing game. Lives left: {lives_remaining}")
                                # Reset episode_starts for the next life
                                episode_starts = np.ones((env.num_envs,), dtype=bool)
                                # Continue without resetting the environment - it should handle life transitions automatically
                        else:
                            # If no life info available, treat any done as game over
                            print(f"    GAME OVER. Final game score: {true_game_score:.2f}")
                            break
                    
                    episode_starts = dones
                
                # Record the final score for this video
                checkpoint_scores.append(true_game_score)
                print(f"    Video {seed_idx + 1} completed:")
                print(f"      - Final game score: {true_game_score:.2f}")
                print(f"      - Total timesteps: {step_idx + 1 if 'step_idx' in locals() else video_length}")
                
            except KeyboardInterrupt:
                print("    Recording interrupted by user")
                break
            finally:
                env.close()

        # Calculate and report comprehensive statistics for this checkpoint
        if checkpoint_scores:
            checkpoint_avg_score = np.mean(checkpoint_scores)
            checkpoint_std_score = np.std(checkpoint_scores)
            checkpoint_max_score = np.max(checkpoint_scores)
            checkpoint_min_score = np.min(checkpoint_scores)
            if model_type == 'best':
                print(f"\n--- BEST MODEL summary ---")
            else:
                print(f"\n--- Checkpoint {steps:,} steps summary ---")
            print(f"Videos analyzed: {len(checkpoint_scores)}")
            print(f"Best game score across all videos: {checkpoint_max_score:.2f}")
            print(f"Average game score across all videos: {checkpoint_avg_score:.2f} ± {checkpoint_std_score:.2f}")
            print(f"Worst game score from any video: {checkpoint_min_score:.2f}")
            print(f"Individual video scores: {[f'{score:.2f}' for score in checkpoint_scores]}")
            print(f"Score range: {checkpoint_min_score:.2f} - {checkpoint_max_score:.2f}")
        else:
            if model_type == 'best':
                print(f"\n--- BEST MODEL summary ---")
            else:
                print(f"\n--- Checkpoint {steps:,} steps summary ---")
            print("No complete games recorded for scoring")
            print("This may indicate the agent performs very poorly at this checkpoint")

        # Check if user wants to stop
        try:
            pass
        except KeyboardInterrupt:
            print("\nStopping checkpoint processing...")
            break

    print(f"\nAll videos saved to: {output_folder}")
    print("Done!")
