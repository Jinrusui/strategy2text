import argparse
import os
import sys
import time

import numpy as np
import yaml
import gymnasium as gym
from huggingface_sb3 import EnvironmentName
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import VecVideoRecorder, DummyVecEnv

from rl_zoo3.exp_manager import ExperimentManager
from rl_zoo3.utils import ALGOS, StoreDict, create_test_env, get_model_path, get_saved_hyperparams

# This is the key to getting the correct score
from stable_baselines3.common.atari_wrappers import AtariWrapper


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", help="Environment ID", type=EnvironmentName, default="BreakoutNoFrameskip-v4")
    parser.add_argument("-f", "--folder", help="Log folder", type=str, default="rl-trained-agents")
    parser.add_argument("-o", "--output-folder", help="Output folder", type=str)
    parser.add_argument("--algo", help="RL Algorithm", default="ppo", type=str, required=False, choices=list(ALGOS.keys()))
    parser.add_argument("-n", "--n-timesteps", help="Number of timesteps", default=5000, type=int)
    parser.add_argument("--n-envs", help="Number of environments", default=1, type=int)
    parser.add_argument("--deterministic", action="store_true", default=False, help="Use deterministic actions")
    parser.add_argument("--stochastic", action="store_true", default=False, help="Use stochastic actions")
    parser.add_argument("--seed", help="Random generator seed", type=int, default=0)
    parser.add_argument("--no-render", action="store_true", default=False, help="Do not render the environment")
    parser.add_argument("--exp-id", help="Experiment ID", default=0, type=int)
    parser.add_argument("--load-best", action="store_true", default=False, help="Load best model")
    parser.add_argument("--load-checkpoint", type=int, help="Load checkpoint")
    parser.add_argument("--load-last-checkpoint", action="store_true", default=False, help="Load last checkpoint")
    parser.add_argument("--env-kwargs", type=str, nargs="+", action=StoreDict, help="Env kwargs")
    parser.add_argument("--custom-objects", action="store_true", default=False, help="Use custom objects")

    args = parser.parse_args()

    env_name: EnvironmentName = args.env
    algo = args.algo
    folder = args.folder
    video_folder = args.output_folder
    seed = args.seed
    video_length = args.n_timesteps

    name_prefix, model_path, log_path = get_model_path(
        args.exp_id, folder, algo, env_name, args.load_best, args.load_checkpoint, args.load_last_checkpoint
    )

    print(f"Loading {model_path}")
    print(f"Using seed: {args.seed}")
    set_random_seed(args.seed)

    stats_path = os.path.join(log_path, env_name)
    hyperparams, maybe_stats_path = get_saved_hyperparams(stats_path, test_mode=True)
    
    env_kwargs = {}
    args_path = os.path.join(log_path, env_name, "args.yml")
    if os.path.isfile(args_path):
        with open(args_path) as f:
            loaded_args = yaml.load(f, Loader=yaml.UnsafeLoader)
            if loaded_args["env_kwargs"] is not None:
                env_kwargs = loaded_args["env_kwargs"]
    if args.env_kwargs is not None:
        env_kwargs.update(args.env_kwargs)
        
    # Force rendering for video recording
    env_kwargs["render_mode"] = "rgb_array"

    # Modify hyperparams to use AtariWrapper with clip_reward=False for true scoring
    # This preserves the original game score while keeping the same environment structure
    if "env_wrapper" in hyperparams and hyperparams["env_wrapper"]:
        # Replace the standard AtariWrapper with one that doesn't clip rewards
        # and ensure we don't treat each life as a separate episode
        hyperparams = hyperparams.copy()
        hyperparams["env_wrapper"] = [
            {"stable_baselines3.common.atari_wrappers.AtariWrapper": {
                "clip_reward": False,
                "terminal_on_life_loss": False  # Don't treat each life as a separate episode
            }}
        ]

    # Use the create_test_env function to properly set up the environment
    # This ensures we get the same environment configuration as during training
    env = create_test_env(
        str(env_name),
        n_envs=1,
        stats_path=maybe_stats_path,
        seed=seed,
        log_dir=None,
        should_render=False,  # We'll handle rendering through VecVideoRecorder
        hyperparams=hyperparams,
        env_kwargs=env_kwargs,
    )

    # Wrap it for video recording
    env = VecVideoRecorder(
        env,
        video_folder,
        record_video_trigger=lambda x: x == 0,
        video_length=video_length,
        name_prefix=name_prefix,
    )
    
    # Ensure environment is properly seeded for different episodes
    env.seed(seed)

    custom_objects = {}
    if args.custom_objects:
        custom_objects = {"learning_rate": 0.0, "lr_schedule": lambda _: 0.0, "clip_range": lambda _: 0.0}

    model = ALGOS[algo].load(model_path, env=env, custom_objects=custom_objects)

    # Reset environment to ensure different starting states
    obs = env.reset()
    lstm_states = None
    episode_starts = np.ones((env.num_envs,), dtype=bool)
    
    # For Atari games, add some random no-op actions at the start to create variation
    if "NoFrameskip" in str(env_name):
        # Add 0-30 random no-op actions at the start based on seed
        np.random.seed(seed)
        n_noops = np.random.randint(0, 31)
        print(f"Adding {n_noops} no-op actions for variation")
        for _ in range(n_noops):
            obs, _, done, _ = env.step(np.array([0]))  # 0 is typically the no-op action in Atari
            if done[0]:
                obs = env.reset()
                break
    
    # Manually track the score
    true_game_score = 0
    
    try:
        for step in range(video_length):
            action, lstm_states = model.predict(
                obs, state=lstm_states, episode_start=episode_starts, deterministic=args.deterministic
            )
            obs, reward, done, infos = env.step(action)
            # The reward is now the un-clipped, real reward from the game
            true_game_score += float(reward[0])

            # Check if this is a true game over (all lives lost) or just a life lost
            if done[0]:
                # Check if we have life information in infos
                if len(infos) > 0 and 'lives' in infos[0]:
                    lives_remaining = infos[0]['lives']
                    print(f"Episode done at step {step}. Lives remaining: {lives_remaining}, Score: {true_game_score}")
                    
                    # Only break if no lives remaining (true game over)
                    if lives_remaining == 0:
                        print(f"TRUE GAME OVER. Final Board Score: {true_game_score}")
                        break
                    else:
                        print(f"Life lost, continuing game. Lives left: {lives_remaining}")
                        # Reset episode_starts for the next life
                        episode_starts = np.ones((env.num_envs,), dtype=bool)
                        # Continue without resetting the environment - it should handle life transitions automatically
                else:
                    # If no life info available, treat any done as game over
                    print(f"GAME OVER. Final Board Score: {true_game_score}")
                    break
    
    except KeyboardInterrupt:
        pass
    
    # This is the line that sample_videos_for_hva.py will parse
    print(f"Total reward: {true_game_score}")

    env.close()


if __name__ == "__main__":
    main()