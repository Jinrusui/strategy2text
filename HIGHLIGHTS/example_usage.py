#!/usr/bin/env python3
"""
Example usage of HIGHLIGHTS with rl-baselines3-zoo

This script demonstrates how to use the updated HIGHLIGHTS implementation
to analyze a trained DQN agent from rl-baselines3-zoo.
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

def check_model_exists(folder, env, algo, exp_id=0):
    """Check if a trained model exists"""
    model_path = Path(folder) / f"{algo}" / f"{env}_{exp_id}"
    return model_path.exists()

def train_example_model(env="CartPole-v1", algo="dqn", timesteps=50000):
    """Train an example model if none exists"""
    print(f"Training {algo} on {env} for {timesteps} timesteps...")
    try:
        subprocess.run([
            sys.executable, "-m", "rl_zoo3.train",
            "--algo", algo,
            "--env", env,
            "--total-timesteps", str(timesteps),
            "--save-freq", "1000",
            "--verbose", "1"
        ], check=True)
        print(f"Model trained successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error training model: {e}")
        return False

def run_highlights_analysis(env="CartPole-v1", algo="dqn", **kwargs):
    """Run HIGHLIGHTS analysis on a trained model"""
    cmd = [
        sys.executable, "run.py",
        "--env", env,
        "--algo", algo,
        "--verbose"
    ]
    
    # Add optional parameters
    for key, value in kwargs.items():
        if value is not None:
            cmd.extend([f"--{key.replace('_', '-')}", str(value)])
    
    print(f"Running HIGHLIGHTS analysis...")
    print(f"Command: {' '.join(cmd)}")
    
    try:
        subprocess.run(cmd, check=True)
        print("HIGHLIGHTS analysis completed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error running HIGHLIGHTS: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Example HIGHLIGHTS usage")
    parser.add_argument("--env", default="CartPole-v1", help="Environment ID")
    parser.add_argument("--algo", default="dqn", help="Algorithm")
    parser.add_argument("--folder", default="rl-trained-agents", help="Model folder")
    parser.add_argument("--train-if-missing", action="store_true", 
                       help="Train model if it doesn't exist")
    parser.add_argument("--training-timesteps", type=int, default=50000,
                       help="Training timesteps if training")
    parser.add_argument("--n-traces", type=int, default=10,
                       help="Number of traces to collect")
    parser.add_argument("--num-highlights", type=int, default=5,
                       help="Number of highlights to generate")
    parser.add_argument("--trajectory-length", type=int, default=10,
                       help="Length of highlight trajectories")
    parser.add_argument("--state-importance", default="second",
                       choices=["second", "worst", "variance"],
                       help="State importance method")
    parser.add_argument("--highlights-div", action="store_true",
                       help="Use diversity-based selection")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    args = parser.parse_args()
    
    print("="*60)
    print("HIGHLIGHTS Example Usage")
    print("="*60)
    print(f"Environment: {args.env}")
    print(f"Algorithm: {args.algo}")
    print(f"Model folder: {args.folder}")
    print("="*60)
    
    # Check if model exists
    if not check_model_exists(args.folder, args.env, args.algo):
        print(f"Model not found in {args.folder}")
        if args.train_if_missing:
            print("Training model...")
            if not train_example_model(args.env, args.algo, args.training_timesteps):
                print("Failed to train model. Exiting.")
                return False
        else:
            print("Use --train-if-missing to train a model automatically")
            print("Or train manually with:")
            print(f"python -m rl_zoo3.train --algo {args.algo} --env {args.env}")
            return False
    
    # Run HIGHLIGHTS analysis
    highlights_kwargs = {
        "folder": args.folder,
        "n_traces": args.n_traces,
        "num_highlights": args.num_highlights,
        "trajectory_length": args.trajectory_length,
        "state_importance": args.state_importance,
        "seed": args.seed
    }
    
    if args.highlights_div:
        highlights_kwargs["highlights_div"] = True
    
    success = run_highlights_analysis(args.env, args.algo, **highlights_kwargs)
    
    if success:
        print("\n" + "="*60)
        print("HIGHLIGHTS Analysis Complete!")
        print("="*60)
        print("Check the highlights/results/ directory for:")
        print("- Highlight videos (HL_*.mp4)")
        print("- Trace data (Traces.pkl)")
        print("- State data (States.pkl)")
        print("- Trajectory data (Trajectories.pkl)")
        print("- Run metadata (metadata.json)")
        print("="*60)
    
    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 