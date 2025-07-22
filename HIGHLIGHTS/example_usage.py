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
            if key == 'trace_seeds' and isinstance(value, list):
                # Handle list of seeds
                cmd.extend([f"--{key.replace('_', '-')}", ','.join(map(str, value))])
            elif isinstance(value, bool) and value:
                # Handle boolean flags
                cmd.append(f"--{key.replace('_', '-')}")
            elif not isinstance(value, bool):
                # Handle other parameters
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

def compare_methods_example():
    """Example of comparing different methods on the same game scenarios"""
    print("\n" + "="*60)
    print("FAIR METHOD COMPARISON EXAMPLE")
    print("="*60)
    
    # Configuration for fair comparison
    base_config = {
        "env": "BreakoutNoFrameskip-v4",
        "n_traces": 5,
        "num_highlights": 3,
        "trajectory_length": 10,
        "base_seed": 42,
        "seed_mode": "sequential",  # Use sequential seeds for reproducible scenarios
        "deterministic_eval": True,  # Use deterministic policy evaluation
        "save_seeds": True  # Save seed information for reproducibility
    }
    
    # Methods to compare
    methods = ["dqn", "qrdqn", "ddqn"]
    
    print("Comparing methods on the same game scenarios:")
    print(f"Base seed: {base_config['base_seed']}")
    print(f"Seed mode: {base_config['seed_mode']}")
    print(f"Methods: {methods}")
    print(f"Traces per method: {base_config['n_traces']}")
    
    for method in methods:
        print(f"\nAnalyzing {method.upper()}...")
        config = base_config.copy()
        config["algo"] = method
        
        success = run_highlights_analysis(**config)
        if not success:
            print(f"Failed to analyze {method}")
            continue
        
        print(f"{method.upper()} analysis completed!")
    
    print("\n" + "="*60)
    print("COMPARISON COMPLETE")
    print("="*60)
    print("All methods were evaluated on the same game scenarios using:")
    print(f"  Sequential seeds starting from {base_config['base_seed']}")
    print(f"  Deterministic policy evaluation")
    print("Check highlights/results/ for individual method results")
    print("Seed information is saved in metadata.json for reproducibility")

def specific_scenarios_example():
    """Example of using specific predefined scenarios"""
    print("\n" + "="*60)
    print("SPECIFIC SCENARIOS EXAMPLE")
    print("="*60)
    
    # Define specific scenarios (seeds) that are interesting
    interesting_seeds = [123, 456, 789, 101112, 131415]
    
    config = {
        "env": "BreakoutNoFrameskip-v4",
        "algo": "dqn",
        "seed_mode": "trace_specific",
        "trace_seeds": interesting_seeds,
        "n_traces": len(interesting_seeds),
        "num_highlights": 3,
        "deterministic_eval": True,
        "save_seeds": True
    }
    
    print("Using predefined interesting scenarios:")
    print(f"Scenario seeds: {interesting_seeds}")
    
    success = run_highlights_analysis(**config)
    
    if success:
        print("\nSpecific scenarios analysis completed!")
        print("To analyze other methods on the same scenarios, use:")
        print(f"  --seed-mode trace-specific")
        print(f"  --trace-seeds {','.join(map(str, interesting_seeds))}")

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
    
    # New seeding options
    parser.add_argument("--seed-mode", default="fixed",
                       choices=["fixed", "sequential", "random", "trace-specific"],
                       help="Seeding strategy")
    parser.add_argument("--base-seed", type=int, help="Base seed for seeding")
    parser.add_argument("--trace-seeds", type=str, help="Comma-separated trace seeds")
    parser.add_argument("--compare-methods", action="store_true",
                       help="Run method comparison example")
    parser.add_argument("--specific-scenarios", action="store_true",
                       help="Run specific scenarios example")
    
    args = parser.parse_args()
    
    # Handle special examples
    if args.compare_methods:
        compare_methods_example()
        return True
    
    if args.specific_scenarios:
        specific_scenarios_example()
        return True
    
    print("="*60)
    print("HIGHLIGHTS Example Usage")
    print("="*60)
    print(f"Environment: {args.env}")
    print(f"Algorithm: {args.algo}")
    print(f"Model folder: {args.folder}")
    print(f"Seed mode: {args.seed_mode}")
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
    
    # Prepare HIGHLIGHTS arguments
    highlights_kwargs = {
        "folder": args.folder,
        "n_traces": args.n_traces,
        "num_highlights": args.num_highlights,
        "trajectory_length": args.trajectory_length,
        "state_importance": args.state_importance,
        "seed": args.seed,
        "seed_mode": args.seed_mode,
        "deterministic_eval": True,
        "save_seeds": True
    }
    
    # Handle base seed
    if args.base_seed:
        highlights_kwargs["base_seed"] = args.base_seed
    
    # Handle trace seeds
    if args.trace_seeds:
        trace_seeds = [int(s.strip()) for s in args.trace_seeds.split(',')]
        highlights_kwargs["trace_seeds"] = trace_seeds
    
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
        print("\nFor method comparison, check metadata.json for reproduction commands")
        print("="*60)
    
    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 