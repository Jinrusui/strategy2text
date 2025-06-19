#!/usr/bin/env python3
"""
Download pre-trained Breakout agents from RL Baselines3 Zoo.

This script downloads PPO and DQN models specifically trained for Breakout.
Based on RL Baselines3 Zoo: https://github.com/DLR-RM/rl-baselines3-zoo
"""

import os
import sys
import requests
from pathlib import Path
from typing import List, Tuple
import argparse


# Breakout-specific models from RL Zoo
BREAKOUT_MODELS = {
    'ppo': 'BreakoutNoFrameskip-v4',
    'dqn': 'BreakoutNoFrameskip-v4',
    'a2c': 'BreakoutNoFrameskip-v4',
    'qr-dqn': 'BreakoutNoFrameskip-v4',
}


def download_breakout_model(algorithm: str, output_dir: str = "rl-trained-agents") -> bool:
    """
    Download a specific Breakout model from RL Baselines3 Zoo.
    
    Args:
        algorithm: Algorithm name (ppo, dqn, a2c, qr-dqn)
        output_dir: Output directory
        
    Returns:
        True if successful, False otherwise
    """
    if algorithm not in BREAKOUT_MODELS:
        print(f"Unsupported algorithm: {algorithm}")
        print(f"Available algorithms: {list(BREAKOUT_MODELS.keys())}")
        return False
    
    env_id = BREAKOUT_MODELS[algorithm]
    
    # GitHub raw URL for the model
    base_url = "https://github.com/DLR-RM/rl-trained-agents/raw/master"
    model_url = f"{base_url}/{algorithm}/{env_id}_1/{env_id}.zip"
    
    # Create output directory structure
    output_path = Path(output_dir) / algorithm / f"{env_id}_1"
    output_path.mkdir(parents=True, exist_ok=True)
    
    model_file = output_path / f"{env_id}.zip"
    
    if model_file.exists():
        print(f"✓ Model already exists: {model_file}")
        return True
    
    try:
        print(f"Downloading {algorithm.upper()} model for Breakout...")
        print(f"URL: {model_url}")
        
        response = requests.get(model_url, stream=True)
        response.raise_for_status()
        
        # Get file size for progress
        total_size = int(response.headers.get('content-length', 0))
        
        with open(model_file, 'wb') as f:
            downloaded = 0
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
                downloaded += len(chunk)
                if total_size > 0:
                    percent = (downloaded / total_size) * 100
                    print(f"\rProgress: {percent:.1f}%", end='', flush=True)
        
        print(f"\n✓ Downloaded: {model_file}")
        print(f"  Size: {model_file.stat().st_size / (1024*1024):.1f} MB")
        
        return True
        
    except Exception as e:
        print(f"\n✗ Failed to download {algorithm} Breakout model: {e}")
        if model_file.exists():
            model_file.unlink()  # Remove partial download
        return False


def download_all_breakout_models(output_dir: str = "rl-trained-agents") -> int:
    """
    Download all available Breakout models.
    
    Args:
        output_dir: Output directory
        
    Returns:
        Number of successfully downloaded models
    """
    print("Downloading all Breakout models from RL Baselines3 Zoo...")
    print("=" * 60)
    
    successful = 0
    total = len(BREAKOUT_MODELS)
    
    for algorithm in BREAKOUT_MODELS.keys():
        print(f"\n[{successful + 1}/{total}] Processing {algorithm.upper()}...")
        if download_breakout_model(algorithm, output_dir):
            successful += 1
        print("-" * 40)
    
    print(f"\n✓ Successfully downloaded {successful}/{total} Breakout models")
    
    if successful > 0:
        print(f"\nModels saved to: {Path(output_dir).absolute()}")
        print("\nTo use a model:")
        print("  python scripts/demo_breakout.py --algorithm ppo")
        print("  python scripts/demo_breakout.py --algorithm dqn")
    
    return successful


def verify_downloads(output_dir: str = "rl-trained-agents") -> None:
    """Verify that downloaded models exist and are valid."""
    print("Verifying downloaded models...")
    
    for algorithm, env_id in BREAKOUT_MODELS.items():
        model_path = Path(output_dir) / algorithm / f"{env_id}_1" / f"{env_id}.zip"
        
        if model_path.exists():
            size_mb = model_path.stat().st_size / (1024*1024)
            print(f"✓ {algorithm.upper()}: {model_path} ({size_mb:.1f} MB)")
        else:
            print(f"✗ {algorithm.upper()}: Missing")


def main():
    parser = argparse.ArgumentParser(
        description="Download pre-trained Breakout agents from RL Baselines3 Zoo",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Download all Breakout models
  python scripts/download_rl_zoo_models.py --all
  
  # Download specific algorithm
  python scripts/download_rl_zoo_models.py --algorithm ppo
  python scripts/download_rl_zoo_models.py --algorithm dqn
  
  # Verify downloads
  python scripts/download_rl_zoo_models.py --verify
        """
    )
    
    parser.add_argument("--all", action="store_true", 
                       help="Download all Breakout models (PPO, DQN, A2C, QR-DQN)")
    parser.add_argument("--algorithm", "-a", choices=list(BREAKOUT_MODELS.keys()),
                       help="Download specific algorithm")
    parser.add_argument("--output-dir", "-o", default="rl-trained-agents", 
                       help="Output directory (default: rl-trained-agents)")
    parser.add_argument("--verify", action="store_true",
                       help="Verify downloaded models")
    
    args = parser.parse_args()
    
    if args.verify:
        verify_downloads(args.output_dir)
    elif args.all:
        download_all_breakout_models(args.output_dir)
    elif args.algorithm:
        download_breakout_model(args.algorithm, args.output_dir)
    else:
        # Default behavior: download PPO and DQN (most common)
        print("No specific option provided. Downloading PPO and DQN models...")
        success_count = 0
        for alg in ['ppo', 'dqn']:
            if download_breakout_model(alg, args.output_dir):
                success_count += 1
        
        print(f"\n✓ Downloaded {success_count}/2 default models")
        if success_count > 0:
            print("\nTo run a demo:")
            print("  python scripts/demo_breakout.py")


if __name__ == "__main__":
    main() 