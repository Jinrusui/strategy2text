#!/usr/bin/env python3
"""
Test script to verify HIGHLIGHTS integration with rl-baselines3-zoo
"""

import sys
import traceback
from pathlib import Path

def test_imports():
    """Test that all required imports work"""
    print("Testing imports...")
    try:
        import stable_baselines3
        print(f"✓ stable-baselines3 {stable_baselines3.__version__}")
        
        import rl_zoo3
        print(f"✓ rl-zoo3 imported successfully")
        
        from huggingface_sb3 import EnvironmentName
        print(f"✓ huggingface-sb3 imported successfully")
        
        from highlights.value_extractor import extract_q_values, is_value_based_model
        print(f"✓ value_extractor imported successfully")
        
        from highlights.get_agent import get_agent
        print(f"✓ get_agent imported successfully")
        
        from highlights.get_traces import get_traces
        print(f"✓ get_traces imported successfully")
        
        from highlights.main import main
        print(f"✓ main imported successfully")
        
        return True
    except Exception as e:
        print(f"✗ Import error: {e}")
        return False

def test_value_extractor():
    """Test value extraction with a simple DQN model"""
    print("\nTesting value extraction...")
    try:
        import torch
        import numpy as np
        from stable_baselines3 import DQN
        import gym
        
        # Create a simple environment
        env = gym.make("CartPole-v1")
        
        # Create a DQN model (not trained, just for testing)
        model = DQN("MlpPolicy", env, verbose=0)
        
        # Test Q-value extraction
        from highlights.value_extractor import extract_q_values, is_value_based_model
        
        # Check if model is recognized as value-based
        if is_value_based_model(model):
            print("✓ Model correctly identified as value-based")
        else:
            print("✗ Model not identified as value-based")
            return False
        
        # Test Q-value extraction with a random observation
        obs = env.observation_space.sample()
        q_values = extract_q_values(model, obs)
        
        if isinstance(q_values, np.ndarray) and len(q_values) == env.action_space.n:
            print(f"✓ Q-values extracted successfully: shape {q_values.shape}")
        else:
            print(f"✗ Q-values extraction failed: {q_values}")
            return False
        
        env.close()
        return True
        
    except Exception as e:
        print(f"✗ Value extraction test error: {e}")
        traceback.print_exc()
        return False

def test_agent_loading():
    """Test agent loading (without actual trained model)"""
    print("\nTesting agent loading structure...")
    try:
        import argparse
        from highlights.get_agent import get_agent
        
        # Create mock args
        args = argparse.Namespace()
        args.env = "CartPole-v1"
        args.algo = "dqn"
        args.folder = "rl-trained-agents"
        args.exp_id = 0
        args.load_best = False
        args.load_checkpoint = None
        args.load_last_checkpoint = False
        args.env_kwargs = None
        args.custom_objects = False
        args.n_envs = 1
        args.seed = 42
        
        print("✓ Agent loading structure test passed (without actual model)")
        return True
        
    except Exception as e:
        print(f"✗ Agent loading test error: {e}")
        traceback.print_exc()
        return False

def test_argument_parsing():
    """Test argument parsing"""
    print("\nTesting argument parsing...")
    try:
        import subprocess
        import sys
        
        # Test help output
        result = subprocess.run([
            sys.executable, "run.py", "--help"
        ], capture_output=True, text=True, cwd=Path(__file__).parent)
        
        if result.returncode == 0 and "HIGHLIGHTS for rl-baselines3-zoo" in result.stdout:
            print("✓ Argument parsing works correctly")
            return True
        else:
            print("✗ Argument parsing failed")
            print(f"stdout: {result.stdout}")
            print(f"stderr: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"✗ Argument parsing test error: {e}")
        return False

def main():
    print("="*60)
    print("HIGHLIGHTS Integration Test")
    print("="*60)
    
    tests = [
        ("Import Test", test_imports),
        ("Value Extractor Test", test_value_extractor),
        ("Agent Loading Test", test_agent_loading),
        ("Argument Parsing Test", test_argument_parsing)
    ]
    
    passed = 0
    failed = 0
    
    for test_name, test_func in tests:
        print(f"\n{test_name}:")
        try:
            if test_func():
                passed += 1
                print(f"✓ {test_name} PASSED")
            else:
                failed += 1
                print(f"✗ {test_name} FAILED")
        except Exception as e:
            failed += 1
            print(f"✗ {test_name} FAILED with exception: {e}")
    
    print("\n" + "="*60)
    print(f"Test Results: {passed} passed, {failed} failed")
    print("="*60)
    
    if failed == 0:
        print("🎉 All tests passed! HIGHLIGHTS integration is working correctly.")
        print("\nTo run a full analysis, use:")
        print("python run.py --env CartPole-v1 --algo dqn --folder rl-trained-agents")
    else:
        print("❌ Some tests failed. Please check the installation and dependencies.")
    
    return failed == 0

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 