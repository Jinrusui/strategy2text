#!/usr/bin/env python3
"""
Installation Test Script
========================

This script tests that all required dependencies are installed correctly
and that the basic functionality works.

Usage:
    python test_installation.py
"""

import sys
import os
from pathlib import Path

def test_imports():
    """Test that all required packages can be imported."""
    print("🔍 Testing package imports...")
    
    required_packages = [
        ('torch', 'PyTorch'),
        ('torchvision', 'TorchVision'),
        ('numpy', 'NumPy'),
        ('matplotlib', 'Matplotlib'),
        ('cv2', 'OpenCV'),
        ('gymnasium', 'Gymnasium'),
        ('stable_baselines3', 'Stable Baselines3'),
        ('pytorch_grad_cam', 'PyTorch GradCAM'),
        ('yaml', 'PyYAML'),
        ('PIL', 'Pillow'),
    ]
    
    failed_imports = []
    
    for package, name in required_packages:
        try:
            __import__(package)
            print(f"✅ {name}")
        except ImportError as e:
            print(f"❌ {name}: {e}")
            failed_imports.append(name)
    
    # Test optional packages
    optional_packages = [
        ('sb3_contrib', 'SB3 Contrib'),
        ('ale_py', 'ALE Python Interface'),
    ]
    
    print("\n🔍 Testing optional packages...")
    for package, name in optional_packages:
        try:
            __import__(package)
            print(f"✅ {name}")
        except ImportError:
            print(f"⚠️  {name} (optional)")
    
    return failed_imports

def test_atari_env():
    """Test that Atari environments can be created."""
    print("\n🎮 Testing Atari environment creation...")
    
    try:
        import gymnasium as gym
        from stable_baselines3.common.atari_wrappers import AtariWrapper
        
        # Try to create Breakout environment
        env = gym.make('BreakoutNoFrameskip-v4')
        env = AtariWrapper(env)
        
        print("✅ Atari environment creation successful")
        
        # Test basic functionality
        obs = env.reset()
        print(f"✅ Environment reset successful, observation shape: {obs[0].shape}")
        
        # Test random action
        action = env.action_space.sample()
        obs, reward, done, info = env.step(action)
        print(f"✅ Environment step successful, reward: {reward}")
        
        env.close()
        return True
        
    except Exception as e:
        print(f"❌ Atari environment test failed: {e}")
        return False

def test_gradcam():
    """Test that GradCAM can be imported and basic functionality works."""
    print("\n🔥 Testing GradCAM functionality...")
    
    try:
        import torch
        import torch.nn as nn
        from pytorch_grad_cam import GradCAM
        from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
        
        # Create a simple CNN model for testing
        class SimpleCNN(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
                self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
                self.pool = nn.AdaptiveAvgPool2d(1)
                self.fc = nn.Linear(32, 4)
                
            def forward(self, x):
                x = torch.relu(self.conv1(x))
                x = torch.relu(self.conv2(x))
                x = self.pool(x)
                x = x.view(x.size(0), -1)
                return self.fc(x)
        
        model = SimpleCNN()
        model.eval()
        
        # Create GradCAM
        target_layers = [model.conv2]
        cam = GradCAM(model=model, target_layers=target_layers)
        
        # Test with dummy input
        input_tensor = torch.randn(1, 3, 84, 84)
        targets = [ClassifierOutputTarget(0)]
        
        grayscale_cam = cam(input_tensor=input_tensor, targets=targets)
        
        print("✅ GradCAM functionality test successful")
        print(f"✅ GradCAM output shape: {grayscale_cam.shape}")
        
        return True
        
    except Exception as e:
        print(f"❌ GradCAM test failed: {e}")
        return False

def test_stable_baselines3():
    """Test that Stable Baselines3 can load a simple model."""
    print("\n🤖 Testing Stable Baselines3 functionality...")
    
    try:
        import gymnasium as gym
        from stable_baselines3 import PPO
        from stable_baselines3.common.vec_env import DummyVecEnv
        
        # Create a simple environment
        env = gym.make('CartPole-v1')
        env = DummyVecEnv([lambda: env])
        
        # Create a simple PPO model
        model = PPO('MlpPolicy', env, verbose=0)
        
        # Test prediction
        obs = env.reset()
        action, _ = model.predict(obs)
        
        print("✅ Stable Baselines3 functionality test successful")
        print(f"✅ Model prediction successful, action: {action}")
        
        return True
        
    except Exception as e:
        print(f"❌ Stable Baselines3 test failed: {e}")
        return False

def test_our_modules():
    """Test that our custom modules can be imported."""
    print("\n📦 Testing custom modules...")
    
    try:
        # Test inference utils
        from inference_utils import ModelLoader, InferenceRunner, ALGOS
        print("✅ inference_utils import successful")
        print(f"✅ Available algorithms: {list(ALGOS.keys())}")
        
        # Test gradcam visualizer (import only, don't instantiate)
        from gradcam_visualizer import AtariGradCAMVisualizer, PPOGradCAMWrapper
        print("✅ gradcam_visualizer import successful")
        
        return True
        
    except Exception as e:
        print(f"❌ Custom modules test failed: {e}")
        return False

def check_model_availability():
    """Check if the default Breakout model is available."""
    print("\n🎯 Checking model availability...")
    
    model_path = "rl-baselines3-zoo/rl-trained-agents/ppo/BreakoutNoFrameskip-v4_1/BreakoutNoFrameskip-v4.zip"
    
    if os.path.exists(model_path):
        print(f"✅ Default Breakout PPO model found at: {model_path}")
        return True
    else:
        print(f"⚠️  Default Breakout PPO model not found at: {model_path}")
        print("   You can download it from the rl-baselines3-zoo repository")
        return False

def main():
    """Run all tests."""
    print("🧪 GradCAM RL Visualization - Installation Test")
    print("=" * 50)
    
    all_passed = True
    
    # Test imports
    failed_imports = test_imports()
    if failed_imports:
        print(f"\n❌ Failed imports: {', '.join(failed_imports)}")
        print("Please install missing packages using: pip install -r requirements.txt")
        all_passed = False
    
    # Test Atari environment
    if not test_atari_env():
        print("\n💡 If Atari test failed, try:")
        print("   pip install 'gymnasium[atari,accept-rom-license]'")
        all_passed = False
    
    # Test GradCAM
    if not test_gradcam():
        all_passed = False
    
    # Test Stable Baselines3
    if not test_stable_baselines3():
        all_passed = False
    
    # Test our modules
    if not test_our_modules():
        all_passed = False
    
    # Check model availability
    model_available = check_model_availability()
    
    # Final summary
    print("\n" + "=" * 50)
    if all_passed:
        print("🎉 All tests passed! Installation is working correctly.")
        if model_available:
            print("🚀 You can now run: python example_gradcam.py")
        else:
            print("📥 Download a trained model to run the full example")
    else:
        print("❌ Some tests failed. Please fix the issues above.")
        return 1
    
    return 0

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code) 