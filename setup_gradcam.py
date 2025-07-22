#!/usr/bin/env python3
"""
Setup script for Grad-CAM visualization of RL agents.

This script checks dependencies, installs missing packages, and verifies
that the environment is ready for Grad-CAM visualization.
"""

import subprocess
import sys
import os
import importlib
from pathlib import Path


def check_package(package_name, import_name=None):
    """Check if a package is installed."""
    if import_name is None:
        import_name = package_name
    
    try:
        importlib.import_module(import_name)
        print(f"✅ {package_name} is installed")
        return True
    except ImportError:
        print(f"❌ {package_name} is not installed")
        return False


def install_package(package_name):
    """Install a package using pip."""
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package_name])
        print(f"✅ Successfully installed {package_name}")
        return True
    except subprocess.CalledProcessError:
        print(f"❌ Failed to install {package_name}")
        return False


def check_pytorch_gradcam():
    """Check if pytorch-grad-cam is properly installed."""
    try:
        from pytorch_grad_cam import GradCAM
        from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
        from pytorch_grad_cam.utils.image import show_cam_on_image
        print("✅ pytorch-grad-cam is properly installed")
        return True
    except ImportError as e:
        print(f"❌ pytorch-grad-cam import failed: {e}")
        return False


def check_rl_environment():
    """Check RL-related dependencies."""
    packages = [
        ("stable-baselines3", "stable_baselines3"),
        ("sb3-contrib", "sb3_contrib"),
        ("gymnasium", "gymnasium"),
    ]
    
    all_installed = True
    for pkg_name, import_name in packages:
        if not check_package(pkg_name, import_name):
            all_installed = False
    
    return all_installed


def check_rl_zoo():
    """Check if rl-baselines3-zoo is available."""
    rl_zoo_path = Path("rl-baselines3-zoo")
    
    if not rl_zoo_path.exists():
        print("❌ rl-baselines3-zoo directory not found")
        print("Please clone it with:")
        print("git clone https://github.com/DLR-RM/rl-baselines3-zoo.git")
        return False
    
    # Check if it has the required modules
    sys.path.insert(0, str(rl_zoo_path))
    
    try:
        from rl_zoo3.utils import ALGOS
        from rl_zoo3.exp_manager import ExperimentManager
        print("✅ rl-baselines3-zoo is properly set up")
        return True
    except ImportError as e:
        print(f"❌ rl-baselines3-zoo import failed: {e}")
        return False


def check_cv2():
    """Check OpenCV installation."""
    try:
        import cv2
        print(f"✅ OpenCV version {cv2.__version__} is installed")
        return True
    except ImportError:
        print("❌ OpenCV (cv2) is not installed")
        return False


def check_models():
    """Check if there are trained models available."""
    rl_zoo_path = Path("rl-baselines3-zoo")
    models_path = rl_zoo_path / "rl-trained-agents"
    
    if not models_path.exists():
        print("⚠️  No trained models directory found")
        print("You may need to download or train models first")
        return False
    
    # Check for DQN models
    dqn_path = models_path / "dqn"
    if dqn_path.exists():
        model_files = list(dqn_path.rglob("*.zip"))
        if model_files:
            print(f"✅ Found {len(model_files)} DQN model file(s)")
            return True
    
    print("⚠️  No DQN models found in rl-trained-agents/dqn/")
    print("You may need to download or train DQN models for Breakout")
    return False


def main():
    """Main setup function."""
    print("=" * 60)
    print("🔥 GRAD-CAM FOR RL AGENTS - SETUP CHECK")
    print("=" * 60)
    print()
    
    print("📦 Checking core dependencies...")
    core_packages = [
        ("torch", "torch"),
        ("torchvision", "torchvision"),
        ("numpy", "numpy"),
        ("matplotlib", "matplotlib"),
    ]
    
    missing_core = []
    for pkg_name, import_name in core_packages:
        if not check_package(pkg_name, import_name):
            missing_core.append(pkg_name)
    
    print("\n🔍 Checking OpenCV...")
    if not check_cv2():
        missing_core.append("opencv-python")
    
    print("\n🎯 Checking pytorch-grad-cam...")
    gradcam_ok = check_pytorch_gradcam()
    if not gradcam_ok:
        missing_core.append("grad-cam")
    
    print("\n🤖 Checking RL dependencies...")
    rl_ok = check_rl_environment()
    
    print("\n🎮 Checking RL Baselines3 Zoo...")
    zoo_ok = check_rl_zoo()
    
    print("\n🎯 Checking for trained models...")
    models_ok = check_models()
    
    print("\n" + "=" * 60)
    print("📊 SETUP SUMMARY")
    print("=" * 60)
    
    if missing_core:
        print("❌ Missing core packages:")
        for pkg in missing_core:
            print(f"   - {pkg}")
        print("\nInstall them with:")
        print(f"pip install {' '.join(missing_core)}")
        print()
    
    if not rl_ok:
        print("❌ RL dependencies not fully available")
        print("Install with:")
        print("pip install stable-baselines3[extra] sb3-contrib gymnasium")
        print()
    
    if not zoo_ok:
        print("❌ RL Baselines3 Zoo not available")
        print("Clone it with:")
        print("git clone https://github.com/DLR-RM/rl-baselines3-zoo.git")
        print()
    
    if not models_ok:
        print("⚠️  No trained models found")
        print("Download pre-trained models or train your own:")
        print("https://github.com/DLR-RM/rl-baselines3-zoo#pretrained-agents")
        print()
    
    # Overall status
    all_ready = (
        not missing_core and 
        gradcam_ok and 
        rl_ok and 
        zoo_ok and 
        models_ok
    )
    
    if all_ready:
        print("🎉 ALL CHECKS PASSED! You're ready to use Grad-CAM visualization!")
        print()
        print("Quick start:")
        print("python record_video_with_gradcam.py --env BreakoutNoFrameskip-v4 --algo dqn --seed 42 -n 500")
        print()
        print("Test reproducibility:")
        print("python test_gradcam_reproducibility.py --create-minimal")
        print("python test_minimal_reproducibility.py")
    else:
        print("❌ Setup incomplete. Please address the issues above.")
        return 1
    
    return 0


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code) 