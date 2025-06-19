#!/usr/bin/env python3
"""
Setup script for Breakout Agent Demo.

This script helps install dependencies and set up the environment.
Uses a stable installation order to prevent segmentation faults.
"""

import subprocess
import sys
import os
from pathlib import Path


def run_command(command, description=""):
    """Run a command and handle errors."""
    print(f"{'=' * 50}")
    if description:
        print(f"🔧 {description}")
    print(f"Running: {' '.join(command)}")
    print(f"{'=' * 50}")
    
    try:
        result = subprocess.run(command, check=True, capture_output=True, text=True)
        if result.stdout:
            print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error: {e}")
        if e.stderr:
            print(f"Error output: {e.stderr}")
        return False


def check_python_version():
    """Check if Python version is compatible."""
    version = sys.version_info
    print(f"Python version: {version.major}.{version.minor}.{version.micro}")
    
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("❌ Python 3.8 or higher is required!")
        return False
    
    print("✅ Python version is compatible")
    return True


def install_packages_stable_order():
    """Install packages in stable order to prevent segmentation faults."""
    print("📦 Installing packages in stable order...")
    
    # Step-by-step installation to prevent conflicts
    installation_steps = [
        (["pip", "install", "numpy"], "Installing NumPy (foundation)"),
        (["pip", "install", "torch", "torchvision", "torchaudio", "--index-url", "https://download.pytorch.org/whl/cpu"], "Installing PyTorch (CPU-only for stability)"),
        (["pip", "install", "opencv-python-headless"], "Installing OpenCV (headless version)"),
        (["pip", "install", "gymnasium[classic-control]"], "Installing Gymnasium"),
        (["pip", "install", "stable-baselines3"], "Installing Stable Baselines3"),
        (["pip", "install", "ale-py", "shimmy"], "Installing ALE-Py and Shimmy"),
        (["pip", "install", "gym", "pillow", "matplotlib", "imageio"], "Installing additional dependencies"),
        (["pip", "install", "requests", "tqdm", "pyyaml", "jupyter"], "Installing utilities"),
    ]
    
    for command, description in installation_steps:
        if not run_command(command, description):
            print(f"❌ Failed at step: {description}")
            return False
        print(f"✅ {description} completed\n")
    
    return True


def install_requirements_fallback():
    """Fallback: Install from requirements.txt (may cause issues)."""
    requirements_file = Path(__file__).parent.parent / "requirements.txt"
    
    if not requirements_file.exists():
        print("❌ requirements.txt not found!")
        return False
    
    print("📦 Installing from requirements.txt (fallback method)...")
    print("⚠️  This may cause segmentation faults in some environments")
    
    return run_command([
        sys.executable, "-m", "pip", "install", "-r", str(requirements_file)
    ], "Installing dependencies from requirements.txt")


def test_imports():
    """Test if key packages can be imported."""
    print("🧪 Testing package imports...")
    
    test_packages = [
        ("numpy", "NumPy"),
        ("torch", "PyTorch"),
        ("cv2", "OpenCV"),
        ("gymnasium", "OpenAI Gymnasium"),
        ("stable_baselines3", "Stable Baselines3"),
        ("ale_py", "Arcade Learning Environment"),
        ("shimmy", "Shimmy"),
    ]
    
    all_good = True
    for package, name in test_packages:
        try:
            print(f"  Testing {name}...")
            __import__(package)
            print(f"  ✅ {name}")
        except ImportError as e:
            print(f"  ❌ {name}: {e}")
            all_good = False
        except Exception as e:
            print(f"  ⚠️  {name}: Segmentation fault or other error - {e}")
            all_good = False
    
    return all_good


def create_directories():
    """Create necessary directories."""
    print("📁 Creating directories...")
    
    directories = [
        "rl-trained-agents",
        "videos",
    ]
    
    for directory in directories:
        dir_path = Path(directory)
        dir_path.mkdir(exist_ok=True)
        print(f"  ✅ {directory}/")
    
    return True


def download_sample_models():
    """Download sample models for testing."""
    print("🤖 Downloading sample models...")
    
    script_path = Path(__file__).parent / "download_rl_zoo_models.py"
    
    if not script_path.exists():
        print("❌ Download script not found!")
        return False
    
    # Download PPO model as a test
    return run_command([
        sys.executable, str(script_path), "--algorithm", "ppo"
    ], "Downloading PPO Breakout model")


def run_quick_test():
    """Run a quick test to ensure everything works."""
    print("🚀 Running quick test...")
    
    demo_script = Path(__file__).parent / "demo_breakout.py"
    
    if not demo_script.exists():
        print("❌ Demo script not found!")
        return False
    
    # Test listing available models
    return run_command([
        sys.executable, str(demo_script), "--list"
    ], "Testing demo script")


def main():
    """Main setup function."""
    print("🎮 Breakout Agent Demo Setup")
    print("=" * 50)
    print("This setup uses a stable installation order to prevent segmentation faults.")
    print("Recommended for WSL2, Docker, and other problematic environments.")
    print("=" * 50)
    
    # Ask user for installation method
    print("\nChoose installation method:")
    print("1. Stable order installation (recommended for WSL2/Docker)")
    print("2. Standard requirements.txt installation")
    
    while True:
        choice = input("Enter choice (1 or 2): ").strip()
        if choice in ['1', '2']:
            break
        print("Please enter 1 or 2")
    
    use_stable_order = choice == '1'
    
    steps = [
        ("Checking Python version", check_python_version),
    ]
    
    if use_stable_order:
        steps.append(("Installing packages (stable order)", install_packages_stable_order))
    else:
        steps.append(("Installing requirements", install_requirements_fallback))
    
    steps.extend([
        ("Testing imports", test_imports),
        ("Creating directories", create_directories),
        ("Downloading sample models", download_sample_models),
        ("Running quick test", run_quick_test),
    ])
    
    success_count = 0
    
    for step_name, step_func in steps:
        print(f"\n🔄 {step_name}...")
        if step_func():
            print(f"✅ {step_name} completed successfully!")
            success_count += 1
        else:
            print(f"❌ {step_name} failed!")
            
            # Ask if user wants to continue
            response = input("Continue with setup? (y/n): ").lower().strip()
            if response not in ['y', 'yes']:
                print("Setup aborted.")
                return False
    
    print("\n" + "=" * 50)
    print("🎉 SETUP SUMMARY")
    print("=" * 50)
    print(f"Completed steps: {success_count}/{len(steps)}")
    
    if success_count == len(steps):
        print("✅ Setup completed successfully!")
        print("\n🚀 Next steps:")
        print("  1. Run a demo:")
        print("     python scripts/demo_breakout.py")
        print("  2. Download more models:")
        print("     python scripts/download_rl_zoo_models.py --all")
        print("  3. Try different algorithms:")
        print("     python scripts/demo_breakout.py --algorithm dqn")
        
    else:
        print("⚠️  Setup completed with some issues.")
        print("You may need to manually install missing dependencies.")
        
        print("\n🔧 Manual installation commands (stable order):")
        print("  pip install numpy")
        print("  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu")
        print("  pip install opencv-python-headless")
        print("  pip install gymnasium[classic-control]")
        print("  pip install stable-baselines3")
        print("  pip install ale-py shimmy")
        
    return success_count == len(steps)


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 