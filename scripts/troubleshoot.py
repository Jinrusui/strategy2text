#!/usr/bin/env python3
"""
Troubleshooting script for Breakout Agent Demo.

This script helps diagnose and fix common issues.
"""

import sys
import subprocess
from pathlib import Path


def check_python_version():
    """Check Python version compatibility."""
    print("🐍 Checking Python version...")
    version = sys.version_info
    print(f"   Python {version.major}.{version.minor}.{version.micro}")
    
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("   ❌ Python 3.8+ required")
        return False
    else:
        print("   ✅ Python version OK")
        return True


def check_imports():
    """Check if required packages can be imported."""
    print("\n📦 Checking package imports...")
    
    packages = [
        ("torch", "PyTorch", "pip install torch"),
        ("numpy", "NumPy", "pip install numpy"),
        ("gym", "OpenAI Gym", "pip install gym"),
        ("gymnasium", "Gymnasium", "pip install gymnasium"),
        ("stable_baselines3", "Stable Baselines3", "pip install stable-baselines3"),
        ("ale_py", "ALE", "pip install ale-py"),
        ("shimmy", "Shimmy", "pip install shimmy"),
        ("cv2", "OpenCV", "pip install opencv-python"),
    ]
    
    failed_imports = []
    
    for package, name, install_cmd in packages:
        try:
            __import__(package)
            print(f"   ✅ {name}")
        except ImportError:
            print(f"   ❌ {name} - Missing")
            failed_imports.append((name, install_cmd))
    
    if failed_imports:
        print(f"\n🔧 To fix missing packages, run:")
        for name, cmd in failed_imports:
            print(f"   {cmd}")
        return False
    
    return True


def check_models():
    """Check if models are available."""
    print("\n🤖 Checking available models...")
    
    models_dir = Path("rl-trained-agents")
    if not models_dir.exists():
        print("   ❌ No models directory found")
        print("   🔧 Run: python scripts/download_rl_zoo_models.py")
        return False
    
    algorithms = ['ppo', 'dqn', 'a2c', 'qr-dqn']
    found_models = []
    
    for alg in algorithms:
        model_path = models_dir / alg / "BreakoutNoFrameskip-v4_1" / "BreakoutNoFrameskip-v4.zip"
        if model_path.exists():
            size_mb = model_path.stat().st_size / (1024 * 1024)
            print(f"   ✅ {alg.upper()}: {size_mb:.1f} MB")
            found_models.append(alg)
        else:
            print(f"   ❌ {alg.upper()}: Missing")
    
    if not found_models:
        print("   🔧 Run: python scripts/download_rl_zoo_models.py --all")
        return False
    
    return True


def test_agent_loading():
    """Test if agents can be loaded successfully."""
    print("\n🧪 Testing agent loading...")
    
    try:
        # Add src to path
        sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))
        from rl_agents.sb3_agent import SB3Agent
        
        # Find a model to test
        models_dir = Path("rl-trained-agents")
        for alg in ['ppo', 'dqn', 'a2c']:
            model_path = models_dir / alg / "BreakoutNoFrameskip-v4_1" / "BreakoutNoFrameskip-v4.zip"
            if model_path.exists():
                print(f"   Testing {alg.upper()} model...")
                
                # Try to load the agent
                agent = SB3Agent(
                    agent_path=str(model_path),
                    algorithm=alg,
                    env_id="ALE/Breakout-v4"
                )
                
                print(f"   ✅ {alg.upper()} agent loaded successfully")
                return True
                
    except Exception as e:
        print(f"   ❌ Agent loading failed: {e}")
        
        # Common fixes
        print("   🔧 Possible fixes:")
        print("      - pip install gym gymnasium")
        print("      - pip install ale-py shimmy")
        print("      - pip install stable-baselines3")
        return False
    
    print("   ❌ No models available for testing")
    return False


def test_environment():
    """Test if environments can be created."""
    print("\n🎮 Testing environment creation...")
    
    try:
        import gymnasium as gym
        import ale_py
        
        # Register ALE environments
        gym.register_envs(ale_py)
        
        # Test environment creation
        env_ids = ["ALE/Breakout-v4", "ALE/Breakout-v5", "CartPole-v1"]
        
        for env_id in env_ids:
            try:
                env = gym.make(env_id)
                env.close()
                print(f"   ✅ {env_id}")
                return True
            except Exception as e:
                print(f"   ❌ {env_id}: {e}")
                continue
        
        print("   ❌ No environments could be created")
        return False
        
    except Exception as e:
        print(f"   ❌ Environment test failed: {e}")
        print("   🔧 Run: pip install gymnasium ale-py")
        return False


def run_quick_fix():
    """Run common fixes automatically."""
    print("\n🔧 Running quick fixes...")
    
    commands = [
        (["pip", "install", "-r", "requirements.txt"], "Installing requirements"),
        (["pip", "install", "gym", "gymnasium", "ale-py", "shimmy"], "Installing compatibility packages"),
    ]
    
    for cmd, description in commands:
        try:
            print(f"   {description}...")
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            print(f"   ✅ {description} completed")
        except subprocess.CalledProcessError as e:
            print(f"   ❌ {description} failed: {e}")
            return False
    
    return True


def main():
    """Main troubleshooting function."""
    print("🔍 Breakout Agent Demo Troubleshooter")
    print("=" * 50)
    
    tests = [
        ("Python Version", check_python_version),
        ("Package Imports", check_imports),
        ("Model Files", check_models),
        ("Agent Loading", test_agent_loading),
        ("Environment Creation", test_environment),
    ]
    
    failed_tests = []
    
    for test_name, test_func in tests:
        try:
            if not test_func():
                failed_tests.append(test_name)
        except Exception as e:
            print(f"   ❌ {test_name} test crashed: {e}")
            failed_tests.append(test_name)
    
    print("\n" + "=" * 50)
    print("📋 TROUBLESHOOTING SUMMARY")
    print("=" * 50)
    
    if not failed_tests:
        print("🎉 All tests passed! Your setup should work correctly.")
        print("\n🚀 Try running a demo:")
        print("   python scripts/demo_breakout.py --list")
        print("   python scripts/demo_breakout.py --algorithm ppo")
    else:
        print(f"⚠️  {len(failed_tests)} issue(s) found:")
        for test in failed_tests:
            print(f"   - {test}")
        
        print(f"\n🔧 Quick fix attempt:")
        response = input("Run automatic fixes? (y/n): ").lower().strip()
        if response in ['y', 'yes']:
            if run_quick_fix():
                print("\n✅ Quick fixes completed. Try running the troubleshooter again.")
            else:
                print("\n❌ Some fixes failed. Manual intervention may be required.")
        
        print(f"\n📚 Manual fix suggestions:")
        print("   1. Install missing packages: pip install -r requirements.txt")
        print("   2. Download models: python scripts/download_rl_zoo_models.py")
        print("   3. Check Python version (3.8+ required)")
        print("   4. Try: pip install gym gymnasium ale-py shimmy")


if __name__ == "__main__":
    main() 