# Installation Guide

This guide helps you set up the Strategy2Text environment without encountering segmentation faults, which are common in WSL2, Docker, and some Linux environments.

## Quick Setup (Recommended)

### Option 1: Automated Setup Script
```bash
# Run the improved setup script
python scripts/setup.py
# Choose option 1 for stable installation order
```

### Option 2: Manual Stable Installation
If you encounter segmentation faults, use this proven installation order:

```bash
# 1. Create fresh conda environment (recommended)
conda deactivate
conda env remove -n s2t  # if exists
conda create -n s2t python=3.11 -y
conda activate s2t

# 2. Install packages in stable order
pip install numpy
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install opencv-python-headless
pip install gymnasium[classic-control]
pip install stable-baselines3
pip install ale-py shimmy
pip install gym pillow matplotlib imageio
pip install requests tqdm pyyaml jupyter
```

## Troubleshooting Segmentation Faults

### Common Causes
1. **PyTorch GPU/CPU mismatch**: WSL2 often has issues with CUDA PyTorch
2. **OpenCV system conflicts**: Regular opencv-python conflicts with system libraries
3. **ALE-Py C++ dependencies**: Missing system dependencies
4. **Package installation order**: Some packages conflict if installed simultaneously

### Solutions

#### 1. Use CPU-only PyTorch
```bash
pip uninstall torch torchvision torchaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

#### 2. Use Headless OpenCV
```bash
pip uninstall opencv-python opencv-contrib-python
pip install opencv-python-headless
```

#### 3. Install System Dependencies (Ubuntu/Debian)
```bash
sudo apt update
sudo apt install -y build-essential cmake zlib1g-dev
```

#### 4. Test Individual Packages
```bash
python -c "import torch; print('torch OK')"
python -c "import cv2; print('opencv OK')"
python -c "import gymnasium; print('gymnasium OK')"
python -c "import ale_py; print('ale_py OK')"
python -c "import stable_baselines3; print('sb3 OK')"
```

## Alternative: Docker Setup

If you continue having issues, use Docker for a consistent environment:

```bash
# Create Dockerfile
cat > Dockerfile << 'EOF'
FROM python:3.11-slim

RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    zlib1g-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .

# Install in stable order
RUN pip install numpy && \
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu && \
    pip install opencv-python-headless && \
    pip install gymnasium[classic-control] && \
    pip install stable-baselines3 && \
    pip install ale-py shimmy && \
    pip install gym pillow matplotlib imageio && \
    pip install requests tqdm pyyaml jupyter

COPY . .
CMD ["python", "scripts/demo_breakout.py", "--list"]
EOF

# Build and run
docker build -t strategy2text .
docker run -it strategy2text bash
```

## Verification

After installation, verify everything works:

```bash
# Run troubleshooter
python scripts/troubleshoot.py

# Test demo
python scripts/demo_breakout.py --list
```

## Environment Information

This setup has been tested with:
- **Python**: 3.11 (recommended), 3.8-3.12 supported
- **OS**: Ubuntu 20.04+, WSL2, macOS, Docker
- **PyTorch**: CPU-only version for maximum compatibility
- **OpenCV**: Headless version to avoid GUI conflicts

## Getting Help

If you still encounter issues:
1. Check Python version: `python --version` (3.8+ required)
2. Try the Docker setup above
3. Run the troubleshooter: `python scripts/troubleshoot.py`
4. Test individual package imports as shown above

## GPU Support (Advanced)

If you need GPU support and have a compatible setup:

```bash
# Replace CPU PyTorch with GPU version
pip uninstall torch torchvision torchaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

Note: GPU support may reintroduce segmentation fault risks in some environments. 