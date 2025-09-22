#!/bin/bash

# Tekken 3 Gameplay Analysis - Setup Script
# This script sets up the complete environment for the Tekken analysis system

set -e  # Exit on any error

echo "🎮 Setting up Tekken 3 Gameplay Analysis System..."
echo "=================================================="

# Check if conda is available
if ! command -v conda &> /dev/null; then
    echo "❌ Error: conda is not installed or not in PATH"
    echo "Please install Miniconda or Anaconda first"
    exit 1
fi

# Check if NVIDIA GPU is available
if ! command -v nvidia-smi &> /dev/null; then
    echo "⚠️  Warning: nvidia-smi not found. GPU acceleration may not be available"
else
    echo "✅ NVIDIA GPU detected:"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits
fi

# Create conda environment
echo "📦 Creating conda environment 'benchmark-env'..."
if conda env list | grep -q "benchmark-env"; then
    echo "Environment 'benchmark-env' already exists. Updating..."
    conda env update -f environment.yml
else
    conda env create -f environment.yml
fi

echo "✅ Environment created successfully!"

# Activate environment and verify installation
echo "🔍 Verifying installation..."
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate benchmark-env

# Check PyTorch CUDA
python -c "
import torch
import sys

print(f'Python version: {sys.version}')
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')

if torch.cuda.is_available():
    print(f'CUDA version: {torch.version.cuda}')
    print(f'GPU count: {torch.cuda.device_count()}')
    for i in range(torch.cuda.device_count()):
        print(f'GPU {i}: {torch.cuda.get_device_name(i)}')
else:
    print('⚠️  CUDA not available - CPU-only mode')
"

# Check transformers
python -c "
try:
    import transformers
    print(f'Transformers version: {transformers.__version__}')
    print('✅ Transformers installed successfully')
except ImportError:
    print('❌ Error: Transformers not installed properly')
    exit(1)
"

# Check other key dependencies
python -c "
import cv2, PIL, numpy, matplotlib, tqdm
print('✅ All key dependencies verified')
"

echo ""
echo "🎉 Setup completed successfully!"
echo ""
echo "📋 Next steps:"
echo "1. Activate environment: conda activate benchmark-env"
echo "2. Run analysis: python analyse1.py"
echo ""
echo "💾 Model Information:"
echo "- First run will download Qwen2-VL-72B-Instruct (~137GB)"
echo "- Download time: 30-60 minutes depending on connection"
echo "- Model will be cached in: ~/.cache/huggingface/"
echo ""
echo "🔧 Troubleshooting:"
echo "- For CUDA issues: Check NVIDIA driver and CUDA installation"
echo "- For memory issues: Ensure sufficient GPU memory (24GB+ recommended)"
echo "- For model download issues: Check internet connection and Hugging Face access"
echo ""
echo "Happy analyzing! 🥊"