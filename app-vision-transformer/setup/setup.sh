#!/bin/bash

# Vision Transform# Check if vit environment already exists
if conda env list | grep -q "vit"; then
    echo "⚠️  vit environment already exists."
    read -p "Do you want to remove and recreate it? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "Removing existing vit environment..."
        conda env remove -n vit -y
    else
        echo "Setup cancelled."
        exit 0
    fi
fi

echo "Creating vit conda environment..."
# Creates dedicated ViT conda environment for AI inference energy profiling

echo "=== Vision Transformer (ViT) Setup ==="
echo "Setting up dedicated ViT conda environment..."

# Check if conda is available
if ! command -v conda &> /dev/null; then
    echo "❌ Conda not found. Please install Miniconda/Anaconda first."
    exit 1
fi

# Determine which environment file to use
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
ENV_FILE=""

# Check for system-specific environment files
if [ -f "$SCRIPT_DIR/vit-env-hpcc.yml" ]; then
    ENV_FILE="$SCRIPT_DIR/vit-env-hpcc.yml"
    echo "Using HPCC environment configuration"
elif [ -f "$SCRIPT_DIR/vit-env-repacss.yml" ]; then
    ENV_FILE="$SCRIPT_DIR/vit-env-repacss.yml"
    echo "Using RepaCCS environment configuration"
else
    echo "❌ No environment configuration file found."
    echo "Please ensure vit-env-hpcc.yml or vit-env-repacss.yml exists in the setup directory."
    exit 1
fi

# Check if vit-energy environment already exists
if conda env list | grep -q "vit-energy"; then
    echo "⚠️  vit-energy environment already exists."
    read -p "Do you want to remove and recreate it? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "Removing existing vit-energy environment..."
        conda env remove -n vit-energy -y
    else
        echo "Setup cancelled."
        exit 0
    fi
fi

echo "Creating vit-energy conda environment..."
conda env create -f "$ENV_FILE"

if [ $? -eq 0 ]; then
    echo "✅ Environment created successfully!"
else
    echo "❌ Failed to create environment."
    exit 1
fi

# Activate the new environment
eval "$(conda shell.bash hook)"
conda activate vit

# Verify installation
echo ""
echo "=== Verifying ViT Installation ==="
python -c "
import torch
import transformers
from PIL import Image
import requests
import torchvision
import numpy as np
print('✅ PyTorch:', torch.__version__)
print('✅ Transformers:', transformers.__version__)
print('✅ Pillow:', Image.__version__)
print('✅ Torchvision:', torchvision.__version__)
print('✅ CUDA available:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('✅ GPU:', torch.cuda.get_device_name(0))
    print('✅ CUDA version:', torch.version.cuda)
print('✅ All dependencies verified!')
"

if [ $? -eq 0 ]; then
    echo ""
    echo "🎉 ViT environment setup completed successfully!"
    echo ""
    echo "To use the ViT environment:"
    echo "  conda activate vit"
    echo ""
    echo "To test the ViT application:"
    echo "  cd /path/to/ai-inference-energy"
    echo "  conda activate vit"
    echo "  python -m app_vision_transformer.ViTViaHF --help"
    echo ""
else
    echo "❌ Environment verification failed."
    exit 1
fi
print('✅ PIL (Pillow):', Image.__version__ if hasattr(Image, '__version__') else 'Available')
print('✅ Requests: Available')
print('✅ CUDA Available:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('✅ CUDA Device:', torch.cuda.get_device_name(0))
"

echo ""
echo "=== ViT Setup Complete ==="
echo "Environment: whisper-energy"
echo "Ready for Vision Transformer energy profiling!"
echo ""
echo "Test the installation:"
echo "conda activate whisper-energy"
echo "python app-vision-transformer/ViTViaHF.py --benchmark --num-images 5"
