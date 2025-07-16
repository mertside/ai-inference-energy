#!/bin/bash
"""
Interactive H100 LLaMA Setup and Testing Script
Runs on H100 compute node to setup environment and test LLaMA
"""

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

log_info() { echo -e "${BLUE}[INFO]${NC} $1"; }
log_success() { echo -e "${GREEN}[SUCCESS]${NC} $1"; }
log_warning() { echo -e "${YELLOW}[WARNING]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

echo "=========================================="
log_info "H100 LLaMA Environment Setup & Testing"
echo "=========================================="

# Check node and GPU information
log_info "Node: $(hostname)"
log_info "GPU Information:"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader

# Navigate to workspace
cd /mnt/DISCL/home/meside/ai-inference-energy/app-llama

log_info "Current directory: $(pwd)"

# Check if conda is available
if ! command -v conda &> /dev/null; then
    log_error "Conda not found. Loading conda module..."
    # Try common module loading commands
    if command -v module &> /dev/null; then
        module load anaconda3 || module load miniconda3 || module load conda
    fi
fi

# Verify conda is now available
if ! command -v conda &> /dev/null; then
    log_error "Conda still not available. Please load conda manually."
    exit 1
fi

log_info "Conda version: $(conda --version)"

# Setup H100 environment
log_info "Setting up H100 LLaMA environment..."
cd setup/

# Check if environment already exists
if conda env list | grep -q "llama-h100"; then
    log_warning "Environment llama-h100 already exists. Activating..."
else
    log_info "Creating new H100 environment..."
    conda env create -f conda_env_h100.yml
fi

# Activate environment
log_info "Activating llama-h100 environment..."
source $(conda info --base)/etc/profile.d/conda.sh
conda activate llama-h100

# Verify environment
log_info "Python version: $(python --version)"
log_info "PyTorch version: $(python -c 'import torch; print(torch.__version__)')"
log_info "CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')"
log_info "GPU count: $(python -c 'import torch; print(torch.cuda.device_count())')"
log_info "GPU name: $(python -c 'import torch; print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A")')"

# Navigate back to app directory
cd ../

# Test basic LLaMA functionality without model loading
log_info "Testing LLaMA application CLI..."
python LlamaViaHF.py --help | head -20

log_info "Testing basic imports..."
python -c "
import torch
from transformers import AutoTokenizer
print('Basic imports successful')
print(f'CUDA devices: {torch.cuda.device_count()}')
if torch.cuda.is_available():
    print(f'GPU 0: {torch.cuda.get_device_name(0)}')
    print(f'CUDA version: {torch.version.cuda}')
"

# Check Hugging Face authentication
log_info "Checking Hugging Face authentication..."
if python -c "from huggingface_hub import HfApi; HfApi().whoami()" &> /dev/null; then
    log_success "Hugging Face authentication verified."
    HF_USER=$(python -c "from huggingface_hub import HfApi; print(HfApi().whoami()['name'])")
    log_info "Logged in as: $HF_USER"
else
    log_warning "Hugging Face authentication not found."
    log_info "You can authenticate later with: huggingface-cli login"
fi

# Test basic model access (quick check without loading)
log_info "Testing model access (tokenizer only)..."
python -c "
from transformers import AutoTokenizer
try:
    tokenizer = AutoTokenizer.from_pretrained('huggyllama/llama-7b', use_fast=False)
    print('✓ LLaMA-7B tokenizer access successful')
except Exception as e:
    print(f'✗ LLaMA-7B access failed: {e}')
    print('This is expected if not authenticated with Hugging Face')
"

log_success "H100 Environment setup completed!"
echo
log_info "To test LLaMA text generation:"
echo "  python LlamaViaHF.py --model llama-7b --prompt 'Hello world' --max-tokens 10"
echo
log_info "To run benchmark for profiling:"
echo "  python LlamaViaHF.py --benchmark --num-generations 3 --quiet --metrics"
echo
log_info "Current session details:"
echo "  Node: $(hostname)"
echo "  Environment: llama-h100"
echo "  Working directory: $(pwd)"
