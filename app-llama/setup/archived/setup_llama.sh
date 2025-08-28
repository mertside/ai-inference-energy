#!/bin/bash
"""
LLaMA Setup Script for Energy Profiling Framework
Creates conda environment and validates LLaMA model access.

Usage:
    ./setup_llama.sh [h100|a100|v100|generic]

If no argument provided, will auto-detect based on available GPU.
"""

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Detect GPU type
detect_gpu() {
    if command -v nvidia-smi &> /dev/null; then
        GPU_INFO=$(nvidia-smi --query-gpu=name --format=csv,noheader,nounits | head -1)
        if [[ $GPU_INFO == *"H100"* ]]; then
            echo "h100"
        elif [[ $GPU_INFO == *"A100"* ]]; then
            echo "a100"
        elif [[ $GPU_INFO == *"V100"* ]]; then
            echo "v100"
        else
            echo "generic"
        fi
    else
        echo "generic"
    fi
}

# Main setup function
setup_llama() {
    local env_type=$1

    log_info "Setting up LLaMA environment for: $env_type"

    # Check if conda is available
    if ! command -v conda &> /dev/null; then
        log_error "Conda not found. Please install Anaconda or Miniconda first."
        exit 1
    fi

    # Environment file path
    local env_file="conda_env_${env_type}.yml"

    if [[ ! -f $env_file ]]; then
        log_error "Environment file not found: $env_file"
        exit 1
    fi

    # Environment name
    local env_name="llama-${env_type}"

    # Check if environment already exists
    if conda env list | grep -q "^${env_name} "; then
        log_warning "Environment $env_name already exists."
        read -p "Do you want to remove and recreate it? (y/N): " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            log_info "Removing existing environment..."
            conda env remove -n $env_name -y
        else
            log_info "Using existing environment. Activating..."
            conda activate $env_name
            return 0
        fi
    fi

    # Create conda environment
    log_info "Creating conda environment from $env_file..."
    conda env create -f $env_file

    # Activate environment
    log_info "Activating environment $env_name..."
    source $(conda info --base)/etc/profile.d/conda.sh
    conda activate $env_name

    # Verify installation
    log_info "Verifying installation..."
    python -c "import torch; print(f'PyTorch version: {torch.__version__}')"
    python -c "import transformers; print(f'Transformers version: {transformers.__version__}')"

    if [[ $env_type != "generic" ]]; then
        python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
        if python -c "import torch; exit(0 if torch.cuda.is_available() else 1)"; then
            python -c "import torch; print(f'GPU count: {torch.cuda.device_count()}')"
            python -c "import torch; print(f'GPU name: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"
        fi
    fi

    log_success "Environment setup completed successfully!"

    # Hugging Face authentication check
    log_info "Checking Hugging Face authentication..."
    if python -c "from huggingface_hub import HfApi; HfApi().whoami()" &> /dev/null; then
        log_success "Hugging Face authentication verified."
    else
        log_warning "Hugging Face authentication not found."
        log_info "Please run: huggingface-cli login"
        log_info "You'll need a Hugging Face token with access to LLaMA models."
    fi

    # Test LLaMA model access
    log_info "Testing LLaMA model access (this may take a moment)..."
    if timeout 30s python -c "from transformers import AutoTokenizer; AutoTokenizer.from_pretrained('huggyllama/llama-7b', use_fast=False)" &> /dev/null; then
        log_success "LLaMA model access verified!"
    else
        log_warning "Could not verify LLaMA model access."
        log_info "This might be due to:"
        log_info "  1. No Hugging Face authentication"
        log_info "  2. No access to LLaMA models"
        log_info "  3. Network connectivity issues"
    fi

    # Display usage instructions
    echo
    log_success "Setup complete! Usage instructions:"
    echo "  1. Activate environment: conda activate $env_name"
    echo "  2. Test basic functionality:"
    echo "     cd ../"
    echo "     python LlamaViaHF.py --model llama-7b --prompt 'Hello world' --max-tokens 10 --metrics"
    echo "  3. Run benchmark:"
    echo "     python LlamaViaHF.py --benchmark --num-generations 3 --quiet"
}

# Main execution
main() {
    local env_type=$1

    # Auto-detect if no argument provided
    if [[ -z $env_type ]]; then
        env_type=$(detect_gpu)
        log_info "Auto-detected GPU type: $env_type"
    fi

    # Validate environment type
    case $env_type in
        h100|a100|v100|generic)
            setup_llama $env_type
            ;;
        *)
            log_error "Invalid environment type: $env_type"
            log_info "Supported types: h100, a100, v100, generic"
            exit 1
            ;;
    esac
}

# Script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
cd "$SCRIPT_DIR"

# Run main function
main "$@"
