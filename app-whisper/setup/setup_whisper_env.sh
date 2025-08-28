#!/bin/bash
#
# Whisper Environment Setup Script for AI Inference Energy Profiling.
#
# This script sets up the Whisper speech recognition environment for energy
# profiling experiments across different GPU architectures.
#
# Usage: ./setup_whisper_env.sh [options]
#
# Options:
#   --env-name NAME     Conda environment name (default: whisper-energy)
#   --python-version    Python version (default: 3.10)
#   --cuda-version      CUDA version (default: 11.8)
#   --force            Force recreate environment
#   --export-only      Only export existing environment to YAML
#   --help             Show this help message
#
# Author: Mert Side
#

set -euo pipefail

# Default values
DEFAULT_ENV_NAME="whisper-energy"
DEFAULT_PYTHON_VERSION="3.10"
DEFAULT_CUDA_VERSION="11.8"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${GREEN}[INFO]${NC} $*"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $*"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $*"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $*"
}

# Help function
show_help() {
    echo "Whisper Environment Setup Script"
    echo
    echo "Usage: ./setup_whisper_env.sh [options]"
    echo
    echo "Options:"
    echo "  --env-name NAME       Conda environment name (default: $DEFAULT_ENV_NAME)"
    echo "  --python-version VER  Python version (default: $DEFAULT_PYTHON_VERSION)"
    echo "  --cuda-version VER    CUDA version (default: $DEFAULT_CUDA_VERSION)"
    echo "  --force               Force recreate environment"
    echo "  --export-only         Only export existing environment to YAML"
    echo "  --help                Show this help message"
    echo
    echo "Examples:"
    echo "  ./setup_whisper_env.sh                                    # Default setup"
    echo "  ./setup_whisper_env.sh --env-name whisper-h100           # Custom environment name"
    echo "  ./setup_whisper_env.sh --python-version 3.9 --force      # Force recreate with Python 3.9"
    echo "  ./setup_whisper_env.sh --export-only                     # Export existing environment"
    echo
}

# Parse command line arguments
ENV_NAME="$DEFAULT_ENV_NAME"
PYTHON_VERSION="$DEFAULT_PYTHON_VERSION"
CUDA_VERSION="$DEFAULT_CUDA_VERSION"
FORCE_RECREATE=false
EXPORT_ONLY=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --env-name)
            ENV_NAME="$2"
            shift 2
            ;;
        --python-version)
            PYTHON_VERSION="$2"
            shift 2
            ;;
        --cuda-version)
            CUDA_VERSION="$2"
            shift 2
            ;;
        --force)
            FORCE_RECREATE=true
            shift
            ;;
        --export-only)
            EXPORT_ONLY=true
            shift
            ;;
        --help)
            show_help
            exit 0
            ;;
        *)
            log_error "Unknown option: $1"
            show_help
            exit 1
            ;;
    esac
done

# Main setup function
main() {
    # Handle export-only mode
    if [ "$EXPORT_ONLY" = true ]; then
        log_info "Export-only mode: Exporting environment $ENV_NAME to YAML files..."

        # Check if environment exists
        if ! conda env list | grep -q "^$ENV_NAME "; then
            log_error "Environment $ENV_NAME does not exist. Please create it first."
            exit 1
        fi

        # Export environment
        export_environment

        # Display export completion message
        display_export_completion_message
        return 0
    fi

    log_info "Setting up Whisper environment for AI inference energy profiling..."
    log_info "Environment: $ENV_NAME"
    log_info "Python: $PYTHON_VERSION"
    log_info "CUDA: $CUDA_VERSION"

    # Check prerequisites
    check_prerequisites

    # Setup conda environment
    setup_conda_environment

    # Install dependencies
    install_dependencies

    # Test installation
    test_installation

    # Display completion message
    display_completion_message
}

# Check prerequisites
check_prerequisites() {
    log_info "Checking prerequisites..."

    # Check conda
    if ! command -v conda &> /dev/null; then
        log_error "Conda not found. Please install Miniconda or Anaconda."
        exit 1
    fi

    # Check CUDA availability
    if command -v nvidia-smi &> /dev/null; then
        log_info "NVIDIA GPU detected:"
        nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1
    else
        log_warning "nvidia-smi not found. GPU support may be limited."
    fi

    # Check if environment exists
    if conda env list | grep -q "^$ENV_NAME "; then
        if [ "$FORCE_RECREATE" = true ]; then
            log_info "Removing existing environment: $ENV_NAME"
            conda env remove -n "$ENV_NAME" -y
        else
            log_warning "Environment $ENV_NAME already exists. Use --force to recreate."
            log_info "Activating existing environment..."
            return 0
        fi
    fi

    log_success "Prerequisites check completed"
}

# Setup conda environment
setup_conda_environment() {
    log_info "Creating conda environment: $ENV_NAME"

    # Create environment with specific Python version
    conda create -n "$ENV_NAME" python="$PYTHON_VERSION" -y

    # Activate environment
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate "$ENV_NAME"

    # Update conda and pip
    conda update -n "$ENV_NAME" -c defaults conda -y
    pip install --upgrade pip

    log_success "Conda environment created and activated"
}

# Install dependencies
install_dependencies() {
    log_info "Installing dependencies..."

    # Ensure we're in the right environment
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate "$ENV_NAME"

    # Install PyTorch with CUDA support
    log_info "Installing PyTorch with CUDA $CUDA_VERSION..."
    if [ "$CUDA_VERSION" = "11.8" ]; then
        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
    elif [ "$CUDA_VERSION" = "12.1" ]; then
        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
    else
        log_warning "Using default PyTorch installation for CUDA $CUDA_VERSION"
        pip install torch torchvision torchaudio
    fi

    # Install transformers and related packages
    log_info "Installing transformers and Hugging Face ecosystem..."
    pip install transformers>=4.21.0 accelerate>=0.12.0

    # Install audio processing libraries
    log_info "Installing audio processing libraries..."
    pip install librosa soundfile datasets

    # Install additional dependencies
    log_info "Installing additional dependencies..."
    pip install numpy scipy

    # Install requirements file if available
    if [ -f "$SCRIPT_DIR/../requirements.txt" ]; then
        log_info "Installing from requirements.txt..."
        pip install -r "$SCRIPT_DIR/../requirements.txt"
    fi

    log_success "Dependencies installed successfully"

    # Export environment to YAML
    export_environment
}

# Export environment to YAML file
export_environment() {
    log_info "Exporting environment to YAML file..."

    # Ensure we're in the right environment
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate "$ENV_NAME"

    # Create exports directory if it doesn't exist
    EXPORT_DIR="$SCRIPT_DIR/../exports"
    mkdir -p "$EXPORT_DIR"

    # Generate timestamp for unique filename
    TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

    # Export full environment (includes all dependencies)
    FULL_EXPORT_FILE="$EXPORT_DIR/${ENV_NAME}_full_${TIMESTAMP}.yml"
    conda env export > "$FULL_EXPORT_FILE"
    log_success "Full environment exported to: $FULL_EXPORT_FILE"

    # Export minimal environment (only explicit dependencies)
    MINIMAL_EXPORT_FILE="$EXPORT_DIR/${ENV_NAME}_minimal_${TIMESTAMP}.yml"
    conda env export --from-history > "$MINIMAL_EXPORT_FILE"
    log_success "Minimal environment exported to: $MINIMAL_EXPORT_FILE"

    # Create a generic export without timestamp for easy sharing
    GENERIC_EXPORT_FILE="$EXPORT_DIR/${ENV_NAME}_environment.yml"
    conda env export > "$GENERIC_EXPORT_FILE"
    log_success "Generic environment exported to: $GENERIC_EXPORT_FILE"

    # Create a cross-platform compatible version
    CROSS_PLATFORM_FILE="$EXPORT_DIR/${ENV_NAME}_cross_platform.yml"
    conda env export --no-builds > "$CROSS_PLATFORM_FILE"
    log_success "Cross-platform environment exported to: $CROSS_PLATFORM_FILE"

    log_info "Environment export completed"
}

# Test installation
test_installation() {
    log_info "Testing installation..."

    # Ensure we're in the right environment
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate "$ENV_NAME"

    # Test basic imports
    python -c "
import torch
import torchaudio
import transformers
import librosa
import soundfile
import numpy as np
print('✓ All core libraries imported successfully')
"

    # Test CUDA availability
    python -c "
import torch
print(f'✓ PyTorch version: {torch.__version__}')
print(f'✓ CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'✓ CUDA version: {torch.version.cuda}')
    print(f'✓ GPU count: {torch.cuda.device_count()}')
    print(f'✓ GPU name: {torch.cuda.get_device_name(0)}')
"

    # Test Whisper model loading (quick test)
    log_info "Testing Whisper model access..."
    python -c "
from transformers import WhisperProcessor
try:
    processor = WhisperProcessor.from_pretrained('openai/whisper-tiny')
    print('✓ Whisper model access successful')
except Exception as e:
    print(f'✗ Whisper model access failed: {e}')
    print('This may be normal if no internet connection')
"

    # Test the main application
    if [ -f "$SCRIPT_DIR/../WhisperViaHF.py" ]; then
        log_info "Testing main application..."
        python "$SCRIPT_DIR/../WhisperViaHF.py" --help > /dev/null
        log_success "Main application test passed"
    fi

    log_success "Installation test completed"
}

# Display completion message
display_completion_message() {
    log_success "Whisper environment setup completed successfully!"
    echo
    echo "┌─────────────────────────────────────────────────────────────┐"
    echo "│                    Setup Complete                           │"
    echo "├─────────────────────────────────────────────────────────────┤"
    echo "│ Environment: $ENV_NAME"
    echo "│ Python: $PYTHON_VERSION"
    echo "│ CUDA: $CUDA_VERSION"
    echo "│ Location: $(conda info --base)/envs/$ENV_NAME"
    echo "│ Exports: app-whisper/exports/"
    echo "└─────────────────────────────────────────────────────────────┘"
    echo
    echo "Environment Files Created:"
    echo "  • ${ENV_NAME}_environment.yml          - Main environment file"
    echo "  • ${ENV_NAME}_cross_platform.yml       - Cross-platform compatible"
    echo "  • ${ENV_NAME}_minimal_*.yml             - Minimal dependencies only"
    echo "  • ${ENV_NAME}_full_*.yml                - All dependencies with versions"
    echo
    echo "To use the environment:"
    echo "  conda activate $ENV_NAME"
    echo
    echo "To recreate environment elsewhere:"
    echo "  conda env create -f app-whisper/exports/${ENV_NAME}_environment.yml"
    echo
    echo "To test Whisper:"
    echo "  python app-whisper/WhisperViaHF.py --help"
    echo "  python app-whisper/WhisperViaHF.py --benchmark --num-samples 1"
    echo
    echo "To run energy profiling:"
    echo "  cd sample-collection-scripts"
    echo "  ./launch_v2.sh --app-name Whisper --app-executable ../app-whisper/WhisperViaHF.py"
    echo
    echo "For more information, see: app-whisper/README.md"
}

# Display export completion message
display_export_completion_message() {
    log_success "Environment export completed successfully!"
    echo
    echo "┌─────────────────────────────────────────────────────────────┐"
    echo "│                   Export Complete                           │"
    echo "├─────────────────────────────────────────────────────────────┤"
    echo "│ Environment: $ENV_NAME"
    echo "│ Export Location: app-whisper/exports/"
    echo "└─────────────────────────────────────────────────────────────┘"
    echo
    echo "Environment Files Created:"
    echo "  • ${ENV_NAME}_environment.yml          - Main environment file"
    echo "  • ${ENV_NAME}_cross_platform.yml       - Cross-platform compatible"
    echo "  • ${ENV_NAME}_minimal_*.yml             - Minimal dependencies only"
    echo "  • ${ENV_NAME}_full_*.yml                - All dependencies with versions"
    echo
    echo "To recreate environment elsewhere:"
    echo "  conda env create -f app-whisper/exports/${ENV_NAME}_environment.yml"
    echo
    echo "To recreate with a different name:"
    echo "  conda env create -f app-whisper/exports/${ENV_NAME}_environment.yml -n new-env-name"
    echo
}

# Error handling
trap 'log_error "Setup failed. Check the error messages above."' ERR

# Run main function
main "$@"
