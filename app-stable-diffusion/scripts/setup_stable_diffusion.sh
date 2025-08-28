#!/bin/bash
#
# Quick setup and validation script for revised Stable Diffusion application
#

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
NC='\033[0m' # No Color

log_info() {
    echo -e "[$(date '+%H:%M:%S')] ${GREEN}[INFO]${NC} $*"
}

log_warn() {
    echo -e "[$(date '+%H:%M:%S')] ${YELLOW}[WARN]${NC} $*"
}

log_error() {
    echo -e "[$(date '+%H:%M:%S')] ${RED}[ERROR]${NC} $*"
}

log_info "🚀 Stable Diffusion Revision - Setup & Validation"
echo "========================================================="

# Check if we're in the right directory
if [[ ! -f "app-stable-diffusion/StableDiffusionViaHF.py" ]]; then
    log_error "Please run this script from the ai-inference-energy root directory"
    exit 1
fi

# Remove xformers if installed (not needed for CPU execution and causes compatibility issues)
log_info "Checking for xformers package..."
if pip show xformers &>/dev/null; then
    log_warn "Uninstalling xformers (not needed for CPU mode and causes compatibility issues)"
    pip uninstall -y xformers
else
    log_info "xformers not installed (good for CPU mode)"
fi

# Also check for fsspec version conflict with datasets
log_info "Checking for fsspec version conflicts..."
if pip show datasets &>/dev/null && pip show fsspec | grep "Version: 2025.5.1" &>/dev/null; then
    log_warn "Found fsspec version conflict, downgrading to compatible version"
    pip install "fsspec<=2025.3.0"
fi

# Check for scipy/glibc compatibility issues
log_info "Checking scipy compatibility with system glibc..."
if python3 -c "import scipy" 2>&1 | grep -q "GLIBCXX_3.4.29"; then
    log_warn "scipy incompatible with system glibc, downgrading..."
    pip uninstall -y scipy
    pip install scipy==1.9.3
    pip install "numpy<1.25"
fi

# Check Python version
python_version=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
log_info "Python version: $python_version"

# Check if Python version is 3.7 or higher using Python itself
python3 -c "
import sys
if sys.version_info < (3, 7):
    print('ERROR: Python 3.7+ required')
    sys.exit(1)
else:
    print('Python version check passed')
" || {
    log_error "Python 3.7+ required, found $python_version"
    exit 1
}

# Check dependencies
log_info "Checking dependencies..."

check_package() {
    local package=$1
    # Set CUDA_VISIBLE_DEVICES='' to avoid CUDA issues during import checks
    if CUDA_VISIBLE_DEVICES='' python3 -c "import $package" 2>/dev/null; then
        local version=$(CUDA_VISIBLE_DEVICES='' python3 -c "import $package; print(getattr($package, '__version__', 'unknown'))" 2>/dev/null)
        log_info "✅ $package: $version"
        return 0
    else
        log_warn "❌ $package: not installed"
        return 1
    fi
}

missing_packages=()

check_package "torch" || missing_packages+=("torch")
check_package "diffusers" || missing_packages+=("diffusers")
check_package "transformers" || missing_packages+=("transformers")
check_package "PIL" || missing_packages+=("Pillow")
check_package "numpy" || missing_packages+=("numpy")

if [[ ${#missing_packages[@]} -gt 0 ]]; then
    log_warn "Missing packages: ${missing_packages[*]}"
    log_info "Installing missing packages with compatible versions..."

    # Install torch first if needed - check for CUDA support
    if [[ " ${missing_packages[*]} " =~ " torch " ]]; then
        log_info "Installing PyTorch..."
        # Check if system has CUDA available
        if command -v nvidia-smi &> /dev/null; then
            log_info "NVIDIA GPU detected, installing CUDA-enabled PyTorch..."
            # Use CUDA 11.8 build which is compatible with CUDA 11.0+
            pip install torch==2.0.1 torchvision==0.15.2 --index-url https://download.pytorch.org/whl/cu118
        else
            log_info "No NVIDIA GPU detected, installing CPU-only PyTorch..."
            pip install torch==2.0.1+cpu torchvision==0.15.2+cpu --index-url https://download.pytorch.org/whl/cpu
        fi
    fi

    # Install compatible ML packages that work together
    if [[ " ${missing_packages[*]} " =~ " transformers " ]] || [[ " ${missing_packages[*]} " =~ " diffusers " ]]; then
        log_info "Installing compatible transformers and diffusers versions..."
        # Install specific versions that work together
        pip install transformers==4.30.0
        pip install diffusers==0.29.2
        pip install accelerate safetensors requests tqdm pyyaml regex tokenizers==0.13.3
        pip install huggingface-hub==0.33.2
        # Ensure numpy and scipy compatibility with older glibc
        pip install "numpy<1.25"
        pip install scipy==1.9.3
    fi

    # Install remaining packages
    pip install pillow numpy requests
else
    log_info "All required packages are installed"
fi

# Check GPU availability
log_info "Checking GPU availability..."
python3 -c "
import os
# Force CPU mode if CUDA compatibility issues
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import torch
print('PyTorch version:', torch.__version__)
print('CUDA available:', torch.cuda.is_available())
if torch.cuda.is_available():
    try:
        gpu_name = torch.cuda.get_device_name(0)
        memory_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f'✅ GPU: {gpu_name} ({memory_gb:.1f}GB)')

        # Check memory for different models
        if memory_gb >= 20:
            print('✅ Sufficient memory for SDXL (20GB+)')
        elif memory_gb >= 12:
            print('✅ Sufficient memory for SD v2.x (12GB+)')
        elif memory_gb >= 6:
            print('✅ Sufficient memory for SD v1.x (6GB+)')
        else:
            print('⚠️ Limited GPU memory - may need CPU offloading')
    except Exception as e:
        print(f'⚠️ GPU detected but CUDA compatibility issue: {e}')
        print('⚠️ Will run in CPU mode (slow but functional)')
else:
    print('⚠️ Running in CPU mode - will be slow but functional')
"

# Check Hugging Face authentication
log_info "Checking Hugging Face authentication..."
if python3 -c "from huggingface_hub import HfApi; api = HfApi(); user = api.whoami(); print(f'✅ Authenticated as: {user[\"name\"]}')"; then
    log_info "Hugging Face authentication successful"
else
    log_warn "Hugging Face authentication failed"
    log_info "Please run: huggingface-cli login"
    log_info "You'll need a Hugging Face account and accept the Stable Diffusion license"
fi

# Test import
log_info "Testing revised Stable Diffusion import..."
if python3 -c "
import sys
sys.path.append('app-stable-diffusion')
from StableDiffusionViaHF import StableDiffusionGenerator
print('✅ Import successful')
configs = StableDiffusionGenerator.MODEL_CONFIGS
print(f'✅ Available models: {list(configs.keys())}')
"; then
    log_info "Import test passed"
else
    log_error "Import test failed - check the revised implementation"
    exit 1
fi

# Quick validation test
log_info "Running quick validation test..."
if python3 test_stable_diffusion_revised.py --quick; then
    log_info "✅ Quick validation passed"
else
    log_warn "❌ Quick validation failed - see output above"
fi

# Test CLI interface
log_info "Testing CLI interface..."
if python3 app-stable-diffusion/StableDiffusionViaHF.py --help > /dev/null; then
    log_info "✅ CLI interface working"
else
    log_error "❌ CLI interface failed"
fi

# Framework integration test
log_info "Testing framework integration..."
if [[ -f "sample-collection-scripts/launch_v2.sh" ]]; then
    if ./sample-collection-scripts/launch_v2.sh --help > /dev/null; then
        log_info "✅ Framework integration ready"
    else
        log_warn "❌ Framework integration issues"
    fi
else
    log_warn "launch_v2.sh not found - framework integration not tested"
fi

echo ""
log_info "🎉 Setup and validation complete!"
echo "========================================================="
echo ""
echo "Next steps:"
echo "1. Run full test: python3 test_stable_diffusion_revised.py --model-variant sd-v1.4"
echo "2. Test with framework: cd sample-collection-scripts && ./launch_v2.sh --app-name StableDiffusion --help"
echo "3. Start energy profiling studies with different model variants"
echo ""
echo "Available model variants:"
echo "  - sd-v1.4    : 512x512, good for V100/A100"
echo "  - sd-v1.5    : 512x512, enhanced version"
echo "  - sd-v2.0    : 768x768, higher resolution"
echo "  - sd-v2.1    : 768x768, latest SD2 family"
echo "  - sdxl       : 1024x1024, requires H100 or high-memory GPU"
echo ""
echo "Example usage:"
echo "  python3 app-stable-diffusion/StableDiffusionViaHF.py --model-variant sd-v1.4 --prompt 'a test image' --num-images 1"
echo ""
echo "For detailed documentation, see: documentation/STABLE_DIFFUSION_REVISION.md"
