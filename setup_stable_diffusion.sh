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

# Check Python version
python_version=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
log_info "Python version: $python_version"

if [[ $(echo "$python_version >= 3.7" | bc -l) -eq 0 ]]; then
    log_error "Python 3.7+ required, found $python_version"
    exit 1
fi

# Check dependencies
log_info "Checking dependencies..."

check_package() {
    local package=$1
    if python3 -c "import $package" 2>/dev/null; then
        local version=$(python3 -c "import $package; print(getattr($package, '__version__', 'unknown'))" 2>/dev/null)
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
    log_info "Installing missing packages..."
    pip install "${missing_packages[@]}" accelerate xformers --upgrade
else
    log_info "All required packages are installed"
fi

# Check GPU availability
log_info "Checking GPU availability..."
python3 -c "
import torch
if torch.cuda.is_available():
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
else:
    print('⚠️ No GPU available - will use CPU (very slow)')
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
if [[ -f "sample-collection-scripts/launch.sh" ]]; then
    if ./sample-collection-scripts/launch.sh --help > /dev/null; then
        log_info "✅ Framework integration ready"
    else
        log_warn "❌ Framework integration issues"
    fi
else
    log_warn "launch.sh not found - framework integration not tested"
fi

echo ""
log_info "🎉 Setup and validation complete!"
echo "========================================================="
echo ""
echo "Next steps:"
echo "1. Run full test: python3 test_stable_diffusion_revised.py --model-variant sd-v1.4"
echo "2. Test with framework: cd sample-collection-scripts && ./launch.sh --app-name StableDiffusion --help"
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
