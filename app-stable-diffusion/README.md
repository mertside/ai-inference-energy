# Stable Diffusion for AI Inference Energy Research

## Overview
GPU-enabled Stable Diffusion implementation optimized for energy profiling and performance analysis on Texas Tech HPCC.

## Main Application
- **`StableDiffusionViaHF.py`** - Primary Stable Diffusion image generation application

## Directory Structure
```
app-stable-diffusion/
├── StableDiffusionViaHF.py          # Main application
├── StableDiffusionViaHF_original.py # Original implementation
├── README.md                        # This file
├── __init__.py                      # Package initialization
├── tests/                          # Test scripts
│   ├── test_stable_diffusion.sh    # Environment testing
│   └── validate_stable_diffusion.py # Validation scripts
└── scripts/                        # Setup and utility scripts
    └── setup_stable_diffusion.sh  # Setup script
```

## Quick Start

### Prerequisites
```bash
conda activate stable-diffusion-gpu
```

### Basic Usage

**⚠️ IMPORTANT: Must run on GPU nodes, not login nodes!**

#### Interactive GPU Session
```bash
# Start interactive GPU session
interactive -p matador                      # For V100
interactive -p toreador -g 1                # For A100
interactive -p h100-build -g 1 -w rpg-93-9  # For H100

# Activate environment
conda activate stable-diffusion-gpu

# Run Stable Diffusion
python StableDiffusionViaHF.py --model-variant sd-v1.4 --device cuda --prompt "your prompt here"
```

#### Batch Job Submission
```bash
# Submit as SLURM job (recommended for longer runs)
sbatch scripts/run_stable_diffusion_job.sh
```

### Testing
```bash
# Must be run on GPU nodes
interactive -p matador                      # For V100
interactive -p toreador -g 1                # For A100
interactive -p h100-build -g 1 -w rpg-93-9  # For H100

# Then run tests
cd tests/
conda activate stable-diffusion-gpu

# Run environment validation
./test_stable_diffusion.sh

# Run comprehensive validation
python validate_stable_diffusion.py
```

## GPU Requirements
- Tesla V100 (or compatible GPU)
- CUDA 11.0+ support
- 8GB+ VRAM recommended

## Common Issues & Troubleshooting

### GLIBCXX/Library Compatibility Issues
If you encounter `GLIBCXX_3.4.29` or similar library errors on HPC systems:

**Method 1: conda-forge approach**
```bash
conda uninstall pillow -y
conda install -c conda-forge pillow -y
```

**Method 2: Specific Pillow version**
```bash
pip install Pillow==9.5.0 --force-reinstall
```

**Method 3: Alternative Pillow implementation**
```bash
pip uninstall pillow -y
pip install pillow-simd
```

**Testing after fixes:**
```bash
python -c "import PIL.Image; print('Pillow OK')"
python -c "import torch; print('PyTorch OK')"
```

### Persistent GLIBCXX Issues
If the above fixes don't work, this indicates a fundamental system library incompatibility on the HPC cluster. Options:

1. **Contact HPC Support**: Request GLIBCXX_3.4.29+ system library updates
2. **Alternative Implementation**: Use a different Stable Diffusion framework
3. **Container Solution**: Use Singularity/Docker containers with compatible libraries
4. **Build from Source**: Compile packages against the system's specific GLIBC version

### Known Working Configurations
- **Tesla V100 + CUDA 11.0 + PyTorch 1.12.1**: ✅ **CONFIRMED WORKING**
  - transformers==4.33.2, diffusers==0.21.4, huggingface_hub==0.16.4 
  - conda-forge Pillow, safetensors==0.3.3
- **A100 + CUDA 11.8**: Generally better compatibility
- **H100 + CUDA 12.0**: Best compatibility with recent packages

### First-time Setup on GPU Node
```bash
# Get GPU session and set up environment
interactive -p matador
conda activate stable-diffusion-gpu

# Install missing dependencies
pip install typing_extensions tqdm requests
conda install packaging -y  # Use conda for packaging to avoid pip conflicts

# Fix NumPy and SciPy compatibility (CRITICAL for PyTorch 1.12.1)
pip install "numpy==1.21.6" "scipy==1.9.3" --force-reinstall

# Fix transformers and diffusers (compatible versions for PyTorch 1.12.1)
pip install transformers==4.33.2 --force-reinstall
pip install diffusers==0.21.4 --force-reinstall
pip install huggingface_hub==0.16.4 --force-reinstall

# Try Method 1: conda-forge for Pillow
conda uninstall pillow -y && conda install -c conda-forge pillow -y

# If that fails, try Method 2: specific version
pip install Pillow==9.5.0 --force-reinstall

# Fix safetensors
pip install safetensors==0.3.3 --force-reinstall

# Test basic functionality
python -c "import typing_extensions; print('typing_extensions OK')"
python -c "import tqdm; print('tqdm OK')"
python -c "import requests; print('requests OK')"
python -c "import packaging; print('packaging OK')"
python -c "import torch; print('PyTorch version:', torch.__version__)"
python -c "import torch; print('CUDA available:', torch.cuda.is_available())"
python -c "import transformers; print('transformers OK')"
python -c "from diffusers import DiffusionPipeline; print('diffusers OK')"
```

### Complete Environment Verification
```bash
# Run these tests in order:
python -c "import typing_extensions; print('✅ typing_extensions')"
python -c "import tqdm; print('✅ tqdm')"
python -c "import requests; print('✅ requests')"
python -c "import packaging; print('✅ packaging')"
python -c "import torch; print('✅ PyTorch:', torch.__version__)"
python -c "import torch; print('✅ CUDA available:', torch.cuda.is_available())"
python -c "import transformers; print('✅ transformers:', transformers.__version__)"
python -c "import PIL.Image; print('✅ Pillow')"
python -c "import safetensors; print('✅ safetensors')"
python -c "from diffusers import DiffusionPipeline; print('✅ diffusers')"
```

### Success Indicators
✅ **Working**: PyTorch 1.12.1+cu113, CUDA, Tesla V100-PCIE-32GB  
✅ **Working**: NumPy 1.21.6, SciPy 1.9.3 (CRITICAL for PyTorch 1.12.1)  
✅ **Working**: transformers, tqdm, packaging, typing_extensions, safetensors, requests  
✅ **Working**: diffusers 0.21.4 + transformers 4.33.2 + huggingface_hub 0.16.4 (compatible with PyTorch 1.12.1)  
✅ **Working**: Pillow (multiple fix methods available)  
🚀 **Ready**: Stable Diffusion image generation functional!

### Workaround for Persistent Issues
If GLIBCXX issues persist, consider these alternatives:

1. **Use Hugging Face Spaces**: Run Stable Diffusion in the cloud
2. **LSTM/Transformer alternatives**: Focus on energy profiling with simpler models
3. **Local development**: Test on systems with compatible libraries, deploy results

## Alternative AI Models for Energy Profiling
If Stable Diffusion remains problematic, these models work well for energy research:
- **LSTM networks** (`../app-lstm/`): Excellent for energy profiling
- **LLaMA models** (`../app-llama/`): Good GPU utilization patterns  
- **Custom transformers**: Controllable complexity for energy studies

### Environment Setup Issues
- **Missing dependencies**: Install with `pip install typing_extensions tqdm requests` and `conda install packaging -y`
- **Corrupted packages**: Use `pip install --force-reinstall --no-deps package_name` for pip conflicts
- **Version compatibility**: Use `transformers==4.33.2`, `diffusers==0.21.4` and `huggingface_hub==0.16.4` for PyTorch 1.12.1
- **NumPy/SciPy compatibility**: CRITICAL - Use `numpy==1.21.6` and `scipy==1.9.3` for PyTorch 1.12.1 compatibility
- **Version drift**: If packages get updated, re-run the exact version installs from "First-time Setup"
- **Ensure you're on a GPU node**: `nvidia-smi` should show GPU info
- **Verify environment activation**: `which python` should show conda path
- **Check CUDA**: `python -c "import torch; print(torch.cuda.is_available())"`
- **Environment incomplete**: Run the complete setup in "First-time Setup" section above

## Energy Profiling
This implementation includes energy measurement capabilities for AI inference research.

## ✅ Final Status: FULLY FUNCTIONAL
**Environment successfully configured for Tesla V100 + CUDA 11.0 + PyTorch 1.12.1**

All components working:
- ✅ PyTorch 1.12.1+cu113 with CUDA support
- ✅ NumPy 1.21.6 + SciPy 1.9.3 (compatible with PyTorch 1.12.1)
- ✅ Stable Diffusion pipeline (diffusers 0.21.4)
- ✅ All dependencies resolved and compatible
- ✅ Image generation confirmed working
- ✅ Ready for energy profiling and research

---
**Part of the AI Inference Energy Research project at Texas Tech HPCC**
