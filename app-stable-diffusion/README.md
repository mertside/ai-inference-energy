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
# Start interactive GPU session (using unified helper)
./interactive_gpu.sh v100                   # For V100 on HPCC
./interactive_gpu.sh a100                   # For A100 on HPCC (requires reservation)
./interactive_gpu.sh h100                   # For H100 on REPACSS

# Once in session, activate environment
conda activate stable-diffusion-gpu
```

#### Quick Start Examples

**1. Simple Image Generation**
```bash
python StableDiffusionViaHF.py --prompt "a photograph of an astronaut riding a horse" --steps 10 --log-level INFO
```

**2. High Quality Generation**
```bash
python StableDiffusionViaHF.py --prompt "a majestic mountain landscape at sunset" --steps 30 --guidance-scale 7.5 --width 768 --height 768
```

**3. Fast Generation with Turbo Model**
```bash
python StableDiffusionViaHF.py --model-variant sd-turbo --prompt "a futuristic city skyline" --steps 4 --guidance-scale 1.0
```

**4. Multiple Images with Different Styles**
```bash
python StableDiffusionViaHF.py --prompt "a cute cat wearing a wizard hat" --negative-prompt "blurry, low quality" --num-images 4 --steps 20
```

**5. Energy Research Benchmark**
```bash
python StableDiffusionViaHF.py --benchmark --benchmark-type energy_research --export-metrics energy_metrics.json
```

#### Advanced Examples

**Memory-Optimized Generation (for limited VRAM)**
```bash
python StableDiffusionViaHF.py --prompt "detailed fantasy artwork" --enable-cpu-offload --dtype float16 --steps 15
```

**High-Resolution Generation (requires sufficient VRAM)**
```bash
python StableDiffusionViaHF.py --model-variant sdxl --prompt "photorealistic portrait of a person" --width 1024 --height 1024 --steps 25
```

**Scheduler Comparison for Research**
```bash
python StableDiffusionViaHF.py --prompt "test prompt for scheduler analysis" --scheduler-comparison --export-metrics scheduler_comparison.json
```

**Multi-Resolution Analysis**
```bash
python StableDiffusionViaHF.py --prompt "benchmark image" --multi-resolution --export-metrics resolution_analysis.json
```

#### Model Options
- `sd-v1.4`: Stable Diffusion v1.4 (default)
- `sd-v1.5`: Stable Diffusion v1.5
- `sd-v2.0`: Stable Diffusion v2.0 
- `sd-v2.1`: Stable Diffusion v2.1
- `sdxl`: Stable Diffusion XL (high quality, requires more VRAM)
- `sdxl-turbo`: SDXL Turbo (fast generation)
- `sd-turbo`: SD Turbo (ultra-fast generation)

#### Scheduler Options
- `dpm++`: DPM++ (default, good quality-speed balance)
- `euler`: Euler scheduler
- `ddim`: DDIM scheduler
- `unipc`: UniPC scheduler (fast)
- `heun`: Heun scheduler (high quality)

Use `--list-schedulers` to see all available options with descriptions.

#### Batch Job Submission
```bash
# Submit as SLURM job (recommended for longer runs)
sbatch scripts/run_stable_diffusion_job.sh
```

### Testing
```bash
# Must be run on GPU nodes - use unified helper
./interactive_gpu.sh v100                   # For V100 on HPCC
./interactive_gpu.sh a100                   # For A100 on HPCC (requires reservation)
./interactive_gpu.sh h100                   # For H100 on REPACSS

# Once in session, activate environment and test
conda activate stable-diffusion-gpu

# Quick functionality test
python StableDiffusionViaHF.py --prompt "test image generation" --steps 5 --log-level INFO

# Run environment validation
cd tests/
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

#### Research and Energy Profiling Examples

**Energy Efficiency Analysis**
```bash
# Quick energy baseline (10 steps)
python StableDiffusionViaHF.py --prompt "energy efficiency test image" --steps 10 --export-metrics baseline_10steps.json

# Quality comparison (30 steps)
python StableDiffusionViaHF.py --prompt "energy efficiency test image" --steps 30 --export-metrics quality_30steps.json

# Turbo model comparison
python StableDiffusionViaHF.py --model-variant sd-turbo --prompt "energy efficiency test image" --steps 4 --export-metrics turbo_4steps.json
```

**GPU Memory Scaling Studies**
```bash
# Small resolution (512x512)
python StableDiffusionViaHF.py --prompt "memory scaling test" --width 512 --height 512 --export-metrics mem_512.json

# Medium resolution (768x768)  
python StableDiffusionViaHF.py --prompt "memory scaling test" --width 768 --height 768 --export-metrics mem_768.json

# Large resolution (1024x1024) - requires SDXL
python StableDiffusionViaHF.py --model-variant sdxl --prompt "memory scaling test" --width 1024 --height 1024 --export-metrics mem_1024.json
```

**Cross-Architecture Performance Analysis**
```bash
# V100 optimal settings
python StableDiffusionViaHF.py --prompt "architecture comparison" --dtype float16 --steps 20 --export-metrics v100_perf.json

# A100 optimal settings
python StableDiffusionViaHF.py --prompt "architecture comparison" --dtype float16 --steps 30 --batch-size 2 --export-metrics a100_perf.json

# H100 optimal settings  
python StableDiffusionViaHF.py --model-variant sdxl --prompt "architecture comparison" --dtype bfloat16 --steps 25 --export-metrics h100_perf.json
```

**Integration with Energy Profiling Framework**
```bash
# Use with the sample-collection-scripts framework
cd ../sample-collection-scripts

# V100 profiling
./launch.sh --gpu-type V100 --app-name StableDiffusion --app-executable "../app-stable-diffusion/StableDiffusionViaHF" --app-params "--prompt 'profiling test image' --steps 15 --log-level WARNING"

# A100 profiling
./launch.sh --gpu-type A100 --app-name StableDiffusion --app-executable "../app-stable-diffusion/StableDiffusionViaHF" --app-params "--prompt 'profiling test image' --steps 20 --dtype float16"

# H100 profiling with SDXL
./launch.sh --gpu-type H100 --app-name StableDiffusion --app-executable "../app-stable-diffusion/StableDiffusionViaHF" --app-params "--model-variant sdxl --prompt 'profiling test image' --steps 15"
```

## Quick Reference

### Essential Parameters
| Parameter | Description | Common Values |
|-----------|-------------|---------------|
| `--prompt` | Text description of desired image | Any descriptive text |
| `--steps` | Number of inference steps | 4 (turbo), 10 (fast), 20 (balanced), 30+ (quality) |
| `--model-variant` | Model to use | `sd-v1.4`, `sd-v1.5`, `sdxl`, `sd-turbo` |
| `--scheduler` | Sampling algorithm | `dpm++`, `euler`, `ddim`, `unipc` |
| `--width`, `--height` | Image dimensions | 512, 768, 1024 (SDXL) |
| `--guidance-scale` | Prompt adherence strength | 1.0 (turbo), 7.5 (default), 15.0 (strong) |
| `--dtype` | Model precision | `float16` (memory efficient), `float32` (quality) |
| `--log-level` | Verbosity | `WARNING` (quiet), `INFO` (normal), `DEBUG` (verbose) |

### Performance Tips
- **Fast Generation**: Use `sd-turbo` with `--steps 4 --guidance-scale 1.0`
- **Memory Constrained**: Add `--enable-cpu-offload --dtype float16`
- **High Quality**: Use `sdxl` with `--steps 25-30`
- **Research Reproducibility**: Always specify `--seed <number>`
- **Energy Profiling**: Use `--log-level WARNING` to reduce output noise
