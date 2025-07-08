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
```bash
python StableDiffusionViaHF.py --model-variant sd-v1.4 --device cuda --prompt "your prompt here"
```

### Testing
```bash
cd tests/
# Run environment validation
./test_stable_diffusion.sh

# Run comprehensive validation
python validate_stable_diffusion.py
```

## GPU Requirements
- Tesla V100 (or compatible GPU)
- CUDA 11.0+ support
- 8GB+ VRAM recommended

## Energy Profiling
This implementation includes energy measurement capabilities for AI inference research.

---
**Part of the AI Inference Energy Research project at Texas Tech HPCC**
