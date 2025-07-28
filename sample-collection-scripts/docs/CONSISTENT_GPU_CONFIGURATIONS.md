# GPU Job Submission Scripts - Consistent Configuration Structure

## Overview

All three GPU job submission scripts (`submit_job_h100.sh`, `submit_job_a100.sh`, `submit_job_v100.sh`) now follow a consistent 32-configuration structure with standardized enumeration across all GPU types.

## Configuration Categories (1-32)

### 📋 BASELINE CONFIGURATIONS (1-5)
**Purpose**: Quick validation and basic benchmarking
- **1**: LSTM - Basic deep learning benchmark
- **2**: Stable Diffusion - Image generation benchmark  
- **3**: LLaMA - Text generation benchmark
- **4**: Whisper - Speech recognition benchmark
- **5**: Vision Transformer - Image classification benchmark

**Timing**: `--time=01:00:00` (1 hour)

### 📊 CUSTOM FREQUENCY CONFIGURATIONS (6-10)
**Purpose**: Three-point frequency analysis (low/mid/high)
- **6**: LSTM Custom
- **7**: Stable Diffusion Custom
- **8**: LLaMA Custom  
- **9**: Whisper Custom
- **10**: Vision Transformer Custom

**Timing**: 
- H100/A100: `--time=01:00:00` (1 hour)
- V100: `--time=02:00:00` (2 hours)

### 🔄 DVFS STUDY CONFIGURATIONS (11-13)
**Purpose**: Complete frequency sweep analysis
- **11**: Comprehensive DVFS - All frequencies
- **12**: Efficient DVFS - Reduced runs for faster completion
- **13**: Statistical DVFS - High statistical power

**Timing**:
- H100: `--time=06:00:00` (6 hours)
- A100: `--time=05:00:00` (5 hours) 
- V100: `--time=12:00:00` (12 hours)

### 🔬 APPLICATION-SPECIFIC DVFS STUDIES (14-18)
**Purpose**: Complete frequency analysis for specific applications
- **14**: LSTM DVFS
- **15**: Stable Diffusion DVFS
- **16**: LLaMA DVFS
- **17**: Whisper DVFS
- **18**: Vision Transformer DVFS

**Timing**:
- H100: `--time=08:00:00` (8 hours)
- A100: `--time=06:00:00` (6 hours)
- V100: `--time=10:00:00` (10 hours)

### 🎓 RESEARCH STUDY CONFIGURATIONS (19-21)
**Purpose**: Advanced research configurations
- **19**: Energy Efficiency Study - Seven-point frequency analysis
- **20**: Extended Baseline - Higher statistical significance
- **21**: Scaling Analysis - Batch size impact study

**Timing**: `--time=02:00:00` (2 hours)

### 🚀 ADVANCED GPU-SPECIFIC CONFIGURATIONS (22-25)
**Purpose**: GPU-specific advanced features

#### H100 (Hopper Architecture)
- **22**: Transformer Engine - Advanced LLM optimization
- **23**: 4th Gen Tensor Cores - FP8 precision
- **24**: Memory Stress Test - 80GB HBM3
- **25**: Flagship Performance - Maximum capability

#### A100 (Ampere Architecture)  
- **22**: Tensor Cores - Mixed precision optimization
- **23**: 3rd Gen Tensor Cores - Maximum performance
- **24**: Memory Stress Test - 40GB HBM2e
- **25**: Flagship Performance - Maximum capability

#### V100 (Volta Architecture)
- **22**: Tensor Cores - Mixed precision optimization
- **23**: 1st Gen Tensor Cores - Mixed precision
- **24**: Memory Stress Test - 32GB HBM2
- **25**: Flagship Performance - Maximum capability

**Timing**:
- H100: `--time=04:00:00` (4 hours)
- A100/V100: `--time=03:00:00` (3 hours)

### 🛠️ UTILITY AND DEBUG CONFIGURATIONS (26-28)
**Purpose**: Debugging and utility functions
- **26**: NVIDIA-SMI Fallback - When DCGMI unavailable
- **27**: Debug Mode - Reduced workload with debug logging
- **28**: NVIDIA-SMI Debug - Minimal workload fallback

**Timing**: `--time=01:00:00` (1 hour)

### ⚡ SAMPLING INTERVAL AND MULTI-GPU CONFIGURATIONS (29-32)
**Purpose**: New features for fine-grained monitoring
- **29**: High-Frequency Sampling - 10ms interval
- **30**: Low-Frequency Sampling - 200ms interval  
- **31**: Multi-GPU Monitoring - All available GPUs
- **32**: Ultra-Fine Monitoring - 25ms + all GPUs

**Timing**: `--time=02:00:00` (2 hours)

## GPU-Specific Frequency Ranges

### H100 (Hopper)
- **Frequencies**: 86 available (510-1785 MHz)
- **Memory**: 2619 MHz
- **Custom Examples**: `'510,960,1785'`

### A100 (Ampere)
- **Frequencies**: 61 available (510-1410 MHz)
- **Memory**: 1215 MHz (fixed)
- **Custom Examples**: `'510,960,1410'`

### V100 (Volta)
- **Frequencies**: 117 available (405-1380 MHz)
- **Memory**: 877 MHz (fixed)
- **Custom Examples**: `'405,892,1380'`

## New Features (v2.1)

All scripts now support:

### Sampling Interval Control
```bash
--sampling-interval MS    # 10-1000ms, default: 50ms
```

### Multi-GPU Monitoring
```bash
--all-gpus               # Monitor all available GPUs
```

### Usage Examples
```bash
# Fast sampling
LAUNCH_ARGS="--gpu-type H100 --sampling-interval 10 --app-name LSTM ..."

# Multi-GPU monitoring  
LAUNCH_ARGS="--gpu-type A100 --all-gpus --app-name ViT ..."

# Combined advanced monitoring
LAUNCH_ARGS="--gpu-type V100 --sampling-interval 25 --all-gpus --app-name StableDiffusion ..."
```

## Consistency Benefits

1. **Unified Structure**: All GPU types follow identical enumeration (1-32)
2. **Predictable Categories**: Same configuration types at same numbers
3. **Scalable Framework**: Easy to add new configurations consistently
4. **Cross-GPU Comparison**: Direct comparison between GPU architectures
5. **Standardized Documentation**: Common reference structure
6. **Enhanced Features**: Sampling interval and multi-GPU support across all GPUs

## Usage Instructions

1. Choose your GPU type: `submit_job_h100.sh`, `submit_job_a100.sh`, or `submit_job_v100.sh`
2. Uncomment desired configuration (1-32)
3. Adjust SLURM timing based on configuration category
4. Submit: `sbatch submit_job_<gpu>.sh`
5. Monitor results in auto-generated directories with job IDs
