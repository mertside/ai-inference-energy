# Sample Collection Scripts - Enhanced Profiling Framework

This directory contains the core scripts for running AI inference energy profiling experiments with **complete command-line interface support**.

## ✨ New Features

- 🔧 **Complete CLI Interface**: Configure all experiments via command-line arguments
- 🎯 **Dual GPU Support**: Native A100 and V100 configurations
- 🛠️ **Intelligent Tool Fallback**: Automatic fallback from DCGMI to nvidia-smi
- 📊 **Flexible Modes**: DVFS (full frequency sweep) or baseline (single frequency)
- 🚀 **Enhanced SLURM Integration**: Multiple job submission scripts

## Quick Start

### CLI-Based Usage (Recommended)
```bash
# Show all available options
./launch.sh --help

# Default A100 DVFS experiment
./launch.sh

# V100 baseline experiment 
./launch.sh --gpu-type V100 --profiling-mode baseline

# Custom application profiling
./launch.sh \
  --app-name "StableDiffusion" \
  --app-executable "stable_diffusion" \
  --app-params "--prompt 'A beautiful landscape' --steps 20"
```

### Legacy Usage (Still Supported)
```bash
# Clean previous results
./clean.sh -f

# Run with defaults
./launch.sh
```

## Command-Line Interface

The `launch.sh` script now accepts comprehensive command-line arguments:

```bash
./launch.sh [OPTIONS]

Options:
  --gpu-type TYPE          GPU type: A100 or V100 (default: A100)
  --profiling-tool TOOL    Profiling tool: dcgmi or nvidia-smi (default: dcgmi)
  --profiling-mode MODE    Mode: dvfs or baseline (default: dvfs)
  --num-runs NUM           Number of runs per frequency (default: 2)
  --sleep-interval SEC     Sleep between runs in seconds (default: 1)
  --app-name NAME          Application display name (default: LSTM)
  --app-executable PATH    Application executable path (default: lstm)
  --app-params "PARAMS"    Application parameters (default: "")
  -h, --help              Show help and examples
```

### GPU Configuration

#### A100 GPU (Toreador Partition)
```bash
./launch.sh --gpu-type A100
```
- Architecture: GA100
- Memory: 1215 MHz
- Core frequencies: 1410-510 MHz (61 frequencies)
- SLURM partition: toreador
- Cluster: HPCC at Texas Tech University
- Interactive helper: `./interactive_a100.sh`

#### V100 GPU (Matador Partition)
```bash
./launch.sh --gpu-type V100
```
- Architecture: GV100  
- Memory: 877 MHz
- Core frequencies: 1380-510 MHz (117 frequencies)
- SLURM partition: matador
- Interactive helper: `./interactive_v100.sh`

#### H100 GPU (REPACSS h100-build Partition)
```bash
./launch.sh --gpu-type H100
```
- Architecture: GH100  
- Memory: 1593 MHz
- Core frequencies: 1785-510 MHz (86 frequencies in 15MHz steps)
- SLURM partition: h100-build (node: rpg-93-9)
- Cluster: REPACSS at Texas Tech University
- Interactive helper: `./interactive_h100.sh`

### Profiling Tool Selection

#### DCGMI (Default with Automatic Fallback)
```bash
./launch.sh --profiling-tool dcgmi
```
- Uses: `profile.py` and `control.sh`
- Requires: DCGMI tools installed
- Features: Comprehensive GPU metrics
- **Automatic fallback** to nvidia-smi if DCGMI unavailable

#### nvidia-smi (Alternative)
```bash
./launch.sh --profiling-tool nvidia-smi
```
- Uses: `profile_smi.py` and `control_smi.sh`
- Requires: NVIDIA drivers (nvidia-smi)
- Features: Standard GPU monitoring
### Experiment Modes

#### DVFS Mode (Default)
```bash
./launch.sh --profiling-mode dvfs
```
- **Full frequency sweep** across all supported frequencies
- Comprehensive energy analysis
- Longer execution time (1-6 hours depending on configuration)

#### Baseline Mode
```bash
./launch.sh --profiling-mode baseline
```
- **Single frequency** at default GPU settings
- Quick profiling for testing and validation
- Shorter execution time (~30 minutes)

## Script Overview

### Core Scripts
- **`launch.sh`** - 🎯 Main experiment orchestration (CLI enhanced)
- **`profile.py`** - DCGMI-based GPU profiler  
- **`profile_smi.py`** - nvidia-smi-based GPU profiler
- **`control.sh`** - DCGMI-based frequency control
- **`control_smi.sh`** - nvidia-smi-based frequency control
- **`clean.sh`** - Enhanced workspace cleanup
- **`lstm.py`** - LSTM benchmark application

### Interactive Helpers
- **`interactive_a100.sh`** - A100 interactive session helper (HPCC toreador)
- **`interactive_v100.sh`** - V100 interactive session helper (HPCC matador)  
- **`interactive_h100.sh`** - H100 interactive session helper (REPACSS h100-build)

Each interactive helper provides:
- Quick interactive session startup
- GPU detection and framework testing
- Node status checking
- Usage examples and troubleshooting

### SLURM Scripts
- **`submit_job.sh`** - Main A100 SLURM submission (toreador)
- **`submit_job_v100_baseline.sh`** - V100 baseline profiling (matador)
- **`submit_job.sh`** - Main A100 DVFS experiment (original)
- **`submit_job_a100_baseline.sh`** - A100 baseline profiling (quick test)
- **`submit_job_a100_comprehensive.sh`** - A100 comprehensive DVFS study
- **`submit_job_a100_custom_app.sh`** - A100 custom application examples
- **`submit_job_v100_baseline.sh`** - V100 baseline profiling
- **`submit_job_v100_comprehensive.sh`** - V100 comprehensive profiling
- **`submit_job_v100_custom_app.sh`** - V100 custom application examples
- **`submit_job_h100_baseline.sh`** - H100 baseline profiling
- **`submit_job_h100_comprehensive.sh`** - H100 comprehensive profiling  
- **`submit_job_h100_custom_app.sh`** - H100 custom application examples
- **`submit_job_custom_app.sh`** - Custom application examples
- **`submit_job_comprehensive.sh`** - Full DVFS study

## Configuration Matrix

| GPU Type | Tool | Profile Script | Control Script | Memory Freq | Default Core |
|----------|------|----------------|----------------|-------------|--------------|
| A100 | dcgmi | profile.py | control.sh | 1215 MHz | 1410 MHz |
| A100 | nvidia-smi | profile_smi.py | control_smi.sh | 1215 MHz | 1410 MHz |
| V100 | dcgmi | profile.py | control.sh | 877 MHz | 1380 MHz |
| V100 | nvidia-smi | profile_smi.py | control_smi.sh | 877 MHz | 1380 MHz |
| H100 | dcgmi | profile.py | control.sh | 1593 MHz | 1755 MHz |
| H100 | nvidia-smi | profile_smi.py | control_smi.sh | 1593 MHz | 1755 MHz |

## Individual Script Usage

### GPU Profiling
```bash
# DCGMI profiling
./profile.py "python lstm.py"

# nvidia-smi profiling  
./profile_smi.py "python lstm.py"
```

### Frequency Control
```bash
# DCGMI control
./control.sh 1215 1410    # memory_freq core_freq

# nvidia-smi control
./control_smi.sh 1215 1410
```

### Cleanup
```bash
# Interactive cleanup
./clean.sh

# Force cleanup (no prompts)
./clean.sh -f

# Verbose cleanup
./clean.sh -v
```

## Output Structure

Results are saved in the `results/` directory:
```
results/
├── GA100-dvfs-LSTM-1410-0     # GPU_ARCH-mode-app-freq-iteration
├── GA100-dvfs-LSTM-1410-1
├── GA100-dvfs-LSTM-1395-0
└── GA100-dvfs-lstm-perf.csv   # Performance summary
```

## Requirements

### For DCGMI
- NVIDIA GPU (A100/V100)
- DCGMI tools installed
- Python 3.8+
- Permissions for GPU control

### For nvidia-smi  
- NVIDIA GPU (A100/V100)
- NVIDIA drivers (nvidia-smi)
- Python 3.8+
- May require sudo for frequency control

## Troubleshooting

### Permission Issues
```bash
# For DCGMI
sudo usermod -a -G nvidia $USER

# For nvidia-smi
sudo nvidia-smi -pm 1  # Enable persistence mode
```

### Script Not Found
```bash
# Make scripts executable
chmod +x *.sh *.py
```

### GPU Detection
```bash
# Check GPU availability
nvidia-smi
dcgmi discovery --list
```

## Examples

### Basic Usage Examples

#### Default A100 DVFS Experiment
```bash
./launch.sh
```

#### V100 Baseline Testing
```bash
./launch.sh --gpu-type V100 --profiling-mode baseline --num-runs 1
```

#### Custom Application Profiling
```bash
# Stable Diffusion
./launch.sh \
  --app-name "StableDiffusion" \
  --app-executable "../app-stable-diffusion/StableDiffusionViaHF.py" \
  --app-params "--prompt 'A beautiful landscape' --steps 20"

# LLaMA
./launch.sh \
  --app-name "LLaMA" \
  --app-executable "../app-llama-collection/LlamaViaHF" \
  --app-params "--max-length 100"
```

#### Quick Testing Configuration
```bash
./launch.sh --num-runs 1 --sleep-interval 0 --profiling-mode baseline
```

#### Comprehensive Experiment
```bash
./launch.sh --gpu-type A100 --profiling-mode dvfs --num-runs 3
```

### SLURM Usage Examples

#### Submit A100 Job
```bash
sbatch submit_job.sh
```

#### Submit V100 Baseline Job
```bash
sbatch submit_job_v100_baseline.sh
```

#### Submit V100 Custom Application Job
```bash
sbatch submit_job_v100_custom_app.sh
```

#### Submit Custom Application Job
```bash
sbatch submit_job_custom_app.sh
```

### Legacy Configuration (Still Supported)

If you prefer to edit the script directly instead of using CLI arguments:

#### A100 with DCGMI
```bash
# Edit launch.sh configuration section:
GPU_TYPE="A100"
PROFILING_TOOL="dcgmi"
PROFILING_MODE="dvfs"
```

#### V100 with nvidia-smi
```bash  
# Edit launch.sh configuration section:
GPU_TYPE="V100"
PROFILING_TOOL="nvidia-smi"
PROFILING_MODE="baseline"
```

## 📚 Documentation

For detailed information, see:
- **[`../documentation/USAGE_EXAMPLES.md`](../documentation/USAGE_EXAMPLES.md)** - Comprehensive CLI usage examples
- **[`../documentation/SUBMIT_JOBS_README.md`](../documentation/SUBMIT_JOBS_README.md)** - SLURM job submission guide
- **[`../documentation/GPU_USAGE_GUIDE.md`](../documentation/GPU_USAGE_GUIDE.md)** - Complete GPU support and troubleshooting guide

For more details, see the main project README.md
