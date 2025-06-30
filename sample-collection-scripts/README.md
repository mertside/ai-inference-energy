# Sample Collection Scripts - README

This directory contains the core scripts for running AI inference energy profiling experiments.

## Quick Start

### Basic Usage
```bash
# Clean previous results
./clean.sh -f

# Run complete experiment
./launch.sh
```

### GPU Configuration

#### A100 GPU (Default)
```bash
# Edit launch.sh and set:
GPU_TYPE="A100"
```
- Architecture: GA100
- Memory: 1215 MHz
- Core frequencies: 1410-510 MHz (61 frequencies)

#### V100 GPU
```bash
# Edit launch.sh and set:
GPU_TYPE="V100"
```
- Architecture: GV100  
- Memory: 877 MHz
- Core frequencies: 1380-405 MHz (105 frequencies)

### Profiling Tool Selection

#### DCGMI (Default)
```bash
# Edit launch.sh and set:
PROFILING_TOOL="dcgmi"
```
- Uses: `profile.py` and `control.sh`
- Requires: DCGMI tools installed
- Features: Comprehensive GPU metrics

#### nvidia-smi (Alternative)
```bash
# Edit launch.sh and set:
PROFILING_TOOL="nvidia-smi"
```
- Uses: `profile_smi.py` and `control_smi.sh`
- Requires: NVIDIA drivers (nvidia-smi)
- Features: Standard GPU monitoring

## Script Overview

### Core Scripts
- **`launch.sh`** - Main experiment orchestration
- **`profile.py`** - DCGMI-based GPU profiling  
- **`profile_smi.py`** - nvidia-smi-based GPU profiling
- **`control.sh`** - DCGMI-based frequency control
- **`control_smi.sh`** - nvidia-smi-based frequency control
- **`clean.sh`** - Workspace cleanup

### SLURM Scripts
- **`submit_job.sh`** - SLURM job submission
- **`test.sh`** - MPI test template

## Configuration Matrix

| GPU Type | Tool | Profile Script | Control Script | Memory Freq | Default Core |
|----------|------|----------------|----------------|-------------|--------------|
| A100 | dcgmi | profile.py | control.sh | 1215 MHz | 1410 MHz |
| A100 | nvidia-smi | profile_smi.py | control_smi.sh | 1215 MHz | 1410 MHz |
| V100 | dcgmi | profile.py | control.sh | 877 MHz | 1380 MHz |
| V100 | nvidia-smi | profile_smi.py | control_smi.sh | 877 MHz | 1380 MHz |

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

### A100 with DCGMI
```bash
# Edit launch.sh:
GPU_TYPE="A100"
PROFILING_TOOL="dcgmi"

# Run
./launch.sh
```

### V100 with nvidia-smi
```bash  
# Edit launch.sh:
GPU_TYPE="V100"
PROFILING_TOOL="nvidia-smi"

# Run
./launch.sh
```

### Custom Application
```bash
# Edit launch.sh:
declare -A APPLICATIONS=(
    ["LSTM"]="lstm"
    ["MyApp"]="my_custom_app"
)

declare -A APP_PARAMS=(
    ["LSTM"]=" > results/LSTM_RUN_OUT"
    ["MyApp"]=" > results/MY_APP_OUT"
)
```

For more details, see the main project README.md
