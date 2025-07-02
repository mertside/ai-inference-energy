# Interactive Scripts Summary

This document provides a quick reference for all interactive session helper scripts available in the AI Inference Energy Profiling Framework.

## Interactive Scripts Overview

| GPU Type | Script | Cluster | Partition | GPU Resource | CPU Tasks |
|----------|--------|---------|-----------|--------------|-----------|
| A100 | `interactive_a100.sh` | HPCC | toreador | --gpus-per-node=1 --reservation=ghazanfar | 16 |
| V100 | `interactive_v100.sh` | HPCC | matador | --gres=gpu:v100:1 | 40 |
| H100 | `interactive_h100.sh` | REPACSS | h100-build | --nodelist=rpg-93-9 --gpus-per-node=1 | 48 |

## Usage Patterns

### Quick Start
```bash
# Start interactive session
./interactive_a100.sh      # For A100
./interactive_v100.sh      # For V100  
./interactive_h100.sh      # For H100

# Test framework (run inside interactive session)
./interactive_a100.sh test # For A100
./interactive_v100.sh test # For V100
./interactive_h100.sh test # For H100

# Check node status
./interactive_a100.sh status # For A100
./interactive_v100.sh status # For V100
./interactive_h100.sh status # For H100
```

### Common Commands

All scripts support:
- `help` - Show usage information
- `status` - Check cluster node availability
- `test` - Run framework validation (in interactive session)
- No arguments - Start interactive session

Additional commands:
- `info` (V100 only) - Show detailed V100 specifications

## GPU Specifications

### A100 (HPCC - Toreador)
- **Memory**: 80GB HBM2
- **Architecture**: GA100
- **Graphics Frequencies**: 210-1410 MHz (7 steps)
- **Memory Frequency**: 1593 MHz
- **Cluster**: HPCC at Texas Tech University

### V100 (HPCC - Matador)  
- **Memory**: 32GB HBM2
- **Architecture**: GV100
- **Graphics Frequencies**: 405-1380 MHz (137 steps)
- **Memory Frequency**: 877 MHz
- **Cluster**: HPCC at Texas Tech University

### H100 (REPACSS - h100-build)
- **Memory**: 80GB HBM3
- **Architecture**: GH100
- **Graphics Frequencies**: 210-1980 MHz (estimated range)
- **Memory Frequency**: High-speed HBM3
- **Cluster**: REPACSS at Texas Tech University

## Interactive Session Commands

### Generated SLURM Commands

**A100 (HPCC)**:
```bash
srun --partition=toreador --gpus-per-node=1 --reservation=ghazanfar --pty bash
```

**V100 (HPCC)**:
```bash
srun --partition=matador --gres=gpu:v100:1 --ntasks=40 --pty bash
```

**H100 (REPACSS)**:
```bash
srun --partition=h100-build --nodelist=rpg-93-9 --gpus-per-node=1 --ntasks=48 --pty bash
```

## Quick Testing Examples

### A100 Testing
```bash
# Start session
./interactive_a100.sh

# In interactive session:
./launch.sh --gpu-type A100 --profiling-mode baseline --num-runs 1
nvidia-smi
```

### V100 Testing
```bash
# Start session  
./interactive_v100.sh

# In interactive session:
./launch.sh --gpu-type V100 --profiling-mode baseline --num-runs 1
nvidia-smi -q -d SUPPORTED_CLOCKS
```

### H100 Testing
```bash
# Start session
./interactive_h100.sh

# In interactive session:
./launch.sh --gpu-type H100 --profiling-mode baseline --num-runs 1
nvidia-smi
```

## Troubleshooting

### Common Issues

1. **Permission Denied**
   ```bash
   chmod +x interactive_*.sh
   ```

2. **SLURM Not Available**
   - Ensure you're on the correct cluster login node
   - Check SLURM module loading

3. **GPU Resource Unavailable**
   - Check node status with `status` command
   - Try different times of day
   - Consider using different GPU types

### Error Messages

- **"This test should be run in an interactive SLURM session"**
  - You tried to run `test` outside an interactive session
  - First run the script without arguments to start a session

- **"SLURM commands not available"**
  - You're not on a SLURM cluster or modules not loaded
  - Load SLURM modules or connect to cluster

### Getting Help

1. Run `./interactive_[gpu].sh help` for usage information
2. Check cluster documentation for node availability
3. Use `squeue` and `sinfo` commands for cluster status
4. Contact cluster administrators for resource allocation issues

## Integration with Framework

These interactive scripts work seamlessly with:
- Enhanced `launch.sh` with full CLI support
- Automatic DCGMI/nvidia-smi fallback
- All profiling modes (baseline, comprehensive, custom)
- Python 3.6+ compatibility
- All example applications (LSTM, Stable Diffusion, etc.)

The scripts provide the easiest way to get started with the framework and troubleshoot any configuration issues before submitting batch jobs.
