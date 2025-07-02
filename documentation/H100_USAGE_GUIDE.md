# H100 GPU Support - Usage Guide

This guide provides comprehensive information for using the AI Inference Energy Profiling Framework with NVIDIA H100 GPUs.

## 🚀 Quick Start with H100

### Basic Commands

```bash
# H100 baseline profiling (single frequency)
./launch.sh --gpu-type H100 --profiling-mode baseline

# H100 comprehensive DVFS study (all 104 frequencies)
./launch.sh --gpu-type H100 --profiling-mode dvfs

# H100 with custom application
./launch.sh --gpu-type H100 --app-name "CustomApp" --app-executable "my_app"
```

### SLURM Job Submission

```bash
# Quick baseline test (REPACSS H100 node)
sbatch submit_job_h100_baseline.sh

# Custom application profiling
sbatch submit_job_h100_custom_app.sh

# Full DVFS study (WARNING: 12+ hours)
sbatch submit_job_h100_comprehensive.sh

# Interactive session for testing
interactive -p h100-build -g 1 -w rpg-93-9

# OR use the helper script
./h100_interactive.sh          # Start interactive session
./h100_interactive.sh test     # Run quick test (in session)
./h100_interactive.sh status   # Check node availability
```

## 🔧 H100 GPU Specifications

### Architecture Details
- **GPU Architecture**: GH100 (Hopper)
- **Memory Frequency**: 1593 MHz
- **Default Core Frequency**: 1755 MHz (maximum)
- **Minimum Core Frequency**: 210 MHz
- **Frequency Step Size**: 15 MHz
- **Total Frequencies**: 104 different settings

### Frequency Range
```
1755, 1740, 1725, 1710, 1695, 1680, ..., 270, 255, 240, 225, 210 MHz
```

### SLURM Configuration
- **Cluster**: REPACSS at Texas Tech University
- **Partition**: h100-build
- **Node**: rpg-93-9 (specific H100 node)
- **GPU Resource**: `--gres=gpu:1`
- **Recommended CPUs**: 8 (`--cpus-per-task=8`)
- **Interactive Sessions**: `interactive -p h100-build -g 1 -w rpg-93-9`

## 📊 Performance Characteristics

### Expected Runtime Estimates

| Mode | Frequencies | Runs per Freq | Est. Time |
|------|-------------|---------------|-----------|
| Baseline | 1 (1755 MHz) | 3 | 5-10 min |
| Baseline Extended | 1 (1755 MHz) | 10 | 15-30 min |
| DVFS Study | 104 (all) | 3 | 12-15 hours |
| DVFS Extended | 104 (all) | 5 | 20-25 hours |

### Memory and Storage Requirements
- **GPU Memory**: H100s typically have 80GB or 96GB HBM3
- **Disk Space**: 
  - Baseline: ~100 MB
  - Full DVFS: ~5-10 GB
- **System Memory**: 32+ GB recommended for comprehensive studies

## 🛠️ Configuration Examples

### 1. Quick Baseline Test
```bash
./launch.sh \
  --gpu-type H100 \
  --profiling-mode baseline \
  --num-runs 3 \
  --sleep-interval 1
```

### 2. Custom Application with H100
```bash
./launch.sh \
  --gpu-type H100 \
  --profiling-mode baseline \
  --app-name "LLaMA" \
  --app-executable "llama_inference" \
  --app-params "--model llama-13b --prompt 'test' > results/llama_output.log" \
  --num-runs 5
```

### 3. Selective Frequency Testing
For custom frequency selection, modify the `CORE_FREQUENCIES` array in `launch.sh`:
```bash
# Example: Test only high frequencies (1755-1500 MHz)
CORE_FREQUENCIES=(1755 1740 1725 1710 1695 1680 1665 1650 1635 1620 1605 1590 1575 1560 1545 1530 1515 1500)
```

### 4. Comprehensive DVFS Study
```bash
./launch.sh \
  --gpu-type H100 \
  --profiling-mode dvfs \
  --num-runs 3 \
  --sleep-interval 2 \
  --profiling-tool dcgmi
```

## 🔍 Monitoring and Results

### Real-time Monitoring
```bash
# Monitor GPU usage during experiment
watch -n 1 'nvidia-smi --query-gpu=name,utilization.gpu,memory.used,power.draw,clocks.gr,clocks.mem --format=csv'

# Monitor SLURM job
squeue -u $USER
tail -f LSTM_H100_*.out
```

### Result Files
```
results/
├── GH100-baseline-LSTM-1755-0    # Individual run results
├── GH100-baseline-LSTM-1755-1    # GPU metrics and power data
├── GH100-baseline-LSTM-1755-2    
└── GH100-baseline-lstm-perf.csv  # Performance summary
```

### Performance Analysis
```bash
# Quick performance summary
grep "performance saved" *.err | tail -10

# Power consumption analysis
python -c "
import pandas as pd
data = pd.read_csv('results/GH100-baseline-lstm-perf.csv')
print(f'Average execution time: {data.iloc[:, 1].mean():.2f}s')
"
```

## ⚠️ Important Considerations

### 1. Runtime Warnings
- **Comprehensive DVFS studies take 12-15 hours** with 104 frequencies
- Plan accordingly for cluster time limits
- Consider running baseline tests first

### 2. Frequency Control
- H100 frequency control requires appropriate permissions
- DCGMI is preferred for H100 (with nvidia-smi fallback)
- Test frequency control before long experiments:
```bash
./control.sh 1593 1755  # Test setting H100 frequencies
```

### 3. Power Limits
- H100s have high power consumption (400-700W typical)
- Monitor cluster power policies
- Some frequencies may be power-limited

### 4. Cluster-Specific Adjustments
The scripts are now configured for REPACSS at Texas Tech University:
- **Partition**: `h100-build` 
- **Node**: `rpg-93-9` (specific H100 node)
- **GPU Resource**: `--gres=gpu:1`
- **Modules**: Only `cuda` module loaded
- **Interactive**: `interactive -p h100-build -g 1 -w rpg-93-9`

If using a different cluster, you may need to adjust:
- Partition name and node specification
- GPU resource specification
- Module names and availability

## 🐛 Troubleshooting

### Common Issues

1. **"GPU type set to H100 but detected GPU doesn't appear to be H100"**
   - Check with `nvidia-smi` that you have an H100
   - Verify SLURM allocation: `squeue -u $USER`

2. **DCGMI not available**
   - Framework automatically falls back to nvidia-smi
   - Install DCGMI for better profiling: `module load dcgmi`

3. **Frequency control fails**
   - Check permissions: `dcgmi config --help`
   - Try nvidia-smi fallback: `--profiling-tool nvidia-smi`

4. **Job timeout**
   - Increase SLURM time limit: `#SBATCH --time=15:00:00`
   - Reduce frequencies or runs for testing

### Getting Help
```bash
# Check H100 capabilities
nvidia-smi -q -d SUPPORTED_CLOCKS

# Verify DCGMI
dcgmi discovery --list

# Test framework
./launch.sh --help
./launch.sh --gpu-type H100 --profiling-mode baseline --num-runs 1
```

## 📚 Additional Resources

- **Main README**: [`../README.md`](../README.md)
- **General Usage**: [`../documentation/USAGE_EXAMPLES.md`](../documentation/USAGE_EXAMPLES.md)
- **SLURM Guide**: [`../documentation/SUBMIT_JOBS_README.md`](../documentation/SUBMIT_JOBS_README.md)
- **Troubleshooting**: [`../documentation/QUICK_FIX_GUIDE.md`](../documentation/QUICK_FIX_GUIDE.md)

---

**Note**: H100 support is based on the frequency range you discovered (210-1755 MHz in 15MHz steps). Actual availability and performance may vary depending on your specific cluster configuration and H100 variant.
