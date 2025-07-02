# A100 GPU Support - Usage Guide

This guide provides comprehensive information for using the AI Inference Energy Profiling Framework with NVIDIA A100 GPUs on HPCC at Texas Tech University.

## 🚀 Quick Start with A100

### Basic Commands

```bash
# A100 baseline profiling (single frequency)
./launch.sh --gpu-type A100 --profiling-mode baseline

# A100 comprehensive DVFS study (all 61 frequencies)
./launch.sh --gpu-type A100 --profiling-mode dvfs

# A100 with custom application
./launch.sh --gpu-type A100 --app-name "CustomApp" --app-executable "my_app"
```

### SLURM Job Submission

```bash
# Quick baseline test (HPCC A100 nodes)
sbatch submit_job_a100_baseline.sh

# Custom application profiling
sbatch submit_job_a100_custom_app.sh

# Full DVFS study (6-8 hours)
sbatch submit_job_a100_comprehensive.sh

# Interactive session for testing
./a100_interactive.sh

# OR use srun directly
srun --partition=toreador --gpus-per-node=1 --reservation=ghazanfar --pty bash
```

## 🔧 A100 GPU Specifications

### Architecture Details
- **GPU Architecture**: GA100 (Ampere)
- **Memory Frequency**: 1215 MHz
- **Default Core Frequency**: 1410 MHz (maximum)
- **Minimum Core Frequency**: 510 MHz
- **Total Frequencies**: 61 different settings

### Frequency Range
```
1410, 1395, 1380, 1365, 1350, 1335, ..., 570, 555, 540, 525, 510 MHz
```

### SLURM Configuration
- **Cluster**: HPCC at Texas Tech University
- **Partition**: toreador
- **GPU Resource**: `--gpus-per-node=1`
- **Recommended CPUs**: 16 (`--ntasks-per-node=16`)
- **Reservation**: ghazanfar
- **Interactive Sessions**: `srun --partition=toreador --gpus-per-node=1 --reservation=ghazanfar --pty bash`

## 📊 Performance Characteristics

### Expected Runtime Estimates

| Mode | Frequencies | Runs per Freq | Est. Time |
|------|-------------|---------------|-----------|
| Baseline | 1 (1410 MHz) | 3 | 5-10 min |
| Baseline Extended | 1 (1410 MHz) | 10 | 15-30 min |
| DVFS Study | 61 (all) | 3 | 6-8 hours |
| DVFS Extended | 61 (all) | 5 | 10-12 hours |

### Memory and Storage Requirements
- **GPU Memory**: A100s typically have 40GB or 80GB HBM2e
- **Disk Space**: 
  - Baseline: ~100 MB
  - Full DVFS: ~3-5 GB
- **System Memory**: 64+ GB recommended for comprehensive studies

## 🛠️ Configuration Examples

### 1. Quick Baseline Test
```bash
./launch.sh \
  --gpu-type A100 \
  --profiling-mode baseline \
  --num-runs 3 \
  --sleep-interval 1
```

### 2. Custom Application with A100
```bash
./launch.sh \
  --gpu-type A100 \
  --profiling-mode baseline \
  --app-name "LLaMA" \
  --app-executable "llama_inference" \
  --app-params "--model llama-7b --prompt 'test' > results/llama_output.log" \
  --num-runs 5
```

### 3. Selective Frequency Testing
For custom frequency selection, modify the `CORE_FREQUENCIES` array in `launch.sh`:
```bash
# Example: Test only high frequencies (1410-1200 MHz)
CORE_FREQUENCIES=(1410 1395 1380 1365 1350 1335 1320 1305 1290 1275 1260 1245 1230 1215 1200)
```

### 4. Comprehensive DVFS Study
```bash
./launch.sh \
  --gpu-type A100 \
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
tail -f LSTM_A100_*.out
```

### Result Files
```
results/
├── GA100-baseline-LSTM-1410-0    # Individual run results
├── GA100-baseline-LSTM-1410-1    # GPU metrics and power data
├── GA100-baseline-LSTM-1410-2    
└── GA100-baseline-lstm-perf.csv  # Performance summary
```

### Performance Analysis
```bash
# Quick performance summary
grep "performance saved" *.err | tail -10

# Power consumption analysis
python -c "
import pandas as pd
data = pd.read_csv('results/GA100-baseline-lstm-perf.csv')
print(f'Average execution time: {data.iloc[:, 1].mean():.2f}s')
"
```

## ⚠️ Important Considerations

### 1. Runtime Warnings
- **Comprehensive DVFS studies take 6-8 hours** with 61 frequencies
- Plan accordingly for HPCC cluster time limits
- Consider running baseline tests first

### 2. Frequency Control
- A100 frequency control requires appropriate permissions
- DCGMI is preferred for A100 (with nvidia-smi fallback)
- Test frequency control before long experiments:
```bash
./control.sh 1215 1410  # Test setting A100 frequencies
```

### 3. Power Limits
- A100s have high power consumption (250-400W typical)
- Monitor cluster power policies
- Some frequencies may be power-limited

### 4. HPCC-Specific Considerations
- **Reservation**: Uses `ghazanfar` reservation for guaranteed access
- **Modules**: Load `gcc cuda cudnn` for full functionality
- **Conda**: Framework expects `tensorflow` conda environment

## 🐛 Troubleshooting

### Common Issues

1. **"GPU type set to A100 but detected GPU doesn't appear to be A100"**
   - Check with `nvidia-smi` that you have an A100
   - Verify SLURM allocation: `squeue -u $USER`

2. **DCGMI not available**
   - Framework automatically falls back to nvidia-smi
   - Install DCGMI for better profiling: `module load dcgmi`

3. **Frequency control fails**
   - Check permissions: `dcgmi config --help`
   - Try nvidia-smi fallback: `--profiling-tool nvidia-smi`

4. **Job timeout**
   - Increase SLURM time limit: `#SBATCH --time=10:00:00`
   - Reduce frequencies or runs for testing

5. **Reservation issues**
   - Check reservation availability: `sinfo -T`
   - Remove `--reservation=ghazanfar` if not available

### Getting Help
```bash
# Check A100 capabilities
nvidia-smi -q -d SUPPORTED_CLOCKS

# Verify DCGMI
dcgmi discovery --list

# Test framework
./launch.sh --help
./launch.sh --gpu-type A100 --profiling-mode baseline --num-runs 1

# Interactive session helper
./a100_interactive.sh help
```

## 📚 Additional Resources

- **Main README**: [`../README.md`](../README.md)
- **General Usage**: [`../documentation/USAGE_EXAMPLES.md`](../documentation/USAGE_EXAMPLES.md)
- **SLURM Guide**: [`../documentation/SUBMIT_JOBS_README.md`](../documentation/SUBMIT_JOBS_README.md)
- **Troubleshooting**: [`../documentation/QUICK_FIX_GUIDE.md`](../documentation/QUICK_FIX_GUIDE.md)
- **H100 Guide**: [`../documentation/H100_USAGE_GUIDE.md`](../documentation/H100_USAGE_GUIDE.md)

---

**Note**: A100 support is optimized for HPCC at Texas Tech University with the toreador partition and ghazanfar reservation. Adjust configuration for other clusters as needed.
