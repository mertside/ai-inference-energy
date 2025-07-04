# GPU Usage Guide

This guide provides comprehensive information for using the AI Inference Energy Profiling Framework with NVIDIA A100, V100, and H100 GPUs across different HPC clusters.

## 🚀 Quick Start by GPU Type

### A100 (HPCC at Texas Tech University)
```bash
# Basic commands
./launch.sh --gpu-type A100 --profiling-mode baseline
./launch.sh --gpu-type A100 --profiling-mode dvfs  # All 61 frequencies

# SLURM job submission
sbatch submit_job_a100_baseline.sh      # Quick test (1 hour)
sbatch submit_job_a100_comprehensive.sh # Full study (6-8 hours)
sbatch submit_job_a100_custom_app.sh    # Custom applications

# Interactive session
./interactive_a100.sh                   # Helper script
# OR manually:
srun --partition=toreador --gpus-per-node=1 --reservation=ghazanfar --pty bash
```

### V100 (HPCC at Texas Tech University)
```bash
# Basic commands (⚠️ Note: 103 frequencies - longer runtime)
./launch.sh --gpu-type V100 --profiling-mode baseline
./launch.sh --gpu-type V100 --profiling-mode dvfs  # All 103 frequencies (8-10 hours)

# SLURM job submission
sbatch submit_job_v100_baseline.sh      # Quick test (1 hour)
sbatch submit_job_v100_comprehensive.sh # Full study (8-10 hours)
sbatch submit_job_v100_custom_app.sh    # Custom applications

# Interactive session
./interactive_v100.sh                   # Helper script
# OR manually:
srun --partition=matador --gres=gpu:v100:1 --ntasks=40 --pty bash
```

### H100 (REPACSS at Texas Tech University)
```bash
# Basic commands
./launch.sh --gpu-type H100 --profiling-mode baseline
./launch.sh --gpu-type H100 --profiling-mode dvfs  # All 104 frequencies

# SLURM job submission
sbatch submit_job_h100_baseline.sh      # Quick test (1 hour)
sbatch submit_job_h100_comprehensive.sh # Full study (12-15 hours)
sbatch submit_job_h100_custom_app.sh    # Custom applications

# Interactive session
./interactive_h100.sh                   # Helper script
# OR manually:
interactive -p h100-build -g 1 -w rpg-93-9
```

## 📊 GPU Specifications Comparison

| Feature | A100 (HPCC) | V100 (HPCC) | H100 (REPACSS) |
|---------|-------------|-------------|----------------|
| **Cluster** | HPCC TTU | HPCC TTU | REPACSS TTU |
| **Partition** | toreador | matador | h100-build |
| **Architecture** | GA100 (Ampere) | GV100 (Volta) | GH100 (Hopper) |
| **Memory** | 80GB HBM2e | 32GB HBM2 | 80GB HBM3 |
| **Memory Freq** | 1215 MHz | 877 MHz | 1593 MHz |
| **Core Freq Range** | 1410-510 MHz | 1380-405 MHz | 1755-210 MHz |
| **Total Frequencies** | 61 | 103 | 104 |
| **DVFS Runtime** | 6-8 hours | 8-10 hours | 12-15 hours |
| **Memory Size** | 80GB | 32GB | 80GB |
| **Power Consumption** | 250-400W | 250-300W | 400-700W |

## 🛠️ Configuration Examples

### A100 Examples
```bash
# Quick baseline test
./launch.sh --gpu-type A100 --profiling-mode baseline --num-runs 3

# Custom application
./launch.sh --gpu-type A100 --app-name "LLaMA" --app-executable "llama_inference" \
  --app-params "--model llama-7b > results/llama_output.log"

# Custom frequency selection (modify launch.sh CORE_FREQUENCIES array)
CORE_FREQUENCIES=(1410 1395 1380 1365 1350 1335 1320 1305 1290 1275 1260 1245 1230 1215 1200)
```

### V100 Examples (⚠️ Recommendation: Use custom frequencies instead of full DVFS)
```bash
# Quick baseline test
./launch.sh --gpu-type V100 --profiling-mode baseline --num-runs 3

# Custom frequencies (RECOMMENDED for V100 due to 103 frequencies)
./launch.sh --gpu-type V100 --profiling-mode custom \
  --custom-frequencies '405,840,1200,1380' --num-runs 7

# Full DVFS study (LONG RUNTIME - 8-10 hours)
./launch.sh --gpu-type V100 --profiling-mode dvfs --num-runs 3
```

### H100 Examples
```bash
# Quick baseline test
./launch.sh --gpu-type H100 --profiling-mode baseline --num-runs 3

# Custom application
./launch.sh --gpu-type H100 --app-name "LLaMA" --app-executable "llama_inference" \
  --app-params "--model llama-13b > results/llama_output.log"

# Custom frequency selection (modify launch.sh CORE_FREQUENCIES array)
CORE_FREQUENCIES=(1755 1740 1725 1710 1695 1680 1665 1650 1635 1620 1605 1590 1575 1560 1545 1530 1515 1500)
```

## 🔧 Cluster-Specific SLURM Configuration

### A100 (HPCC)
```bash
#SBATCH --partition=toreador
#SBATCH --gpus-per-node=1
#SBATCH --reservation=ghazanfar
#SBATCH --ntasks-per-node=16

# Module loading
module load gcc cuda cudnn
source "$HOME/conda/etc/profile.d/conda.sh"
conda activate tensorflow
```

### V100 (HPCC)
```bash
#SBATCH --partition=matador
#SBATCH --gres=gpu:v100:1
#SBATCH --ntasks=40

# Module loading
module load gcc cuda cudnn
source "$HOME/conda/etc/profile.d/conda.sh"
conda activate tensorflow
```

### H100 (REPACSS)
```bash
#SBATCH --partition=h100-build
#SBATCH --nodelist=rpg-93-9
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8

# Module loading
module load cuda
source "$HOME/conda/etc/profile.d/conda.sh"
conda activate tensorflow
```

## 📈 Performance Monitoring and Results

### Real-time Monitoring
```bash
# Monitor GPU usage during experiment
watch -n 1 'nvidia-smi --query-gpu=name,utilization.gpu,memory.used,power.draw,clocks.gr,clocks.mem --format=csv'

# Monitor SLURM job
squeue -u $USER
tail -f LSTM_*_*.out  # Follow job output
```

### Result Files Structure
```
results/
├── GA100-baseline-LSTM-1410-0    # A100 individual run results
├── GV100-baseline-LSTM-1380-0    # V100 individual run results  
├── GH100-baseline-LSTM-1755-0    # H100 individual run results
└── *-baseline-lstm-perf.csv      # Performance summary files
```

### Performance Analysis
```bash
# Quick performance summary
grep "performance saved" *.err | tail -10

# Power consumption analysis
python -c "
import pandas as pd
data = pd.read_csv('results/GA100-baseline-lstm-perf.csv')  # Adjust filename
print(f'Average execution time: {data.iloc[:, 1].mean():.2f}s')
"
```

## ⚠️ Important Considerations by GPU Type

### A100 Considerations
- **Runtime**: 6-8 hours for comprehensive DVFS studies (61 frequencies)
- **Reservation**: Uses `ghazanfar` reservation for guaranteed access on HPCC
- **Power**: Moderate power consumption (250-400W)
- **Memory**: 80GB ideal for large models
- **Cluster**: HPCC toreador partition

### V100 Considerations  
- **Runtime**: 8-10 hours for comprehensive DVFS studies (103 frequencies) ⚠️
- **Memory**: Limited to 32GB vs 80GB on A100/H100
- **Recommendation**: Use custom frequency mode instead of full DVFS
- **Cluster**: HPCC matador partition
- **Legacy**: Good for compatibility testing and custom frequency studies

### H100 Considerations
- **Runtime**: 12-15 hours for comprehensive DVFS studies (104 frequencies) ⚠️
- **Power**: High power consumption (400-700W)
- **Node**: Single dedicated node (rpg-93-9) on REPACSS
- **Latest**: Most advanced architecture for cutting-edge research
- **Cluster**: REPACSS h100-build partition

## 🐛 Troubleshooting

### Common Issues Across All GPUs

1. **"GPU type set to [GPU] but detected GPU doesn't appear to be [GPU]"**
   - Check with `nvidia-smi` that you have the correct GPU
   - Verify SLURM allocation: `squeue -u $USER`

2. **DCGMI not available**
   - Framework automatically falls back to nvidia-smi
   - Install DCGMI for better profiling: `module load dcgmi`

3. **Frequency control fails**
   - Check permissions: `dcgmi config --help`
   - Try nvidia-smi fallback: `--profiling-tool nvidia-smi`

4. **Job timeout**
   - Increase SLURM time limit in job scripts
   - Reduce frequencies or runs for testing

### GPU-Specific Issues

**A100**:
- Reservation availability (ghazanfar)
- Check `sinfo -T` for reservation status

**V100**:
- Long runtimes with DVFS mode (8-10 hours)
- Memory limitations with 32GB
- Use custom frequency mode for faster experiments

**H100**:
- Node availability (single rpg-93-9 node)
- Different module loading on REPACSS vs HPCC
- Requires latest driver compatibility

### Testing and Validation Commands
```bash
# Check GPU capabilities
nvidia-smi -q -d SUPPORTED_CLOCKS

# Verify DCGMI availability
dcgmi discovery --list

# Test framework help
./launch.sh --help

# Quick test run
./launch.sh --gpu-type [A100|V100|H100] --profiling-mode baseline --num-runs 1

# Interactive session helpers
./interactive_[a100|v100|h100].sh help
```

## 📚 Usage Recommendations

### For Quick Testing
- **A100**: Fastest option with only 61 frequencies
- **V100**: Use baseline mode to avoid long runtimes
- **H100**: Good for cutting-edge model testing

### For Comprehensive Studies
- **A100**: Full DVFS (6-8 hours) - good balance
- **V100**: Custom frequencies recommended over full DVFS
- **H100**: Full DVFS (12-15 hours) - plan accordingly

### For Large Models
- **A100**: 80GB memory, good for most large models
- **V100**: 32GB memory limit - may not support largest models
- **H100**: 80GB memory + latest architecture for best performance

## 🔗 Related Documentation

- **[Main README](../README.md)** - Project overview and installation
- **[Usage Examples](USAGE_EXAMPLES.md)** - CLI usage examples and automation
- **[SLURM Guide](SUBMIT_JOBS_README.md)** - Job submission and HPC cluster usage
- **[Power Modeling](README_POWER_MODELING.md)** - Advanced power modeling framework

---

**Note**: GPU configurations are optimized for their respective clusters (HPCC for A100/V100, REPACSS for H100). Adjust partition names, resource specifications, and module loading for other cluster environments.
