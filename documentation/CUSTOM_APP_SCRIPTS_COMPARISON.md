# Custom Application SLURM Scripts Comparison

This document compares the custom application SLURM scripts across different GPU types to highlight their key differences and use cases.

## Script Overview

| Script | GPU Type | Cluster | Partition | Resource Spec | Default Time |
|--------|----------|---------|-----------|---------------|--------------|
| `submit_job_a100_custom_app.sh` | A100 | HPCC | toreador | --gpus-per-node=1 --reservation=ghazanfar | 3:00:00 |
| `submit_job_v100_custom_app.sh` | V100 | HPCC | matador | --gres=gpu:v100:1 --ntasks=40 | 3:00:00 |
| `submit_job_h100_custom_app.sh` | H100 | REPACSS | h100-build | --nodelist=rpg-93-9 --gres=gpu:1 | 3:00:00 |
| `submit_job_custom_app.sh` | A100 | HPCC | toreador | --gpus-per-node=1 | 2:00:00 |

## GPU Specifications

### A100 (HPCC Toreador)
- **Memory**: 80GB HBM2
- **Frequencies**: 7 steps (210-1410 MHz)
- **DVFS Runtime**: ~2-3 hours
- **Recommended for**: Large models, high-memory applications

### V100 (HPCC Matador)
- **Memory**: 32GB HBM2
- **Frequencies**: 137 steps (405-1380 MHz)
- **DVFS Runtime**: 6-10 hours ⚠️
- **Recommended for**: Legacy compatibility, custom frequency studies

### H100 (REPACSS h100-build)
- **Memory**: 80GB HBM3
- **Frequencies**: ~104 steps (210-1980 MHz estimated)
- **DVFS Runtime**: ~4-6 hours
- **Recommended for**: Cutting-edge research, maximum performance

## Configuration Examples

### A100 Examples
```bash
# Quick baseline test
LAUNCH_ARGS="--gpu-type A100 --profiling-mode baseline --num-runs 5"

# Custom application
LAUNCH_ARGS="--gpu-type A100 --profiling-mode baseline --app-name CustomApp --app-executable my_app"

# Full DVFS study
LAUNCH_ARGS="--gpu-type A100 --profiling-mode dvfs --num-runs 3"
```

### V100 Examples (⚠️ Note: 137 frequencies)
```bash
# Quick baseline test
LAUNCH_ARGS="--gpu-type V100 --profiling-mode baseline --num-runs 5"

# Custom frequencies (RECOMMENDED for V100)
LAUNCH_ARGS="--gpu-type V100 --profiling-mode custom --custom-frequencies '405,840,1200,1380' --num-runs 7"

# Full DVFS study (LONG RUNTIME - 6-10 hours)
LAUNCH_ARGS="--gpu-type V100 --profiling-mode dvfs --num-runs 3"
```

### H100 Examples
```bash
# Quick baseline test
LAUNCH_ARGS="--gpu-type H100 --profiling-mode baseline --num-runs 5"

# Custom application
LAUNCH_ARGS="--gpu-type H100 --profiling-mode baseline --app-name LLaMA --app-executable llama_inference"

# Full DVFS study
LAUNCH_ARGS="--gpu-type H100 --profiling-mode dvfs --num-runs 3"
```

## Key Differences

### SLURM Resource Allocation

**A100 (HPCC)**:
```bash
#SBATCH --partition=toreador
#SBATCH --gpus-per-node=1
#SBATCH --reservation=ghazanfar
#SBATCH --ntasks-per-node=16
```

**V100 (HPCC)**:
```bash
#SBATCH --partition=matador
#SBATCH --gres=gpu:v100:1
#SBATCH --ntasks=40
```

**H100 (REPACSS)**:
```bash
#SBATCH --partition=h100-build
#SBATCH --nodelist=rpg-93-9
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
```

### Module Loading

**A100/V100 (HPCC)**:
```bash
module load gcc cuda cudnn
```

**H100 (REPACSS)**:
```bash
module load cuda
```

### Special Considerations

#### V100 Specific Warnings
- **137 frequencies**: Much larger frequency range than A100 (7) or H100 (~104)
- **Long DVFS runtime**: 6-10 hours for comprehensive studies
- **Memory limitation**: 32GB vs 80GB on A100/H100
- **Recommendation**: Use custom frequency mode instead of full DVFS

#### A100 Specific Features
- **Fastest DVFS**: Only 7 frequencies to test
- **High memory**: 80GB for large models
- **Reservation**: Uses ghazanfar reservation on HPCC

#### H100 Specific Features
- **Single node**: Dedicated rpg-93-9 node
- **Latest architecture**: Most advanced GPU
- **Different cluster**: REPACSS vs HPCC

## Usage Recommendations

### For Quick Testing
```bash
# A100: Fastest option
sbatch submit_job_a100_custom_app.sh

# V100: Use baseline mode
# Edit script: LAUNCH_ARGS="--gpu-type V100 --profiling-mode baseline --num-runs 3"
sbatch submit_job_v100_custom_app.sh

# H100: Cutting-edge testing
sbatch submit_job_h100_custom_app.sh
```

### For Comprehensive Studies
```bash
# A100: Full DVFS (2-3 hours)
# Edit script: LAUNCH_ARGS="--gpu-type A100 --profiling-mode dvfs --num-runs 5"
sbatch submit_job_a100_custom_app.sh

# V100: Custom frequencies (recommended over full DVFS)
# Edit script: LAUNCH_ARGS="--gpu-type V100 --profiling-mode custom --custom-frequencies '405,600,900,1200,1380'"
sbatch submit_job_v100_custom_app.sh

# H100: Full DVFS (4-6 hours)
# Edit script: LAUNCH_ARGS="--gpu-type H100 --profiling-mode dvfs --num-runs 3"
sbatch submit_job_h100_custom_app.sh
```

### For Custom Applications
All scripts support custom applications through the same interface:
```bash
LAUNCH_ARGS="--gpu-type [GPU] --profiling-mode baseline --app-name MyApp --app-executable my_executable --app-params 'custom parameters'"
```

## Troubleshooting

### Common Issues by GPU Type

**A100**:
- Reservation availability (ghazanfar)
- Memory allocation for 80GB models

**V100**:
- Long runtimes with DVFS mode
- Memory limitations with 32GB
- Frequency control permissions

**H100**:
- Node availability (single rpg-93-9 node)
- Different module loading on REPACSS
- Latest driver requirements

### Error Resolution
1. Check GPU-specific resource allocation
2. Verify cluster-specific module loading
3. Adjust time limits based on profiling mode
4. Consider memory limitations for your application

---

Choose the appropriate script based on your GPU availability, time constraints, and research requirements. V100 offers the most detailed frequency analysis but requires careful time management due to its 137 frequency steps.
