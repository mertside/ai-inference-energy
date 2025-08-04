# SLURM Job Submission Scripts for AI Inference Energy Profiling

This directory contains SLURM job submission scripts for running AI inference energy profiling experiments on HPC clusters. All scripts have been updated to work with the enhanced `launch_v2.sh` command-line interface.

## Available Submit Scripts

### 1. `submit_job.sh` - Main Submit Script
The primary SLURM job submission script with configurable `LAUNCH_ARGS`.

**Features:**
- Configurable experiment parameters via `LAUNCH_ARGS` variable
- Comprehensive environment setup and validation
- Support for all launch_v2.sh command-line options
- Detailed logging and error handling

**Usage:**
```bash
# Edit LAUNCH_ARGS in the script, then submit
sbatch submit_job.sh
```

### 2. `submit_job_v100_baseline.sh` - V100 Baseline Profiling
Quick baseline profiling on V100 GPUs.

**Configuration:**
- GPU Type: V100
- Profiling Mode: baseline (default frequency only)
- Runs: 2 per frequency
- Estimated Time: ~1 hour

**Usage:**
```bash
sbatch submit_job_v100_baseline.sh
```

### 3. `submit_job_custom_app.sh` - Custom Application Example
Example for running custom applications (Stable Diffusion, LLaMA, etc.).

**Configuration:**
- Demonstrates custom application setup
- Includes multiple example configurations
- Application executable validation

**Usage:**
```bash
# Edit the LAUNCH_ARGS for your application, then submit
sbatch submit_job_custom_app.sh
```

### 4. `submit_job_comprehensive.sh` - Comprehensive DVFS Study
Full DVFS experiment across all frequencies with multiple runs.

**Configuration:**
- GPU Type: A100 (all 61 frequencies)
- Profiling Mode: dvfs (full frequency sweep)
- Runs: 5 per frequency
- Estimated Time: 6-8 hours

**Usage:**
```bash
sbatch submit_job_comprehensive.sh
```

## Common SLURM Configuration

All scripts use the following SLURM settings (modify as needed):

```bash
#SBATCH --partition=toreador
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=16
#SBATCH --gpus-per-node=1
#SBATCH --reservation=ghazanfar
#SBATCH --mail-user=mert.side@ttu.edu
```

## Launch Script Arguments

The submit scripts pass arguments to `launch_v2.sh`. Available options:

| Argument | Description | Default |
|----------|-------------|---------|
| `--gpu-type` | GPU type: A100 or V100 | A100 |
| `--profiling-tool` | Profiling tool: dcgmi or nvidia-smi | dcgmi |
| `--profiling-mode` | Mode: dvfs or baseline | dvfs |
| `--num-runs` | Number of runs per frequency | 3 |
| `--sleep-interval` | Sleep between runs (seconds) | 1 |
| `--app-name` | Application display name | LSTM |
| `--app-executable` | Application executable path | lstm |
| `--app-params` | Application parameters with output | "> results/LSTM_RUN_OUT" |

## Example Configurations

### Default LSTM on A100 with DCGMI
```bash
LAUNCH_ARGS=""
```

### V100 Baseline Profiling
```bash
LAUNCH_ARGS="--gpu-type V100 --profiling-mode baseline --num-runs 2"
```

### Custom Stable Diffusion Application
```bash
LAUNCH_ARGS="--app-name StableDiffusion --app-executable stable_diffusion --app-params '--prompt \"test image\" > results/SD_output.log'"
```

### Quick Test Configuration
```bash
LAUNCH_ARGS="--profiling-mode baseline --num-runs 1 --sleep-interval 0"
```

### nvidia-smi Alternative
```bash
LAUNCH_ARGS="--profiling-tool nvidia-smi --profiling-mode baseline"
```

## Environment Requirements

All scripts automatically set up:
- GCC, CUDA, cuDNN modules
- Conda environment (tensorflow)
- GPU profiling tools verification
- Python package availability

## Output Files

Results are saved in the `results/` directory with naming convention:
- `$ARCH-$MODE-$APP-$FREQ-$ITERATION` (profiling data)
- `$ARCH-$MODE-$APP-perf.csv` (performance metrics)

## Tips for HPC Usage

1. **Time Allocation**: Adjust `#SBATCH --time` based on experiment scope:
   - Baseline: 1-2 hours
   - DVFS: 6-8 hours for comprehensive studies

2. **Disk Space**: Ensure sufficient space for results:
   - Baseline: ~100MB
   - Full DVFS: ~1-2GB

3. **GPU Availability**: Check available GPU types on your cluster and adjust `--gpu-type` accordingly.

4. **Profiling Tools**: If DCGMI is not available, use `--profiling-tool nvidia-smi`.

5. **Custom Applications**: Make sure your application Python files are present before submission.

## Troubleshooting

- **Module Load Failures**: Check if module names match your cluster (gcc, cuda, cudnn)
- **Conda Environment**: Ensure the conda environment name matches your setup
- **DCGMI Issues**: Use nvidia-smi profiling as alternative
- **Permission Errors**: Some frequency control operations may require special permissions

For more details, see the main project documentation and `launch_v2.sh --help`.
