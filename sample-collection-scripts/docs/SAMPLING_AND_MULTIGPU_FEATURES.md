# New Features: Sampling Interval and Multi-GPU Support

## Overview

Two new parameters have been added to the AI Inference Energy Profiling Framework:

1. **`--sampling-interval MS`** - Control DCGMI sampling frequency
2. **`--all-gpus`** - Monitor all available GPUs instead of just GPU 0

## Parameter Details

### Sampling Interval (`--sampling-interval`)

- **Purpose**: Controls how frequently DCGMI collects power and energy data
- **Range**: 10-1000 milliseconds
- **Default**: 50ms
- **Usage**: `--sampling-interval 25`

**Recommendations:**
- **Fast sampling (10-25ms)**: For detailed power transitions, short experiments
- **Normal sampling (50ms)**: Default, good balance of detail and performance  
- **Slow sampling (100-200ms)**: For long experiments, reduced data volume

### Multi-GPU Monitoring (`--all-gpus`)

- **Purpose**: Monitor all available GPUs instead of just GPU 0
- **Default**: Disabled (monitors GPU 0 only)
- **Usage**: `--all-gpus` (no argument needed)

**Use Cases:**
- Multi-GPU inference workloads
- Distributed training experiments
- System-wide power analysis on H100 nodes (GPUs 0-3)

## New Configuration Examples

The following configurations have been added to `submit_job_h100.sh`:

### Configuration #29: High-Frequency Sampling
```bash
LAUNCH_ARGS="--gpu-type H100 --profiling-mode baseline --sampling-interval 10 --app-name StableDiffusion --app-executable ../app-stable-diffusion/StableDiffusionViaHF.py --app-params '--prompt \"a photograph of an astronaut riding a horse\" --steps 50 --job-id ${SLURM_JOB_ID} --log-level INFO' --num-runs 3"
```

### Configuration #30: Low-Frequency Sampling
```bash
LAUNCH_ARGS="--gpu-type H100 --profiling-mode dvfs --sampling-interval 200 --app-name LLaMA --app-executable ../app-llama/LlamaViaHF.py --app-params '--benchmark --num-generations 10 --quiet --metrics' --num-runs 3"
```

### Configuration #31: Multi-GPU Monitoring
```bash
LAUNCH_ARGS="--gpu-type H100 --profiling-mode baseline --all-gpus --app-name ViT --app-executable ../app-vision-transformer/ViTViaHF.py --app-params '--benchmark --num-images 1000 --model google/vit-base-patch16-224 --precision float16' --num-runs 3"
```

### Configuration #32: Ultra-Fine Monitoring
```bash
LAUNCH_ARGS="--gpu-type H100 --profiling-mode baseline --sampling-interval 25 --all-gpus --app-name StableDiffusion --app-executable ../app-stable-diffusion/StableDiffusionViaHF.py --app-params '--prompt \"a cyberpunk cityscape\" --steps 100 --job-id ${SLURM_JOB_ID} --log-level INFO' --num-runs 3"
```

## Usage Instructions

1. **Edit submit_job_h100.sh**: Uncomment one of configurations #29-32 or add the parameters to any existing configuration
2. **Submit job**: `sbatch submit_job_h100.sh`
3. **Monitor progress**: Check the SLURM output files for configuration summary

## Implementation Details

- Parameters are parsed in `lib/args_parser.sh`
- Configuration summary shows sampling rate and GPU monitoring status
- Profile.py already supported these parameters via `--interval` and `--all-gpus`
- Full backward compatibility maintained with existing configurations

## Validation

The features have been tested with:
- ✅ Argument parsing and validation
- ✅ Configuration summary display
- ✅ Integration with existing profiling pipeline
- ✅ SLURM job ID integration
- ✅ Application validation (LSTM, Stable Diffusion)

## SLURM Timing Recommendations

For configurations #29-32: `--time=02:00:00` (2 hours)

These configurations are suitable for detailed analysis and may generate larger datasets with fine-grained sampling or multi-GPU data.
