# Launch Script CLI Enhancement Summary

## Overview

The `launch.sh` script and related submission scripts have been enhanced to support complete command-line configuration, making the AI inference energy profiling framework much more flexible and suitable for automated experiments.

## ✅ **Updated Scripts**

### 1. `launch.sh` - Main Launch Script
**STATUS: ✅ UPDATED WITH FULL CLI SUPPORT**

The main profiling orchestration script now accepts all configuration parameters as command-line arguments:

#### New Command-Line Arguments:
- `--gpu-type TYPE`: Switch between A100 and V100 GPU configurations
- `--profiling-tool TOOL`: Choose dcgmi or nvidia-smi profiling tools
- `--profiling-mode MODE`: Select dvfs (full sweep) or baseline (single frequency)
- `--num-runs NUM`: Number of runs per frequency
- `--sleep-interval SEC`: Sleep time between runs in seconds
- `--app-name NAME`: Application display name
- `--app-executable PATH`: Path to application executable (without .py extension)
- `--app-params "PARAMS"`: Application parameters with automatic output redirection
- `-h, --help`: Show comprehensive help and usage examples

#### Key Features:
- **Automatic Output Redirection**: Ensures application output is always captured
- **Parameter Validation**: Validates all inputs for correctness
- **GPU-Specific Configuration**: Automatically configures frequencies and settings
- **Backward Compatibility**: Works with defaults when no parameters provided
- **Comprehensive Help**: Detailed usage information and examples

### 2. `submit_job.sh` - Main SLURM Submit Script
**STATUS: ✅ UPDATED WITH CLI SUPPORT**

Enhanced SLURM job submission script with:
- Configurable `LAUNCH_ARGS` variable for experiment customization
- Support for all `launch.sh` command-line options
- Enhanced environment validation and logging
- Better profiling tool availability checking

### 3. Example Submit Scripts - **NEW**
**STATUS: ✅ CREATED**

Created three example submit scripts demonstrating different use cases:

#### `submit_job_v100_baseline.sh`
- V100 GPU baseline profiling
- Quick testing configuration
- ~1 hour runtime

#### `submit_job_custom_app.sh`
- Custom application examples (Stable Diffusion, LLaMA)
- Application executable validation
- Multiple configuration examples

#### `submit_job_comprehensive.sh`
- Full DVFS experiment (all frequencies)
- Multiple runs for statistical significance
- 6-8 hour comprehensive study

### 4. Documentation - **NEW**
**STATUS: ✅ CREATED**

#### `USAGE_EXAMPLES.md`
Comprehensive usage examples covering:
- Basic and advanced CLI usage
- GPU configuration examples
- Application configuration examples
- Automation scripts

#### `SUBMIT_JOBS_README.md`
SLURM-specific documentation covering:
- All available submit scripts
- Configuration options
- HPC usage tips
- Troubleshooting guide

## ✅ **Scripts That Don't Need Updates**

### 1. `control.sh` / `control_smi.sh`
**STATUS: ✅ NO CHANGES NEEDED**
- These scripts are called by `launch.sh` with fixed arguments
- Interface remains the same (memory_freq, core_freq)

### 2. `profile.py` / `profile_smi.py`
**STATUS: ✅ NO CHANGES NEEDED**
- These scripts are executed by `launch.sh` with application commands
- Interface remains the same (application command as arguments)

### 3. `clean.sh`
**STATUS: ✅ NO CHANGES NEEDED**
- Utility script for cleaning results directory
- Already enhanced in previous refactoring
- No interaction with CLI interface

### 4. `test.sh`
**STATUS: ✅ NO CHANGES NEEDED**
- Template for MPI applications
- Not part of energy profiling framework
- Reference template only

## ✅ **Application Scripts**

### AI Inference Applications
**STATUS: ✅ NO CHANGES NEEDED**
- `app-lstm/lstm.py`
- `app-llama-collection/LlamaViaHF.py`
- `app-stable-diffusion-collection/StableDiffusionViaHF.py`

These scripts don't need changes because:
- They are executed by the profiling framework
- Input/output handled through command-line parameters
- Already enhanced in previous refactoring

## 🎯 **Usage Examples**

### Default Usage (LSTM on A100 with DCGMI)
```bash
./launch.sh
```

### V100 Baseline Profiling
```bash
./launch.sh --gpu-type V100 --profiling-mode baseline
```

### Custom Application
```bash
./launch.sh \
  --app-name "StableDiffusion" \
  --app-executable "stable_diffusion" \
  --app-params "--prompt 'test image' > results/SD_output.log"
```

### SLURM Job Submission
```bash
# Edit LAUNCH_ARGS in submit_job.sh, then:
sbatch submit_job.sh

# Or use example scripts:
sbatch submit_job_v100_baseline.sh
sbatch submit_job_custom_app.sh
sbatch submit_job_comprehensive.sh
```

## 🔧 **Migration from Old Scripts**

### Before (Old Interface)
```bash
# Had to edit variables inside launch.sh
GPU_TYPE="V100"
PROFILING_TOOL="dcgmi"
# ... edit script and run
./launch.sh
```

### After (New CLI Interface)
```bash
# Pass everything as arguments
./launch.sh --gpu-type V100 --profiling-tool dcgmi --profiling-mode baseline
```

## 📊 **Benefits of the New Interface**

1. **Automation-Friendly**: Easy to script and automate experiments
2. **No File Editing**: All configuration through command-line
3. **Parameter Validation**: Immediate feedback on invalid inputs
4. **Multiple Configurations**: Easy to run different experiments
5. **HPC Integration**: Seamless SLURM job customization
6. **Documentation**: Built-in help and comprehensive examples
7. **Safety**: Automatic output redirection ensures data capture
8. **Flexibility**: Support for any application with proper parameters

## 🎉 **Summary**

The enhancement is complete! The AI inference energy profiling framework now provides:

- ✅ Full command-line interface for all configuration
- ✅ Multiple SLURM submission options
- ✅ Comprehensive documentation and examples
- ✅ Backward compatibility with existing workflows
- ✅ Enhanced automation capabilities
- ✅ Better error handling and validation

All scripts are ready for production use with the new interface while maintaining compatibility with existing setups.
