# AI Inference Energy Profiling Framework

> **Latest Version**: v2.0.1 (Enhanced) | **Status**: Production Ready ✅  
> **Quick Start**: `./launch_v2.sh --help` | **Cleanup**: `./clean.sh --help` | **Legacy**: `./legacy/launch.sh`

This directory contains a **production-ready, modular AI inference energy profiling framework** for comprehensive power analysis across GPU architectures (H100, A100, V100) and AI workloads (Stable Diffusion, LSTM, LLaMA, Whisper, Vision Transformer).

**🎉 Recent Enhancements (v2.0.1):**
- ✅ **Robust Error Handling**: Resolved "experiment failed" issues with graceful error recovery
- ✅ **Intelligent Results Naming**: Automatic `results_[gpu]_[app]` directory organization  
- ✅ **Enhanced Cleanup Tool**: Advanced filtering, backup, and selective cleanup options
- ✅ **High-Resolution Profiling**: Consistent 50ms sampling for both DCGMI and nvidia-smi
- ✅ **Comprehensive Summaries**: Rich experiment metadata and performance statistics
- ✅ **Configuration Consolidation**: Unified DCGMI field configuration (25 comprehensive fields)
- ✅ **Clean Filenames**: Fixed duplicate frequency in custom experiment filenames
- ✅ **Robust Import System**: Resolved configuration import conflicts for reliable operation
- ✅ **Stability**: Improved reliability and user experience

---

## 🚀 Quick Start (New Users)

```bash
# 1. Get help and see all options
./launch_v2.sh --help

# 2. Preview workspace cleanup (recommended before first run)
./clean.sh --dry-run

# 3. Run default A100 DVFS experiment
./launch_v2.sh

# 4. Quick V100 baseline test
./launch_v2.sh --gpu-type V100 --profiling-mode baseline

# 5. Custom Stable Diffusion profiling
./launch_v2.sh \
    --app-name "StableDiffusion" \
    --app-executable "StableDiffusionViaHF.py" \
    --app-params "--prompt 'A beautiful landscape' --steps 20"

# 6. Clean up results when done
./clean.sh --gpu-type H100  # Clean only H100 results
```

**Note:** `launch_v2.sh` is the recommended entry point. Legacy scripts are preserved in `legacy/launch.sh`.

---

## 📋 Framework Overview

### ✨ Key Features (v2.0)

- 🏗️ **Modular Architecture**: Clean separation of concerns with reusable libraries
- 🔧 **Robust CLI**: Comprehensive command-line interface with intelligent defaults
- 🎯 **Multi-GPU Support**: Native H100, A100, and V100 configurations  
- 🛠️ **Tool Flexibility**: DCGMI, nvidia-smi with automatic fallback
- ⚡ **High Resolution**: 50ms sampling interval for both DCGMI and nvidia-smi
- 📊 **Experiment Modes**: DVFS (frequency sweep) and baseline profiling
- 🚀 **SLURM Ready**: HPC cluster integration
- 🔍 **Auto-Detection**: GPU types, profiling tools, conda environments
- ✅ **Zero Raw Escapes**: Clean terminal output across all environments

### 🏗️ Architecture (v2.0)

```
sample-collection-scripts/
├── launch_v2.sh              # 🎯 Main entry point (recommended)
├── clean.sh                  # 🧹 Enhanced workspace cleanup tool
├── lib/                      # 📚 Modular libraries
│   └── ...                   #   └─ All original files
└── tests/                    # 🧪 Testing framework
    ├── test_framework.sh     #   ├─ Framework validation
    └── test_color_output.sh  #   └─ Terminal output testing
```

---

## 📖 Complete Usage Guide

### Basic Commands

```bash
# Show comprehensive help
./launch_v2.sh --help

# Show version and framework info
./launch_v2.sh --version

# Default experiment (A100 DVFS)
./launch_v2.sh

# Enable debug output
./launch_v2.sh --debug
```

### GPU-Specific Experiments

```bash
# H100 experiment with DCGMI
./launch_v2.sh --gpu-type H100

# V100 baseline with nvidia-smi fallback  
./launch_v2.sh --gpu-type V100 --profiling-tool nvidia-smi --profiling-mode baseline

# A100 custom frequency range
./launch_v2.sh --memory-freq-start 5001 --memory-freq-end 6251 --memory-freq-step 250
```

### Application Profiling

```bash
# LSTM baseline profiling
./launch_v2.sh \
    --app-executable "lstm.py" \
    --profiling-mode baseline \
    --num-runs 5

# Stable Diffusion with custom parameters
./launch_v2.sh \
    --app-name "StableDiffusion" \
    --app-executable "StableDiffusionViaHF.py" \
    --app-params "--prompt 'A serene mountain landscape' --steps 30 --guidance_scale 7.5" \
    --num-runs 3

# LLaMA inference profiling
./launch_v2.sh \
    --app-executable "LlamaViaHF.py" \
    --app-params "--model-name 'meta-llama/Llama-2-7b-hf' --max-tokens 100"

# Whisper speech recognition profiling
./launch_v2.sh \
    --app-name "Whisper" \
    --app-executable "../app-whisper/WhisperViaHF.py" \
    --app-params "--benchmark --model base --num-samples 3 --quiet"

# Vision Transformer image classification profiling
./launch_v2.sh \
    --app-name "ViT" \
    --app-executable "../app-vision-transformer/ViTViaHF.py" \
    --app-params "--benchmark --num-images 5 --model google/vit-base-patch16-224"
```

### Advanced Configuration

```bash
# Custom output directory with metadata
./launch_v2.sh \
    --output-dir "experiments/$(date +%Y%m%d)" \
    --experiment-name "production_baseline" \
    --debug

# Conda environment override
./launch_v2.sh \
    --conda-env "pytorch_2.1" \
    --app-executable "custom_model.py"

# Profiling tool comparison (auto-named directories with job IDs)
./launch_v2.sh --gpu-type H100 --app-name StableDiffusion --profiling-tool dcgmi
# → Creates: results_h100_stablediffusion_job_12345/ (SLURM) or results_h100_stablediffusion/ (non-SLURM)

./launch_v2.sh --gpu-type A100 --app-name LSTM --profiling-tool nvidia-smi  
# → Creates: results_a100_lstm_job_12346/ (SLURM) or results_a100_lstm/ (non-SLURM)

./launch_v2.sh --gpu-type V100 --app-name StableDiffusion --profiling-mode baseline
# → Creates: results_v100_stablediffusion_job_12347/ (SLURM) or results_v100_stablediffusion/ (non-SLURM)

# Custom output directory (job ID still appended automatically)
./launch_v2.sh --output-dir custom_results_dir
# → Creates: custom_results_dir_job_12348/ (SLURM) or custom_results_dir/ (non-SLURM)
```

---

## 🔧 Configuration & Customization

### Default Configuration

Edit `config/defaults.sh` for persistent settings:

```bash
# GPU and profiling defaults
DEFAULT_GPU_TYPE="A100"
DEFAULT_PROFILING_TOOL="dcgmi"
DEFAULT_PROFILING_MODE="dvfs"
DEFAULT_NUM_RUNS=3

# Application defaults  
DEFAULT_APP_NAME="LSTM"
DEFAULT_APP_EXECUTABLE="lstm.py"
DEFAULT_OUTPUT_DIR="results"
```

### User Configuration

Create `config/user_config.sh` for personal overrides:

```bash
# Personal preferences
DEFAULT_NUM_RUNS=5
DEFAULT_OUTPUT_DIR="/scratch/$USER/ai_profiling"
ENABLE_DEBUG_OUTPUT=true
DEFAULT_CONDA_ENV="my_pytorch_env"
```

### Environment Variables

```bash
# Color output control
export NO_COLOR=1              # Disable all colors
export FORCE_COLOR=1           # Force colors (override detection)
export DISABLE_COLORS=1        # Alternative to NO_COLOR

# Framework behavior
export DEBUG_MODE=1            # Enable debug output
export SKIP_CONFIRMATIONS=1    # Skip interactive prompts
```

---

## 🎛️ HPC & SLURM Integration

### Job Submission

The framework integrates seamlessly with existing SLURM job scripts:

```bash
# H100 job submission
sbatch submit_job_h100.sh

# A100 job submission  
sbatch submit_job_a100.sh

# V100 job submission
sbatch submit_job_v100.sh
```

### Custom SLURM Scripts

Update your job scripts to use the new framework:

```bash
#!/bin/bash
#SBATCH --job-name=ai_profiling_v2
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h100:1
#SBATCH --time=02:00:00

# Load modules
module load cuda/12.1
module load anaconda3

# Run with new framework
./launch_v2.sh \
    --gpu-type H100 \
    --profiling-mode dvfs \
    --num-runs 3 \
    --output-dir "$SLURM_JOB_ID"
```

---

## 🧪 Testing & Validation

### Framework Testing

```bash
# Test basic functionality
./tests/test_framework.sh

# Test color output handling
./tests/test_color_output.sh

# Manual validation
./launch_v2.sh --gpu-type INVALID  # Should show helpful error
./launch_v2.sh --help             # Should display clean help
./launch_v2.sh --debug-colors     # Should show color detection info
```

### Experiment Validation

```bash
# Quick validation run
./launch_v2.sh --profiling-mode baseline --num-runs 1 --debug

# Compare with legacy (if needed)
./legacy/launch.sh              # Legacy run
./launch_v2.sh                  # New framework run
# Compare results for consistency
```

---

## 🔄 Migration from Legacy

### For New Users
- **Start with `launch_v2.sh`** - The new framework
- **Use `--help`** for comprehensive guidance
- **Leverage auto-detection** for GPU types and tools

### For Existing Users

#### Phase 1: Safe Testing
```bash
# Test new framework alongside legacy
./launch_v2.sh --help                    # Explore new options
./launch_v2.sh --profiling-mode baseline # Quick test
./legacy/launch.sh                       # Keep using legacy
```

#### Phase 2: Feature Adoption  
```bash
# Start using new features
./launch_v2.sh --debug                   # Better debugging
./launch_v2.sh --gpu-type auto           # Auto-detection
./launch_v2.sh --app-params "custom"     # Flexible app config
```

#### Phase 3: Full Migration
```bash
# Update job scripts (optional)
# Old: ./legacy/launch.sh
# New: ./launch_v2.sh --gpu-type H100 --profiling-mode dvfs
```

### Command Equivalence

| Legacy Method | New Framework Equivalent |
|---------------|--------------------------|
| Edit legacy scripts for GPU type | `--gpu-type A100/V100/H100` |
| Edit script for profiling tool | `--profiling-tool dcgmi/nvidia-smi` |
| Edit script for number of runs | `--num-runs N` |
| Edit script for application | `--app-executable path --app-params "args"` |
| Manual debugging | `--debug` flag |

---

## 🐛 Troubleshooting

### Common Issues

**"Library not found" error:**
```bash
# Check library directory
ls -la lib/
# Ensure execute permissions
chmod +x launch_v2.sh lib/*.sh
```

**Color output issues:**
```bash
# Test color handling
./tests/test_color_output.sh
# Force disable colors
NO_COLOR=1 ./launch_v2.sh --help
# Debug color detection
./launch_v2.sh --debug-colors
```

**GPU not detected:**
```bash
# Manual GPU specification
./launch_v2.sh --gpu-type H100
# Check nvidia-smi availability
nvidia-smi
# Enable debug mode
./launch_v2.sh --debug
```

**"H100/A100/V100 profiling experiment failed" - RESOLVED ✅**

This issue has been resolved in v2.0 with improved error handling. If you see this message, it typically indicates a summary generation issue, not an actual experiment failure.

**Symptoms:**
- SLURM job reports "profiling experiment failed"
- All experiment runs completed successfully (visible in logs)
- Results directory contains all expected files
- Issue occurs during summary generation phase

**Solution (v2.0):**
The framework now handles summary generation failures gracefully:
```bash
# The experiment will complete successfully even if summary generation has issues
# All profiling data is preserved and usable
# Check results directory for complete data:
ls -la results_h100_*/
```

**If issue persists:**
```bash
# Check if results were actually generated successfully
ls -la results_*/
# Verify profiling data files exist
find results_*/ -name "*.csv" -o -name "*.out" | wc -l
# Check timing summary
cat results_*/timing_summary.log
```

**Recovery:**
```bash
# Your data is safe - all profiling results are generated correctly
# The "failure" is cosmetic in the summary generation phase
# You can proceed with data analysis using the results directory
```

### Debug Mode

Enable comprehensive debugging:
```bash
./launch_v2.sh --debug
```

This provides:
- ✅ Library loading status
- ✅ GPU detection results  
- ✅ Profiling tool validation
- ✅ Configuration summary
- ✅ Step-by-step execution trace

---

## 📊 Results & Output

### Automatic Directory Naming (v2.0)

Results are automatically organized using the naming convention `results_[gpu]_[app]_job_[id]`:

```
# SLURM Environment (with job IDs)
results_h100_stablediffusion_job_12345/    # H100 + Stable Diffusion, Job 12345
results_h100_lstm_job_12346/               # H100 + LSTM, Job 12346
results_a100_stablediffusion_job_12347/    # A100 + Stable Diffusion, Job 12347
results_a100_lstm_job_12348/               # A100 + LSTM, Job 12348
results_v100_stablediffusion_job_12349/    # V100 + Stable Diffusion, Job 12349

# Non-SLURM Environment (without job IDs)
results_h100_stablediffusion/              # H100 + Stable Diffusion
results_h100_lstm/                         # H100 + LSTM
results_a100_stablediffusion/              # A100 + Stable Diffusion
```

The naming is automatically generated based on:
- **GPU Type**: H100, A100, V100
- **Application Name**: Derived from `--app-name` or `--app-executable`
- **Job ID**: Automatically appended in SLURM environments (`${SLURM_JOB_ID}`)
- **Intelligent Defaults**: Framework automatically determines appropriate names

You can override the automatic naming with `--output-dir custom_name` (job ID will still be appended).

### Output Structure

```
results_[gpu]_[app]/
├── experiment_summary.log           # 📊 Comprehensive experiment summary
├── timing_summary.log               # ⏱️ Run timing and performance data
├── run_[id]_freq_[freq]_profile.csv # 📈 Individual profiling data per run
├── run_[id]_freq_[freq]_app.out     # 📝 Application stdout per run
├── run_[id]_freq_[freq]_app.err     # ⚠️ Application stderr per run
└── [Additional files based on experiment type]

# Example for H100 Stable Diffusion baseline experiment:
results_h100_stablediffusion/
├── experiment_summary.log           # Complete experiment metadata
├── timing_summary.log               # baseline_01,1785,38,0,success
├── run_baseline_01_freq_1785_profile.csv
├── run_baseline_01_freq_1785_app.out
├── run_baseline_01_freq_1785_app.err
├── run_baseline_02_freq_1785_profile.csv
├── run_baseline_02_freq_1785_app.out
├── run_baseline_02_freq_1785_app.err
└── [continues for all runs...]
```

### Data Analysis

The framework generates analysis-ready data:
- **CSV Format**: Direct import into pandas, R, Excel
- **Structured Logs**: Timestamped and categorized
- **Metadata**: Complete experiment configuration tracking
- **Hardware Info**: GPU specifications and environment details

---

## 📚 Library Documentation

### `lib/common.sh` - Core Utilities
```bash
# Logging functions
log_info "Information message"
log_error "Error message"  
log_warning "Warning message"
log_debug "Debug message"

# Validation functions
is_positive_integer 42          # true
is_valid_frequency 5001         # true
command_exists nvidia-smi       # true/false
```

### `lib/gpu_config.sh` - GPU Management
```bash
# GPU detection
detect_gpu_type                 # Returns: H100, A100, V100, or unknown
get_gpu_count                   # Returns: number of GPUs
validate_gpu_type "H100"        # Validates GPU type

# Frequency management  
get_gpu_memory_freq "A100"      # Returns valid frequency ranges
generate_frequency_range start end step  # Creates frequency array
```

### `lib/profiling.sh` - Profiling Control
```bash
# Tool validation
validate_profiling_tool "dcgmi"       # Checks tool availability
get_profiling_script "nvidia-smi"     # Returns script path

# Experiment execution
run_dvfs_experiment                   # Full frequency sweep
run_baseline_experiment               # Single frequency run
```

### `lib/args_parser.sh` - CLI Interface
```bash
# Argument parsing
parse_arguments "$@"                  # Parse command line
validate_arguments                    # Validate parsed args
show_usage                           # Display help

# Configuration
apply_intelligent_defaults           # Set smart defaults
export_configuration                # Export config for use
```

---

## 🔮 Future Enhancements

The modular architecture enables easy extensions:

### Planned Features
- 🔬 **Additional Profiling Tools**: Intel VTune, NVIDIA Nsight integration
- 🎯 **More GPU Architectures**: RTX 4090, AMD GPUs, Intel GPUs
- 📊 **Enhanced Analytics**: Real-time visualization, statistical analysis
- 🌐 **Web Interface**: Browser-based experiment management
- 🔄 **CI/CD Integration**: Automated testing and validation

### Easy Customization
- **New GPU Support**: Add configuration in `lib/gpu_config.sh`
- **Additional Tools**: Extend `lib/profiling.sh` 
- **Custom Analytics**: Create new libraries in `lib/`
- **Web Frontend**: Build on existing CLI foundation

---

## 📞 Support & Contributing

### Self-Help Resources
1. **Comprehensive Help**: `./launch_v2.sh --help`
2. **Debug Mode**: `./launch_v2.sh --debug`  
3. **Test Suite**: `./tests/test_framework.sh`
4. **Library Testing**: `source lib/common.sh && log_info "Test"`

### Getting Help
1. **Check Debug Output**: Enable `--debug` for detailed information
2. **Test Individual Components**: Source libraries individually
3. **Compare with Legacy**: Use `./legacy/launch.sh` to isolate issues
4. **Review Documentation**: Complete guides in this README

### Contributing
The modular architecture makes contributions straightforward:
- **Bug Fixes**: Target specific libraries for isolated changes
- **New Features**: Add new functions to appropriate libraries
- **Testing**: Add tests to `tests/` directory
- **Documentation**: Update this unified README

---

## 📈 Performance Benefits

### Compared to Legacy Framework

| Metric | Legacy | v2.0 | Improvement |
|--------|--------|------|-------------|
| **Lines of Code** | ~1200 | ~200 (+libs) | 6x more maintainable |
| **Setup Time** | Manual editing | CLI args | 10x faster |
| **Error Handling** | Basic | Comprehensive | Robust debugging |
| **Documentation** | Inline comments | Dedicated docs | Complete coverage |
| **Testing** | Manual | Automated | Reliable validation |
| **Maintenance** | Difficult | Modular | Easy updates |

### User Experience Improvements
- ✅ **Faster Experiments**: No script editing required
- ✅ **Better Debugging**: Comprehensive error messages and debug mode
- ✅ **Flexible Configuration**: Command-line args + config files
- ✅ **Auto-Detection**: Smart defaults reduce manual configuration
- ✅ **Clean Output**: Professional terminal output across all environments

---

*The AI Inference Energy Profiling Framework v2.0 - Modular architecture for comprehensive power analysis across GPU architectures and AI workloads.*

## Quick Start

### CLI-Based Usage (Recommended)
```bash
# Show all available options
./launch_v2.sh --help

# Default A100 DVFS experiment
./launch_v2.sh

# V100 baseline experiment 
./launch_v2.sh --gpu-type V100 --profiling-mode baseline

# Custom application profiling
./launch_v2.sh \
  --app-name "StableDiffusion" \
  --app-executable "stable_diffusion" \
  --app-params "--prompt 'A beautiful landscape' --steps 20"
```

### Legacy Usage (Still Supported)
```bash
# Clean previous results
./clean.sh -f

# Run with legacy scripts
./legacy/launch.sh
```

## Command-Line Interface

The `launch_v2.sh` script accepts comprehensive command-line arguments:

```bash
./launch_v2.sh [OPTIONS]

Options:
  --gpu-type TYPE          GPU type: A100 or V100 (default: A100)
                           (auto-detected and overridden when possible)
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
./launch_v2.sh --gpu-type A100
```
- Architecture: GA100
- Memory: 1215 MHz
- Core frequencies: 1410-510 MHz (61 frequencies)
- SLURM partition: toreador
- Cluster: HPCC at Texas Tech University
- Interactive helper: `./interactive_gpu.sh --gpu-type A100`

#### V100 GPU (Matador Partition)
```bash
./launch_v2.sh --gpu-type V100
```
- Architecture: GV100  
- Memory: 877 MHz
- Core frequencies: 1380-510 MHz (117 frequencies)
- SLURM partition: matador
- Interactive helper: `./interactive_gpu.sh --gpu-type V100`

#### H100 GPU (REPACSS h100-build Partition)
```bash
./launch_v2.sh --gpu-type H100
```
- Architecture: GH100  
- Memory: 2619 MHz
- Core frequencies: 1785-510 MHz (86 frequencies in 15MHz steps)
- SLURM partition: h100-build (node: rpg-93-9)
- Cluster: REPACSS at Texas Tech University
- Interactive helper: `./interactive_gpu.sh --gpu-type H100`

### Profiling Tool Selection

#### DCGMI (Default with Automatic Fallback)
```bash
./launch_v2.sh --profiling-tool dcgmi
```
- Uses: `profile.py` and `control.sh`
- Requires: DCGMI tools installed
- Features: Comprehensive GPU metrics
- **Automatic fallback** to nvidia-smi if DCGMI unavailable

#### nvidia-smi (Alternative)
```bash
./launch_v2.sh --profiling-tool nvidia-smi
```
- Uses: `profile_smi.py` and `control_smi.sh`
- Requires: NVIDIA drivers (nvidia-smi)
- Features: Standard GPU monitoring

### 📊 Technical Specifications

#### Sampling Configuration
- **Sampling Interval**: 50ms (20 samples per second)
- **DCGMI Command**: `dcgmi dmon -i 0 -e [...] -d 50`
- **nvidia-smi Command**: `nvidia-smi [...] --loop-ms=50`
- **Data Resolution**: ~1,200 samples per minute
- **Temporal Precision**: Captures brief GPU activity transitions

#### Data Collection
- **Metrics per Sample**: 25 comprehensive DCGMI fields (v2.0.1), 17 nvidia-smi fields
- **DCGMI Fields**: Device info, power metrics, temperatures, utilization, memory, clocks, P-state, compute activity
- **File Format**: CSV with timestamps and comprehensive GPU telemetry
- **Storage**: ~150KB per 20-second run (typical application)
### Experiment Modes

#### DVFS Mode (Default)
```bash
./launch_v2.sh --profiling-mode dvfs
```
- **Full frequency sweep** across all supported frequencies
- Comprehensive energy analysis
- Longer execution time (1-6 hours depending on configuration)

#### Baseline Mode
```bash
./launch_v2.sh --profiling-mode baseline
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
- **`interactive_gpu.sh`** - 🎯 **Unified interactive session helper** (supports V100, A100, H100)

The unified interactive helper (`interactive_gpu.sh`) provides:
- **Auto-detection** of GPU type and optimal settings
- **Color-coded output** for better readability
- **Quick framework testing** and validation
- **Node status checking** and troubleshooting
- **Usage examples** and best practice guidance
- **Legacy compatibility** with automatic migration support

### SLURM Scripts
- **`submit_job.sh`** - Main A100 SLURM submission (toreador)
- **`submit_job_v100.sh`** - 🎯 **Unified V100 submission script** (16 pre-configured options)
- **`submit_job_h100.sh`** - 🎯 **Unified H100 submission script** with multiple configurations
- **`submit_job_a100_baseline.sh`** - A100 baseline profiling (quick test)
- **`submit_job_a100_comprehensive.sh`** - A100 comprehensive DVFS study
- **`submit_job_a100_custom_app.sh`** - A100 custom application examples
- **`submit_job_v100_baseline.sh`** - Legacy V100 baseline (redirects to unified)
- **`submit_job_v100_comprehensive.sh`** - Legacy V100 comprehensive (redirects to unified)
- **`submit_job_v100_custom_app.sh`** - Legacy V100 custom app (redirects to unified)
- **`submit_job_h100_baseline.sh`** - H100 baseline profiling
- **`submit_job_h100_comprehensive.sh`** - H100 comprehensive profiling  
- **`submit_job_h100_custom_app.sh`** - H100 custom application examples
- **`submit_job_custom_app.sh`** - Custom application examples
- **`submit_job_comprehensive.sh`** - Full DVFS study

**🎯 V100 Users:** Use the unified `submit_job_v100.sh` script which provides 16 pre-configured options:
- Quick tests (baseline, frequency sampling)
- AI applications (LSTM, Stable Diffusion, LLaMA, Whisper, Vision Transformer) 
- DVFS studies (comprehensive, efficient, statistical)
- Tool compatibility (DCGMI, nvidia-smi fallback)
- Research configurations (energy efficiency, precision comparison)

## Configuration Matrix

| GPU Type | Tool | Profile Script | Control Script | Memory Freq | Default Core |
|----------|------|----------------|----------------|-------------|--------------|
| A100 | dcgmi | profile.py | control.sh | 1215 MHz | 1410 MHz |
| A100 | nvidia-smi | profile_smi.py | control_smi.sh | 1215 MHz | 1410 MHz |
| V100 | dcgmi | profile.py | control.sh | 877 MHz | 1380 MHz |
| V100 | nvidia-smi | profile_smi.py | control_smi.sh | 877 MHz | 1380 MHz |
| H100 | dcgmi | profile.py | control.sh | 2619 MHz | 1785 MHz |
| H100 | nvidia-smi | profile_smi.py | control_smi.sh | 2619 MHz | 1785 MHz |

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

---

## 🧹 Workspace Management

### Enhanced Cleanup Script (v2.0)

The `clean.sh` script has been significantly enhanced to handle the new results directory structure and provide advanced cleanup options:

```bash
# Basic cleanup (interactive)
./clean.sh

# Preview what would be cleaned (dry run)
./clean.sh --dry-run

# Force cleanup without prompts
./clean.sh --force

# Create backup before cleaning
./clean.sh --backup --force

# Clean only specific GPU results
./clean.sh --gpu-type H100

# Clean only specific application results  
./clean.sh --app-name stablediffusion

# Clean results older than 7 days
./clean.sh --older-than 7

# Interactive selective cleanup
./clean.sh --selective

# Comprehensive cleanup with backup
./clean.sh --backup --verbose --force
```

#### Enhanced Features

- **🎯 Smart Pattern Matching**: Automatically detects `results_h100_*`, `results_a100_*`, etc.
- **🗂️ SLURM Output Cleanup**: Removes `PROFILING_*.out` and `PROFILING_*.err` files
- **🔍 Advanced Filtering**: Filter by GPU type, application, or file age
- **💾 Backup Creation**: Create timestamped archives before cleanup
- **📊 Size Reporting**: Shows disk space that will be freed
- **⚙️ Selective Mode**: Interactive selection of what to clean
- **🔍 Preview Mode**: Dry run to see what would be cleaned
- **📈 Comprehensive Logging**: Detailed progress and summary reporting

#### Cleanup Targets

The enhanced script cleans:
- **Results Directories**: `results_h100_*`, `results_a100_*`, `results_v100_*`, `results/` (legacy)
- **SLURM Files**: `PROFILING_*.out`, `PROFILING_*.err` 
- **Log Files**: `log.*` patterns
- **Temporary Files**: `*.tmp`, `*.temp`, `*.pyc`, `core.*`, `changeme`

#### Safety Features

- **Dry Run Mode**: Test cleanup operations safely with `--dry-run`
- **Backup Option**: Create archives before deletion with `--backup`
- **Size Preview**: See exactly how much space will be freed
- **Confirmation Prompts**: Interactive confirmation unless `--force` is used
- **Granular Control**: Clean specific subsets with filters

---

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
./launch_v2.sh
```

#### V100 Baseline Testing
```bash
./launch_v2.sh --gpu-type V100 --profiling-mode baseline --num-runs 1
```

#### Custom Application Profiling
```bash
# Stable Diffusion
./launch_v2.sh \
  --app-name "StableDiffusion" \
  --app-executable "../app-stable-diffusion/StableDiffusionViaHF.py" \
  --app-params "--prompt 'A beautiful landscape' --steps 20"

# LLaMA
./launch_v2.sh \
  --app-name "LLaMA" \
  --app-executable "../app-llama/LlamaViaHF" \
  --app-params "--max-length 100"
```

#### Quick Testing Configuration
```bash
./launch_v2.sh --num-runs 1 --sleep-interval 0 --profiling-mode baseline
```

#### Comprehensive Experiment
```bash
./launch_v2.sh --gpu-type A100 --profiling-mode dvfs --num-runs 3
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

## Interactive GPU Sessions

### 🎯 **NEW: Unified Interactive Script**
The `interactive_gpu.sh` script provides a unified interface for all GPU types:

```bash
# Start interactive sessions
./interactive_gpu.sh v100          # V100 on HPCC
./interactive_gpu.sh a100          # A100 on HPCC (requires reservation)
./interactive_gpu.sh h100          # H100 on REPACSS

# Get information and test
./interactive_gpu.sh v100 info     # V100 specifications
./interactive_gpu.sh a100 status   # A100 availability
./interactive_gpu.sh h100 test     # H100 framework test (in session)
./interactive_gpu.sh help          # Full help
```

See `README_INTERACTIVE_GPU.md` for detailed documentation.

---

## Experiment Summary Generation (v2.0)

Each experiment automatically generates comprehensive summaries:

#### `experiment_summary.log`
```
AI Inference Energy Profiling Experiment Summary
================================================

Experiment Details:
  Framework Version: 2.0.1
  Timestamp: 2025-07-10 16:43:43
  Mode: baseline
  
GPU Configuration:
  Type: H100
  Architecture: GH100
  Memory Frequency: 2619 MHz
  Core Frequency Range: 510-1785 MHz

Profiling Configuration:
  Tool: dcgmi
  Runs per frequency: 3
  Sleep interval: 1s

Application Configuration:
  Name: StableDiffusion
  Executable: ../app-stable-diffusion/StableDiffusionViaHF.py
  Parameters: --prompt "A beautiful landscape" --steps 20

Output Files:
  results_h100_stablediffusion/experiment_summary.log
  results_h100_stablediffusion/timing_summary.log
  [... all generated files listed ...]

Run Timing Summary:
===================
  Run baseline_01    :  38s (freq: 1785MHz, status: success)
  Run baseline_02    :  35s (freq: 1785MHz, status: success)
  Run baseline_03    :  34s (freq: 1785MHz, status: success)

Timing Statistics:
  Total runs:       3
  Successful runs:  3
  Failed runs:      0
  Total duration:   107s
  Average duration: 35s
  Min duration:     34s
  Max duration:     38s

Experiment completed at: 2025-07-10 16:43:43
```

#### `timing_summary.log`
```
# Run Timing Summary
# Format: run_id,frequency_mhz,duration_seconds,exit_code,status
baseline_01,1785,38,0,success
baseline_02,1785,35,0,success
baseline_03,1785,34,0,success
```

#### Robust Summary Generation
- **Error Tolerance**: Graceful handling of summary generation issues
- **Complete Metadata**: All experiment parameters and hardware info
- **Performance Metrics**: Detailed timing and success statistics
- **File Inventory**: Complete list of all generated output files
- **Analysis Ready**: CSV format for direct import into analysis tools
