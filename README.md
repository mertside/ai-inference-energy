# AI Inference Energy

A **comprehensive framework** for studying energy-efficient GPU frequency selection for AI inference workloads. This framework provides **complete command-line interfaces**, **triple GPU architecture support (A100/V100/H100)**, **intelligent tool fallback**, **comprehensive profiling tools**, and **multiple AI model support** for conducting systematic DVFS (Dynamic Voltage and Frequency Scaling) research on modern AI workloads.

## 🎯 Project Overview

As AI workloads grow in complexity and energy demand, static frequency settings on GPUs often result in sub-optimal trade-offs between performance and power consumption. This framework provides tools for conducting comprehensive energy profiling experiments on **NVIDIA A100, V100, and H100 GPUs** across various AI inference tasks.

### ✨ Key Features

- 🔧 **Complete CLI Interface**: Configure all experiments via command-line arguments with --help support
- 🎯 **Triple GPU Support**: Native A100, V100, and H100 configurations 
- 🛠️ **Multiple Profiling Tools**: Support for both DCGMI and nvidia-smi profiling with automatic fallback
- 📊 **Flexible Experiment Modes**: DVFS (full frequency sweep) or baseline (single frequency) modes
- 🚀 **HPC Integration**: Ready-to-use SLURM submission scripts for cluster environments
- ⚡ **Intelligent Fallback**: Automatic tool selection when DCGMI is unavailable
- 📈 **Comprehensive Logging**: Error handling and progress tracking
- 🔄 **Professional Architecture**: Modular, maintainable, and extensible codebase
- 🐍 **Python 3.8+ Compatible**: Works with modern cluster environments
- 📊 **Profiling Infrastructure**: Comprehensive GPU profiling with DCGMI and nvidia-smi support
- 🔍 **Data Collection**: Systematic energy and performance data collection across AI workloads  
- �️ **Framework Foundation**: Extensible foundation for power modeling and analysis (modules in development)
- 🎨 **Modernized AI Models**: Latest Stable Diffusion variants (SDXL, Turbo, Lightning) with comprehensive benchmarking

### 🎉 Latest Updates (v2.0.1)

- ✅ **Configuration Consolidation**: Unified DCGMI monitoring with 25 comprehensive fields (vs 17 previously)
- ✅ **Clean Filenames**: Fixed duplicate frequency naming in custom experiments (`run_01_freq_510` vs `run_freq_510_01_freq_510`)
- ✅ **Robust Imports**: Resolved configuration import conflicts for reliable operation
- ✅ **Enhanced Compatibility**: Improved PyTorch/torchvision compatibility in AI model environments
- ✅ **Visualization Module Fixes**: Resolved syntax errors and duplicate code in `power_plots.py`, enhanced module reliability

### 🔋 Profiling Infrastructure Foundation

This framework provides a robust foundation for GPU energy profiling and comprehensive data collection:

#### **Core Infrastructure (Available)**
- **Profiling Data Collection**: Comprehensive GPU profiling with DCGMI and nvidia-smi across V100, A100, and H100
- **Application Integration**: Support for LLaMA, Stable Diffusion, Whisper, Vision Transformer, and LSTM workloads
- **Job Automation**: Complete SLURM integration with automated frequency sweeping
- **Data Export**: Structured CSV output for analysis and visualization
- **Comprehensive Testing**: Full test coverage for profiling infrastructure and AI applications

#### **Planned Extensions (Future Work)**
- **FGCS Power Models**: ML-based power prediction models from FGCS 2023 methodology
- **EDP Optimization**: Energy-Delay Product and ED²P optimization algorithms
- **Advanced Analytics**: Statistical analysis and model validation frameworks
- **Visualization Suite**: Comprehensive plotting and analysis tools

```bash
# Quick profiling example - Available now!
cd sample-collection-scripts
./launch_v2.sh --app-name "StableDiffusion" --profiling-mode baseline
# Results saved to structured CSV files for analysis
```

### Supported AI Models & Applications

- **[LLaMA](https://github.com/meta-llama/llama)**: Text generation via transformer-based large language models
- **[Stable Diffusion](https://github.com/CompVis/stable-diffusion)**: **Modernized** latent diffusion model with latest variants (SD v1.x, v2.x, SDXL, Turbo, Lightning) for high-quality image generation  
- **[Whisper](https://github.com/openai/whisper)**: OpenAI Whisper automatic speech recognition for audio processing energy profiling
- **[Vision Transformer (ViT)](https://github.com/huggingface/transformers)**: Transformer-based image classification for computer vision energy profiling
- **LSTM Sentiment Analysis**: Binary classification benchmark for consistent profiling
- **Custom Applications**: Framework supports any Python-based AI inference workload

### Research & Experimental Capabilities

- 📊 **Comprehensive Profiling**: GPU power consumption, utilization, temperature, and performance metrics
- 🔄 **Frequency Scaling**: Support for 61 A100 frequencies (1410-510 MHz) and 117 V100 frequencies (1380-510 MHz)
- ⚡ **Energy Analysis**: Detailed power vs performance trade-off analysis across frequency ranges
- 📈 **Statistical Rigor**: Multiple runs per frequency with configurable parameters for statistical significance
- 📝 **Reproducible Research**: Standardized output formats and comprehensive experiment documentation

## 🏗️ Repository Structure

```
ai-inference-energy/
├── README.md                            # Project documentation
├── requirements.txt                     # Python dependencies  
├── setup.py                             # Package installation
├── config.py                            # Centralized configuration (Python 3.8+ compatible)
├── utils.py                             # Utility functions and helpers
│
├── app-llama/                           # LLaMA inference applications
│   ├── README.md                        # LLaMA application documentation
│   └── LlamaViaHF.py                    # LLaMA text generation via Hugging Face
│  
├── app-stable-diffusion/                # 🎨 Modernized Stable Diffusion applications  
│   ├── README.md                        # Comprehensive Stable Diffusion documentation
│   ├── StableDiffusionViaHF.py          # **Modernized** image generation with latest models
│   ├── scripts/                         # Setup and utility scripts
│   │   └── setup_stable_diffusion.sh    # Complete setup and validation script
│   ├── test_stable_diffusion_*.py       # Comprehensive test suites
│   └── validate_stable_diffusion.py    # Quick validation script
│
├── app-whisper/                         # 🎤 Whisper speech recognition applications
│   ├── README.md                        # Comprehensive Whisper documentation
│   ├── WhisperViaHF.py                  # OpenAI Whisper speech-to-text via Hugging Face
│   ├── __init__.py                      # Python package initialization
│   ├── setup/                           # Environment setup and configuration
│   │   ├── setup_whisper_env.sh         # Automated conda environment setup
│   │   ├── requirements.txt             # Python dependencies
│   │   └── whisper-repacss.yml          # REPACSS cluster environment
│   └── tests/                           # Test suite for Whisper implementation
│       └── test_whisper.py              # Comprehensive test suite
│
├── app-vision-transformer/              # 🖼️ Vision Transformer applications
│   ├── README.md                        # Comprehensive ViT documentation
│   ├── ViTViaHF.py                      # Vision Transformer image classification via Hugging Face
│   ├── __init__.py                      # Python package initialization
│   └── setup/                           # Environment setup and configuration
│       ├── setup.sh                     # Automated conda environment setup
│       ├── requirements.txt             # Python dependencies
│       ├── vit-env-repacss.yml          # REPACSS cluster environment
│       └── vit-env-hpcc.yml             # HPCC cluster environment
│
├── app-lstm/                            # LSTM benchmark application
│   ├── README.md                        # LSTM benchmark documentation
│   └── lstm.py                          # Sentiment analysis benchmark
│
├── examples/                            # 📋 Usage examples and demonstrations
│   ├── README.md                        # Comprehensive examples documentation
│   ├── example_usage.py                 # Framework usage demonstration
│   ├── simple_power_modeling_demo.py    # Basic demonstration (planned feature preview)
│   └── submit_helper.sh                 # SLURM submission helper
│
├── tests/                               # 🧪 Comprehensive test suite
│   ├── README.md                        # Test documentation and coverage
│   ├── test_integration.py              # Integration and system tests
│   ├── test_configuration.py            # Configuration and compatibility tests
│   ├── test_hardware_module.py          # Hardware detection tests
│   ├── test_utils.py                    # Utility function tests
│   └── test_python_compatibility.sh   # Python compatibility test
│
├── documentation/                       # 📚 Essential documentation (streamlined)
│   ├── README.md                        # Documentation index and quick reference
│   ├── GPU_USAGE_GUIDE.md               # Complete GPU support guide (A100/V100/H100)
│   ├── USAGE_EXAMPLES.md                # CLI usage examples and automation
│   └── SUBMIT_JOBS_README.md            # SLURM usage and HPC deployment
│
└── sample-collection-scripts/           # 🚀 Enhanced profiling framework
    ├── README.md                        # Profiling framework documentation
    ├── launch_v2.sh                     # 🎯 Main experiment orchestration (enhanced CLI)
    ├── profile.py                       # DCGMI-based GPU profiler
    ├── profile_smi.py                   # nvidia-smi alternative profiler  
    ├── control.sh                       # DCGMI frequency control
    ├── control_smi.sh                   # nvidia-smi frequency control
    ├── clean.sh                         # Enhanced workspace cleanup
    ├── lstm.py                          # LSTM benchmark application
    │
    ├── interactive_gpu.sh               # 🎯 Unified interactive GPU session helper (V100/A100/H100)
    │
    ├── submit_job_v100.sh               # 🎯 Unified V100 submission (16 configurations)
    ├── submit_job_a100.sh               # 🎯 Unified A100 submission (16 configurations) 
    ├── submit_job_h100.sh               # 🎯 Unified H100 submission (16 configurations)
    │
    ├── submit_job_v100_baseline.sh      # Legacy V100 baseline (redirects to unified)
    ├── submit_job_v100_comprehensive.sh # Legacy V100 comprehensive (redirects to unified)
    ├── submit_job_v100_custom_app.sh    # Legacy V100 custom app (redirects to unified)
    ├── submit_job_a100_baseline.sh      # Legacy A100 baseline (redirects to unified)
    ├── submit_job_a100_comprehensive.sh # Legacy A100 comprehensive (redirects to unified)
    ├── submit_job_a100_custom_app.sh    # Legacy A100 custom app (redirects to unified)
    ├── submit_job_h100_baseline.sh      # Legacy H100 baseline (redirects to unified)
    ├── submit_job_h100_comprehensive.sh # Legacy H100 comprehensive (redirects to unified)
    ├── submit_job_h100_custom_app.sh    # Legacy H100 custom app (redirects to unified)
    └── submit_job*.sh                   # Additional legacy scripts
```

## 🚀 Quick Start

### Prerequisites

#### Hardware Requirements
- NVIDIA GPU with DCGMI support (A100/H100 recommended)
- Sufficient GPU memory for AI models (8GB+ recommended)
- CUDA-compatible driver

#### Software Requirements
- Python 3.8+ (tested on Python 3.8-3.11)
- CUDA Toolkit 11.0+
- NVIDIA DCGMI tools (automatically falls back to nvidia-smi if unavailable)
- Hugging Face account with model access

**Framework Note:** This project provides two profiling frameworks:
- **`launch_v2.sh`** - Enhanced framework with modular architecture (recommended)
- **`launch_v2.sh`** - Enhanced framework with modular architecture

#### HPC Environment (Optional)
- SLURM workload manager
- Environment modules (GCC, CUDA, cuDNN)
- Conda/Miniconda

### Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd ai-inference-energy
   ```

2. **Install Python dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up Hugging Face authentication**
   ```bash
   huggingface-cli login
   # Follow prompts to enter your HF token
   ```

4. **Verify GPU and profiling tool setup**
   ```bash
   nvidia-smi                    # Check GPU status
   dcgmi discovery --list        # Verify DCGMI access (optional - will fallback to nvidia-smi)
   
   # Use unified interactive helper for quick setup validation
   cd sample-collection-scripts
   ./interactive_gpu.sh          # Auto-detects GPU type and provides setup guidance
   ```

5. **Make scripts executable**
   ```bash
   chmod +x sample-collection-scripts/*.sh
   chmod +x sample-collection-scripts/profile.py
   chmod +x app-stable-diffusion/scripts/setup_stable_diffusion.sh
   ```

### Basic Usage

#### 1. Individual Model Testing

**Run LLaMA inference:**
```bash
cd app-llama
python LlamaViaHF.py
```

**Run Stable Diffusion inference:**
```bash
cd app-stable-diffusion
python StableDiffusionViaHF.py
```

**Run Whisper speech recognition:**
```bash
cd app-whisper
python WhisperViaHF.py --benchmark --num-samples 3
```

#### 2. Power Profiling

**Profile a single application:**
```bash
cd sample-collection-scripts
./profile.py "python ../app-llama/LlamaViaHF.py"
```

**Set specific GPU frequencies:**
```bash
# Set memory=1215MHz, core=1200MHz
./control.sh 1215 1200
```

#### 3. Enhanced Full Experiment Suite

**Complete CLI-driven experiments:**
```bash
cd sample-collection-scripts

# Show all available options
./launch_v2.sh --help

# Default A100 DVFS experiment with DCGMI
./launch_v2.sh

# V100 baseline experiment with nvidia-smi fallback
./launch_v2.sh --gpu-type V100 --profiling-mode baseline --profiling-tool nvidia-smi

# Custom application profiling
./launch_v2.sh \
  --app-name "StableDiffusion" \
  --app-executable "../app-stable-diffusion/StableDiffusionViaHF.py" \
  --app-params "--prompt 'A beautiful landscape' --steps 20"

# Quick test configuration
./launch_v2.sh --num-runs 1 --sleep-interval 0
```

#### 4. HPC Cluster Deployment

**Multiple SLURM submission options:**
```bash
# A100 unified submission (toreador partition) - Edit script first to uncomment desired config
sbatch submit_job_a100.sh

# V100 unified submission (matador partition) - Edit script first to uncomment desired config
sbatch submit_job_v100.sh

# H100 unified submission (h100-build partition) - Edit script first to uncomment desired config
sbatch submit_job_h100.sh
```

**Unified Script Features:**
- 📋 **16+ pre-configured options** in each GPU-specific script
- 🎯 **Easy selection**: Just uncomment one configuration
- ⏱️ **Timing guidance**: Built-in recommendations for SLURM `--time` parameter
- 🔧 **GPU-optimized**: Configurations tailored for each GPU architecture
 - 📖 **Full guide**: See `sample-collection-scripts/JOB_SCRIPT_GUIDE_V100.md`

**Legacy Scripts (Deprecated):**
```bash
# A100 legacy scripts (redirect to unified)
sbatch submit_job_a100_baseline.sh      # → use submit_job_a100.sh config #1
sbatch submit_job_a100_comprehensive.sh # → use submit_job_a100.sh config #8
sbatch submit_job_a100_custom_app.sh    # → use submit_job_a100.sh config #5

# V100 legacy scripts (redirect to unified)
sbatch submit_job_v100_baseline.sh      # → use submit_job_v100.sh config #1
sbatch submit_job_v100_comprehensive.sh # → use submit_job_v100.sh config #8
sbatch submit_job_v100_custom_app.sh    # → use submit_job_v100.sh config #7

# H100 legacy scripts (redirect to unified)
sbatch submit_job_h100_baseline.sh      # → use submit_job_h100.sh config #1
sbatch submit_job_h100_comprehensive.sh # → use submit_job_h100.sh config #4
sbatch submit_job_h100_custom_app.sh    # → use submit_job_h100.sh config #5

# Custom application profiling
sbatch submit_job_custom_app.sh
sbatch submit_job_h100_custom_app.sh

# Comprehensive DVFS study (all frequencies)
sbatch submit_job_comprehensive.sh
sbatch submit_job_v100_comprehensive.sh
sbatch submit_job_h100_comprehensive.sh
```

#### 5. Profiling Data Analysis

**Analyze profiling results:**
```bash
cd sample-collection-scripts

# Basic analysis with built-in tools
./launch_v2.sh --help  # See analysis options

# View profiling results
ls -la results_*/
head results_*/profiling_*.csv

# Use visualization tools
cd visualization
python plot_metric_vs_time.py --gpu V100 --app LLAMA --metric POWER
```

**📚 For detailed examples, see [`documentation/USAGE_EXAMPLES.md`](documentation/USAGE_EXAMPLES.md) and [`documentation/SUBMIT_JOBS_README.md`](documentation/SUBMIT_JOBS_README.md)**

## 🔧 Configuration

### GPU Frequency Settings

The framework supports comprehensive frequency scaling for all three GPU architectures:

#### **A100 GPU (Toreador Partition)**
- **Memory Frequency**: 1215 MHz (A100 default)
- **Core Frequencies**: 61 different settings from 1410 MHz down to 510 MHz
- **Frequency Control**: Via DCGMI interface with nvidia-smi fallback

#### **V100 GPU (Matador Partition)**  
- **Memory Frequency**: 877 MHz (V100 default)
- **Core Frequencies**: 117 different settings from 1380 MHz down to 510 MHz
- **Frequency Control**: Via nvidia-smi interface

#### **H100 GPU (REPACSS)**
- **Memory Frequency**: 2619 MHz (H100 maximum)
- **Core Frequencies**: 86 different settings from 1785 MHz down to 510 MHz in 15MHz steps
- **Frequency Control**: Via DCGMI interface with nvidia-smi fallback
- **Cluster**: REPACSS at Texas Tech University (node: rpg-93-9)

### Command-Line Interface

The `launch_v2.sh` script accepts comprehensive command-line arguments for flexible experiment configuration:

```bash
./launch_v2.sh [OPTIONS]
```

Options:
  --gpu-type TYPE          GPU type: A100 or V100 (default: A100)
  --profiling-tool TOOL    Profiling tool: dcgmi or nvidia-smi (default: dcgmi)
  --profiling-mode MODE    Mode: dvfs or baseline (default: dvfs)
  --num-runs NUM           Number of runs per frequency (default: 2)
  --sleep-interval SEC     Sleep between runs in seconds (default: 1)
  --app-name NAME          Application display name (default: LSTM)
  --app-executable PATH    Application executable path (default: lstm)
  --app-params "PARAMS"    Application parameters (default: "")
  -h, --help              Show help and examples
```

### Experiment Parameters

Key configuration options in `config.py` (Python 3.8+ compatible):

```python
# Profiling settings
DEFAULT_NUM_RUNS = 2              # Runs per frequency
DEFAULT_INTERVAL_MS = 50          # Sampling interval
DCGMI_FIELDS = [52, 50, 155, 160, ...]  # Comprehensive GPU metrics to collect (25 fields)
                                  # ✅ v2.0.1: Consolidated comprehensive field set
                                  # ✅ Includes: device info, power, temps, clocks, utilization, activity metrics

# Model settings
LLAMA_MODEL_NAME = "huggyllama/llama-7b"
STABLE_DIFFUSION_MODEL_NAME = "CompVis/stable-diffusion-v1-4"

# A100 GPU settings (Toreador partition)
A100_MEMORY_FREQ = 1215           # MHz
A100_DEFAULT_CORE_FREQ = 1410     # MHz

# V100 GPU settings (Matador partition)
V100_MEMORY_FREQ = 877            # MHz
V100_DEFAULT_CORE_FREQ = 1380     # MHz
```

## 📊 Output and Results

### Data Collection

The framework collects comprehensive GPU metrics during inference:

- **Power consumption** (watts)
- **GPU utilization** (%)
- **Memory utilization** (%)
- **Temperature** (°C)
- **Clock frequencies** (MHz)
- **Execution time** (seconds)

### Output Files

Results are saved in the `results/` directory with structured naming:

```
results/
├── GA100-dvfs-LSTM-1410-0        # Architecture-Mode-App-Freq-Iteration
├── GA100-dvfs-LSTM-1410-1
├── GA100-dvfs-LSTM-1395-0
└── GA100-dvfs-lstm-perf.csv      # Performance summary
```

### Analysis Scripts

The collected data can be analyzed using standard data science tools:

```python
import pandas as pd
import matplotlib.pyplot as plt

# Load performance data
perf_data = pd.read_csv('results/GA100-dvfs-lstm-perf.csv')

# Plot frequency vs execution time
plt.plot(perf_data['frequency'], perf_data['execution_time'])
plt.xlabel('GPU Frequency (MHz)')
plt.ylabel('Execution Time (s)')
plt.title('LLaMA Inference: Frequency vs Performance')
```

## 🛠️ Advanced Usage

### Custom Applications

To add new AI applications to the framework:

1. **Create application script** following the pattern:
   ```python
   # my_app.py
   import sys, os
   sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
   from utils import setup_logging
   
   def main():
       logger = setup_logging()
       # Your AI inference code here
       logger.info("Inference completed")
   
   if __name__ == "__main__":
       main()
   ```

2. **Run with launch script**:
   ```bash
   ./launch_v2.sh \
     --app-name "MyApp" \
     --app-executable "my_app" \
     --app-params "--model bert-base --batch-size 32"
   
   # For Stable Diffusion (modernized)
   ./launch_v2.sh \
     --app-name "StableDiffusion" \
     --app-executable "../app-stable-diffusion/StableDiffusionViaHF.py" \
     --app-params "--model-variant sdxl --steps 30"
   ```

### GPU-Specific Configurations

#### **A100 Configuration (Toreador)**
```bash
./launch_v2.sh \
  --gpu-type A100 \
  --profiling-tool dcgmi \
  --profiling-mode dvfs
```

#### **V100 Configuration (Matador)**
```bash
./launch_v2.sh \
  --gpu-type V100 \
  --profiling-tool nvidia-smi \
  --profiling-mode baseline
```

#### **H100 Configuration (REPACSS)**
```bash
./launch_v2.sh \
  --gpu-type H100 \
  --profiling-tool dcgmi \
  --profiling-mode dvfs
```

### Profiling Tool Selection & Fallback

The framework supports intelligent profiling tool selection:

```bash
# Prefer DCGMI (will fallback to nvidia-smi if unavailable)
./launch_v2.sh --profiling-tool dcgmi

# Force nvidia-smi usage
./launch_v2.sh --profiling-tool nvidia-smi

# Test profiling tool availability
dcgmi discovery --list  # Check DCGMI
nvidia-smi              # Check nvidia-smi
```

### Experiment Automation

#### **Batch Testing Multiple Configurations**
```bash
#!/bin/bash
# Test script for multiple GPU types and applications

for gpu in A100 V100; do
  for app in "LSTM" "StableDiffusion"; do
    ./launch_v2.sh \
      --gpu-type $gpu \
      --app-name $app \
      --profiling-mode baseline \
      --num-runs 1
  done
done
```

## 🔍 Troubleshooting

### Common Issues

#### GPU Access & Tool Problems
```bash
# Check GPU visibility and type
nvidia-smi

# Check DCGMI availability (optional)
dcgmi discovery --list

# Test profiling tool fallback
./launch_v2.sh --profiling-tool dcgmi  # Will auto-fallback to nvidia-smi if needed

# Reset GPU if needed
sudo nvidia-smi --gpu-reset
```

#### Python & Environment Issues
```bash
# Check Python version (3.8+ required)
python --version

# Test config module compatibility
python -c "import config; print('Config loaded successfully')"

# Check HuggingFace authentication
huggingface-cli whoami
```

#### SLURM & Partition Issues
```bash
# Check available partitions
sinfo

# Check A100 nodes (toreador)
sinfo -p toreador

# Check V100 nodes (matador)
sinfo -p matador

# Test SLURM job submission
sbatch --test-only submit_job.sh
```

**📚 For detailed troubleshooting, see [`documentation/GPU_USAGE_GUIDE.md`](documentation/GPU_USAGE_GUIDE.md) troubleshooting sections**

### Performance Optimization

#### For Better Profiling Accuracy
- Ensure stable GPU temperature before experiments
- Run experiments during low system load
- Use dedicated GPU nodes when possible
- Increase sampling interval for longer workloads
- Use `--profiling-mode baseline` for quick testing

#### For Faster Experiments
- Use `--num-runs 1` for quick tests
- Set `--sleep-interval 0` to reduce delays
- Use `--profiling-mode baseline` (single frequency)
- Test with smaller model variants first
- Use V100 nodes with `--gpu-type V100` for availability

## 📚 Documentation

The framework includes **streamlined documentation** focused on practical usage:

### 🎯 **Essential Guides**
- **[GPU_USAGE_GUIDE.md](documentation/GPU_USAGE_GUIDE.md)**: Complete GPU support guide for A100, V100, and H100 across HPCC and REPACSS clusters
- **[USAGE_EXAMPLES.md](documentation/USAGE_EXAMPLES.md)**: Complete CLI usage examples and automation scripts
- **[SUBMIT_JOBS_README.md](documentation/SUBMIT_JOBS_README.md)**: SLURM submission guide and HPC cluster deployment

### 📋 **Additional Module Documentation**
- **[sample-collection-scripts/README.md](sample-collection-scripts/README.md)**: Profiling framework documentation
- **[app-stable-diffusion/README.md](app-stable-diffusion/README.md)**: Modernized Stable Diffusion application with latest models
- **[app-whisper/README.md](app-whisper/README.md)**: OpenAI Whisper speech recognition for audio processing energy profiling
- **[app-llama/README.md](app-llama/README.md)**: LLaMA text generation application for language model energy profiling

All documentation follows consistent patterns with **practical examples**, **real commands**, and **comprehensive troubleshooting** sections.

## 📝 Citation

If you use this framework in your research, please cite:

```bibtex
@misc{Side:2025:AIEnergy:GitHub,
  title={AI Inference Energy Profiling Framework},
  author={Side, Mert},
  year={2025},
  url={https://github.com/mertside/ai-inference-energy}
}
```
---

**Happy profiling! ⚡🔬**
