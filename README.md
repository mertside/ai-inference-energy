# AI Inference Energy Profi### Research & Experimental Capabilities

- 📊 **Comprehensive Profiling**: GPU power consumption, utilization, temperature, and performance metrics
- 🔄 **Frequency Scaling**: Support for 61 A100 frequencies (1410-510 MHz) and 103 V100 frequencies (1380-405 MHz)
- ⚡ **Energy Analysis**: Detailed power vs performance trade-off analysis across frequency ranges
- 📈 **Statistical Rigor**: Multiple runs per frequency with configurable parameters for statistical significance
- 🔬 **Reproducible Research**: Standardized output formats and comprehensive experiment documentation
- 🏗️ **HPC Cluster Ready**: Native SLURM integration with partition-specific configurationsamework

A **comprehensive, production-ready framework** for studying energy-efficient GPU frequency selection for AI inference workloads. This framework provides **complete command-line interfaces**, **dual GPU architecture support (A100/V100)**, **intelligent tool fallback**, and **multiple profiling tools** for conducting systematic DVFS (Dynamic Voltage and Frequency Scaling) research on modern AI workloads.

## 🎯 Project Overview

As AI workloads grow in complexity and energy demand, static frequency settings on GPUs often result in sub-optimal trade-offs between performance and power consumption. This framework provides enterprise-grade tools for conducting comprehensive energy profiling experiments on **NVIDIA A100 and V100 GPUs** across various AI inference tasks.

### ✨ Key Features

- 🔧 **Complete CLI Interface**: Configure all experiments via command-line arguments with --help support
- 🎯 **Dual GPU Support**: Native A100 (toreador partition) and V100 (matador partition) configurations 
- 🛠️ **Multiple Profiling Tools**: Support for both DCGMI and nvidia-smi profiling with automatic fallback
- 📊 **Flexible Experiment Modes**: DVFS (full frequency sweep) or baseline (single frequency) modes
- 🚀 **HPC Integration**: Ready-to-use SLURM submission scripts for cluster environments
- ⚡ **Intelligent Fallback**: Automatic tool selection when DCGMI is unavailable
- 📈 **Comprehensive Logging**: Enterprise-grade error handling and progress tracking
- 🔄 **Professional Architecture**: Modular, maintainable, and extensible codebase
- 🐍 **Python 3.6+ Compatible**: Works on older cluster environments

### Supported AI Models & Applications

- **[LLaMA](https://github.com/meta-llama/llama)**: Text generation via transformer-based large language models
- **[Stable Diffusion](https://github.com/CompVis/stable-diffusion)**: Latent diffusion model for high-quality image generation  
- **LSTM Sentiment Analysis**: Binary classification benchmark for consistent profiling
- **Custom Applications**: Framework supports any Python-based AI inference workload

### Research & Experimental Capabilities

- 📊 **Comprehensive Profiling**: GPU power consumption, utilization, temperature, and performance metrics
- 🔄 **Frequency Scaling**: Support for 61 A100 frequencies (1410-510 MHz) and 103 V100 frequencies (1380-405 MHz)
- ⚡ **Energy Analysis**: Detailed power vs performance trade-off analysis across frequency ranges
- 📈 **Statistical Rigor**: Multiple runs per frequency with configurable parameters for statistical significance
- � **Reproducible Research**: Standardized output formats and comprehensive experiment documentation

## 🏗️ Repository Structure

```
ai-inference-energy/
├── README.md                           # Project documentation
├── requirements.txt                    # Python dependencies  
├── setup.py                           # Package installation
├── config.py                          # Centralized configuration (Python 3.6+ compatible)
├── utils.py                           # Utility functions and helpers
│
├── app-llama-collection/               # LLaMA inference applications
│   └── LlamaViaHF.py                  # LLaMA text generation via Hugging Face
│
├── app-stable-diffusion-collection/    # Stable Diffusion applications  
│   └── StableDiffusionViaHF.py        # Image generation via Hugging Face
│
├── app-lstm/                          # LSTM benchmark application
│   └── lstm.py                        # Sentiment analysis benchmark
│
├── documentation/                      # 📚 Comprehensive documentation
│   ├── USAGE_EXAMPLES.md              # CLI usage examples and automation
│   ├── SUBMIT_JOBS_README.md          # SLURM usage documentation
│   ├── CLI_ENHANCEMENT_SUMMARY.md     # Technical implementation details
│   ├── REFACTORING_SUMMARY.md         # Complete refactoring overview
│   ├── PYTHON36_COMPATIBILITY_FIX.md  # Python 3.6 compatibility guide
│   └── QUICK_FIX_GUIDE.md             # Troubleshooting and fixes
│
└── sample-collection-scripts/          # 🚀 Enhanced profiling framework
    ├── launch.sh                      # 🎯 Main experiment orchestration (CLI enhanced)
    ├── profile.py                     # DCGMI-based GPU profiler
    ├── profile_smi.py                 # nvidia-smi alternative profiler  
    ├── control.sh                     # DCGMI frequency control
    ├── control_smi.sh                 # nvidia-smi frequency control
    ├── clean.sh                       # Enhanced workspace cleanup
    │
    ├── submit_job.sh                  # 🎯 Main SLURM submission (A100/toreador)
    ├── submit_job_v100_baseline.sh    # V100 baseline profiling (matador)
    ├── submit_job_custom_app.sh       # Custom application examples
    ├── submit_job_comprehensive.sh    # Full DVFS study
    └── submit_job_v100_comprehensive.sh # V100 comprehensive profiling
```

## 🚀 Quick Start

### Prerequisites

#### Hardware Requirements
- NVIDIA GPU with DCGMI support (A100/H100 recommended)
- Sufficient GPU memory for AI models (8GB+ recommended)
- CUDA-compatible driver

#### Software Requirements
- Python 3.6+ (tested on Python 3.6-3.11)
- CUDA Toolkit 11.0+
- NVIDIA DCGMI tools (automatically falls back to nvidia-smi if unavailable)
- Hugging Face account with model access

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
   ```

5. **Make scripts executable**
   ```bash
   chmod +x sample-collection-scripts/*.sh
   chmod +x sample-collection-scripts/profile.py
   ```

### Basic Usage

#### 1. Individual Model Testing

**Run LLaMA inference:**
```bash
cd app-llama-collection
python LlamaViaHF.py
```

**Run Stable Diffusion inference:**
```bash
cd app-stable-diffusion-collection
python StableDiffusionViaHF.py
```

#### 2. Power Profiling

**Profile a single application:**
```bash
cd sample-collection-scripts
./profile.py "python ../app-llama-collection/LlamaViaHF.py"
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
./launch.sh --help

# Default A100 DVFS experiment with DCGMI
./launch.sh

# V100 baseline experiment with nvidia-smi fallback
./launch.sh --gpu-type V100 --profiling-mode baseline --profiling-tool nvidia-smi

# Custom application profiling
./launch.sh \
  --app-name "StableDiffusion" \
  --app-executable "stable_diffusion" \
  --app-params "--prompt 'A beautiful landscape' --steps 20"

# Quick test configuration
./launch.sh --num-runs 1 --sleep-interval 0
```

#### 4. HPC Cluster Deployment

**Multiple SLURM submission options:**
```bash
# Main A100 submission (toreador partition)
sbatch submit_job.sh

# V100 baseline profiling (matador partition)
sbatch submit_job_v100_baseline.sh

# Custom application profiling
sbatch submit_job_custom_app.sh

# Comprehensive DVFS study (all frequencies)
sbatch submit_job_comprehensive.sh
```

**📚 For detailed examples, see [`documentation/USAGE_EXAMPLES.md`](documentation/USAGE_EXAMPLES.md) and [`documentation/SUBMIT_JOBS_README.md`](documentation/SUBMIT_JOBS_README.md)**

## 🔧 Configuration

### GPU Frequency Settings

The framework supports comprehensive frequency scaling for both GPU architectures:

#### **A100 GPU (Toreador Partition)**
- **Memory Frequency**: 1215 MHz (A100 default)
- **Core Frequencies**: 61 different settings from 1410 MHz down to 510 MHz
- **Frequency Control**: Via DCGMI interface with nvidia-smi fallback

#### **V100 GPU (Matador Partition)**  
- **Memory Frequency**: 877 MHz (V100 default)
- **Core Frequencies**: 103 different settings from 1380 MHz down to 405 MHz
- **Frequency Control**: Via nvidia-smi interface

### Command-Line Interface

The `launch.sh` script accepts comprehensive command-line arguments for flexible experiment configuration:

```bash
./launch.sh [OPTIONS]

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

Key configuration options in `config.py` (Python 3.6+ compatible):

```python
# Profiling settings
DEFAULT_NUM_RUNS = 2              # Runs per frequency
DEFAULT_INTERVAL_MS = 50          # Sampling interval
DCGMI_FIELDS = [1001, 1002, ...]  # GPU metrics to collect

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
   ./launch.sh \
     --app-name "MyApp" \
     --app-executable "my_app" \
     --app-params "--model bert-base --batch-size 32"
   ```

### GPU-Specific Configurations

#### **A100 Configuration (Toreador)**
```bash
./launch.sh \
  --gpu-type A100 \
  --profiling-tool dcgmi \
  --profiling-mode dvfs
```

#### **V100 Configuration (Matador)**
```bash
./launch.sh \
  --gpu-type V100 \
  --profiling-tool nvidia-smi \
  --profiling-mode baseline
```

### Profiling Tool Selection & Fallback

The framework supports intelligent profiling tool selection:

```bash
# Prefer DCGMI (will fallback to nvidia-smi if unavailable)
./launch.sh --profiling-tool dcgmi

# Force nvidia-smi usage
./launch.sh --profiling-tool nvidia-smi

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
    ./launch.sh \
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
./launch.sh --profiling-tool dcgmi  # Will auto-fallback to nvidia-smi if needed

# Reset GPU if needed
sudo nvidia-smi --gpu-reset
```

#### Python & Environment Issues
```bash
# Check Python version (3.6+ required)
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

**📚 For detailed troubleshooting, see [`documentation/QUICK_FIX_GUIDE.md`](documentation/QUICK_FIX_GUIDE.md)**

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

The framework includes comprehensive documentation:

- **[USAGE_EXAMPLES.md](documentation/USAGE_EXAMPLES.md)**: Complete CLI usage examples and automation scripts
- **[SUBMIT_JOBS_README.md](documentation/SUBMIT_JOBS_README.md)**: SLURM submission guide and HPC usage
- **[CLI_ENHANCEMENT_SUMMARY.md](documentation/CLI_ENHANCEMENT_SUMMARY.md)**: Technical implementation details
- **[REFACTORING_SUMMARY.md](documentation/REFACTORING_SUMMARY.md)**: Complete refactoring overview
- **[PYTHON36_COMPATIBILITY_FIX.md](documentation/PYTHON36_COMPATIBILITY_FIX.md)**: Python 3.6 compatibility guide
- **[QUICK_FIX_GUIDE.md](documentation/QUICK_FIX_GUIDE.md)**: Troubleshooting and quick fixes

## 🤝 Contributing

We welcome contributions to improve the framework:

### Development Setup
```bash
# Install development dependencies
pip install -r requirements.txt
pip install pytest black flake8

# Run code formatting
black . --line-length 88

# Run linting
flake8 --max-line-length 88

# Run tests
pytest tests/
```

### Adding New Features
1. Follow the existing code structure and documentation patterns
2. Add comprehensive error handling and logging
3. Update configuration files as needed
4. Include usage examples in docstrings
5. Add tests for new functionality

## 📝 Citation

If you use this framework in your research, please cite:

```bibtex
@misc{Side:2025:AIEngergy:GitHub,
  title={AI Inference Energy Profiling Framework},
  author={AI Inference Energy Research Team},
  year={2025},
  url={https://github.com/mertside/ai-inference-energy}
}
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- NVIDIA for DCGMI profiling tools
- Hugging Face for AI model hosting and libraries
- The open-source AI community for model development
- HPC centers providing computational resources

## 📞 Support

For questions and support:

- 📧 **Email**: research-team@example.com
- 💬 **Issues**: Use GitHub issues for bug reports and feature requests
- 📖 **Documentation**: Comprehensive docs available in the repository
- 🤝 **Community**: Join our research community discussions

---

**Happy profiling! ⚡🔬**
