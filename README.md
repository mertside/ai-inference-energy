# AI Inference Energy Profiling Framework

This repository contains a comprehensive framework for studying **energy-efficient GPU frequency selection for AI inference workloads**. Building on prior work in analytical and machine learning-based DVFS (Dynamic Voltage and Frequency Scaling), this project investigates how modern AI inference tasks—such as large language models (LLMs), diffusion-based image generation, and retrieval-augmented generation (RAG)—respond to core frequency scaling on **NVIDIA A100 and H100 GPUs**.

## 🎯 Project Overview

As AI workloads grow in complexity and energy demand, static frequency settings on GPUs often result in sub-optimal trade-offs between performance and power consumption. This framework extends prior DVFS optimization approaches to emerging AI inference scenarios and evaluates their effectiveness using state-of-the-art open-source models.

### Supported AI Models

- **[LLaMA](https://github.com/meta-llama/llama)**: Text generation via transformer-based large language models
- **[Stable Diffusion](https://github.com/CompVis/stable-diffusion)**: Latent diffusion model for high-quality image generation
- **[ICEAGE](https://...)**: Retrieval-augmented inference pipeline for scientific data (future work)

### Research Goals

- 📊 Profile GPU power consumption, utilization, and performance across DVFS settings
- 🔄 Adapt analytical and ML-based frequency prediction models from HPC studies to AI workloads
- ⚡ Evaluate energy savings and throughput impact on modern inference tasks
- 📈 Provide reproducible benchmarks and analysis for A100 and H100 platforms
- 🔬 Enable research into energy-efficient AI deployment strategies

## 🏗️ Repository Structure

```
ai-inference-energy/
├── README.md                           # Project documentation
├── requirements.txt                    # Python dependencies
├── config.py                          # Configuration settings
├── utils.py                           # Utility functions
│
├── app-llama-collection/               # LLaMA inference applications
│   └── LlamaViaHF.py                  # LLaMA text generation via Hugging Face
│
├── app-stable-diffusion-collection/    # Stable Diffusion applications
│   └── StableDiffusionViaHF.py        # Image generation via Hugging Face
│
└── sample-collection-scripts/          # Profiling and experiment scripts
    ├── profile.py                     # GPU power/performance profiler
    ├── control.sh                     # GPU frequency control
    ├── launch.sh                      # Experiment orchestration
    ├── clean.sh                       # Workspace cleanup
    ├── submit_job.sh                  # SLURM job submission
    └── test.sh                        # MPI test template
```

## 🚀 Quick Start

### Prerequisites

#### Hardware Requirements
- NVIDIA GPU with DCGMI support (A100/H100 recommended)
- Sufficient GPU memory for AI models (8GB+ recommended)
- CUDA-compatible driver

#### Software Requirements
- Python 3.8+
- CUDA Toolkit 11.0+
- NVIDIA DCGMI tools
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

4. **Verify GPU and DCGMI setup**
   ```bash
   nvidia-smi                    # Check GPU status
   dcgmi discovery --list        # Verify DCGMI access
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

#### 3. Full Experiment Suite

**Clean workspace and run complete frequency sweep:**
```bash
cd sample-collection-scripts
./clean.sh -f                   # Clean previous results
./launch.sh                     # Run complete experiment
```

#### 4. HPC Cluster Deployment

**Submit to SLURM scheduler:**
```bash
sbatch submit_job.sh
```

## 🔧 Configuration

### GPU Frequency Settings

The framework supports comprehensive frequency scaling for NVIDIA A100 GPUs:

- **Memory Frequency**: 1215 MHz (A100 default)
- **Core Frequencies**: 61 different settings from 1410 MHz down to 510 MHz
- **Frequency Control**: Via DCGMI interface

### Experiment Parameters

Key configuration options in `config.py`:

```python
# Profiling settings
DEFAULT_NUM_RUNS = 2              # Runs per frequency
DEFAULT_INTERVAL_MS = 50          # Sampling interval
DCGMI_FIELDS = [1001, 1002, ...]  # GPU metrics to collect

# Model settings
LLAMA_MODEL_NAME = "huggyllama/llama-7b"
STABLE_DIFFUSION_MODEL_NAME = "CompVis/stable-diffusion-v1-4"

# GPU settings
A100_MEMORY_FREQ = 1215           # MHz
A100_DEFAULT_CORE_FREQ = 1410     # MHz
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

To add new AI applications:

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

2. **Update configuration** in `launch.sh`:
   ```bash
   declare -A APPLICATIONS=(
       ["LSTM"]="lstm"
       ["MyApp"]="my_app"  # Add your application
   )
   ```

### Custom Profiling

For specialized profiling needs:

```python
from sample-collection-scripts.profile import GPUProfiler

# Custom profiling setup
profiler = GPUProfiler(
    output_file="custom_profile.csv",
    interval_ms=25,  # Higher sampling rate
    gpu_id=0
)

# Profile custom command
result = profiler.profile_command("my_custom_inference_script.py")
print(f"Execution time: {result['duration']:.2f}s")
```

### Frequency Optimization

The framework supports custom frequency optimization strategies:

```bash
# Test specific frequency range
CORE_FREQUENCIES=(1410 1350 1290 1230 1170 1110)  # Edit in launch.sh

# Run subset of applications
APPLICATIONS=(["LLaMA"]="llama")  # Edit in launch.sh
```

## 🔍 Troubleshooting

### Common Issues

#### GPU Access Problems
```bash
# Check GPU visibility
nvidia-smi

# Verify DCGMI permissions
dcgmi discovery --list

# Reset GPU if needed
sudo nvidia-smi --gpu-reset
```

#### Model Download Issues
```bash
# Re-authenticate with Hugging Face
huggingface-cli logout
huggingface-cli login

# Check model access
python -c "from transformers import AutoTokenizer; AutoTokenizer.from_pretrained('huggyllama/llama-7b')"
```

#### Memory Issues
```bash
# Monitor GPU memory
watch -n 1 nvidia-smi

# Clear GPU memory in Python
import torch
torch.cuda.empty_cache()
```

### Performance Optimization

#### For Better Profiling Accuracy
- Ensure stable GPU temperature before experiments
- Run experiments during low system load
- Use dedicated GPU nodes when possible
- Increase sampling interval for longer workloads

#### For Faster Experiments
- Reduce number of frequencies tested
- Decrease number of runs per frequency
- Use smaller model variants for initial testing
- Parallelize experiments across multiple GPUs

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
