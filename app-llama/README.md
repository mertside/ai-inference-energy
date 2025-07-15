# LLaMA Text Generation for Energy Profiling

This directory contains a production-ready LLaMA text generation application designed for comprehensive energy profiling studies on GPU inference workloads.

## Features

- **Multiple Model Support**: Supports various LLaMA variants (7B to 70B parameters)
- **Comprehensive CLI**: Full command-line interface with extensive configuration options
- **Energy Profiling Ready**: Designed for integration with DCGMI and nvidia-smi profiling
- **Cluster Optimized**: Separate conda environments for H100, A100, V100, and generic setups
- **Performance Metrics**: Built-in token/second measurements and inference timing
- **Flexible Generation**: Configurable prompts, generation parameters, and batch processing

## Quick Start

### 1. Environment Setup

Choose the appropriate environment for your hardware:

```bash
cd setup/

# Auto-detect GPU and setup environment
./setup_llama.sh

# Or specify GPU type manually
./setup_llama.sh h100    # For H100 GPUs
./setup_llama.sh a100    # For A100 GPUs  
./setup_llama.sh v100    # For V100 GPUs
./setup_llama.sh generic # For development/CPU
```

### 2. Hugging Face Authentication

```bash
huggingface-cli login
```

You'll need a Hugging Face token with access to LLaMA models.

### 3. Basic Usage

```bash
# Activate environment (adjust name based on your setup)
conda activate llama-h100

# Basic text generation
python LlamaViaHF.py --prompt "The future of AI is" --max-tokens 100

# Use specific model variant
python LlamaViaHF.py --model llama2-13b --prompt "Climate change" --max-tokens 50

# Run benchmark with multiple prompts
python LlamaViaHF.py --benchmark --num-generations 3 --quiet --metrics

# Energy profiling mode (minimal output)
python LlamaViaHF.py --benchmark --num-generations 5 --log-level WARNING --quiet
```

## Command Line Interface

### Model Configuration

```bash
--model MODEL_NAME          # Model name or shorthand (default: llama-7b)
--device {cuda,cpu,auto}     # Target device (default: auto)
--precision {float16,float32,int8}  # Model precision (default: float16)
--max-memory-mb MB           # Maximum GPU memory to use in MB
```

### Generation Parameters

```bash
--prompt PROMPT              # Input text prompt
--max-tokens NUM             # Maximum new tokens to generate (default: 50)
--temperature FLOAT          # Sampling temperature (default: 0.7)
--top-p FLOAT               # Nucleus sampling parameter (default: 0.9)
--top-k INT                 # Top-k sampling parameter (default: 50)
--repetition-penalty FLOAT  # Repetition penalty (default: 1.1)
--no-sampling               # Use greedy decoding instead of sampling
```

### Benchmark Options

```bash
--benchmark                 # Run benchmark with default prompts
--num-generations NUM       # Number of generations per prompt (default: 1)
```

### Output Options

```bash
--output-file FILE          # Save results to JSON file
--log-level LEVEL           # Logging level (DEBUG, INFO, WARNING, ERROR)
--quiet                     # Suppress generated text output
--metrics                   # Display performance metrics
```

## Supported Models

### Model Shortcuts

| Shortcut | Full Model Name | Parameters | VRAM Required |
|----------|----------------|------------|---------------|
| `llama-7b` | `huggyllama/llama-7b` | 7B | 14GB+ |
| `llama-13b` | `huggyllama/llama-13b` | 13B | 26GB+ |
| `llama-30b` | `huggyllama/llama-30b` | 30B | 60GB+ |
| `llama-65b` | `huggyllama/llama-65b` | 65B | 120GB+ |
| `llama2-7b` | `meta-llama/Llama-2-7b-hf` | 7B | 14GB+ |
| `llama2-13b` | `meta-llama/Llama-2-13b-hf` | 13B | 26GB+ |
| `llama2-70b` | `meta-llama/Llama-2-70b-hf` | 70B | 140GB+ |

### Custom Models

You can also specify any Hugging Face model directly:

```bash
python LlamaViaHF.py --model "microsoft/DialoGPT-large" --prompt "Hello"
```

## Usage Examples

### Research and Benchmarking

```bash
# Standard benchmark for energy profiling
python LlamaViaHF.py --benchmark --num-generations 10 --max-tokens 100 --quiet --metrics

# Specific model evaluation
python LlamaViaHF.py --model llama2-13b --benchmark --num-generations 5 --max-tokens 150

# Single prompt with multiple runs
python LlamaViaHF.py --prompt "Artificial intelligence" --num-generations 10 --max-tokens 75
```

### Development and Testing

```bash
# Quick test with minimal resource usage
python LlamaViaHF.py --model llama-7b --prompt "Hello world" --max-tokens 10 --metrics

# Debug mode with detailed logging
python LlamaViaHF.py --log-level DEBUG --prompt "Test prompt" --max-tokens 20

# Save detailed results to file
python LlamaViaHF.py --benchmark --output-file results.json --metrics
```

### Production Profiling

```bash
# Minimal output for clean profiling logs
python LlamaViaHF.py --benchmark --num-generations 20 --quiet --log-level WARNING

# Memory-constrained environment
python LlamaViaHF.py --model llama-7b --max-memory-mb 12000 --benchmark --quiet

# High-precision generation
python LlamaViaHF.py --precision float32 --benchmark --num-generations 5
```

## Integration with Energy Profiling Framework

### SLURM Job Scripts

The LLaMA application is integrated with the energy profiling framework job scripts:

```bash
# From sample-collection-scripts directory
LAUNCH_ARGS="--gpu-type H100 --profiling-mode baseline --app-name LLaMA --app-executable ../app-llama/LlamaViaHF.py --num-runs 5"
bash launch_v2.sh $LAUNCH_ARGS

# With specific parameters
LAUNCH_ARGS="--gpu-type A100 --profiling-mode baseline --app-name LLaMA --app-executable ../app-llama/LlamaViaHF.py --app-params '--model llama2-13b --max-tokens 100' --num-runs 3"
bash launch_v2.sh $LAUNCH_ARGS
```

### Results Directory

Results are automatically saved to `results_[GPU]_LLaMA/` directory following the framework convention.

## Performance Optimization

### Memory Management

- Use `--max-memory-mb` to limit GPU memory usage
- Use `float16` precision for memory efficiency
- Consider smaller models for memory-constrained environments

### Generation Speed

- Use `--no-sampling` for faster deterministic generation
- Reduce `--max-tokens` for shorter generation times
- Use batch processing with `--benchmark` for efficiency

### Profiling Optimization

- Use `--quiet` to reduce output noise during profiling
- Set `--log-level WARNING` to minimize logging overhead
- Use `--metrics` flag to collect performance statistics

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   ```bash
   # Solution: Use smaller model or limit memory
   python LlamaViaHF.py --model llama-7b --max-memory-mb 12000
   ```

2. **Model Access Denied**
   ```bash
   # Solution: Check Hugging Face authentication
   huggingface-cli login
   huggingface-cli whoami
   ```

3. **Slow Model Loading**
   ```bash
   # Solution: Use local model cache or faster storage
   export HF_HOME="/fast/storage/path"
   ```

### Debug Mode

Enable detailed debugging:

```bash
python LlamaViaHF.py --log-level DEBUG --model llama-7b --prompt "Debug test" --max-tokens 5
```

## File Structure

```
app-llama/
├── LlamaViaHF.py              # Main application script
├── LlamaViaHF_backup.py       # Backup of original version
├── LlamaViaHF_original.py     # Original simple version
├── README.md                  # This documentation
├── __init__.py               # Python package initialization
└── setup/                    # Environment setup
    ├── README.md            # Setup documentation
    ├── setup_llama.sh       # Automated setup script
    ├── conda_env_h100.yml   # H100 environment
    ├── conda_env_a100.yml   # A100 environment
    ├── conda_env_v100.yml   # V100 environment
    └── conda_env_generic.yml # Generic environment
```

## Version Information

- **Version**: 2.0.1
- **Compatible with**: Energy Profiling Framework v2.0.1
- **Python Requirements**: 3.9+ (3.10+ recommended)
- **CUDA Requirements**: 11.7+ (12.1+ for H100)
- **Transformers Version**: 4.25.0+ (4.35.0+ recommended)

## Support

For issues related to:
- **Model access**: Check Hugging Face documentation and model permissions
- **CUDA/GPU issues**: Verify driver compatibility and memory requirements  
- **Framework integration**: See main repository documentation
- **Environment setup**: Review setup script logs and conda environment files
