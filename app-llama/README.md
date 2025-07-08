# LLaMA Text Generation Application

This directory contains the LLaMA (Large Language Model) text generation application for the AI Inference Energy Profiling Framework.

## Overview

The LLaMA application provides text generation capabilities using transformer-based large language models via Hugging Face Transformers. It's designed to be profiled for energy consumption analysis across different GPU frequency settings.

## Files

- **`LlamaViaHF.py`**: Main LLaMA text generation application with enterprise-grade features

## Features

- 🤖 **Model Support**: LLaMA-7B and other LLaMA model variants
- 🔧 **Configurable Parameters**: Adjustable prompt length, temperature, and generation settings
- 🚀 **GPU Optimization**: Automatic device detection with CUDA/CPU fallback
- 📊 **Profiling Ready**: Designed for energy profiling experiments
- 🛡️ **Error Handling**: Comprehensive error handling and logging
- 📝 **Logging**: Detailed execution logging for debugging and analysis

## Usage

### Standalone Usage
```bash
cd app-llama
python LlamaViaHF.py
```

### Profiling Usage
```bash
# From sample-collection-scripts directory
./launch.sh --app-name "LLaMA" --app-executable "llama_inference"
```

### Custom Configuration
```python
from app-llama.LlamaViaHF import LlamaTextGenerator

generator = LlamaTextGenerator(
    model_name="huggyllama/llama-7b",
    max_length=100,
    temperature=0.7
)

result = generator.generate_text("Your prompt here")
print(result)
```

## Configuration

The application uses settings from the main `config.py`:

```python
# Default LLaMA model
LLAMA_MODEL_NAME = "huggyllama/llama-7b"

# Generation parameters
DEFAULT_MAX_LENGTH = 50
DEFAULT_TEMPERATURE = 0.7
DEFAULT_NUM_RETURN_SEQUENCES = 1
```

## Requirements

- Python 3.6+
- PyTorch
- Transformers (Hugging Face)
- CUDA-capable GPU (recommended)
- Hugging Face account with model access

## Authentication

Ensure you're authenticated with Hugging Face:
```bash
huggingface-cli login
```

## Troubleshooting

### Common Issues

1. **Model Download Issues**
   ```bash
   # Re-authenticate
   huggingface-cli logout
   huggingface-cli login
   ```

2. **GPU Memory Issues**
   - Use smaller model variants
   - Reduce max_length parameter
   - Clear GPU cache: `torch.cuda.empty_cache()`

3. **Permission Issues**
   - Ensure proper Hugging Face authentication
   - Check model access permissions

## Output

The application generates:
- Text output to stdout/logs
- Performance metrics in profiling mode
- Error logs for debugging

## Integration

This application integrates with:
- **Main profiling framework** (`sample-collection-scripts/`)
- **Configuration system** (`config.py`)
- **Utility functions** (`utils.py`)
- **SLURM job submission** scripts

For detailed usage examples, see [`documentation/USAGE_EXAMPLES.md`](../documentation/USAGE_EXAMPLES.md).
