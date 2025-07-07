# Stable Diffusion Image Generation Application

This directory contains the Stable Diffusion image generation application for the AI Inference Energy Profiling Framework.

## Overview

The Stable Diffusion application provides high-quality image generation capabilities using latent diffusion models via Hugging Face Diffusers. It's optimized for energy consumption analysis across different GPU frequency settings.

## Files

- **`StableDiffusionViaHF.py`**: Main Stable Diffusion image generation application with advanced features

## Features

- 🎨 **Image Generation**: High-quality image synthesis from text prompts
- ⚡ **Memory Optimization**: Attention slicing and xformers support for reduced memory usage
- 🔧 **Configurable Parameters**: Adjustable steps, guidance scale, and image dimensions
- 🚀 **GPU Optimization**: Automatic device detection with CUDA/CPU fallback
- 📊 **Profiling Ready**: Designed for energy profiling experiments
- 💾 **Automatic Saving**: Smart file naming and automatic image saving
- 🛡️ **Error Handling**: Comprehensive error handling and logging

## Usage

### Standalone Usage
```bash
cd app-stable-diffusion-collection
python StableDiffusionViaHF.py
```

### Custom Prompt Usage
```python
from app-stable-diffusion-collection.StableDiffusionViaHF import StableDiffusionGenerator

generator = StableDiffusionGenerator()
images = generator.generate_image(
    prompt="A beautiful landscape with mountains and lakes",
    num_inference_steps=20,
    guidance_scale=7.5
)
```

### Profiling Usage
```bash
# From sample-collection-scripts directory
./launch.sh \
  --app-name "StableDiffusion" \
  --app-executable "stable_diffusion" \
  --app-params "--prompt 'A beautiful landscape' --steps 20"
```

## Configuration

The application uses settings from the main `config.py`:

```python
# Default Stable Diffusion model
STABLE_DIFFUSION_MODEL_NAME = "CompVis/stable-diffusion-v1-4"

# Generation parameters
DEFAULT_NUM_INFERENCE_STEPS = 20
DEFAULT_GUIDANCE_SCALE = 7.5
DEFAULT_HEIGHT = 512
DEFAULT_WIDTH = 512
```

## Memory Optimization

The application includes several memory optimization features:

- **Attention Slicing**: Reduces memory usage for large images
- **xformers**: Optional memory-efficient attention (if available)
- **CPU Offloading**: Automatic fallback to CPU if GPU memory is insufficient
- **Garbage Collection**: Automatic cleanup after generation

## Requirements

- Python 3.6+
- PyTorch
- Diffusers (Hugging Face)
- Transformers (Hugging Face)
- CUDA-capable GPU (recommended, 8GB+ VRAM)
- Hugging Face account with model access

## Authentication

Ensure you're authenticated with Hugging Face:
```bash
huggingface-cli login
```

## Performance Tips

### For Better Performance
- Use CUDA-enabled GPU with sufficient memory (8GB+)
- Reduce `num_inference_steps` for faster generation
- Use smaller image dimensions (256x256 or 512x512)

### For Lower Memory Usage
- Enable attention slicing (automatic)
- Use CPU fallback if needed
- Reduce batch size for multiple images

## Troubleshooting

### Common Issues

1. **GPU Memory Issues**
   ```python
   # Enable memory optimizations
   generator = StableDiffusionGenerator(enable_attention_slicing=True)
   ```

2. **Model Download Issues**
   ```bash
   # Re-authenticate
   huggingface-cli logout
   huggingface-cli login
   ```

3. **CUDA Issues**
   - Check CUDA installation: `nvidia-smi`
   - Verify PyTorch CUDA support: `python -c "import torch; print(torch.cuda.is_available())"`

## Output

The application generates:
- High-quality PNG images saved to disk
- Automatic file naming with timestamps
- Performance metrics in profiling mode
- Detailed execution logs

## Output Files

Images are saved with descriptive names:
```
sd_output_20250702_143022.png
sd_output_20250702_143022_1.png  # Multiple images
sd_output_20250702_143022_2.png
```

## Integration

This application integrates with:
- **Main profiling framework** (`sample-collection-scripts/`)
- **Configuration system** (`config.py`)
- **Utility functions** (`utils.py`)
- **SLURM job submission** scripts

For detailed usage examples, see [`documentation/USAGE_EXAMPLES.md`](../documentation/USAGE_EXAMPLES.md).
