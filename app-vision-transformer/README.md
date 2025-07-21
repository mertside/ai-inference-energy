# Vision Transformer (ViT) Application

This directory contains the Vision Transformer implementation for AI inference energy profiling experiments.

## Overview

Vision Transformer (ViT) is a transformer-based architecture for image classification that treats images as sequences of patches. This implementation uses Hugging Face's transformers library for easy integration and comprehensive energy profiling capabilities.

## Features

- **Multiple ViT Models**: Support for various pre-trained ViT models from Hugging Face
- **Flexible Precision**: float32, float16, bfloat16 support for different hardware
- **Comprehensive Benchmarking**: Built-in performance and energy profiling
- **Interactive Demo**: Real-time image classification with URL/local file support
- **System Monitoring**: CPU, GPU, memory, and temperature tracking
- **Batch Processing**: Configurable batch sizes for throughput optimization

## Supported Models

- `google/vit-base-patch16-224` (default) - Base ViT model, 86M parameters
- `google/vit-large-patch16-224` - Large ViT model, 307M parameters
- `google/vit-huge-patch14-224-in21k` - Huge ViT model, 632M parameters
- `facebook/deit-base-distilled-patch16-224` - DeiT (Data-efficient image Transformers)
- Custom fine-tuned models compatible with ViT architecture

## Quick Start

### Environment Setup
```bash
# Uses existing whisper-energy conda environment
conda activate whisper-energy
cd app-vision-transformer/setup
./setup.sh
```

### Basic Usage
```bash
# Interactive demo
python ViTViaHF.py

# Benchmark with default settings
python ViTViaHF.py --benchmark

# Custom benchmark
python ViTViaHF.py --benchmark --model google/vit-large-patch16-224 --num-images 20 --precision float16
```

## Configuration Options

### Model Selection
- `--model`: Choose ViT model variant
- `--precision`: float32 (default), float16, bfloat16
- `--device`: cuda (auto-detected), cpu
- `--temperature`: Temperature scaling for predictions (default: 1.0)

### Benchmarking
- `--benchmark`: Run benchmark mode instead of interactive
- `--num-images`: Number of images to process (default: 10)
- `--batch-size`: Batch size for processing (default: 1)

## Energy Profiling Integration

This ViT application is designed to work with the energy profiling framework:

```bash
# Example: Profile ViT on different GPU frequencies
python sample-collection-scripts/collect_samples.py \
    --app-type vit \
    --model google/vit-base-patch16-224 \
    --num-images 50 \
    --precision float16 \
    --frequencies 1410 1695 1980 \
    --iterations 5
```

## Performance Characteristics

### Model Sizes (Approximate)
- ViT-Base: ~86M parameters, ~330MB memory
- ViT-Large: ~307M parameters, ~1.2GB memory  
- ViT-Huge: ~632M parameters, ~2.5GB memory

### Expected Throughput (A100)
- ViT-Base + float16: ~100-150 images/sec
- ViT-Large + float16: ~40-60 images/sec
- ViT-Huge + float16: ~15-25 images/sec

*Note: Actual performance varies by input size, batch size, and hardware configuration.*

## Sample Images

The application includes several sample images for testing:
- HuggingFace documentation images
- Wikipedia commons images
- Support for custom URLs and local files

## Output Format

Classification results include:
- Top-K predictions with confidence scores
- Processing time per image
- System resource utilization
- GPU memory usage and temperature

## Integration with Job Scripts

ViT experiments can be submitted through the job submission system:
- See `sample-collection-scripts/` for GPU-specific job templates
- Configuration templates available for H100, A100, V100
- Automatic frequency scaling and power measurement integration

## Troubleshooting

### Common Issues
- **CUDA out of memory**: Reduce batch size or use float16/bfloat16 precision
- **Slow performance**: Ensure CUDA is available and model is on GPU
- **Import errors**: Verify whisper-energy environment is activated

### Performance Optimization
- Use float16 precision on modern GPUs for 2x speedup
- Increase batch size for better throughput (within memory limits)
- Pre-download sample images to reduce network overhead

## Files Structure

```
app-vision-transformer/
├── __init__.py              # Package initialization
├── ViTViaHF.py             # Main ViT implementation
├── README.md               # This file
└── setup/
    ├── requirements.txt    # Python dependencies
    └── setup.sh           # Environment setup script
```

## Energy Profiling Use Cases

1. **Model Comparison**: Compare energy efficiency across ViT variants
2. **Precision Analysis**: Evaluate float32 vs float16 vs bfloat16 energy trade-offs
3. **Frequency Scaling**: Measure performance/power curves across GPU frequencies
4. **Batch Size Optimization**: Find optimal batch sizes for energy efficiency
5. **Hardware Comparison**: Profile across different GPU architectures

For detailed energy profiling workflows, see the main repository documentation.
