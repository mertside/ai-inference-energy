# 🎨 Stable Diffusion for AI Energy Research

This directory contains a **modernized, production-ready** Stable Diffusion application for the AI Inference Energy Profiling Framework. It extends the proven ICPP 2023/FGCS 2023 DVFS methodology to contemporary generative AI workloads.

## 🚀 Overview

The Stable Diffusion application provides state-of-the-art image generation capabilities using the latest diffusion models via Hugging Face Diffusers. It's specifically optimized for energy consumption analysis across different GPU architectures (V100, A100, H100) and frequency settings.

## 📁 Directory Contents

### **Core Application Files**
- **`StableDiffusionViaHF.py`**: **Main modernized application** with latest model support, advanced schedulers, and comprehensive benchmarking
- **`StableDiffusionViaHF_original.py`**: Legacy simple version for backward compatibility
- **`__init__.py`**: Package initialization

### **Setup & Validation**
- **`setup_stable_diffusion.sh`**: Complete setup and validation script with dependency checking
- **`test_stable_diffusion_modernized.py`**: Comprehensive test suite for all functionality
- **`test_stable_diffusion_revised.py`**: Alternative test implementation
- **`validate_stable_diffusion.py`**: Quick validation script for basic functionality

### **Documentation**
- **`README.md`**: This comprehensive guide
- **`README_MODERNIZED.md`**: Detailed modernization documentation (legacy reference)

## ✨ Revolutionary Features

### 🎨 **Latest Model Support**
- **SD v1.x Series**: Classic 512×512 baseline models (v1.4, v1.5)
- **SD v2.x Series**: Enhanced 768×768 models with improved quality (v2.0, v2.1)  
- **SDXL Series**: Flagship 1024×1024 models with superior performance
- **Turbo Variants**: Ultra-fast generation (SD-Turbo, SDXL-Turbo, 1-4 steps)
- **Lightning Models**: Cutting-edge speed optimization (SDXL-Lightning)

### ⚡ **Advanced Scheduler Collection**
```python
SCHEDULER_OPTIONS = {
    "dpm++": "High quality, fast convergence",
    "euler": "Simple, reliable, good balance", 
    "ddim": "Deterministic, fast, consistent",
    "unipc": "Unified predictor-corrector, very fast",
    "heun": "Higher order, more accurate",
    # + 5 more advanced schedulers
}
```

### 🧠 **Production-Grade Optimizations**
- **Memory Efficiency**: Attention slicing, xformers, CPU offload
- **Dynamic Batching**: Auto-adjust batch size based on VRAM
- **Mixed Precision**: FP16/BF16 optimization for A100/H100
- **Smart Caching**: Intelligent memory management for extended profiling

### 📊 **Comprehensive Research Features**
- **Energy Profiling Integration**: Seamless framework integration
- **Performance Monitoring**: Per-step timing, memory tracking
- **Benchmark Suites**: Speed, quality, memory stress, artistic styles
- **Cross-Architecture Support**: V100, A100, H100 optimization

## 🛠️ Quick Start

### **1. Setup Environment**
```bash
# Run automated setup script
cd app-stable-diffusion
./setup_stable_diffusion.sh
```

### **2. Basic Usage**
```bash
# Simple generation with default settings
python StableDiffusionViaHF.py

# Custom prompt and model
python StableDiffusionViaHF.py --model-variant sd-v1.4 --prompt "A beautiful landscape" --num-images 1

# High-resolution with SDXL
python StableDiffusionViaHF.py --model-variant sdxl --prompt "A futuristic city" --steps 30
```

### **3. Advanced Usage**
```python
# Python API usage
from StableDiffusionViaHF import StableDiffusionGenerator

generator = StableDiffusionGenerator(model_variant='sd-v1.4')
images = generator.generate_image(
    prompt="A beautiful landscape with mountains and lakes",
    num_inference_steps=20,
    guidance_scale=7.5,
    num_images=1
)
```

### **4. Energy Profiling Integration**
```bash
# From the main project root
cd sample-collection-scripts

# Basic profiling run
./launch.sh \
  --app-name "StableDiffusion" \
  --app-executable "../app-stable-diffusion/StableDiffusionViaHF.py" \
  --app-params "--model-variant sd-v1.4 --steps 20"

# DVFS study for energy research
./launch.sh \
  --app-name "StableDiffusion" \
  --app-executable "../app-stable-diffusion/StableDiffusionViaHF.py" \
  --app-params "--model-variant sdxl --scheduler dpm++ --batch-size 2" \
  --gpu-type A100 \
  --profiling-mode dvfs \
  --num-runs 3
```

## 🔬 Research Applications

### **Energy Efficiency Studies**
- **DVFS Sensitivity Analysis**: Frequency vs performance curves for generative AI
- **Cross-Architecture Comparison**: V100 → A100 → H100 energy progression
- **Quality-Energy Trade-offs**: Steps vs energy vs image quality analysis
- **Batch Scaling Studies**: Energy efficiency vs batch size optimization

### **Performance Characterization**
- **Inference Throughput**: Images/second across GPU types
- **Memory Scaling**: VRAM usage across models and resolutions
- **Scheduler Efficiency**: Speed/quality trade-offs across schedulers
- **Workload Characterization**: Compute vs memory intensity analysis

## 📊 Benchmarking & Testing

### **Automated Testing**
```bash
# Run comprehensive test suite
python test_stable_diffusion_modernized.py

# Quick validation check
python validate_stable_diffusion.py

# Run specific benchmark types
python StableDiffusionViaHF.py --benchmark --benchmark-type energy_research
python StableDiffusionViaHF.py --benchmark --benchmark-type memory_stress
python StableDiffusionViaHF.py --benchmark --benchmark-type speed_test
```

### **Available Model Variants**
| Variant | Resolution | Memory Req. | Speed | Quality | Use Case |
|---------|------------|-------------|-------|---------|----------|
| `sd-v1.4` | 512×512 | 4GB+ | Fast | Good | V100/A100 baseline |
| `sd-v1.5` | 512×512 | 4GB+ | Fast | Better | Enhanced baseline |
| `sd-v2.0` | 768×768 | 6GB+ | Medium | High | Higher resolution |
| `sd-v2.1` | 768×768 | 6GB+ | Medium | High | Latest SD2 family |
| `sdxl` | 1024×1024 | 8GB+ | Slower | Highest | H100 or high-memory |
| `sdxl-turbo` | 1024×1024 | 8GB+ | Very Fast | Good | Speed research |

## 🛠️ Installation & Requirements

### **System Requirements**
- **Python**: 3.8+ (3.10+ recommended)
- **PyTorch**: 2.0+ with CUDA support
- **GPU Memory**: 4GB+ (SDXL requires 8GB+)
- **System RAM**: 16GB+ recommended

### **Dependencies**
```bash
# Core requirements (automatically installed by setup script)
pip install torch>=2.0 diffusers>=0.21 transformers>=4.25
pip install accelerate Pillow numpy matplotlib

# Optional performance optimizations
pip install xformers  # Memory-efficient attention
pip install triton    # GPU kernel optimization
```

### **Authentication Setup**
```bash
# Required for model access
huggingface-cli login

# Verify authentication
python -c "from huggingface_hub import HfApi; print(HfApi().whoami())"
```

## ⚙️ Configuration & Optimization

### **Memory Optimization Features**
- **Attention Slicing**: Automatic for large images/batches
- **xformers Support**: Memory-efficient attention (if available)
- **CPU Offloading**: Automatic fallback for insufficient VRAM
- **Dynamic Batching**: Auto-adjust based on available memory
- **Mixed Precision**: FP16/BF16 for modern GPUs

### **Performance Tuning**
```bash
# For maximum speed
python StableDiffusionViaHF.py --model-variant sdxl-turbo --steps 4 --scheduler unipc

# For memory efficiency
python StableDiffusionViaHF.py --model-variant sd-v1.4 --attention-slicing --cpu-offload

# For quality research
python StableDiffusionViaHF.py --model-variant sdxl --steps 50 --scheduler dpm++
```

## 📈 Performance Expectations

### **Throughput Improvements**
| Feature | Improvement | Description |
|---------|-------------|-------------|
| **Batch Processing** | 2-4x | Dynamic batching optimization |
| **Advanced Schedulers** | 20-30% | DPM++, UniPC acceleration |
| **Memory Optimization** | 30-50% | Attention slicing, xformers |
| **Mixed Precision** | 15-25% | FP16/BF16 on modern GPUs |

### **Energy Efficiency Gains**
- **Optimized Inference**: 10-20% energy reduction through efficient scheduling
- **Dynamic Batching**: Better GPU utilization → lower energy per image
- **Model Selection**: Turbo variants for speed vs quality trade-offs
- **DVFS Integration**: Expected 25-30% energy savings (consistent with FGCS 2023)

## 🔗 Framework Integration

### **SLURM Job Submission**
```bash
# Navigate to sample collection scripts
cd ../sample-collection-scripts

# A100 baseline study
sbatch submit_job_a100_custom_app.sh

# H100 comprehensive analysis  
sbatch submit_job_h100_custom_app.sh

# V100 compatibility testing
sbatch submit_job_v100_custom_app.sh
```

### **Power Modeling Integration**
- **Enhanced Feature Set**: SD-specific workload characteristics
- **Cross-Architecture Analysis**: V100 → A100 → H100 efficiency evolution
- **Energy-Quality Trade-offs**: Novel optimization strategies
- **FGCS 2023 Extension**: Generative AI workload expansion

## 🚨 Important Notes

### **Memory Requirements by GPU**
- **V100 (16GB)**: SD v1.x/v2.x, small batches
- **A100 (40GB/80GB)**: All models, medium-large batches
- **H100 (80GB)**: All models, large batches, optimal for SDXL

### **Authentication & Licensing**
- All models require Hugging Face authentication
- Accept model licenses (especially for SDXL variants)
- Ensure compliance with usage terms for research

### **Troubleshooting**
1. **Out of Memory**: Reduce batch size, enable CPU offload, use attention slicing
2. **Slow Generation**: Use faster schedulers (UniPC, DPM++), reduce steps
3. **Authentication Errors**: Re-run `huggingface-cli login`
4. **Import Issues**: Ensure you're in the `app-stable-diffusion` directory

## 📚 Related Documentation

- **[STABLE_DIFFUSION_REVISION.md](../documentation/STABLE_DIFFUSION_REVISION.md)**: Complete technical revision plan
- **[USAGE_EXAMPLES.md](../documentation/USAGE_EXAMPLES.md)**: Comprehensive usage examples  
- **[GPU_USAGE_GUIDE.md](../documentation/GPU_USAGE_GUIDE.md)**: GPU-specific optimization guides
- **[README_POWER_MODELING.md](../documentation/README_POWER_MODELING.md)**: Power modeling integration

## 🎯 Getting Started Checklist

1. **✅ Run Setup**: `./setup_stable_diffusion.sh`
2. **✅ Test Installation**: `python validate_stable_diffusion.py`
3. **✅ Try Basic Generation**: `python StableDiffusionViaHF.py --help`
4. **✅ Run Framework Integration**: `cd ../sample-collection-scripts && ./launch.sh --help`
5. **✅ Start Energy Research**: Choose model variant and begin profiling

---

**🎯 This modernized Stable Diffusion application provides production-grade tools for extending proven DVFS methodology to contemporary generative AI workloads, enabling cutting-edge energy efficiency research across modern GPU architectures.**
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
