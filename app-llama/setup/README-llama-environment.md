# LLaMA Environment Setup

This directory contains the conda environment configuration for running LLaMA inference on H100 GPUs.

## Environment Details

- **Environment Name**: `llama`
- **Python Version**: 3.10.18
- **PyTorch**: 2.5.1+cu124 (CUDA 12.4 support)
- **Torchvision**: 0.20.1+cu124
- **Transformers**: 4.53.2
- **Target GPU**: NVIDIA H100 NVL (80GB HBM3)

## Key Features

✅ **Compatible PyTorch/Torchvision**: Resolved version conflicts for stable operation  
✅ **CUDA 12.4 Support**: Full GPU acceleration with latest CUDA toolkit  
✅ **Transformers Library**: Latest version for LLaMA model support  
✅ **H100 Optimized**: Tested and validated on H100 hardware  

## Quick Setup

### Option 1: Create from YML file
```bash
conda env create -f llama-environment.yml
conda activate llama
```

### Option 2: Create minimal environment
```bash
conda create -n llama python=3.10
conda activate llama
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
pip install transformers accelerate
```

## Validation

Test the environment:
```bash
conda activate llama
python -c "import torch; import torchvision; print(f'PyTorch: {torch.__version__}'); print(f'Torchvision: {torchvision.__version__}')"
```

Test LLaMA application:
```bash
cd /path/to/app-llama
python LlamaViaHF.py --prompt "Hello world" --max-tokens 10
```

## Performance Benchmarks

On NVIDIA H100 NVL:
- **Model Loading**: ~9-10 seconds for LLaMA-7B
- **Inference Speed**: ~60 tokens/second
- **Memory Usage**: ~12.5GB VRAM for LLaMA-7B
- **Benchmark Mode**: 24 generations across 8 prompts in ~20 seconds

## Usage in Job Scripts

This environment is automatically selected when using LLaMA applications in the H100 job submission scripts:

```bash
# In submit_job_h100.sh, LLaMA apps automatically use the 'llama' environment
LAUNCH_ARGS="--gpu-type H100 --app-name LLaMA --app-executable ../app-llama/LlamaViaHF.py --app-params '--benchmark --num-generations 3 --quiet --metrics'"
```

## Troubleshooting

### Common Issues

1. **PyTorch/Torchvision Version Mismatch**
   - Delete environment: `conda remove --name llama --all`
   - Recreate from YML: `conda env create -f llama-environment.yml`

2. **CUDA Version Conflicts**
   - Ensure system CUDA drivers are compatible with CUDA 12.4
   - Check: `nvidia-smi` and verify driver version

3. **Import Errors**
   - Verify environment activation: `conda activate llama`
   - Check package versions: `conda list torch transformers`

### Environment Recreation

If you need to recreate the environment:
```bash
# Remove existing
conda remove --name llama --all

# Recreate from working environment
conda create --name llama --clone llama-h100

# Or create from YML
conda env create -f llama-environment.yml
```

## Files

- `llama-environment.yml`: Complete conda environment export (504 packages)
- `LlamaViaHF.py`: Main LLaMA inference application
- `README-llama-environment.md`: This documentation

## Version History

- **v1.0** (July 2025): Initial H100-optimized environment
  - PyTorch 2.5.1 + CUDA 12.4
  - Transformers 4.53.2
  - Validated on H100 NVL 80GB
  - Energy profiling research
