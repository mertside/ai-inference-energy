# Running Stable Diffusion on HPCC

## Quick Start Guide

Based on your current setup, here's how to run Stable Diffusion on HPCC interactive sessions:

### 1. Request an Interactive Session

Choose the appropriate GPU type:

```bash
# For A100 GPUs (recommended for Stable Diffusion)
cd /home/meside/ai-inference-energy/sample-collection-scripts
./interactive_a100.sh

# For V100 GPUs 
./interactive_v100.sh

# For H100 GPUs (if available)
./interactive_h100.sh
```

### 2. Environment Setup Issues

**Current Status:** Your environment has package compatibility issues that need to be resolved. The main issues are:

- Torch/TorchVision version conflicts
- Transformers/Diffusers dependency conflicts
- GPU driver compatibility (on login nodes)

### 3. Temporary Solution: Use Pre-built Containers

Instead of fighting dependency conflicts, consider using pre-built containers:

```bash
# Option 1: Use Singularity container (if available on HPCC)
singularity exec --nv /path/to/pytorch-container.sif python your_script.py

# Option 2: Use conda environment with conda-forge
conda create -n stable-diffusion python=3.9
conda activate stable-diffusion
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
conda install -c conda-forge diffusers transformers accelerate
```

### 4. Model Variants by GPU Type

Based on your hardware configurations:

#### V100 (16GB VRAM)
- **Recommended:** `sd-v1.4`, `sd-v1.5`
- **Resolution:** 512×512
- **Batch size:** 1-2

#### A100 (40GB/80GB VRAM)  
- **Recommended:** `sd-v2.1`, `sdxl-base`
- **Resolution:** 768×768 or 1024×1024
- **Batch size:** 2-4

#### H100 (80GB VRAM)
- **Recommended:** `sdxl-turbo`, `sdxl-refiner`
- **Resolution:** 1024×1024
- **Batch size:** 4-8

### 5. Example Commands

Once your environment is working:

```bash
# Basic image generation
python app-stable-diffusion/StableDiffusionViaHF.py \
  --model-variant sd-v1.4 \
  --prompt "a beautiful landscape with mountains" \
  --num-images 1 \
  --height 512 \
  --width 512

# Energy profiling with framework
cd sample-collection-scripts
./launch.sh --app-name StableDiffusion \
  --model-variant sd-v1.4 \
  --prompt "a serene lake at sunset" \
  --gpu-freq 1200
```

### 6. Framework Integration

Your project structure shows integration with energy profiling:

```bash
# Test framework integration
./sample-collection-scripts/launch.sh --help

# Run with energy monitoring
./sample-collection-scripts/profile.py --app StableDiffusion
```

### 7. Next Steps for Dependency Resolution

1. **Clean Installation:**
   ```bash
   # Remove current conflicting packages
   pip uninstall torch torchvision transformers diffusers -y
   
   # Install from conda-forge (more stable)
   conda install pytorch=2.0 torchvision=0.15 -c pytorch
   conda install transformers=4.30 diffusers=0.20 -c conda-forge
   ```

2. **Alternative: Use pip-tools for dependency resolution:**
   ```bash
   # Create requirements.in file with compatible versions
   pip install pip-tools
   pip-compile requirements.in
   pip install -r requirements.txt
   ```

### 8. Energy Study Configuration

For your energy optimization research:

```bash
# Test different frequencies
for freq in 800 1000 1200 1400; do
  ./launch.sh --app StableDiffusion --gpu-freq $freq --prompt "test image"
done

# Collect comprehensive metrics
./submit_job_a100_comprehensive.sh StableDiffusion
```

## Troubleshooting

### GPU Driver Issues
- The "CUDA driver too old" warning is expected on login nodes
- GPU functionality will work correctly on compute nodes during interactive sessions

### Memory Issues  
- Start with smaller models (sd-v1.4) and increase resolution gradually
- Use `--cpu-offload` flag if GPU memory is insufficient

### Package Conflicts
- Use conda instead of pip for core ML packages
- Create separate environments for different model types
