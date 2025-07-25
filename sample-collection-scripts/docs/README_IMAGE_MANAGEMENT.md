# 🎨 Stable Diffusion Image Management for DVFS Sweeps

## 📊 **The Challenge**

When running full DVFS frequency sweeps for Stable Diffusion:
- **H100**: 86 frequencies × runs = **86+ images**
- **A100**: 61 frequencies × runs = **61+ images** 
- **V100**: 117 frequencies × runs = **117+ images**

Each image is ~1-5 MB, so a complete DVFS sweep can generate **500MB - 2GB** of images!

## 🔧 **Recommended Solutions**

### 1. **Research Mode (Default) - Save Representative Images**
For energy research, you typically only need:
- **Quality validation**: A few sample images to verify generation quality
- **Performance data**: Timing and energy metrics (CSV files)

### 2. **Job-Organized Mode - Automatic Organization by SLURM Job ID**
Each SLURM job automatically creates its own image subfolder:
- **Structure**: `images/job_<SLURM_JOB_ID>/`
- **Benefits**: Easy to identify which images belong to which experiment
- **Cleanup**: Target specific jobs for cleanup

### 3. **Full Archive Mode - Save All Images**  
For comprehensive analysis or demonstrations

### 4. **Benchmark Mode - No Image Saving**
For pure performance testing without image storage

## 🚀 **Implementation Options**

### Option A: Modify Stable Diffusion Arguments (Recommended)

Add `--no-save-images` flag to DVFS configurations in submit scripts:

```bash
# Energy research - save only metrics, not images
LAUNCH_ARGS="--profiling-mode dvfs --app-name StableDiffusion --app-params '--prompt \"a photograph of an astronaut riding a horse\" --steps 50 --no-save-images --log-level INFO'"

# Quality validation - save 1 sample per 10 frequencies
LAUNCH_ARGS="--profiling-mode custom --custom-frequencies '510,750,1000,1250,1500,1785' --app-name StableDiffusion --app-params '--prompt \"a photograph of an astronaut riding a horse\" --steps 50'"
```

### Option B: Automated Image Management

Use cleanup scripts to manage image storage automatically.

## 📋 **Best Practices**

### For Energy Research (Recommended)
1. **Use research mode**: Generate images only for frequency validation
2. **Focus on metrics**: CSV files contain the energy and performance data you need
3. **Spot checks**: Generate a few sample images to verify quality

### For Complete Studies
1. **Archive important runs**: Save images from key configurations
2. **Clean regularly**: Use automated cleanup between experiments
3. **Compress archives**: Use tar.gz for long-term storage

### For HPC Clusters (REPACSS)
1. **Mind disk quotas**: Check available space before DVFS sweeps
2. **Clean between jobs**: Don't let images accumulate across multiple jobs
3. **Use scratch space**: Store temporary images in /tmp or scratch directories

## 🛠️ **Tools Provided**

### 1. `cleanup_sd_images.sh` - Smart Image Cleanup
- **Job-specific cleanup**: Target images from specific SLURM jobs
- **Archive and compress**: Safe backup before cleaning
- **Size-based cleanup**: Automatic cleanup when storage limits exceeded
- **Age-based cleanup**: Remove old images automatically

### 2. **Automatic Job Organization**
- Images automatically organized by SLURM job ID
- Structure: `app-stable-diffusion/images/job_<JOB_ID>/`
- Easy identification and targeted cleanup

### 3. `sd_research_guide.sh` - Research-focused SD configuration  
### 4. Documentation and examples

## 📊 **Storage Estimates**

| GPU Type | Frequencies | Images/Run | Storage/Run | Full DVFS |
|----------|-------------|------------|-------------|-----------|
| H100     | 86          | 86         | ~400MB      | ~1.2GB    |
| A100     | 61          | 61         | ~300MB      | ~900MB    |
| V100     | 117         | 117        | ~500MB      | ~1.5GB    |

*Estimates assume 50 diffusion steps, 512x512 resolution, ~4MB per image*

## 🎯 **Quick Start**

### Research Mode (Recommended)
```bash
# Add --no-save-images to your SD DVFS configurations
# This saves energy data but skips image files
```

### Validation Mode  
```bash
# Use custom frequencies for spot checks
--custom-frequencies '510,960,1410,1785'  # Sample key frequencies
```

### Archive Mode
```bash
# Run cleanup_sd_images.sh after each major experiment
./cleanup_sd_images.sh --archive --compress
```

---
**💡 Pro Tip**: For energy research, you care about the **performance and energy metrics**, not the generated images. The CSV files contain all the data you need for DVFS analysis!
