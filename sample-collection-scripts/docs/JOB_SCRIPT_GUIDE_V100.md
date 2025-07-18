# V100 Unified Submission Script - Quick Reference

## Overview
The new `submit_job_v100.sh` script unifies all V100 SLURM submission functionality into a single, comprehensive script with 16 pre-configured options.

## 🎯 How to Use
1. Edit `submit_job_v100.sh`
2. Uncomment ONE configuration (lines 40-100)
3. Submit: `sbatch submit_job_v100.sh`

## 📋 Configuration Categories

### Quick Start (Configs 1-3)
- **#1 Quick Test**: Baseline profiling (~5-10 min)
- **#2 Research Baseline**: Extended baseline (~15-20 min) 
- **#3 Frequency Sampling**: Selected frequencies (~30-45 min)

### AI Applications (Configs 4-8)
- **#4 Whisper**: Speech recognition benchmark
- **#5 LSTM**: Sentiment analysis benchmark
- **#6 Stable Diffusion**: Image generation profiling
- **#7 LLaMA**: Text generation profiling
- **#8 Custom App**: Template for your applications

### DVFS Studies (Configs 9-11)
- **#9 Comprehensive**: All 117 frequencies (6-12 hours)
- **#10 Efficient**: Reduced runs (4-6 hours)
- **#11 Statistical**: High statistical power (12-20 hours)

### Tools & Compatibility (Configs 12-14)
- **#12 nvidia-smi**: When DCGMI unavailable
- **#13 Debug**: Minimal config for troubleshooting
- **#14 Memory Test**: Large model testing

### Research Studies (Configs 15-17)
- **#15 Energy Efficiency**: Power vs performance focus
- **#16 Precision Comparison**: Different model precisions
- **#17 Scaling Analysis**: Batch size impact study

## ⏱️ Timing Guidelines
- Configs 1-3: `--time=01:00:00`
- Configs 4-8: `--time=02:00:00`
- Configs 9-10: `--time=08:00:00`
- Config 11: `--time=24:00:00`
- Configs 12-17: `--time=03:00:00`

## 🔄 Legacy Script Migration
- `submit_job_v100_baseline.sh` → Use config #1 or #2
- `submit_job_v100_comprehensive.sh` → Use config #9, #10, or #11
- `submit_job_v100_custom_app.sh` → Use configs #4-17

## 💡 Pro Tips
- V100 has 117 frequencies vs A100's 61 - DVFS takes longer
- Use frequency sampling (#3) instead of full DVFS for faster results
- Monitor disk space - comprehensive studies generate substantial data
- Adjust SLURM `--time` parameter based on configuration chosen

## 🚀 Features
- ✅ Color-coded logging and progress indicators
- ✅ Comprehensive system validation and GPU checks
- ✅ Intelligent warnings for long-running configurations
- ✅ Detailed results summary and next steps
- ✅ Automatic error handling and troubleshooting suggestions
- ✅ V100-specific optimizations and recommendations
