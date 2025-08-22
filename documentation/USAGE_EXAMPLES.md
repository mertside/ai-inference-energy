# Launch Script Usage Examples

The launch_v2.sh script now accepts all configuration and application details as command-line arguments, making it much more flexible and suitable for automated experiments.

## Basic Usage

### 1. Default Configuration (LSTM on A100 with DCGMI and DVFS)
```bash
./launch_v2.sh
```

### 2. Show Help
```bash
./launch_v2.sh --help
```

## GPU Configuration Examples

### 3. Switch to V100 GPU
```bash
./launch_v2.sh --gpu-type V100
```

### 4. Use nvidia-smi instead of DCGMI
```bash
./launch_v2.sh --profiling-tool nvidia-smi
```

### 5. Baseline profiling (single frequency, no frequency control)
```bash
./launch_v2.sh --profiling-mode baseline
```

### 6. V100 baseline with nvidia-smi
```bash
./launch_v2.sh --gpu-type V100 --profiling-tool nvidia-smi --profiling-mode baseline
```

## Application Configuration Examples

### 7. Custom Application - Stable Diffusion
```bash
./launch_v2.sh \
  --app-name "StableDiffusion" \
  --app-executable "stable_diffusion" \
  --app-params "--prompt 'A beautiful landscape' --steps 20 > results/SD_output.log"
```

### 8. Custom Application - LLaMA
```bash
./launch_v2.sh \
  --app-name "LLaMA" \
  --app-executable "llama_inference" \
  --app-params "--model llama-7b --input prompt.txt > results/LLaMA_output.log"
```

### 9. Custom Application - Whisper
```bash
./launch_v2.sh \
  --app-name "Whisper" \
  --app-executable "../app-whisper/WhisperViaHF.py" \
  --app-params "--benchmark --model base --num-samples 3 --quiet > results/Whisper_output.log"
```

### 10. Custom Application - Vision Transformer
```bash
./launch_v2.sh \
  --app-name "ViT" \
  --app-executable "../app-vision-transformer/ViTViaHF.py" \
  --app-params "--benchmark --num-images 5 --model google/vit-base-patch16-224 > results/ViT_output.log"
```

### 11. Multiple Applications Testing Loop
```bash
#!/bin/bash

for app in "LSTM" "StableDiffusion" "LLaMA" "Whisper" "ViT"; do

## Experiment Configuration Examples

### 12. Quick Test (fewer runs)
```bash
./launch_v2.sh --num-runs 1 --sleep-interval 0
```

### 13. Comprehensive Test (more runs)
```bash
./launch_v2.sh --num-runs 5 --sleep-interval 2
```

### 14. Full Custom Configuration
```bash
./launch_v2.sh \
  --gpu-type A100 \
  --profiling-tool dcgmi \
  --profiling-mode dvfs \
  --num-runs 3 \
  --sleep-interval 1 \
  --app-name "CustomApp" \
  --app-executable "custom_inference" \
  --app-params "--model bert-base --input data.txt --output-format json > results/custom_output.log"
```

## Automation Examples

### 15. Script for Testing Multiple Applications
```bash
#!/bin/bash

# Test different applications with the same configuration
for app in "LSTM" "StableDiffusion" "LLaMA" "Whisper"; do
    ./launch_v2.sh \
      --gpu-type A100 \
      --profiling-mode baseline \
      --num-runs 2 \
      --app-name "$app" \
      --app-executable "${app,,}" \
      --app-params "> results/${app}_baseline_output.log"
done
```

### 16. Script for Testing Multiple GPU Types
```bash
#!/bin/bash

# Test the same application on different GPU types
for gpu in "A100" "V100"; do
    ./launch_v2.sh \
      --gpu-type "$gpu" \
      --profiling-mode baseline \
      --app-name "LSTM" \
      --app-executable "lstm" \
      --app-params "> results/LSTM_${gpu}_output.log"
done
```

## Key Features

### Automatic Output Redirection
- If `--app-params` doesn't include output redirection (`>`), it's automatically added
- Default format: `> results/${APP_NAME}_RUN_OUT`
- Always ensures application output is captured

### Parameter Validation
- GPU type must be "A100" or "V100"
- Profiling tool must be "dcgmi" or "nvidia-smi"
- Profiling mode must be "dvfs" or "baseline"
- Number of runs must be a positive integer
- Sleep interval must be a non-negative integer

### Prerequisite Checking
- Verifies required profiling scripts exist and are executable
- Checks if profiling tools (dcgmi/nvidia-smi) are available
- Warns if detected GPU doesn't match specified GPU type
- Creates results directory if it doesn't exist

### Baseline Mode Benefits
- No frequency control scripts required
- No special permissions needed for frequency changes
- Faster execution (single frequency instead of full sweep)
- Useful for reference measurements and quick tests

## Output Files

Results are saved with the naming convention:
```
$ARCH-$MODE-$APP-$FREQ-$ITERATION
```

Examples:
- `GA100-dvfs-LSTM-1410-0` (A100 DVFS mode, LSTM app, 1410MHz, iteration 0)
- `GV100-baseline-StableDiffusion-1380-0` (V100 baseline mode, Stable Diffusion, 1380MHz, iteration 0)

Performance metrics are saved to:
- `$ARCH-$MODE-$APP-perf.csv` (CSV file with frequency and performance data)
