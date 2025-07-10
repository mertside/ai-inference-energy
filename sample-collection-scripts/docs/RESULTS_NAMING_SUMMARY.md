# Results Folder Naming Convention - Implementation Summary

## Overview
Successfully implemented automatic results folder naming based on GPU type and application name.

## Naming Convention
The new naming format is: `results_[gpu]_[app]`

### Examples:
- `results_h100_stablediffusion` - H100 GPU + Stable Diffusion
- `results_a100_lstm` - A100 GPU + LSTM
- `results_v100_llama` - V100 GPU + LLaMA
- `results_h100_customapp` - H100 GPU + Custom Application

## Implementation Details

### 1. Core Logic (args_parser.sh)
- Added auto-generation logic in `apply_intelligent_defaults()`
- Only applies when using default output directory
- Converts GPU/app names to lowercase, removes special characters
- Custom `--output-dir` still overrides auto-generation

### 2. Job Scripts Updated
- **submit_job_h100.sh**: Added `determine_results_dir()` function
- **submit_job_a100.sh**: Added `determine_results_dir()` function  
- **submit_job_v100.sh**: Added `determine_results_dir()` function
- All scripts now use `$RESULTS_DIR` variable instead of hardcoded "results"

### 3. Documentation Updates
- Help text updated to show auto-generation
- README.md updated with examples and naming convention
- Output section shows new directory structure

### 4. Benefits
- ✅ Clear organization by GPU and application
- ✅ No more overwriting results between different experiments
- ✅ Easy to identify experiment type from folder name
- ✅ Maintains backward compatibility with custom --output-dir
- ✅ Works across all GPU types (H100, A100, V100)

## Usage Examples

### Automatic Naming
```bash
# Creates: results_h100_stablediffusion/
./launch_v2.sh --gpu-type H100 --app-name StableDiffusion

# Creates: results_a100_lstm/
./launch_v2.sh --gpu-type A100 --app-name LSTM

# Creates: results_v100_llama/
./launch_v2.sh --gpu-type V100 --app-name LLaMA
```

### Custom Override
```bash
# Creates: my_custom_results/
./launch_v2.sh --gpu-type H100 --app-name StableDiffusion --output-dir my_custom_results
```

### Fallback
```bash
# Creates: results/ (when GPU/app not specified)
./launch_v2.sh --profiling-mode baseline
```

## Files Modified

1. **lib/args_parser.sh**
   - Added auto-generation logic in `apply_intelligent_defaults()`
   - Updated help text for `--output-dir` option
   - Updated output section with examples

2. **submit_job_h100.sh**
   - Added `determine_results_dir()` function
   - Added `RESULTS_DIR` variable in main function
   - Updated all hardcoded "results" references
   - Updated analysis command paths

3. **submit_job_a100.sh**
   - Added `determine_results_dir()` function
   - Added `RESULTS_DIR` variable in main function
   - Updated results directory references

4. **submit_job_v100.sh**
   - Added `determine_results_dir()` function
   - Added `RESULTS_DIR` variable in main function
   - Updated results directory references

5. **README.md**
   - Added "Automatic Directory Naming" section
   - Updated examples to show new naming convention
   - Updated output structure documentation

## Testing
- ✅ Verified naming logic works correctly
- ✅ Confirmed job scripts extract parameters correctly
- ✅ Tested custom output directory override
- ✅ Validated fallback to default "results"

## Status: COMPLETE ✅
The results folder naming convention has been successfully implemented and is ready for use!
