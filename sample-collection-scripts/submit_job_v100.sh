#!/bin/bash
#
# Unified SLURM Job Submission Script - V100 GPU Profiling
#
# This script provides a comprehensive set of V100 profiling configurations.
# Simply uncomment the desired configuration and submit with: sbatch submit_job_v100.sh
#
# NEW FEATURES (v2.1):
#   - Customizable DCGMI sampling intervals (10-1000ms)
#   - Multi-GPU monitoring support (all available V100 GPUs)
#   - Enhanced configuration examples demonstrating new capabilities
#
# V100 Specifications:
#   - GPU: Tesla V100 (32GB HBM2)
#   - Partition: matador (HPCC Texas Tech)
#   - Frequencies: 117 available (405-1380 MHz)
#   - Memory: 877 MHz (fixed)
#   - Architecture: Volta (GV100)
#

#SBATCH --job-name=PROFILING_V100
#SBATCH --output=%x.%j.out
#SBATCH --error=%x.%j.err
#SBATCH --partition=matador
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=40
#SBATCH --gpus-per-node=1
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=mert.side@ttu.edu

# Enable strict error handling (conda-friendly)
set -eo pipefail  # Removed -u to avoid issues with conda environment scripts

# Configuration
readonly LAUNCH_SCRIPT_LEGACY="./launch.sh"
readonly LAUNCH_SCRIPT_V2="./launch_v2.sh"

# Use new framework by default, fallback to legacy if needed
readonly LAUNCH_SCRIPT="${LAUNCH_SCRIPT_V2}"

# Function to determine conda environment based on application
determine_conda_env() {
    local app_name=""
    
    # Extract app name from LAUNCH_ARGS
    if echo "$LAUNCH_ARGS" | grep -q "app-name"; then
        app_name=$(echo "$LAUNCH_ARGS" | sed -n 's/.*--app-name \([^ ]*\).*/\1/p')
    fi
    
    # Map application names to conda environments
    case "$app_name" in
        "StableDiffusion")
            echo "stable-diffusion-gpu"
            ;;
        "LSTM")
            echo "tensorflow"
            ;;
        "LLaMA")
            echo "llama"  # Updated to use the new llama environment
            ;;
        "Whisper")
            echo "whisper"  # Whisper speech recognition environment
            ;;
        "ViT")
            echo "vit"  # Vision Transformer environment
            ;;
        *)
            echo "tensorflow"  # Default environment
            ;;
    esac
}

# Function to determine expected results directory from launch arguments
determine_results_dir() {
    local gpu_type=""
    local app_name=""
    local custom_output=""
    
    # Extract relevant parameters from LAUNCH_ARGS
    if echo "$LAUNCH_ARGS" | grep -q "gpu-type"; then
        gpu_type=$(echo "$LAUNCH_ARGS" | sed -n 's/.*--gpu-type \([^ ]*\).*/\1/p')
    fi
    
    if echo "$LAUNCH_ARGS" | grep -q "app-name"; then
        app_name=$(echo "$LAUNCH_ARGS" | sed -n 's/.*--app-name \([^ ]*\).*/\1/p')
    fi
    
    if echo "$LAUNCH_ARGS" | grep -q "output-dir"; then
        custom_output=$(echo "$LAUNCH_ARGS" | sed -n 's/.*--output-dir \([^ ]*\).*/\1/p')
        # Append job ID to custom output directory if available
        if [[ -n "$SLURM_JOB_ID" ]]; then
            echo "${custom_output}_job_${SLURM_JOB_ID}"
        else
            echo "$custom_output"
        fi
        return
    fi
    
    # Generate auto-generated directory name (same logic as args_parser.sh)
    if [[ -n "$gpu_type" && -n "$app_name" ]]; then
        local gpu_name=$(echo "$gpu_type" | tr '[:upper:]' '[:lower:]')
        local app_name_clean=$(echo "$app_name" | tr '[:upper:]' '[:lower:]' | sed 's/[^a-z0-9]//g')
        local base_name="results_${gpu_name}_${app_name_clean}"
        # Append job ID if available
        if [[ -n "$SLURM_JOB_ID" ]]; then
            echo "${base_name}_job_${SLURM_JOB_ID}"
        else
            echo "$base_name"
        fi
    else
        # Append job ID to default results directory if available
        if [[ -n "$SLURM_JOB_ID" ]]; then
            echo "results_job_${SLURM_JOB_ID}"
        else
            echo "results"
        fi
    fi
}

# ============================================================================
# CONFIGURATION SECTION - Uncomment ONE configuration below
# ============================================================================

# 📋 BASELINE CONFIGURATIONS
# ============================================================================

# 1. 🤖 LSTM - Basic deep learning benchmark
# LAUNCH_ARGS="--gpu-type V100 --profiling-mode baseline --num-runs 3 --sleep-interval 1 --app-name LSTM --app-executable ../app-lstm/lstm"

# 2. 🎨 STABLE DIFFUSION - Image generation benchmark
# LAUNCH_ARGS="--gpu-type V100 --profiling-mode baseline --app-name StableDiffusion --app-executable ../app-stable-diffusion/StableDiffusionViaHF.py --app-params '--prompt \"a photograph of an astronaut riding a horse\" --steps 500 --log-level INFO' --num-runs 3 --sleep-interval 1"

# 3. 📝 LLAMA - Text generation benchmark
# LAUNCH_ARGS="--gpu-type V100 --profiling-mode baseline --app-name LLaMA --app-executable ../app-llama/LlamaViaHF.py --app-params '--benchmark --num-generations 3 --quiet --metrics' --num-runs 3"

# 4. 🎤 WHISPER - Speech recognition benchmark
# LAUNCH_ARGS="--gpu-type V100 --profiling-mode baseline --app-name Whisper --app-executable ../app-whisper/WhisperViaHF.py --app-params '--benchmark --model base --num-samples 3 --quiet' --num-runs 3"

# 5. 🖼️ VISION TRANSFORMER - Image classification benchmark
# LAUNCH_ARGS="--gpu-type V100 --profiling-mode baseline --app-name ViT --app-executable ../app-vision-transformer/ViTViaHF.py --app-params '--benchmark --num-images 2000 --batch-size 4 --model google/vit-large-patch16-224 --precision float16' --num-runs 3"

# 📊 CUSTOM FREQUENCY CONFIGURATIONS (Low/Mid/High Analysis)
# ============================================================================

# 6. 🤖 LSTM CUSTOM - Three-point frequency analysis (low/mid/high)
# LAUNCH_ARGS="--gpu-type V100 --profiling-mode custom --custom-frequencies '510,960,1380' --app-name LSTM --app-executable ../app-lstm/lstm --num-runs 5 --sleep-interval 2"

# 7. 🎨 STABLE DIFFUSION CUSTOM - Three-point frequency analysis (low/mid/high)
# LAUNCH_ARGS="--gpu-type V100 --profiling-mode custom --custom-frequencies '510,960,1380' --app-name StableDiffusion --app-executable ../app-stable-diffusion/StableDiffusionViaHF.py --app-params '--prompt \"a photograph of an astronaut riding a horse\" --steps 500 --log-level INFO' --num-runs 5 --sleep-interval 2"

# 8. 📝 LLAMA CUSTOM - Three-point frequency analysis with benchmark (low/mid/high)
# LAUNCH_ARGS="--gpu-type V100 --profiling-mode custom --custom-frequencies '510,960,1380' --app-name LLaMA --app-executable ../app-llama/LlamaViaHF.py --app-params '--benchmark --num-generations 3 --quiet --metrics' --num-runs 5 --sleep-interval 2"

# 9. 🎤 WHISPER CUSTOM - Three-point frequency analysis (low/mid/high)
# LAUNCH_ARGS="--gpu-type V100 --profiling-mode custom --custom-frequencies '510,960,1380' --app-name Whisper --app-executable ../app-whisper/WhisperViaHF.py --app-params '--benchmark --model base --num-samples 3 --quiet' --num-runs 5 --sleep-interval 2"

# 10. 🖼️ VISION TRANSFORMER CUSTOM - Three-point frequency analysis (low/mid/high)
# LAUNCH_ARGS="--gpu-type V100 --profiling-mode custom --custom-frequencies '510,960,1380' --app-name ViT --app-executable ../app-vision-transformer/ViTViaHF.py --app-params '--benchmark --num-images 2000 --batch-size 4 --model google/vit-large-patch16-224 --precision float16' --num-runs 5 --sleep-interval 2"

# 🔄 DVFS STUDY CONFIGURATIONS
# ============================================================================

# 11. ⚡ COMPREHENSIVE DVFS - All 117 frequencies (⚠️ LONG: 6-12 hours, change --time to 12:00:00)
# LAUNCH_ARGS="--gpu-type V100 --profiling-mode dvfs --num-runs 3 --sleep-interval 2"

# 12. 🎯 EFFICIENT DVFS - Reduced runs for faster completion (~4-6 hours, change --time to 08:00:00)
# LAUNCH_ARGS="--gpu-type V100 --profiling-mode dvfs --num-runs 2 --sleep-interval 1"

# 13. 📈 STATISTICAL DVFS - High statistical power (⚠️ VERY LONG: 12-20 hours, change --time to 24:00:00)
# LAUNCH_ARGS="--gpu-type V100 --profiling-mode dvfs --num-runs 5 --sleep-interval 3"

# 🔬 APPLICATION-SPECIFIC DVFS STUDIES (All 117 Frequencies)
# ============================================================================

# 14. 🤖 LSTM DVFS - Complete frequency analysis for deep learning (~6-8 hours, change --time to 10:00:00)
# LAUNCH_ARGS="--gpu-type V100 --profiling-mode dvfs --app-name LSTM --app-executable ../app-lstm/lstm --num-runs 3 --sleep-interval 2"

# 15. 🎨 STABLE DIFFUSION DVFS - Complete frequency analysis for image generation (~8-12 hours, change --time to 14:00:00)
# Research Mode (no images): Uncomment for energy-focused research
# LAUNCH_ARGS="--gpu-type V100 --profiling-mode dvfs --app-name StableDiffusion --app-executable ../app-stable-diffusion/StableDiffusionViaHF.py --app-params '--prompt \"a photograph of an astronaut riding a horse\" --steps 50 --no-save-images --job-id ${SLURM_JOB_ID} --log-level INFO' --num-runs 3 --sleep-interval 2"
# Full Mode (with images): Uncomment for complete generation study  
# LAUNCH_ARGS="--gpu-type V100 --profiling-mode dvfs --app-name StableDiffusion --app-executable ../app-stable-diffusion/StableDiffusionViaHF.py --app-params '--prompt \"a photograph of an astronaut riding a horse\" --steps 50 --job-id ${SLURM_JOB_ID} --log-level INFO' --num-runs 3 --sleep-interval 2"


# 16. 📝 LLAMA DVFS - Complete frequency analysis for text generation (~6-8 hours, change --time to 10:00:00)
# LAUNCH_ARGS="--gpu-type V100 --profiling-mode dvfs --app-name LLaMA --app-executable ../app-llama/LlamaViaHF.py --app-params '--benchmark --num-generations 3 --quiet --metrics' --num-runs 3 --sleep-interval 2"

# 17. 🎤 WHISPER DVFS - Complete frequency analysis for speech recognition (~6-8 hours, change --time to 10:00:00)
# LAUNCH_ARGS="--gpu-type V100 --profiling-mode dvfs --app-name Whisper --app-executable ../app-whisper/WhisperViaHF.py --app-params '--benchmark --model base --num-samples 3 --quiet' --num-runs 3 --sleep-interval 2"

# 18. 🖼️ VISION TRANSFORMER DVFS - Complete frequency analysis for image classification (~6-8 hours, change --time to 10:00:00)
# LAUNCH_ARGS="--gpu-type V100 --profiling-mode dvfs --app-name ViT --app-executable ../app-vision-transformer/ViTViaHF.py --app-params '--benchmark --num-images 1200 --batch-size 4 --model google/vit-large-patch16-224 --precision float16' --num-runs 3 --sleep-interval 2"

# 🎓 RESEARCH STUDY CONFIGURATIONS
# ============================================================================

# 19. 📊 ENERGY EFFICIENCY STUDY - Seven-point frequency analysis for power vs performance
# LAUNCH_ARGS="--gpu-type V100 --profiling-mode custom --custom-frequencies '405,600,795,990,1185,1380' --num-runs 7 --sleep-interval 2"

# 20. 🔬 EXTENDED BASELINE - Higher statistical significance for applications
# LAUNCH_ARGS="--gpu-type V100 --profiling-mode baseline --app-name LLaMA --app-executable ../app-llama/LlamaViaHF.py --app-params '--benchmark --num-generations 3 --quiet --metrics' --num-runs 5"

# 21. 📈 SCALING ANALYSIS - Batch size impact study
# LAUNCH_ARGS="--gpu-type V100 --profiling-mode custom --custom-frequencies '405,892,1380' --app-name LSTM --app-executable ../app-lstm/lstm --app-params '--batch-size 256' --num-runs 5"

# 🚀 ADVANCED V100 CONFIGURATIONS
# ============================================================================

# 22. 🔥 TENSOR CORES - Advanced mixed precision optimization
# LAUNCH_ARGS="--gpu-type V100 --profiling-mode baseline --app-name LLaMA --app-executable ../app-llama/LlamaViaHF.py --app-params '--benchmark --num-generations 3 --precision float16 --quiet --metrics' --num-runs 3"

# 23. 🧠 1ST GEN TENSOR CORES - Mixed precision configuration
# LAUNCH_ARGS="--gpu-type V100 --profiling-mode custom --custom-frequencies '405,892,1380' --app-name StableDiffusion --app-executable ../app-stable-diffusion/StableDiffusionViaHF.py --app-params '--use-tensor-cores --precision float16 --prompt \"a photograph of an astronaut riding a horse\" --steps 500' --num-runs 5"

# 24. 💾 MEMORY STRESS TEST - Large model testing with 32GB HBM2
# LAUNCH_ARGS="--gpu-type V100 --profiling-mode baseline --app-name LLaMA --app-executable ../app-llama/LlamaViaHF.py --app-params '--model llama2-13b --benchmark --num-generations 3 --quiet --metrics' --num-runs 3"

# 25. 🏆 FLAGSHIP PERFORMANCE - Maximum capability demonstration
# LAUNCH_ARGS="--gpu-type V100 --profiling-mode baseline --app-name LLaMA --app-executable ../app-llama/LlamaViaHF.py --app-params '--benchmark --num-generations 5 --max-tokens 200' --num-runs 2"

# 🛠️ UTILITY AND DEBUG CONFIGURATIONS
# ============================================================================

# 26. 🔧 NVIDIA-SMI FALLBACK - When DCGMI is not available
# LAUNCH_ARGS="--gpu-type V100 --profiling-tool nvidia-smi --profiling-mode baseline --num-runs 3"

# 27. 🔧 DEBUG MODE - Test DVFS mode with reduced image count and debug logging
# LAUNCH_ARGS="--gpu-type V100 --profiling-mode dvfs --app-name ViT --app-executable ../app-vision-transformer/ViTViaHF.py --app-params '--benchmark --num-images 100 --batch-size 4 --model google/vit-large-patch16-224 --precision float16' --num-runs 1 --debug"

# 28. 🔍 NVIDIA-SMI DEBUG - Fallback tool with minimal workload
# LAUNCH_ARGS="--gpu-type V100 --profiling-tool nvidia-smi --profiling-mode baseline --app-name ViT --app-executable ../app-vision-transformer/ViTViaHF.py --app-params '--benchmark --num-images 5 --batch-size 1' --num-runs 1"

# ============================================================================
# SAMPLING INTERVAL AND MULTI-GPU CONFIGURATIONS (29-32)
# ============================================================================
# 
# New parameters available:
# --sampling-interval MS   : Set DCGMI sampling interval (10-1000ms, default: 50ms)
#                            Lower values = more detailed data, higher overhead
#                            Higher values = less detail, lower overhead
# --all-gpus              : Monitor all available GPUs instead of just GPU 0
#                            Useful for multi-GPU systems like V100 nodes
#
# Examples:
# - Fast sampling (10-25ms): For detailed power transitions, short experiments
# - Normal sampling (50ms): Default, good balance of detail and performance  
# - Slow sampling (100-200ms): For long experiments, reduced data volume
# - Multi-GPU: Essential for distributed training or multi-GPU inference
# ============================================================================

# 29. ⚡ HIGH-FREQUENCY SAMPLING - 10ms interval for fine-grained energy data
# LAUNCH_ARGS="--gpu-type V100 --profiling-mode baseline --sampling-interval 10 --app-name StableDiffusion --app-executable ../app-stable-diffusion/StableDiffusionViaHF.py --app-params '--prompt \"a photograph of an astronaut riding a horse\" --steps 50 --log-level INFO' --num-runs 3"

# 30. 🐌 LOW-FREQUENCY SAMPLING - 200ms interval for long-running experiments
# LAUNCH_ARGS="--gpu-type V100 --profiling-mode dvfs --sampling-interval 200 --app-name LLaMA --app-executable ../app-llama/LlamaViaHF.py --app-params '--benchmark --num-generations 10 --quiet --metrics' --num-runs 3"

# 31. 🎯 MULTI-GPU MONITORING - Monitor all V100 GPUs with default 50ms sampling
# LAUNCH_ARGS="--gpu-type V100 --profiling-mode baseline --all-gpus --app-name ViT --app-executable ../app-vision-transformer/ViTViaHF.py --app-params '--benchmark --num-images 1000 --model google/vit-base-patch16-224 --precision float16' --num-runs 3"

# 32. 🔬 ULTRA-FINE MONITORING - 25ms sampling across all GPUs for detailed analysis
# LAUNCH_ARGS="--gpu-type V100 --profiling-mode baseline --sampling-interval 25 --all-gpus --app-name StableDiffusion --app-executable ../app-stable-diffusion/StableDiffusionViaHF.py --app-params '--prompt \"a cyberpunk cityscape\" --steps 100 --log-level INFO' --num-runs 3"

# ============================================================================
# TIMING GUIDELINES FOR SLURM --time PARAMETER
# ============================================================================
# Configuration 1-5:     --time=01:00:00  (1 hour)
# Configuration 6-10:    --time=02:00:00  (2 hours) - Custom frequency mode with 3 frequencies  
# Configuration 11-13:   --time=12:00:00  (12 hours) - DVFS studies, V100 has 117 frequencies
# Configuration 14-18:   --time=10:00:00  (10 hours) - Application-specific DVFS studies
# Configuration 19-21:   --time=02:00:00  (2 hours) - Research studies
# Configuration 22-25:   --time=03:00:00  (3 hours) - Advanced V100 features
# Configuration 26-28:   --time=01:00:00  (1 hour) - Utility configurations
# Configuration 29-32:   --time=02:00:00  (2 hours) - Sampling interval and multi-GPU studies
#
# 💡 TIP: V100 has 117 frequencies (most of all GPU generations), DVFS studies take longer
# 🚀 TIP: Take advantage of V100's tensor cores for mixed precision workloads
# 💡 TIP: For DVFS studies (7-9), consider running during off-peak hours due to long duration
# ============================================================================

# Logging functions with colored output
log_info() {
    echo -e "\033[0;32m[$(date '+%Y-%m-%d %H:%M:%S')] [INFO]\033[0m $*"
}

log_error() {
    echo -e "\033[0;31m[$(date '+%Y-%m-%d %H:%M:%S')] [ERROR]\033[0m $*" >&2
}

log_warning() {
    echo -e "\033[0;33m[$(date '+%Y-%m-%d %H:%M:%S')] [WARNING]\033[0m $*" >&2
}

log_header() {
    echo -e "\033[1;34m[$(date '+%Y-%m-%d %H:%M:%S')] [INFO]\033[0m $*"
}

# Main execution function
main() {
    log_header "🚀 Starting V100 GPU Profiling Job"
    log_info "Configuration: $LAUNCH_ARGS"
    
    # Determine expected results directory
    readonly RESULTS_DIR=$(determine_results_dir)
    log_info "Expected results directory: $RESULTS_DIR"
    
    # Load HPCC modules
    log_info "Loading HPCC modules..."
    module load gcc cuda cudnn
    
    # Determine and activate conda environment based on application
    local CONDA_ENV=$(determine_conda_env)
    log_info "Activating conda environment: $CONDA_ENV (auto-selected for application)"
    if [[ -f "$HOME/miniforge3/etc/profile.d/conda.sh" ]]; then
        source "$HOME/miniforge3/etc/profile.d/conda.sh"
    elif [[ -f "$HOME/conda/etc/profile.d/conda.sh" ]]; then
        source "$HOME/conda/etc/profile.d/conda.sh"
    else
        log_error "❌ Conda initialization script not found"
        exit 1
    fi
    
    # Check if environment exists
    if ! conda info --envs | grep -q "^$CONDA_ENV "; then
        log_error "❌ Conda environment '$CONDA_ENV' not found"
        log_error "📋 Available environments:"
        conda info --envs
        case "$CONDA_ENV" in
            "stable-diffusion-gpu")
                log_error "💡 To create stable-diffusion-gpu environment: conda create -n stable-diffusion-gpu python=3.10"
                log_error "💡 Then install requirements: pip install -r ../app-stable-diffusion/requirements.txt"
                ;;
            "tensorflow")
                log_error "💡 To create tensorflow environment: conda env create -f ../app-lstm/lstm-v100-20250708.yml"
                ;;
        esac
        exit 1
    fi
    
    conda activate "$CONDA_ENV"
    
    # Display V100 system information
    display_v100_info
    
    # Validate configuration and provide warnings
    validate_configuration
    
    # Check system resources
    check_system_resources
    
    # Display GPU status
    check_gpu_status
    
    # Run the profiling experiment
    run_experiment
}

# Display V100 capabilities and system info
display_v100_info() {
    log_header "📊 V100 System Information"
    echo "┌─────────────────────────────────────────────────────────────┐"
    echo "│                    HPCC V100 Specifications                 │"
    echo "├─────────────────────────────────────────────────────────────┤"
    echo "│ Cluster:      HPCC at Texas Tech University                 │"
    echo "│ Partition:    matador                                       │"
    echo "│ Architecture: Volta (GV100)                                 │"
    echo "│ Memory:       32GB HBM2                                     │"
    echo "│ Mem Freq:     877 MHz (fixed)                               │"
    echo "│ Core Freq:    405-1380 MHz (117 frequencies)                │"
    echo "│ DVFS Step:    Variable (7-8 MHz typical)                    │"
    echo "│ Tools:        DCGMI (preferred) or nvidia-smi               │"
    echo "└─────────────────────────────────────────────────────────────┘"
}

# Validate configuration and provide appropriate warnings
validate_configuration() {
    log_header "🔍 Configuration Validation"
    
    # Check for DVFS mode and provide warnings
    if echo "$LAUNCH_ARGS" | grep -q "dvfs"; then
        log_warning "⚠️  DVFS mode detected - this will test ALL 117 V100 frequencies!"
        log_warning "⚠️  Estimated runtime: 6-20 hours depending on runs per frequency"
        log_warning "⚠️  Consider using 'custom' mode with selected frequencies for faster results"
        
        # Calculate estimated runtime
        if echo "$LAUNCH_ARGS" | grep -q "num-runs"; then
            runs=$(echo "$LAUNCH_ARGS" | sed -n 's/.*--num-runs \([0-9]\+\).*/\1/p')
            if [[ -n "$runs" && "$runs" -gt 0 ]]; then
                total_runs=$((117 * runs))
                estimated_hours=$((total_runs / 60))  # Rough estimate: 1 minute per run
                log_warning "⚠️  Estimated total runs: $total_runs"
                log_warning "⚠️  Estimated runtime: ~${estimated_hours} hours"
                
                # Recommend time adjustment
                if (( estimated_hours > 8 )); then
                    log_warning "⚠️  Consider adjusting SLURM --time to ${estimated_hours}:00:00 or higher"
                fi
            fi
        fi
        
        echo ""
        log_info "💡 For faster results, consider configuration #3 (frequency sampling)"
        log_info "💡 Example: custom --custom-frequencies '405,600,800,1000,1200,1380'"
    fi
    
    # Check for custom frequency selection
    if echo "$LAUNCH_ARGS" | grep -q "custom-frequencies"; then
        frequencies=$(echo "$LAUNCH_ARGS" | sed -n "s/.*--custom-frequencies '\([^']*\)'.*/\1/p")
        freq_count=$(echo "$frequencies" | tr ',' '\n' | wc -l)
        log_info "✅ Custom frequency mode: testing $freq_count selected frequencies"
        log_info "📊 Frequencies: $frequencies"
    fi
    
    # Check for specific applications
    if echo "$LAUNCH_ARGS" | grep -q "app-name"; then
        app_name=$(echo "$LAUNCH_ARGS" | sed -n 's/.*--app-name \([^ ]*\).*/\1/p')
        log_info "🎯 Application: $app_name"
        
        # Application-specific notes
        case "$app_name" in
            "StableDiffusion")
                log_info "🎨 Stable Diffusion: Expect high memory usage (~20-25GB on V100)"
                ;;
            "LLaMA")
                log_info "📝 LLaMA: Monitor memory usage, 13B+ models may exceed V100 32GB"
                ;;
            "LSTM")
                log_info "🤖 LSTM: Lightweight benchmark, good for initial testing"
                ;;
        esac
    fi
}

# Check system resources
check_system_resources() {
    log_header "💾 System Resource Check"
    
    # Check available disk space
    local available_space
    available_space=$(df . | awk 'NR==2 {print $4}')
    local available_gb=$((available_space / 1024 / 1024))
    
    if (( available_space < 1000000 )); then  # Less than 1GB
        log_warning "⚠️  Available disk space: ${available_gb}GB (may be insufficient)"
        log_warning "⚠️  Recommended: >2GB for comprehensive studies"
        log_warning "⚠️  V100 DVFS experiments can generate substantial data"
    else
        log_info "✅ Available disk space: ${available_gb}GB"
    fi
    
    # Check if results directory exists
    if [[ ! -d "results" ]]; then
        log_info "📁 Creating results directory..."
        mkdir -p results
    fi
}

# Check GPU status and availability
check_gpu_status() {
    log_header "🖥️  GPU Status Check"
    
    # Display GPU information
    if gpu_info=$(nvidia-smi --query-gpu=name,memory.total,driver_version,power.max_limit --format=csv,noheader,nounits 2>/dev/null); then
        log_info "📊 GPU Information: $gpu_info"
        
        # Verify it's actually a V100
        if echo "$gpu_info" | grep -qi "v100"; then
            log_info "✅ V100 GPU confirmed"
        else
            log_warning "⚠️  Expected V100 but detected: $gpu_info"
            log_warning "⚠️  Configuration may not be optimal"
        fi
    else
        log_warning "⚠️  Could not query GPU info - continuing anyway"
    fi
    
    # Check frequency control availability
    log_info "🔧 Checking frequency control capabilities..."
    if nvidia-smi -q -d SUPPORTED_CLOCKS &>/dev/null; then
        local freq_count
        freq_count=$(nvidia-smi -q -d SUPPORTED_CLOCKS | grep -c "Graphics" 2>/dev/null || echo "0")
        if [[ "$freq_count" -gt 0 ]]; then
            log_info "✅ Frequency control available ($freq_count frequency options detected)"
        else
            log_info "📊 DCGMI/nvidia-smi frequency control will be attempted"
        fi
    else
        log_warning "⚠️  Frequency control query failed"
        log_info "🔄 Framework will attempt DCGMI first, then fallback to nvidia-smi monitoring"
    fi
}

# Run the main profiling experiment
run_experiment() {
    log_header "🚀 Starting V100 Profiling Experiment"
    log_info "Launch command: $LAUNCH_SCRIPT $LAUNCH_ARGS"
    
    local start_time
    start_time=$(date +%s)
    
    # Execute the experiment
    if eval "$LAUNCH_SCRIPT $LAUNCH_ARGS"; then
        # Success path
        local end_time
        end_time=$(date +%s)
        local total_time=$((end_time - start_time))
        local hours=$((total_time / 3600))
        local minutes=$(((total_time % 3600) / 60))
        
        log_header "🎉 V100 Profiling Completed Successfully!"
        log_info "⏱️  Total runtime: ${hours}h ${minutes}m"
        
        # Display results summary
        display_results_summary
        
        # Display completion notes
        display_completion_notes
        
    else
        # Failure path
        log_error "❌ V100 profiling experiment failed"
        log_error "🔍 Check the error logs above for details"
        
        # Common troubleshooting suggestions
        log_error ""
        log_error "🛠️  Common V100 Issues and Solutions:"
        log_error "   • Frequency control permissions → Try nvidia-smi fallback (config #11)"
        log_error "   • DCGMI tool unavailable → Automatic fallback should occur"
        log_error "   • Memory limitations → V100 has 32GB vs A100's 80GB"
        log_error "   • Long DVFS runtimes → Use custom frequency selection (config #3)"
        log_error "   • Module loading issues → Check HPCC environment setup"
        
        exit 1
    fi
}

# Display comprehensive results summary
display_results_summary() {
    log_header "📊 Results Summary"
    
    if [[ -d "$RESULTS_DIR" ]]; then
        local result_count
        result_count=$(find "$RESULTS_DIR" -type f | wc -l)
        log_info "📁 Generated $result_count result files in $RESULTS_DIR"
        
        # Show recent files
        if [[ "$result_count" -gt 0 ]]; then
            log_info "📋 Recent result files:"
            find "$RESULTS_DIR" -type f -newer "$LAUNCH_SCRIPT" 2>/dev/null | head -5 | while read -r file; do
                if [[ -n "$file" ]]; then
                    local size
                    size=$(stat -f%z "$file" 2>/dev/null || stat -c%s "$file" 2>/dev/null || echo "unknown")
                    log_info "   📄 $(basename "$file") - ${size} bytes"
                fi
            done
        fi
        
        # Check for specific output files
        local csv_files
        csv_files=$(find "$RESULTS_DIR" -name "GV100*.csv" 2>/dev/null)
        if [[ -n "$csv_files" ]]; then
            local csv_file
            csv_file=$(echo "$csv_files" | head -1)
            local csv_lines
            csv_lines=$(wc -l < "$csv_file" 2>/dev/null || echo "unknown")
            log_info "📊 Performance data points in $(basename "$csv_file"): $csv_lines"
        fi
        
        # Calculate total data size
        local total_size
        total_size=$(du -sh "$RESULTS_DIR" 2>/dev/null | cut -f1 || echo "unknown")
        log_info "💾 Total results directory size: $total_size"
        
    else
        log_warning "⚠️  No results directory found: $RESULTS_DIR"
    fi
}

# Display completion notes and next steps
display_completion_notes() {
    log_header "📝 V100 Profiling Completion Notes"
    
    echo "┌─────────────────────────────────────────────────────────────┐"
    echo "│                   Profiling Summary                         │"
    echo "├─────────────────────────────────────────────────────────────┤"
    echo "│ GPU:          V100 (Volta GV100) - 32GB HBM2                │"
    echo "│ Cluster:      HPCC matador partition                        │"
    
    # Mode-specific notes
    if echo "$LAUNCH_ARGS" | grep -q "dvfs"; then
        echo "│ Mode:         DVFS (tested across 117 frequency range)      │"
    elif echo "$LAUNCH_ARGS" | grep -q "custom"; then
        echo "│ Mode:         Custom frequency selection                    │"
    else
        echo "│ Mode:         Baseline (single frequency profiling)         │"
    fi
    
    # Tool-specific notes
    if echo "$LAUNCH_ARGS" | grep -q "nvidia-smi"; then
        echo "│ Tool:         nvidia-smi profiling                          │"
    else
        echo "│ Tool:         DCGMI (with nvidia-smi fallback)              │"
    fi
    
    echo "└─────────────────────────────────────────────────────────────┘"
    
    # Next steps
    log_info ""
    log_info "🎯 Next Steps:"
    log_info "   📊 Analyze results with power modeling framework:"
    log_info "      python -c \"from power_modeling import analyze_application; analyze_application('$RESULTS_DIR/GV100*.csv')\""
    log_info "   📈 Run EDP optimization:"
    log_info "      python -c \"from edp_analysis import edp_calculator; edp_calculator.find_optimal_configuration('$RESULTS_DIR/GV100*.csv')\""
    log_info "   🔄 Submit additional configurations by editing this script and resubmitting"
}

# Execute main function
main "$@"
