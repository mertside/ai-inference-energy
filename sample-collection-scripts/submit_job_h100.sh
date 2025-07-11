#!/bin/bash
#
# Unified SLURM Job Submission Script - H100 GPU Profiling
#
# This script provides a comprehensive set of H100 profiling configurations.
# Simply uncomment the desired configuration and submit with: sbatch submit_job_h100.sh
#
# H100 Specifications:
#   - GPU: H100 (80GB HBM3)
#   - Partition: h100-build (REPACSS Texas Tech)
#   - Frequencies: 86 available (510-1785 MHz)
#   - Memory: 2619 MHz (maximum)
#   - Architecture: Hopper (GH100)
#

#SBATCH --job-name=PROFILING_H100
#SBATCH --output=%x.%j.out
#SBATCH --error=%x.%j.err
#SBATCH --partition=h100
# # SBATCH --partition=h100-build
# # SBATCH --nodelist=rpg-93-9
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=mert.side@ttu.edu
# # SBATCH --time=02:00:00  # Adjust based on configuration (see timing notes below)

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
            echo "tensorflow"  # Default for now, can be adjusted
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
        echo "$custom_output"
        return
    fi
    
    # Generate auto-generated directory name (same logic as args_parser.sh)
    if [[ -n "$gpu_type" && -n "$app_name" ]]; then
        local gpu_name=$(echo "$gpu_type" | tr '[:upper:]' '[:lower:]')
        local app_name_clean=$(echo "$app_name" | tr '[:upper:]' '[:lower:]' | sed 's/[^a-z0-9]//g')
        echo "results_${gpu_name}_${app_name_clean}"
    else
        echo "results"
    fi
}

# ============================================================================
# CONFIGURATION SECTION - Uncomment ONE configuration below
# ============================================================================

# 📋 QUICK START CONFIGURATIONS
# ============================================================================

# 1. 🚀 QUICK TEST - Baseline profiling (fastest, ~2-4 minutes) - PyTorch LSTM
# LAUNCH_ARGS="--gpu-type H100 --profiling-mode baseline --num-runs 3 --sleep-interval 1 --app-name LSTM --app-executable ../app-lstm/lstm"

# 2. 🔬 RESEARCH BASELINE - Extended baseline for statistical significance (~6-10 minutes)
# LAUNCH_ARGS="--gpu-type H100 --profiling-mode baseline --num-runs 5 --sleep-interval 2"

# 3. 🎯 FREQUENCY SAMPLING - Selected frequencies for efficient analysis (~12-20 minutes)
# LAUNCH_ARGS="--gpu-type H100 --profiling-mode custom --custom-frequencies '510,750,1000,1250,1500,1785' --num-runs 5 --sleep-interval 2"

# 📊 AI APPLICATION CONFIGURATIONS
# ============================================================================

# 4. 🤖 LSTM PROFILING - PyTorch sentiment analysis benchmark
# LAUNCH_ARGS="--gpu-type H100 --profiling-mode baseline --app-name LSTM --app-executable ../app-lstm/lstm --num-runs 5"

# 5. 🎨 STABLE DIFFUSION - Image generation profiling (1000 steps, 768x768, astronaut riding horse)
LAUNCH_ARGS="--gpu-type H100 --profiling-mode baseline --app-name StableDiffusion --app-executable ../app-stable-diffusion/StableDiffusionViaHF.py --app-params '--prompt \"a photograph of an astronaut riding a horse\" --steps 500 --log-level INFO' --num-runs 3 --sleep-interval 1"

# 6. 📝 LLAMA - Text generation profiling  
# LAUNCH_ARGS="--gpu-type H100 --profiling-mode baseline --app-name LLaMA --app-executable llama_inference --num-runs 5"

# 7. 🔧 CUSTOM APPLICATION - Template for your own applications
# LAUNCH_ARGS="--gpu-type H100 --profiling-mode baseline --app-name CustomApp --app-executable my_app --app-params '--config config.json > results/custom_output.log' --num-runs 3"

# 🔄 DVFS STUDY CONFIGURATIONS
# ============================================================================

# 8. ⚡ COMPREHENSIVE DVFS - All 86 frequencies (~3-5 hours, change --time to 06:00:00)
# LAUNCH_ARGS="--gpu-type H100 --profiling-mode dvfs --num-runs 3 --sleep-interval 2"

# 9. 🎯 EFFICIENT DVFS - Reduced runs for faster completion (~1.5-3 hours, change --time to 04:00:00)
# LAUNCH_ARGS="--gpu-type H100 --profiling-mode dvfs --num-runs 2 --sleep-interval 1"

# 10. 📈 STATISTICAL DVFS - High statistical power (~5-10 hours, change --time to 12:00:00)
# LAUNCH_ARGS="--gpu-type H100 --profiling-mode dvfs --num-runs 5 --sleep-interval 3"

# 🛠️ TOOL AND COMPATIBILITY CONFIGURATIONS  
# ============================================================================

# 11. 🔧 NVIDIA-SMI FALLBACK - When DCGMI is not available
# LAUNCH_ARGS="--gpu-type H100 --profiling-tool nvidia-smi --profiling-mode baseline --num-runs 3"

# 12. 🐛 DEBUG MODE - Minimal configuration for troubleshooting
# LAUNCH_ARGS="--gpu-type H100 --profiling-mode baseline --num-runs 1 --sleep-interval 0"

# 13. 💾 MEMORY STRESS TEST - Large model testing (H100 has 80GB)
# LAUNCH_ARGS="--gpu-type H100 --profiling-mode baseline --app-name LLaMA --app-executable llama_inference --app-params '--model-size 70b' --num-runs 3"

# 🎓 RESEARCH STUDY CONFIGURATIONS
# ============================================================================

# 14. 📊 ENERGY EFFICIENCY STUDY - Focus on power vs performance
# LAUNCH_ARGS="--gpu-type H100 --profiling-mode custom --custom-frequencies '510,700,900,1100,1300,1500,1785' --num-runs 7 --sleep-interval 2"

# 15. 🔬 PRECISION COMPARISON - Different model precisions
# LAUNCH_ARGS="--gpu-type H100 --profiling-mode baseline --app-name StableDiffusion --app-executable stable_diffusion --app-params '--precision fp8' --num-runs 5"

# 16. 📈 SCALING ANALYSIS - Batch size impact study
# LAUNCH_ARGS="--gpu-type H100 --profiling-mode custom --custom-frequencies '800,1200,1600' --app-name LSTM --app-executable ../app-lstm/lstm --app-params '--batch-size 256' --num-runs 5"

# 🚀 ADVANCED H100 CONFIGURATIONS
# ============================================================================

# 17. 🔥 TRANSFORMER ENGINE - Optimized for large language models
# LAUNCH_ARGS="--gpu-type H100 --profiling-mode baseline --app-name LLaMA --app-executable llama_inference --app-params '--use-transformer-engine --precision fp8' --num-runs 3"

# 18. 🧠 4TH GEN TENSOR CORES - Maximum performance configuration
# LAUNCH_ARGS="--gpu-type H100 --profiling-mode custom --custom-frequencies '1200,1500,1785' --app-name StableDiffusion --app-params '--use-4th-gen-tensor-cores --precision fp8' --num-runs 5"

# 19. 💡 HBM3 BANDWIDTH TEST - Memory-intensive workloads
# LAUNCH_ARGS="--gpu-type H100 --profiling-mode custom --custom-frequencies '1000,1400,1785' --app-name CustomApp --app-params '--memory-intensive --hbm3-optimized' --num-runs 5"

# 20. 🏆 FLAGSHIP PERFORMANCE - Maximum capability demonstration
# LAUNCH_ARGS="--gpu-type H100 --profiling-mode baseline --app-name LLaMA --app-executable llama_inference --app-params '--model-size 175b --use-all-features' --num-runs 2"

# ============================================================================
# TIMING GUIDELINES FOR SLURM --time PARAMETER
# ============================================================================
# Configuration 1-3:     --time=01:00:00  (1 hour)
# Configuration 4-7:     --time=02:00:00  (2 hours) 
# Configuration 8-9:     --time=06:00:00  (6 hours)
# Configuration 10:      --time=12:00:00  (12 hours)
# Configuration 11-16:   --time=03:00:00  (3 hours, adjust as needed)
# Configuration 17-20:   --time=04:00:00  (4 hours, cutting-edge features)
#
# 💡 TIP: H100 has 86 frequencies (between A100's 61 and V100's 117)
# 🚀 TIP: Take advantage of H100's advanced features (FP8, Transformer Engine)
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
    log_header "🚀 Starting H100 GPU Profiling Job"
    log_info "Configuration: $LAUNCH_ARGS"
    
    # Determine expected results directory
    readonly RESULTS_DIR=$(determine_results_dir)
    log_info "Expected results directory: $RESULTS_DIR"
    
    # Check REPACSS H100 environment
    log_info "Checking REPACSS H100 environment..."
    # Note: CUDA tools are available system-wide on H100 nodes
    if command -v nvidia-smi &> /dev/null; then
        log_info "✅ nvidia-smi available: $(nvidia-smi --version | head -1)"
    else
        log_warning "⚠️  nvidia-smi not found - may affect profiling capabilities"
    fi
    
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
                log_error "💡 To create tensorflow environment: conda env create -f ../app-lstm/lstm-h100-20250708.yml"
                ;;
        esac
        exit 1
    fi
    
    conda activate "$CONDA_ENV"
    
    # Display H100 system information
    display_h100_info
    
    # Validate configuration and provide warnings
    validate_configuration
    
    # Check system resources
    check_system_resources
    
    # Display GPU status
    check_gpu_status
    
    # Run the profiling experiment
    run_experiment
}

# Display H100 capabilities and system info
display_h100_info() {
    log_header "📊 H100 System Information"
    echo "┌─────────────────────────────────────────────────────────────┐"
    echo "│                   REPACSS H100 Specifications              │"
    echo "├─────────────────────────────────────────────────────────────┤"
    echo "│ Cluster:      REPACSS at Texas Tech University             │"
    echo "│ Partition:    h100-build                                    │"
    echo "│ Architecture: Hopper (GH100)                                │"
    echo "│ Memory:       80GB HBM3                                     │"
    echo "│ Mem Freq:     2619 MHz (maximum)                            │"
    echo "│ Core Freq:    510-1785 MHz (86 frequencies)                 │"
    echo "│ DVFS Step:    ~15 MHz typical                                │"
    echo "│ Features:     4th Gen Tensor Cores, Transformer Engine     │"
    echo "│ Precision:    FP8, FP16, BF16, INT8, INT4                   │"
    echo "│ Tools:        DCGMI (preferred) or nvidia-smi               │"
    echo "└─────────────────────────────────────────────────────────────┘"
}

# Validate configuration and provide appropriate warnings
validate_configuration() {
    log_header "🔍 Configuration Validation"
    
    # Check for DVFS mode and provide warnings
    if echo "$LAUNCH_ARGS" | grep -q "dvfs"; then
        log_warning "⚠️  DVFS mode detected - this will test ALL 86 H100 frequencies"
        log_warning "⚠️  Estimated runtime: 3-10 hours depending on runs per frequency"
        log_warning "⚠️  H100 frequency range: 510-1785 MHz (86 frequencies)"
        
        # Calculate estimated runtime
        if echo "$LAUNCH_ARGS" | grep -q "num-runs"; then
            runs=$(echo "$LAUNCH_ARGS" | sed -n 's/.*--num-runs \([0-9]\+\).*/\1/p')
            if [[ -n "$runs" && "$runs" -gt 0 ]]; then
                total_runs=$((86 * runs))
                estimated_hours=$((total_runs / 60))  # Rough estimate: 1 minute per run
                log_warning "⚠️  Estimated total runs: $total_runs"
                log_warning "⚠️  Estimated runtime: ~${estimated_hours} hours"
                
                # Recommend time adjustment
                if (( estimated_hours > 6 )); then
                    log_warning "⚠️  Consider adjusting SLURM --time to ${estimated_hours}:00:00 or higher"
                fi
            fi
        fi
        
        echo ""
        log_info "💡 For faster results, consider configuration #3 (frequency sampling)"
        log_info "💡 Example: custom --custom-frequencies '510,750,1000,1250,1500,1785'"
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
                log_info "🎨 Stable Diffusion: H100 excels with 4th Gen Tensor Cores and FP8"
                ;;
            "LLaMA")
                log_info "📝 LLaMA: H100 80GB can handle very large models (70B+)"
                log_info "🔥 Consider using Transformer Engine for optimal performance"
                ;;
            "LSTM")
                log_info "🤖 LSTM: Lightweight benchmark for H100 validation"
                ;;
        esac
    fi
    
    # Check for advanced H100 features
    if echo "$LAUNCH_ARGS" | grep -q "transformer-engine"; then
        log_info "🔥 Transformer Engine detected: Optimized for large language models"
        log_info "⚡ Expect significant speedup for transformer-based workloads"
    fi
    
    if echo "$LAUNCH_ARGS" | grep -q "4th-gen-tensor-cores"; then
        log_info "🧠 4th Gen Tensor Cores detected: Maximum H100 performance mode"
    fi
    
    if echo "$LAUNCH_ARGS" | grep -q "fp8"; then
        log_info "🎯 FP8 precision detected: H100's cutting-edge feature"
        log_warning "⚠️  Ensure software stack supports FP8 (requires recent CUDA/cuDNN)"
    fi
    
    if echo "$LAUNCH_ARGS" | grep -q "hbm3-optimized"; then
        log_info "💾 HBM3 bandwidth optimization detected"
        log_info "🚀 H100 HBM3 provides 3TB/s memory bandwidth"
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
        log_warning "⚠️  Recommended: >3GB for comprehensive H100 studies"
        log_warning "⚠️  H100 can generate large datasets with 80GB memory"
    else
        log_info "✅ Available disk space: ${available_gb}GB"
    fi
    
    # Check if results directory exists
    if [[ ! -d "$RESULTS_DIR" ]]; then
        log_info "📁 Creating results directory: $RESULTS_DIR"
        mkdir -p "$RESULTS_DIR"
    else
        log_info "📁 Results directory exists: $RESULTS_DIR"
    fi
    
    # Check for H100-specific requirements
    log_info "🔧 Checking H100-specific requirements..."
    if echo "$LAUNCH_ARGS" | grep -q "fp8"; then
        log_info "🎯 FP8 precision requires CUDA 12.0+ and cuDNN 8.7+"
    fi
    
    if echo "$LAUNCH_ARGS" | grep -q "transformer-engine"; then
        log_info "🔥 Transformer Engine requires compatible software stack"
    fi
}

# Check GPU status and availability
check_gpu_status() {
    log_header "🖥️  GPU Status Check"
    
    # Display GPU information
    if gpu_info=$(nvidia-smi --query-gpu=name,memory.total,driver_version,power.max_limit --format=csv,noheader,nounits 2>/dev/null); then
        log_info "📊 GPU Information: $gpu_info"
        
        # Verify it's actually an H100
        if echo "$gpu_info" | grep -qi "h100"; then
            log_info "✅ H100 GPU confirmed"
            
            # Check memory size to determine variant
            if echo "$gpu_info" | grep -q "81920"; then
                log_info "💾 H100 80GB variant detected"
            elif echo "$gpu_info" | grep -q "40960"; then
                log_info "💾 H100 40GB variant detected"
            fi
        else
            log_warning "⚠️  Expected H100 but detected: $gpu_info"
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
    
    # Check for H100-specific features
    if nvidia-smi --query-gpu=name --format=csv,noheader,nounits 2>/dev/null | grep -qi "h100"; then
        log_info "🧠 4th Gen Tensor Cores available for cutting-edge AI workloads"
        log_info "🔥 Transformer Engine support available"
        log_info "💾 HBM3 memory with 3TB/s bandwidth"
    fi
    
    # Check CUDA driver version for advanced features
    if driver_version=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader,nounits 2>/dev/null); then
        log_info "🔧 Driver version: $driver_version"
        # Check if driver supports H100 advanced features
        if [[ -n "$driver_version" ]]; then
            log_info "💡 Verify driver supports H100 advanced features (FP8, Transformer Engine)"
        fi
    fi
}

# Run the main profiling experiment
run_experiment() {
    log_header "🚀 Starting H100 Profiling Experiment"
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
        
        log_header "🎉 H100 Profiling Completed Successfully!"
        log_info "⏱️  Total runtime: ${hours}h ${minutes}m"
        
        # Display results summary
        display_results_summary
        
        # Display completion notes
        display_completion_notes
        
    else
        # Failure path
        log_error "❌ H100 profiling experiment failed"
        log_error "🔍 Check the error logs above for details"
        
        # Common troubleshooting suggestions
        log_error ""
        log_error "🛠️  Common H100 Issues and Solutions:"
        log_error "   • Frequency control permissions → Try nvidia-smi fallback (config #11)"
        log_error "   • DCGMI tool unavailable → Automatic fallback should occur"
        log_error "   • Node access → Check REPACSS node allocation (rpg-93-9)"
        log_error "   • Advanced features → Verify CUDA/cuDNN versions for FP8/Transformer Engine"
        log_error "   • Module loading issues → Check REPACSS environment setup"
        
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
        csv_files=$(find "$RESULTS_DIR" -name "GH100*.csv" 2>/dev/null)
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
    log_header "📝 H100 Profiling Completion Notes"
    
    echo "┌─────────────────────────────────────────────────────────────┐"
    echo "│                   Profiling Summary                        │"
    echo "├─────────────────────────────────────────────────────────────┤"
    echo "│ GPU:          H100 (Hopper GH100) - 80GB HBM3              │"
    echo "│ Cluster:      REPACSS h100-build partition                 │"
    
    # Mode-specific notes
    if echo "$LAUNCH_ARGS" | grep -q "dvfs"; then
        echo "│ Mode:         DVFS (tested across 86 frequency range)      │"
    elif echo "$LAUNCH_ARGS" | grep -q "custom"; then
        echo "│ Mode:         Custom frequency selection                   │"
    else
        echo "│ Mode:         Baseline (single frequency profiling)        │"
    fi
    
    # Tool-specific notes
    if echo "$LAUNCH_ARGS" | grep -q "nvidia-smi"; then
        echo "│ Tool:         nvidia-smi profiling                         │"
    else
        echo "│ Tool:         DCGMI (with nvidia-smi fallback)             │"
    fi
    
    # Feature-specific notes
    if echo "$LAUNCH_ARGS" | grep -q "transformer-engine"; then
        echo "│ Features:     Transformer Engine enabled                   │"
    fi
    
    if echo "$LAUNCH_ARGS" | grep -q "4th-gen-tensor-cores"; then
        echo "│ Features:     4th Gen Tensor Cores enabled                 │"
    fi
    
    if echo "$LAUNCH_ARGS" | grep -q "fp8"; then
        echo "│ Precision:    FP8 cutting-edge precision                   │"
    fi
    
    echo "└─────────────────────────────────────────────────────────────┘"
    
    # Next steps
    log_info ""
    log_info "🎯 Next Steps:"
    log_info "   📊 Analyze results with power modeling framework:"
    log_info "      python -c \"from power_modeling import analyze_application; analyze_application('$RESULTS_DIR/GH100*.csv')\""
    log_info "   📈 Run EDP optimization:"
    log_info "      python -c \"from edp_analysis import edp_calculator; edp_calculator.find_optimal_configuration('$RESULTS_DIR/GH100*.csv')\""
    log_info "   🔄 Submit additional configurations by editing this script and resubmitting"
    log_info "   🚀 Explore cutting-edge H100 features: configs #17-20 (Transformer Engine, FP8)"
    log_info "   🏆 Compare performance across GPU generations (V100 → A100 → H100)"
}

# Execute main function
main "$@"
