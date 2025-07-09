#!/bin/bash
#
# Unified SLURM Job Submission Script - A100 GPU Profiling
#
# This script provides a comprehensive set of A100 profiling configurations.
# Simply uncomment the desired configuration and submit with: sbatch submit_job_a100.sh
#
# A100 Specifications:
#   - GPU: Tesla A100 (40GB HBM2e)
#   - Partition: toreador (HPCC Texas Tech)
#   - Frequencies: 61 available (510-1410 MHz)
#   - Memory: 1215 MHz (fixed)
#   - Architecture: Ampere (GA100)
#

#SBATCH --job-name=PROFILING_A100
#SBATCH --output=%x.%j.out
#SBATCH --error=%x.%j.err
#SBATCH --partition=toreador
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=16
#SBATCH --gpus-per-node=1
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=mert.side@ttu.edu
#SBATCH --time=02:00:00  # Adjust based on configuration (see timing notes below)

# Enable strict error handling (conda-friendly)
set -eo pipefail  # Removed -u to avoid issues with conda environment scripts

# Configuration
readonly LAUNCH_SCRIPT="./launch.sh"

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

# ============================================================================
# CONFIGURATION SECTION - Uncomment ONE configuration below
# ============================================================================

# 📋 QUICK START CONFIGURATIONS
# ============================================================================

# 1. 🚀 QUICK TEST - Baseline profiling (fastest, ~? minutes) - PyTorch LSTM
# LAUNCH_ARGS="--gpu-type A100 --profiling-mode baseline --num-runs 3 --sleep-interval 1 --app-name LSTM --app-executable ../app-lstm/lstm"

# 2. 🔬 RESEARCH BASELINE - Extended baseline for statistical significance (~8-12 minutes)
# LAUNCH_ARGS="--gpu-type A100 --profiling-mode baseline --num-runs 5 --sleep-interval 2"

# 3. 🎯 FREQUENCY SAMPLING - Selected frequencies for efficient analysis (~15-25 minutes)
# LAUNCH_ARGS="--gpu-type A100 --profiling-mode custom --custom-frequencies '510,700,900,1100,1300,1410' --num-runs 5 --sleep-interval 2"

# 📊 AI APPLICATION CONFIGURATIONS
# ============================================================================

# 4. 🤖 LSTM PROFILING - Default sentiment analysis benchmark
# LAUNCH_ARGS="--gpu-type A100 --profiling-mode baseline --app-name LSTM --app-executable ../app-lstm/lstm --num-runs 5"

# 5. 🎨 STABLE DIFFUSION - Image generation profiling (1000 steps, 768x768, astronaut riding horse)
LAUNCH_ARGS="--gpu-type A100 --profiling-mode baseline --app-name StableDiffusion --app-executable ../app-stable-diffusion/StableDiffusionViaHF.py --app-params '--prompt \"a photograph of an astronaut riding a horse\" --steps 1000 --log-level INFO --width 768 --height 768' --num-runs 3 --sleep-interval 1"

# 6. 📝 LLAMA - Text generation profiling  
# LAUNCH_ARGS="--gpu-type A100 --profiling-mode baseline --app-name LLaMA --app-executable llama_inference --num-runs 5"

# 7. 🔧 CUSTOM APPLICATION - Template for your own applications
# LAUNCH_ARGS="--gpu-type A100 --profiling-mode baseline --app-name CustomApp --app-executable my_app --app-params '--config config.json > results/custom_output.log' --num-runs 3"

# 🔄 DVFS STUDY CONFIGURATIONS
# ============================================================================

# 8. ⚡ COMPREHENSIVE DVFS - All 61 frequencies (~2-4 hours, change --time to 05:00:00)
# LAUNCH_ARGS="--gpu-type A100 --profiling-mode dvfs --num-runs 3 --sleep-interval 2"

# 9. 🎯 EFFICIENT DVFS - Reduced runs for faster completion (~1-2 hours, change --time to 03:00:00)
# LAUNCH_ARGS="--gpu-type A100 --profiling-mode dvfs --num-runs 2 --sleep-interval 1"

# 10. 📈 STATISTICAL DVFS - High statistical power (~4-8 hours, change --time to 10:00:00)
# LAUNCH_ARGS="--gpu-type A100 --profiling-mode dvfs --num-runs 5 --sleep-interval 3"

# 🛠️ TOOL AND COMPATIBILITY CONFIGURATIONS  
# ============================================================================

# 11. 🔧 NVIDIA-SMI FALLBACK - When DCGMI is not available
# LAUNCH_ARGS="--gpu-type A100 --profiling-tool nvidia-smi --profiling-mode baseline --num-runs 3"

# 12. 🐛 DEBUG MODE - Minimal configuration for troubleshooting
# LAUNCH_ARGS="--gpu-type A100 --profiling-mode baseline --num-runs 1 --sleep-interval 0"

# 13. 💾 MEMORY STRESS TEST - Large model testing (A100 has 40GB)
# LAUNCH_ARGS="--gpu-type A100 --profiling-mode baseline --app-name LLaMA --app-executable llama_inference --app-params '--model-size 30b' --num-runs 3"

# 🎓 RESEARCH STUDY CONFIGURATIONS
# ============================================================================

# 14. 📊 ENERGY EFFICIENCY STUDY - Focus on power vs performance
# LAUNCH_ARGS="--gpu-type A100 --profiling-mode custom --custom-frequencies '510,650,800,950,1100,1250,1410' --num-runs 7 --sleep-interval 2"

# 15. 🔬 PRECISION COMPARISON - Different model precisions
# LAUNCH_ARGS="--gpu-type A100 --profiling-mode baseline --app-name StableDiffusion --app-executable stable_diffusion --app-params '--precision fp16' --num-runs 5"

# 16. 📈 SCALING ANALYSIS - Batch size impact study
# LAUNCH_ARGS="--gpu-type A100 --profiling-mode custom --custom-frequencies '700,1000,1300' --app-name LSTM --app-executable ../app-lstm/lstm --app-params '--batch-size 128' --num-runs 5"

# 🚀 ADVANCED A100 CONFIGURATIONS
# ============================================================================

# 17. 🔥 HIGH THROUGHPUT - Multi-instance profiling
# LAUNCH_ARGS="--gpu-type A100 --profiling-mode baseline --app-name LSTM --app-executable ../app-lstm/lstm --app-params '--multi-instance 4' --num-runs 3"

# 18. 💡 TENSOR CORE OPTIMIZATION - Mixed precision workloads
# LAUNCH_ARGS="--gpu-type A100 --profiling-mode custom --custom-frequencies '900,1200,1410' --app-name StableDiffusion --app-params '--use-tensor-cores --precision mixed' --num-runs 5"

# ============================================================================
# TIMING GUIDELINES FOR SLURM --time PARAMETER
# ============================================================================
# Configuration 1-3:     --time=01:00:00  (1 hour)
# Configuration 4-7:     --time=02:00:00  (2 hours) 
# Configuration 8-9:     --time=05:00:00  (5 hours)
# Configuration 10:      --time=10:00:00  (10 hours)
# Configuration 11-16:   --time=03:00:00  (3 hours, adjust as needed)
# Configuration 17-18:   --time=04:00:00  (4 hours, advanced features)
#
# 💡 TIP: A100 has fewer frequencies (61) than V100 (117), so DVFS is faster
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
    log_header "🚀 Starting A100 GPU Profiling Job"
    log_info "Configuration: $LAUNCH_ARGS"
    
    # Load HPCC modules
    log_info "Loading HPCC modules..."
    module load gcc cuda cudnn
    
    # Determine and activate conda environment based on application
    local CONDA_ENV=$(determine_conda_env)
    log_info "Activating conda environment: $CONDA_ENV (auto-selected for application)"
    source "$HOME/conda/etc/profile.d/conda.sh"
    
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
                log_error "💡 To create tensorflow environment: conda env create -f ../app-lstm/lstm-a100-20250708.yml"
                ;;
        esac
        exit 1
    fi
    
    conda activate "$CONDA_ENV"
    
    # Display A100 system information
    display_a100_info
    
    # Validate configuration and provide warnings
    validate_configuration
    
    # Check system resources
    check_system_resources
    
    # Display GPU status
    check_gpu_status
    
    # Run the profiling experiment
    run_experiment
}

# Display A100 capabilities and system info
display_a100_info() {
    log_header "📊 A100 System Information"
    echo "┌─────────────────────────────────────────────────────────────┐"
    echo "│                    HPCC A100 Specifications                │"
    echo "├─────────────────────────────────────────────────────────────┤"
    echo "│ Cluster:      HPCC at Texas Tech University                │"
    echo "│ Partition:    toreador                                      │"
    echo "│ Architecture: Ampere (GA100)                                │"
    echo "│ Memory:       40GB HBM2e                                    │"
    echo "│ Mem Freq:     1215 MHz (fixed)                              │"
    echo "│ Core Freq:    510-1410 MHz (61 frequencies)                 │"
    echo "│ DVFS Step:    ~15 MHz typical                                │"
    echo "│ Features:     3rd Gen Tensor Cores, RT Cores               │"
    echo "│ Tools:        DCGMI (preferred) or nvidia-smi               │"
    echo "└─────────────────────────────────────────────────────────────┘"
}

# Validate configuration and provide appropriate warnings
validate_configuration() {
    log_header "🔍 Configuration Validation"
    
    # Check for DVFS mode and provide warnings
    if echo "$LAUNCH_ARGS" | grep -q "dvfs"; then
        log_warning "⚠️  DVFS mode detected - this will test ALL 61 A100 frequencies"
        log_warning "⚠️  Estimated runtime: 2-8 hours depending on runs per frequency"
        log_warning "⚠️  A100 DVFS is faster than V100 (61 vs 117 frequencies)"
        
        # Calculate estimated runtime
        if echo "$LAUNCH_ARGS" | grep -q "num-runs"; then
            runs=$(echo "$LAUNCH_ARGS" | sed -n 's/.*--num-runs \([0-9]\+\).*/\1/p')
            if [[ -n "$runs" && "$runs" -gt 0 ]]; then
                total_runs=$((61 * runs))
                estimated_hours=$((total_runs / 60))  # Rough estimate: 1 minute per run
                log_warning "⚠️  Estimated total runs: $total_runs"
                log_warning "⚠️  Estimated runtime: ~${estimated_hours} hours"
                
                # Recommend time adjustment
                if (( estimated_hours > 4 )); then
                    log_warning "⚠️  Consider adjusting SLURM --time to ${estimated_hours}:00:00 or higher"
                fi
            fi
        fi
        
        echo ""
        log_info "💡 For faster results, consider configuration #3 (frequency sampling)"
        log_info "💡 Example: custom --custom-frequencies '510,700,900,1100,1300,1410'"
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
                log_info "🎨 Stable Diffusion: A100 excels with Tensor Cores for mixed precision"
                ;;
            "LLaMA")
                log_info "📝 LLaMA: A100 40GB can handle 7B-30B models effectively"
                ;;
            "LSTM")
                log_info "🤖 LSTM: Lightweight benchmark, good for A100 validation"
                ;;
        esac
    fi
    
    # Check for advanced A100 features
    if echo "$LAUNCH_ARGS" | grep -q "tensor-cores"; then
        log_info "🔥 Tensor Cores detected: Optimized for A100 3rd Gen Tensor Cores"
    fi
    
    if echo "$LAUNCH_ARGS" | grep -q "multi-instance"; then
        log_info "🚀 Multi-Instance GPU (MIG) configuration detected"
        log_warning "⚠️  Ensure MIG partitioning is configured on the A100"
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
        log_warning "⚠️  A100 DVFS experiments generate substantial data"
    else
        log_info "✅ Available disk space: ${available_gb}GB"
    fi
    
    # Check if results directory exists
    if [[ ! -d "results" ]]; then
        log_info "📁 Creating results directory..."
        mkdir -p results
    fi
    
    # Check for A100-specific requirements
    log_info "🔧 Checking A100-specific requirements..."
    if echo "$LAUNCH_ARGS" | grep -q "multi-instance"; then
        log_info "📊 Multi-Instance GPU mode requires special configuration"
        log_info "💡 Verify MIG partitioning: nvidia-smi -mig"
    fi
}

# Check GPU status and availability
check_gpu_status() {
    log_header "🖥️  GPU Status Check"
    
    # Display GPU information
    if gpu_info=$(nvidia-smi --query-gpu=name,memory.total,driver_version,power.max_limit --format=csv,noheader,nounits 2>/dev/null); then
        log_info "📊 GPU Information: $gpu_info"
        
        # Verify it's actually an A100
        if echo "$gpu_info" | grep -qi "a100"; then
            log_info "✅ A100 GPU confirmed"
            
            # Check memory size to determine variant
            if echo "$gpu_info" | grep -q "40960"; then
                log_info "💾 A100 40GB variant detected"
            elif echo "$gpu_info" | grep -q "81920"; then
                log_info "💾 A100 80GB variant detected"
            fi
        else
            log_warning "⚠️  Expected A100 but detected: $gpu_info"
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
    
    # Check for Tensor Core availability
    if nvidia-smi --query-gpu=name --format=csv,noheader,nounits 2>/dev/null | grep -qi "a100"; then
        log_info "🔥 3rd Gen Tensor Cores available for mixed precision workloads"
    fi
}

# Run the main profiling experiment
run_experiment() {
    log_header "🚀 Starting A100 Profiling Experiment"
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
        
        log_header "🎉 A100 Profiling Completed Successfully!"
        log_info "⏱️  Total runtime: ${hours}h ${minutes}m"
        
        # Display results summary
        display_results_summary
        
        # Display completion notes
        display_completion_notes
        
    else
        # Failure path
        log_error "❌ A100 profiling experiment failed"
        log_error "🔍 Check the error logs above for details"
        
        # Common troubleshooting suggestions
        log_error ""
        log_error "🛠️  Common A100 Issues and Solutions:"
        log_error "   • Frequency control permissions → Try nvidia-smi fallback (config #11)"
        log_error "   • DCGMI tool unavailable → Automatic fallback should occur"
        log_error "   • Reservation access → Check HPCC reservation status"
        log_error "   • MIG configuration → Verify Multi-Instance GPU setup"
        log_error "   • Module loading issues → Check HPCC environment setup"
        
        exit 1
    fi
}

# Display comprehensive results summary
display_results_summary() {
    log_header "📊 Results Summary"
    
    if [[ -d "results" ]]; then
        local result_count
        result_count=$(find results -type f | wc -l)
        log_info "📁 Generated $result_count result files"
        
        # Show recent files
        if [[ "$result_count" -gt 0 ]]; then
            log_info "📋 Recent result files:"
            find results -type f -newer "$LAUNCH_SCRIPT" 2>/dev/null | head -5 | while read -r file; do
                if [[ -n "$file" ]]; then
                    local size
                    size=$(stat -f%z "$file" 2>/dev/null || stat -c%s "$file" 2>/dev/null || echo "unknown")
                    log_info "   📄 $(basename "$file") - ${size} bytes"
                fi
            done
        fi
        
        # Check for specific output files
        local csv_files
        csv_files=$(find results -name "GA100*.csv" 2>/dev/null)
        if [[ -n "$csv_files" ]]; then
            local csv_file
            csv_file=$(echo "$csv_files" | head -1)
            local csv_lines
            csv_lines=$(wc -l < "$csv_file" 2>/dev/null || echo "unknown")
            log_info "📊 Performance data points in $(basename "$csv_file"): $csv_lines"
        fi
        
        # Calculate total data size
        local total_size
        total_size=$(du -sh results 2>/dev/null | cut -f1 || echo "unknown")
        log_info "💾 Total results directory size: $total_size"
        
    else
        log_warning "⚠️  No results directory found"
    fi
}

# Display completion notes and next steps
display_completion_notes() {
    log_header "📝 A100 Profiling Completion Notes"
    
    echo "┌─────────────────────────────────────────────────────────────┐"
    echo "│                   Profiling Summary                        │"
    echo "├─────────────────────────────────────────────────────────────┤"
    echo "│ GPU:          A100 (Ampere GA100) - 40GB HBM2e             │"
    echo "│ Cluster:      HPCC toreador partition                      │"
    
    # Mode-specific notes
    if echo "$LAUNCH_ARGS" | grep -q "dvfs"; then
        echo "│ Mode:         DVFS (tested across 61 frequency range)      │"
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
    if echo "$LAUNCH_ARGS" | grep -q "tensor-cores"; then
        echo "│ Features:     3rd Gen Tensor Cores enabled                 │"
    fi
    
    echo "└─────────────────────────────────────────────────────────────┘"
    
    # Next steps
    log_info ""
    log_info "🎯 Next Steps:"
    log_info "   📊 Analyze results with power modeling framework:"
    log_info "      python -c \"from power_modeling import analyze_application; analyze_application('results/GA100*.csv')\""
    log_info "   📈 Run EDP optimization:"
    log_info "      python -c \"from edp_analysis import edp_calculator; edp_calculator.find_optimal_configuration('results/GA100*.csv')\""
    log_info "   🔄 Submit additional configurations by editing this script and resubmitting"
    log_info "   🔥 Try advanced A100 features: configs #17-18 (Tensor Cores, MIG)"
}

# Execute main function
main "$@"
