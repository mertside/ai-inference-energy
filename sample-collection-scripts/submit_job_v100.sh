#!/bin/bash
#
# Unified SLURM Job Submission Script - V100 GPU Profiling
#
# This script provides a comprehensive set of V100 profiling configurations.
# Simply uncomment the desired configuration and submit with: sbatch submit_job_v100.sh
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
#SBATCH --gres=gpu:v100:1
#SBATCH --ntasks=40
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

# 1. 🚀 QUICK TEST - Baseline profiling (fastest, ~? minutes) - PyTorch LSTM
# LAUNCH_ARGS="--gpu-type V100 --profiling-mode baseline --num-runs 3 --sleep-interval 1 --app-name LSTM --app-executable ../app-lstm/lstm"

# 2. 🔬 RESEARCH BASELINE - Extended baseline for statistical significance (~15-20 minutes)
# LAUNCH_ARGS="--gpu-type V100 --profiling-mode baseline --num-runs 5 --sleep-interval 2"

# 3. 🎯 FREQUENCY SAMPLING - Extended baseline for comparative analysis (~30-45 minutes)
# LAUNCH_ARGS="--gpu-type V100 --profiling-mode dvfs --num-runs 3 --sleep-interval 2"

# 📊 AI APPLICATION CONFIGURATIONS
# ============================================================================

# 4. 🤖 LSTM PROFILING - Default sentiment analysis benchmark
# LAUNCH_ARGS="--gpu-type V100 --profiling-mode baseline --app-name LSTM --app-executable ../app-lstm/lstm --num-runs 5"

# 5. 🎨 STABLE DIFFUSION - Image generation profiling (1000 steps, 768x768, astronaut riding horse)
# LAUNCH_ARGS="--gpu-type V100 --profiling-mode baseline --app-name StableDiffusion --app-executable ../app-stable-diffusion/StableDiffusionViaHF.py --app-params '--prompt \"a photograph of an astronaut riding a horse\" --steps 500 --log-level INFO' --num-runs 3 --sleep-interval 1"

# 6. 📝 LLAMA - Text generation profiling with benchmark suite  
# LAUNCH_ARGS="--gpu-type V100 --profiling-mode baseline --app-name LLaMA --app-executable ../app-llama/LlamaViaHF.py --app-params '--benchmark --num-generations 3 --quiet --metrics' --num-runs 5"

# 7. 🔧 CUSTOM APPLICATION - Template for your own applications
# LAUNCH_ARGS="--gpu-type V100 --profiling-mode baseline --app-name CustomApp --app-executable my_app --app-params '--config config.json > results/custom_output.log' --num-runs 3"

# 🎯 TARGETED DVFS APPLICATION STUDIES
# ============================================================================
# NOTE: Framework v2.0.1 now supports 'custom' mode for targeted frequency analysis.
# Use --custom-frequencies to specify exact frequencies to test.

# 8. 🤖 LSTM CUSTOM - Three-point frequency analysis (low/mid/high, ~20-30 minutes)
LAUNCH_ARGS="--gpu-type V100 --profiling-mode custom --custom-frequencies '510,960,1380' --app-name LSTM --app-executable ../app-lstm/lstm --num-runs 5 --sleep-interval 2"

# 9. 🎨 STABLE DIFFUSION CUSTOM - Three-point frequency analysis (low/mid/high, ~40-60 minutes)
# LAUNCH_ARGS="--gpu-type V100 --profiling-mode custom --custom-frequencies '510,960,1380' --app-name StableDiffusion --app-executable ../app-stable-diffusion/StableDiffusionViaHF.py --app-params '--prompt \"a photograph of an astronaut riding a horse\" --steps 500 --log-level INFO' --num-runs 5 --sleep-interval 2"

# 🔄 DVFS STUDY CONFIGURATIONS
# ============================================================================

# 10. ⚡ COMPREHENSIVE DVFS - All 117 frequencies (⚠️ LONG: 6-12 hours, change --time to 12:00:00)
# LAUNCH_ARGS="--gpu-type V100 --profiling-mode dvfs --num-runs 3 --sleep-interval 2"

# 11. 🎯 EFFICIENT DVFS - Reduced runs for faster completion (~4-6 hours, change --time to 08:00:00)
# LAUNCH_ARGS="--gpu-type V100 --profiling-mode dvfs --num-runs 2 --sleep-interval 1"

# 12. 📈 STATISTICAL DVFS - High statistical power (⚠️ VERY LONG: 12-20 hours, change --time to 24:00:00)
# LAUNCH_ARGS="--gpu-type V100 --profiling-mode dvfs --num-runs 5 --sleep-interval 3"

# 🛠️ TOOL AND COMPATIBILITY CONFIGURATIONS  
# ============================================================================

# 13. 🔧 NVIDIA-SMI FALLBACK - When DCGMI is not available
# LAUNCH_ARGS="--gpu-type V100 --profiling-tool nvidia-smi --profiling-mode baseline --num-runs 3"

# 14. 🐛 DEBUG MODE - Minimal configuration for troubleshooting
# LAUNCH_ARGS="--gpu-type V100 --profiling-mode baseline --num-runs 1 --sleep-interval 0"

# 15. 💾 MEMORY STRESS TEST - Large model testing with benchmark (V100 has 32GB)
# LAUNCH_ARGS="--gpu-type V100 --profiling-mode baseline --app-name LLaMA --app-executable ../app-llama/LlamaViaHF.py --app-params '--model llama-13b --benchmark --num-generations 3 --quiet --metrics' --num-runs 3"

# 🎓 RESEARCH STUDY CONFIGURATIONS
# ============================================================================

# 16. 📊 ENERGY EFFICIENCY STUDY - Focus on power vs performance (Use DVFS mode)
# LAUNCH_ARGS="--gpu-type V100 --profiling-mode dvfs --num-runs 3 --sleep-interval 2"

# 17. 🔬 PRECISION COMPARISON - Different model precisions
# LAUNCH_ARGS="--gpu-type V100 --profiling-mode baseline --app-name StableDiffusion --app-executable stable_diffusion --app-params '--precision fp16' --num-runs 5"

# 18. 📈 SCALING ANALYSIS - Batch size impact study (Use baseline for quick comparison)
# LAUNCH_ARGS="--gpu-type V100 --profiling-mode baseline --app-name LSTM --app-executable ../app-lstm/lstm --app-params '--batch-size 64' --num-runs 5"

# ============================================================================
# TIMING GUIDELINES FOR SLURM --time PARAMETER
# ============================================================================
# Configuration 1-3:     --time=01:00:00  (1 hour)
# Configuration 4-7:     --time=02:00:00  (2 hours) 
# Configuration 8-9:     --time=02:00:00  (2 hours) - Custom frequency mode with 3 frequencies
# Configuration 10-11:   --time=08:00:00  (8 hours)
# Configuration 12:      --time=24:00:00  (24 hours)
# Configuration 13-18:   --time=03:00:00  (3 hours, adjust as needed)
#
# 💡 TIP: For DVFS studies (8-12), consider running during off-peak hours
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
    log_info "      python -c \"from power_modeling import analyze_application; analyze_application('results/GV100*.csv')\""
    log_info "   📈 Run EDP optimization:"
    log_info "      python -c \"from edp_analysis import edp_calculator; edp_calculator.find_optimal_configuration('results/GV100*.csv')\""
    log_info "   🔄 Submit additional configurations by editing this script and resubmitting"
}

# Execute main function
main "$@"
