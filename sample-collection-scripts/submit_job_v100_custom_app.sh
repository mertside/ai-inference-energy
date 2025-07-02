#!/bin/bash
#
# Example SLURM Job Submission Script - V100 Custom Application Profiling
#
# This script demonstrates how to run custom applications on V100 GPUs with
# configurable profiling parameters. Modify the LAUNCH_ARGS to test different
# applications, frequencies, and configurations.
#

#SBATCH --job-name=CUSTOM_V100_PROFILING
#SBATCH --output=%x.%j.out
#SBATCH --error=%x.%j.err
#SBATCH --partition=matador
#SBATCH --nodes=1
#SBATCH --gres=gpu:v100:1
#SBATCH --ntasks=40
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=mert.side@ttu.edu
#SBATCH --time=03:00:00

# Enable strict error handling
set -euo pipefail

# Configuration
readonly CONDA_ENV="tensorflow"
readonly LAUNCH_SCRIPT="./launch.sh"

# Custom V100 application configuration
# Modify these parameters for your specific use case:
#
# Example configurations:
# 1. Custom DVFS study with 5 runs per frequency (LONG RUNTIME - V100 has 137 frequencies)
# LAUNCH_ARGS="--gpu-type V100 --profiling-mode dvfs --num-runs 5 --sleep-interval 3"
#
# 2. Baseline profiling with custom application
# LAUNCH_ARGS="--gpu-type V100 --profiling-mode baseline --app-name CustomApp --app-executable my_app --app-params '--config config.json > results/custom_output.log'"
#
# 3. LLaMA model profiling
# LAUNCH_ARGS="--gpu-type V100 --profiling-mode baseline --app-name LLaMA --app-executable llama_inference --num-runs 5"
#
# 4. Stable Diffusion profiling
# LAUNCH_ARGS="--gpu-type V100 --profiling-mode baseline --app-name StableDiffusion --app-executable stable_diffusion --num-runs 3"
#
# 5. Custom frequency selection (recommended for V100 due to large frequency range)
# LAUNCH_ARGS="--gpu-type V100 --profiling-mode custom --custom-frequencies '405,840,1200,1380' --num-runs 7"
#
# 6. Testing with nvidia-smi (fallback profiling tool)
# LAUNCH_ARGS="--gpu-type V100 --profiling-tool nvidia-smi --profiling-mode baseline --num-runs 3"

# Default: V100 baseline with extended runs for statistical significance
LAUNCH_ARGS="--gpu-type V100 --profiling-mode baseline --num-runs 5 --sleep-interval 2"

# Logging functions
log_info() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] [INFO] $*"
}

log_error() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] [ERROR] $*" >&2
}

log_warning() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] [WARNING] $*" >&2
}

# Main execution
main() {
    log_info "Starting V100 custom application profiling job"
    log_info "Configuration: $LAUNCH_ARGS"
    
    # Load modules (HPCC-specific)
    module load gcc cuda cudnn
    
    # Activate conda
    source "$HOME/conda/etc/profile.d/conda.sh"
    conda activate "$CONDA_ENV"
    
    # Display V100 capabilities
    log_info "=== HPCC V100 Profiling Capabilities ==="
    log_info "Cluster: HPCC at Texas Tech University"
    log_info "Partition: matador"
    log_info "Architecture: GV100 (Volta)"
    log_info "Memory: 32GB HBM2"
    log_info "Memory frequency: 877 MHz (fixed)"
    log_info "Core frequency range: 405-1380 MHz (137 frequencies)"
    log_info "DVFS step size: Variable (7-8 MHz typical)"
    log_info "Profiling tools: DCGMI (preferred) or nvidia-smi"
    log_info "=========================================="
    
    # V100-specific warnings
    if echo "$LAUNCH_ARGS" | grep -q "dvfs"; then
        log_warning "⚠ V100 DVFS mode selected - this will test 137 frequencies!"
        log_warning "⚠ Estimated runtime for DVFS: 6-10 hours depending on runs per frequency"
        log_warning "⚠ Consider using 'custom' mode with selected frequencies for faster results"
        
        # Calculate estimated runtime
        if echo "$LAUNCH_ARGS" | grep -q "num-runs"; then
            runs=$(echo "$LAUNCH_ARGS" | sed -n 's/.*--num-runs \([0-9]\+\).*/\1/p')
            if [[ -n "$runs" ]]; then
                total_runs=$((137 * runs))
                estimated_hours=$((total_runs / 60))  # Rough estimate: 1 minute per run
                log_warning "⚠ Estimated total runs: $total_runs"
                log_warning "⚠ Estimated runtime: ~${estimated_hours} hours"
            fi
        fi
    fi
    
    # Display GPU information
    log_info "=== GPU Detection ==="
    if gpu_info=$(nvidia-smi --query-gpu=name,memory.total,driver_version,power.max_limit --format=csv,noheader,nounits); then
        log_info "GPU information: $gpu_info"
        
        # Verify it's actually a V100
        if echo "$gpu_info" | grep -qi "v100"; then
            log_info "✓ V100 GPU confirmed"
        else
            log_warning "⚠ Expected V100 but detected: $gpu_info"
        fi
    else
        log_warning "Could not query GPU info - continuing anyway"
    fi
    log_info "===================="
    
    # Check frequency control availability
    log_info "=== Frequency Control Check ==="
    if nvidia-smi -q -d SUPPORTED_CLOCKS &>/dev/null; then
        freq_count=$(nvidia-smi -q -d SUPPORTED_CLOCKS | grep -c "Graphics" || echo "0")
        log_info "✓ Frequency control available with $freq_count frequency options"
    else
        log_warning "⚠ Frequency control may not be available"
        log_info "Framework will attempt DCGMI first, then fallback to nvidia-smi monitoring"
    fi
    log_info "=============================="
    
    # Verify sufficient disk space for V100 experiments
    available_space=$(df . | awk 'NR==2 {print $4}')
    if (( available_space < 500000 )); then  # Less than 500MB
        log_warning "⚠ Available disk space may be insufficient: ${available_space}KB"
        log_warning "⚠ V100 experiments can generate substantial data, especially DVFS mode"
        log_warning "⚠ Recommended: >1GB free space for comprehensive studies"
    else
        log_info "✓ Sufficient disk space available: ${available_space}KB"
    fi
    
    # Run experiment
    log_info "Starting V100 profiling experiment..."
    log_info "Launch command: $LAUNCH_SCRIPT $LAUNCH_ARGS"
    
    local start_time
    start_time=$(date +%s)
    
    if eval "$LAUNCH_SCRIPT $LAUNCH_ARGS"; then
        local end_time
        end_time=$(date +%s)
        local total_time=$((end_time - start_time))
        local hours=$((total_time / 3600))
        local minutes=$(((total_time % 3600) / 60))
        
        log_info "V100 custom application profiling completed successfully"
        log_info "Total runtime: ${hours}h ${minutes}m"
        
        # Display results summary
        log_info "=== Results Summary ==="
        if [[ -d "results" ]]; then
            local result_count
            result_count=$(find results -type f | wc -l)
            log_info "Generated $result_count result files"
            
            # Show recent files
            log_info "Recent result files:"
            find results -type f -newer "$LAUNCH_SCRIPT" | head -5 | while read -r file; do
                size=$(stat -f%z "$file" 2>/dev/null || echo "unknown")
                log_info "  $(basename "$file") - ${size} bytes"
            done
            
            # Check for specific output files
            if [[ -f "results/GV100"*".csv" ]]; then
                csv_file=$(find results -name "GV100*.csv" | head -1)
                if [[ -n "$csv_file" ]]; then
                    csv_lines=$(wc -l < "$csv_file")
                    log_info "Performance data points in $(basename "$csv_file"): $csv_lines"
                fi
            fi
            
            # Calculate total data size
            total_size=$(du -sh results 2>/dev/null | cut -f1)
            log_info "Total results directory size: $total_size"
        else
            log_warning "No results directory found"
        fi
        log_info "======================"
        
        # V100-specific completion notes
        log_info "=== V100 Profiling Notes ==="
        log_info "V100 GPU profiling completed on HPCC matador partition"
        log_info "Architecture: Volta (GV100) with 32GB HBM2"
        if echo "$LAUNCH_ARGS" | grep -q "dvfs"; then
            log_info "DVFS mode: Tested across V100's 137 frequency range"
        elif echo "$LAUNCH_ARGS" | grep -q "custom"; then
            log_info "Custom mode: Tested selected frequencies for targeted analysis"
        else
            log_info "Baseline mode: Single frequency profiling completed"
        fi
        log_info "============================="
        
    else
        log_error "V100 custom application profiling failed"
        log_error "Check the error logs above for details"
        log_error "Common V100 issues:"
        log_error "  - Frequency control permissions on matador partition"
        log_error "  - DCGMI tool availability (falls back to nvidia-smi)"
        log_error "  - Memory limitations with 32GB V100 vs 80GB A100"
        log_error "  - Long DVFS runtimes (137 frequencies vs A100's 7)"
        exit 1
    fi
}

main "$@"
