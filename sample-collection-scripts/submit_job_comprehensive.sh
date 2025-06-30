#!/bin/bash
#
# Example SLURM Job Submission Script - Comprehensive DVFS Study
#
# This script demonstrates how to run comprehensive DVFS experiments
# with multiple runs across all frequencies for detailed energy analysis.
#

#SBATCH --job-name=AI_ENERGY_COMPREHENSIVE_DVFS
#SBATCH --output=%x.%j.out
#SBATCH --error=%x.%j.err
#SBATCH --partition=toreador
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=16
#SBATCH --gpus-per-node=1
#SBATCH --reservation=ghazanfar
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=mert.side@ttu.edu
#SBATCH --time=08:00:00  # Longer time for comprehensive study

# Enable strict error handling
set -euo pipefail

# Configuration
readonly CONDA_ENV="tensorflow"
readonly LAUNCH_SCRIPT="./launch.sh"

# Comprehensive DVFS configuration
# This will test all A100 frequencies (61 frequencies) with 5 runs each
# Total runs: 61 * 5 = 305 runs (estimated 6-8 hours)
LAUNCH_ARGS="--gpu-type A100 --profiling-tool dcgmi --profiling-mode dvfs --num-runs 5 --sleep-interval 2 --app-name LSTM --app-executable lstm --app-params '> results/LSTM_comprehensive_output.log'"

# Alternative comprehensive configurations (uncomment one):
# V100 comprehensive study
# LAUNCH_ARGS="--gpu-type V100 --profiling-mode dvfs --num-runs 3 --sleep-interval 2"

# A100 with nvidia-smi (if DCGMI not available)
# LAUNCH_ARGS="--gpu-type A100 --profiling-tool nvidia-smi --profiling-mode dvfs --num-runs 3"

# Logging function
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
    log_info "Starting comprehensive DVFS profiling job"
    log_info "Configuration: $LAUNCH_ARGS"
    log_warning "This is a long-running job (estimated 6-8 hours)"
    log_warning "Make sure you have sufficient time allocation"
    
    # Load modules
    module load gcc cuda cudnn
    
    # Activate conda
    source "$HOME/conda/etc/profile.d/conda.sh"
    conda activate "$CONDA_ENV"
    
    # Display estimated runtime
    log_info "=== Estimated Runtime ==="
    log_info "GPU: A100 (61 frequencies)"
    log_info "Runs per frequency: 5"
    log_info "Total runs: ~305"
    log_info "Estimated time: 6-8 hours"
    log_info "========================="
    
    # Verify sufficient disk space
    available_space=$(df . | awk 'NR==2 {print $4}')
    if (( available_space < 1000000 )); then  # Less than 1GB
        log_warning "Available disk space may be insufficient for comprehensive study"
        log_warning "Available: ${available_space}KB, Recommended: >1GB"
    fi
    
    # Run experiment
    local start_time
    start_time=$(date +%s)
    
    if eval "$LAUNCH_SCRIPT $LAUNCH_ARGS"; then
        local end_time
        end_time=$(date +%s)
        local total_time=$((end_time - start_time))
        local hours=$((total_time / 3600))
        local minutes=$(((total_time % 3600) / 60))
        
        log_info "Comprehensive DVFS profiling completed successfully"
        log_info "Total runtime: ${hours}h ${minutes}m"
        
        # Display results summary
        if [[ -d "results" ]]; then
            local result_count
            result_count=$(find results -type f | wc -l)
            log_info "Generated $result_count result files"
            
            # Check for performance CSV
            if [[ -f "results/GA100-dvfs-lstm-perf.csv" ]]; then
                local csv_lines
                csv_lines=$(wc -l < "results/GA100-dvfs-lstm-perf.csv")
                log_info "Performance data points: $csv_lines"
            fi
        fi
    else
        log_error "Comprehensive DVFS profiling failed"
        exit 1
    fi
}

main "$@"
