#!/bin/bash
#
# Example SLURM Job Submission Script - V100 Comprehensive DVFS Study
#
# This script demonstrates how to run comprehensive DVFS experiments
# on V100 GPUs with multiple runs across all frequencies.
#

#SBATCH --job-name=LSTM_V100_COMPREHENSIVE
#SBATCH --output=%x.%j.o
#SBATCH --error=%x.%j.e
#SBATCH --partition=matador
#SBATCH --nodes=1
#SBATCH --gres=gpu:v100:1
#SBATCH --ntasks=40
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=mert.side@ttu.edu
#SBATCH --time=10:00:00  # Longer time for comprehensive study (V100 has more frequencies)

# Enable strict error handling
set -euo pipefail

# Configuration
readonly CONDA_ENV="tensorflow"
readonly LAUNCH_SCRIPT="./launch.sh"

# Comprehensive DVFS configuration for V100
# This will test all V100 frequencies (105 frequencies) with 3 runs each
# Total runs: 105 * 3 = 315 runs (estimated 8-10 hours)
LAUNCH_ARGS="--gpu-type V100 --profiling-tool dcgmi --profiling-mode dvfs --num-runs 3 --sleep-interval 2 --app-name LSTM --app-executable lstm --app-params '> results/LSTM_V100_comprehensive_output.log'"

# Alternative configurations (uncomment one):
# V100 with nvidia-smi (if DCGMI not available)
# LAUNCH_ARGS="--gpu-type V100 --profiling-tool nvidia-smi --profiling-mode dvfs --num-runs 3"

# V100 baseline for testing
# LAUNCH_ARGS="--gpu-type V100 --profiling-mode baseline --num-runs 2"

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
    log_info "Starting V100 comprehensive DVFS profiling job"
    log_info "Configuration: $LAUNCH_ARGS"
    log_warning "This is a long-running job (estimated 8-10 hours)"
    log_warning "V100 has 105 frequencies vs A100's 61 frequencies"
    
    # Load modules
    module load gcc cuda cudnn
    
    # Activate conda
    source "$HOME/conda/etc/profile.d/conda.sh"
    conda activate "$CONDA_ENV"
    
    # Display estimated runtime
    log_info "=== Estimated Runtime ==="
    log_info "GPU: V100 (105 frequencies)"
    log_info "Runs per frequency: 3"
    log_info "Total runs: ~315"
    log_info "Estimated time: 8-10 hours"
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
        
        log_info "V100 comprehensive DVFS profiling completed successfully"
        log_info "Total runtime: ${hours}h ${minutes}m"
        
        # Display results summary
        if [[ -d "results" ]]; then
            local result_count
            result_count=$(find results -type f | wc -l)
            log_info "Generated $result_count result files"
            
            # Check for performance CSV
            if [[ -f "results/GV100-dvfs-lstm-perf.csv" ]]; then
                local csv_lines
                csv_lines=$(wc -l < "results/GV100-dvfs-lstm-perf.csv")
                log_info "Performance data points: $csv_lines"
            fi
        fi
    else
        log_error "V100 comprehensive DVFS profiling failed"
        exit 1
    fi
}

main "$@"
