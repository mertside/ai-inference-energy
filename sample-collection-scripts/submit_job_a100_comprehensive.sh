#!/bin/bash
#
# Example SLURM Job Submission Script - A100 Comprehensive DVFS Study
#
# This script demonstrates how to run comprehensive DVFS experiments
# on A100 GPUs with multiple runs across all frequencies.
#
# Note: A100 has 61 frequencies from 1410MHz down to 510MHz
# This is a long-running job (estimated 6-8 hours)
#

#SBATCH --job-name=LSTM_A100_COMPREHENSIVE
#SBATCH --output=%x.%j.out
#SBATCH --error=%x.%j.err
#SBATCH --partition=toreador
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=16
#SBATCH --gpus-per-node=1
#SBATCH --reservation=ghazanfar
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=mert.side@ttu.edu
#SBATCH --time=08:00:00  # Longer time for comprehensive A100 study

# Enable strict error handling
set -euo pipefail

# Configuration
readonly CONDA_ENV="tensorflow"
readonly LAUNCH_SCRIPT="./launch.sh"

# Comprehensive DVFS configuration for A100
# This will test all A100 frequencies (61 frequencies) with 3 runs each
LAUNCH_ARGS="--gpu-type A100 --profiling-mode dvfs --num-runs 3 --sleep-interval 2"

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
    log_info "Starting A100 comprehensive DVFS profiling job"
    log_info "Configuration: $LAUNCH_ARGS"
    log_warning "This is a long-running job (estimated 6-8 hours)"
    log_warning "A100 has 61 frequencies vs V100's 103 and H100's 104 frequencies"
    
    # Load modules (HPCC-specific)
    module load gcc cuda cudnn
    
    # Activate conda
    source "$HOME/conda/etc/profile.d/conda.sh"
    conda activate "$CONDA_ENV"
    
    # Display estimated runtime
    log_info "=== Estimated Runtime ==="
    log_info "GPU: A100 (61 frequencies: 1410-510 MHz)"
    log_info "Runs per frequency: 3"
    log_info "Total runs: ~183"
    log_info "Estimated time: 6-8 hours"
    log_info "========================="
    
    # Display GPU information
    log_info "=== HPCC A100 Node Information ==="
    log_info "Cluster: HPCC at Texas Tech University"
    log_info "Partition: toreador"
    log_info "Architecture: GA100"
    log_info "Memory frequency: 1215 MHz"
    log_info "Max core frequency: 1410 MHz"
    log_info "Min core frequency: 510 MHz"
    log_info "Total frequencies: 61"
    nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader,nounits || log_warning "Could not query GPU info"
    log_info "============================="
    
    # Verify sufficient disk space
    log_info "=== Disk Space Check ==="
    df -h . | tail -1 | awk '{print "Available space: " $4 " (Total: " $2 ")"}'
    log_info "Estimated space needed: ~3-5 GB for full A100 DVFS study"
    log_info "========================"
    
    # Final confirmation warning
    log_warning "Starting comprehensive A100 DVFS experiment"
    log_warning "This will test 61 different frequencies with 3 runs each"
    log_warning "Total estimated runtime: 6-8 hours"
    
    # Run experiment
    if eval "$LAUNCH_SCRIPT $LAUNCH_ARGS"; then
        log_info "A100 comprehensive DVFS profiling completed successfully"
        log_info "Results can be found in the results/ directory"
        log_info "Performance summary: results/GA100-dvfs-lstm-perf.csv"
    else
        log_error "A100 comprehensive DVFS profiling failed"
        exit 1
    fi
}

main "$@"
