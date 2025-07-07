#!/bin/bash
#
# Example SLURM Job Submission Script - H100 Comprehensive DVFS Study
#
# This script demonstrates how to run comprehensive DVFS experiments
# on H100 GPUs with multiple runs across all frequencies.
#
# WARNING: This is a very long-running job (estimated 12+ hours)
# H100 has 86 frequencies from 1785MHz down to 510MHz in 15MHz steps
#

#SBATCH --job-name=LSTM_H100_COMPREHENSIVE
#SBATCH --output=%x.%j.out
#SBATCH --error=%x.%j.err
#SBATCH --partition=h100-build
#SBATCH --nodelist=rpg-93-9
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=mert.side@ttu.edu
#SBATCH --time=12:00:00  # Longer time for comprehensive H100 study

# Enable strict error handling
set -euo pipefail

# Configuration
readonly CONDA_ENV="tensorflow"
readonly LAUNCH_SCRIPT="./launch.sh"

# Comprehensive DVFS configuration for H100
# This will test all H100 frequencies (86 frequencies) with 3 runs each
LAUNCH_ARGS="--gpu-type H100 --profiling-mode dvfs --num-runs 3 --sleep-interval 2"

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
    log_info "Starting H100 comprehensive DVFS profiling job"
    log_info "Configuration: $LAUNCH_ARGS"
    log_warning "This is a very long-running job (estimated 12+ hours)"
    log_warning "H100 has 86 frequencies vs A100's 61 and V100's 117 frequencies"
    
    # Load modules (REPACSS-specific)
    # module load cuda
    
    # Activate conda
    source "$HOME/conda/etc/profile.d/conda.sh"
    conda activate "$CONDA_ENV"
    
    # Display estimated runtime
    log_info "=== Estimated Runtime ==="
    log_info "GPU: H100 (86 frequencies: 1785-510 MHz in 15MHz steps)"
    log_info "Runs per frequency: 3"
    log_info "Total runs: ~258"
    log_info "Estimated time: 10-12 hours"
    log_info "========================="
    
    # Display GPU information
    log_info "=== REPACSS H100 Node Information ==="
    log_info "Cluster: REPACSS at Texas Tech University"
    log_info "Partition: h100-build"
    log_info "Node: rpg-93-9"
    log_info "Architecture: GH100"
    log_info "Memory frequency: 1593 MHz"
    log_info "Max core frequency: 1785 MHz"
    log_info "Min core frequency: 510 MHz"
    log_info "Frequency step: 15 MHz"
    log_info "Total frequencies: 86"
    nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader,nounits || log_warning "Could not query GPU info"
    log_info "============================="
    
    # Verify sufficient disk space
    log_info "=== Disk Space Check ==="
    df -h . | tail -1 | awk '{print "Available space: " $4 " (Total: " $2 ")"}'
    log_info "Estimated space needed: ~5-10 GB for full H100 DVFS study"
    log_info "========================"
    
    # Final confirmation warning
    log_warning "Starting comprehensive H100 DVFS experiment"
    log_warning "This will test 86 different frequencies with 3 runs each"
    log_warning "Total estimated runtime: 10-12 hours"
    
    # Run experiment
    if eval "$LAUNCH_SCRIPT $LAUNCH_ARGS"; then
        log_info "H100 comprehensive DVFS profiling completed successfully"
        log_info "Results can be found in the results/ directory"
        log_info "Performance summary: results/GH100-dvfs-lstm-perf.csv"
    else
        log_error "H100 comprehensive DVFS profiling failed"
        exit 1
    fi
}

main "$@"
