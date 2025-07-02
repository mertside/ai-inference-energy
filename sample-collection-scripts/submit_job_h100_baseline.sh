#!/bin/bash
#
# Example SLURM Job Submission Script - H100 Baseline Profiling
#
# This script demonstrates how to run baseline profiling on H100 GPUs.
# Baseline mode runs at default frequency only, making it faster and
# requiring no special frequency control permissions.
#

#SBATCH --job-name=LSTM_H100_BASELINE
#SBATCH --output=%x.%j.out
#SBATCH --error=%x.%j.err
#SBATCH --partition=h100-build
#SBATCH --nodelist=rpg-93-9
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=mert.side@ttu.edu
#SBATCH --time=01:00:00

# Enable strict error handling
set -euo pipefail

# Configuration
readonly CONDA_ENV="tensorflow"
readonly LAUNCH_SCRIPT="./launch.sh"

# H100 baseline configuration with fewer runs for quick testing
LAUNCH_ARGS="--gpu-type H100 --profiling-mode baseline --num-runs 3 --sleep-interval 1"

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
    log_info "Starting H100 baseline profiling job"
    log_info "Configuration: $LAUNCH_ARGS"
    log_info "=== REPACSS H100 Node Information ==="
    log_info "Cluster: REPACSS at Texas Tech University"
    log_info "Partition: h100-build"
    log_info "Node: rpg-93-9"
    log_info "Architecture: GH100"
    log_info "Memory frequency: 1593 MHz"
    log_info "Default core frequency: 1755 MHz"
    log_info "Baseline mode: Single frequency test"
    log_info "============================"
    
    # Load modules (REPACSS-specific)
    module load cuda
    
    # Activate conda
    source "$HOME/conda/etc/profile.d/conda.sh"
    conda activate "$CONDA_ENV"
    
    # Display GPU information
    log_info "=== GPU Detection ==="
    nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader,nounits || log_warning "Could not query GPU info"
    log_info "===================="
    
    # Run experiment
    if eval "$LAUNCH_SCRIPT $LAUNCH_ARGS"; then
        log_info "H100 baseline profiling completed successfully"
    else
        log_error "H100 baseline profiling failed"
        exit 1
    fi
}

main "$@"
