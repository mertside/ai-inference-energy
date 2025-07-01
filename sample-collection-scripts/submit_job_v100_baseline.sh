#!/bin/bash
#
# Example SLURM Job Submission Script - V100 Baseline Profiling
#
# This script demonstrates how to run baseline profiling on V100 GPUs.
# Baseline mode runs at default frequency only, making it faster and
# requiring no special frequency control permissions.
#

#SBATCH --job-name=LSTM_V100_DATA
#SBATCH --output=%x.%j.o
#SBATCH --error=%x.%j.e
#SBATCH --partition=matador
#SBATCH --nodes=1
#SBATCH --gres=gpu:v100:1
#SBATCH --ntasks=40
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=mert.side@ttu.edu
#SBATCH --time=01:00:00

# Enable strict error handling
set -euo pipefail

# Configuration
readonly CONDA_ENV="tensorflow"
readonly LAUNCH_SCRIPT="./launch.sh"

# V100 baseline configuration with fewer runs for quick testing
LAUNCH_ARGS="--gpu-type V100 --profiling-mode baseline --num-runs 3 --sleep-interval 1"

# Logging function
log_info() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] [INFO] $*"
}

log_error() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] [ERROR] $*" >&2
}

# Main execution
main() {
    log_info "Starting V100 baseline profiling job"
    log_info "Configuration: $LAUNCH_ARGS"
    
    # Load modules
    module load gcc cuda cudnn
    
    # Activate conda
    source "$HOME/conda/etc/profile.d/conda.sh"
    conda activate "$CONDA_ENV"
    
    # Run experiment
    if eval "$LAUNCH_SCRIPT $LAUNCH_ARGS"; then
        log_info "V100 baseline profiling completed successfully"
    else
        log_error "V100 baseline profiling failed"
        exit 1
    fi
}

main "$@"
