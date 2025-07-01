#!/bin/bash
#
# Example SLURM Job Submission Script - Custom Application
#
# This script demonstrates how to run profiling experiments with
# custom applications (e.g., Stable Diffusion, LLaMA, etc.)
#

#SBATCH --job-name=LSTM_A100_CUSTOM
#SBATCH --output=%x.%j.o
#SBATCH --error=%x.%j.e
#SBATCH --partition=toreador
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=16
#SBATCH --gpus-per-node=1
#SBATCH --reservation=ghazanfar
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=mert.side@ttu.edu
#SBATCH --time=02:00:00

# Enable strict error handling
set -euo pipefail

# Configuration
readonly CONDA_ENV="tensorflow"
readonly LAUNCH_SCRIPT="./launch.sh"

# Custom application configuration
# Example: Stable Diffusion with baseline profiling for quick testing
LAUNCH_ARGS="--gpu-type A100 --profiling-mode baseline --num-runs 2 --app-name StableDiffusion --app-executable stable_diffusion --app-params '--prompt \"A beautiful landscape\" --steps 20 > results/SD_output.log'"

# Alternative examples (uncomment one):
# LLaMA inference
# LAUNCH_ARGS="--gpu-type A100 --profiling-mode baseline --app-name LLaMA --app-executable llama_inference --app-params '--model llama-7b --input test_prompt.txt > results/LLaMA_output.log'"

# Custom PyTorch model
# LAUNCH_ARGS="--gpu-type A100 --profiling-mode dvfs --num-runs 3 --app-name CustomModel --app-executable custom_inference --app-params '--model resnet50 --batch-size 8 > results/custom_output.log'"

# Logging function
log_info() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] [INFO] $*"
}

log_error() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] [ERROR] $*" >&2
}

# Main execution
main() {
    log_info "Starting custom application profiling job"
    log_info "Configuration: $LAUNCH_ARGS"
    
    # Load modules
    module load gcc cuda cudnn
    
    # Activate conda
    source "$HOME/conda/etc/profile.d/conda.sh"
    conda activate "$CONDA_ENV"
    
    # Verify application executable exists
    app_executable=$(echo "$LAUNCH_ARGS" | grep -o -- '--app-executable [^ ]*' | cut -d' ' -f2)
    if [[ -n "$app_executable" && ! -f "${app_executable}.py" ]]; then
        log_error "Application executable not found: ${app_executable}.py"
        log_error "Make sure your application script is present before running"
        exit 1
    fi
    
    # Run experiment
    if eval "$LAUNCH_SCRIPT $LAUNCH_ARGS"; then
        log_info "Custom application profiling completed successfully"
    else
        log_error "Custom application profiling failed"
        exit 1
    fi
}

main "$@"
