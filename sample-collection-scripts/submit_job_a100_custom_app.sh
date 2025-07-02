#!/bin/bash
#
# Example SLURM Job Submission Script - A100 Custom Application Profiling
#
# This script demonstrates how to run custom applications on A100 GPUs with
# configurable profiling parameters. Modify the LAUNCH_ARGS to test different
# applications, frequencies, and configurations.
#

#SBATCH --job-name=CUSTOM_A100_PROFILING
#SBATCH --output=%x.%j.out
#SBATCH --error=%x.%j.err
#SBATCH --partition=toreador
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=16
#SBATCH --gpus-per-node=1
#SBATCH --reservation=ghazanfar
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=mert.side@ttu.edu
#SBATCH --time=03:00:00

# Enable strict error handling
set -euo pipefail

# Configuration
readonly CONDA_ENV="tensorflow"
readonly LAUNCH_SCRIPT="./launch.sh"

# Custom A100 application configuration
# Modify these parameters for your specific use case:
#
# Example configurations:
# 1. Custom DVFS study with 5 runs per frequency
# LAUNCH_ARGS="--gpu-type A100 --profiling-mode dvfs --num-runs 5 --sleep-interval 3"
#
# 2. Baseline profiling with custom application
# LAUNCH_ARGS="--gpu-type A100 --profiling-mode baseline --app-name CustomApp --app-executable my_app --app-params '--config config.json > results/custom_output.log'"
#
# 3. LLaMA model profiling
# LAUNCH_ARGS="--gpu-type A100 --profiling-mode baseline --app-name LLaMA --app-executable llama_inference --num-runs 5"
#
# 4. Stable Diffusion profiling
# LAUNCH_ARGS="--gpu-type A100 --profiling-mode baseline --app-name StableDiffusion --app-executable stable_diffusion --num-runs 3"

# Default: A100 baseline with extended runs for statistical significance
LAUNCH_ARGS="--gpu-type A100 --profiling-mode baseline --num-runs 5 --sleep-interval 2"

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
    log_info "Starting A100 custom application profiling job"
    log_info "Configuration: $LAUNCH_ARGS"
    
    # Load modules (HPCC-specific)
    module load gcc cuda cudnn
    
    # Activate conda
    source "$HOME/conda/etc/profile.d/conda.sh"
    conda activate "$CONDA_ENV"
    
    # Display A100 capabilities
    log_info "=== HPCC A100 Profiling Capabilities ==="
    log_info "Cluster: HPCC at Texas Tech University"
    log_info "Partition: toreador"
    log_info "Architecture: GA100 (Ampere)"
    log_info "Memory frequency: 1215 MHz"
    log_info "Core frequency range: 1410-510 MHz (61 frequencies)"
    log_info "Profiling tools: DCGMI (preferred) or nvidia-smi"
    log_info "=========================================="
    
    # Display GPU information
    log_info "=== GPU Detection ==="
    if nvidia-smi --query-gpu=name,memory.total,driver_version,power.max_limit --format=csv,noheader,nounits; then
        log_info "GPU information retrieved successfully"
    else
        log_warning "Could not query GPU info - continuing anyway"
    fi
    log_info "===================="
    
    # Run experiment
    log_info "Starting A100 profiling experiment..."
    if eval "$LAUNCH_SCRIPT $LAUNCH_ARGS"; then
        log_info "A100 custom application profiling completed successfully"
        
        # Display results summary
        log_info "=== Results Summary ==="
        if [[ -d "results" ]]; then
            log_info "Results directory contents:"
            ls -la results/ | head -10
            if [[ $(ls results/ | wc -l) -gt 10 ]]; then
                log_info "... and $(($(ls results/ | wc -l) - 10)) more files"
            fi
        fi
        log_info "======================"
        
    else
        log_error "A100 custom application profiling failed"
        log_error "Check the error logs above for details"
        exit 1
    fi
}

main "$@"
