#!/bin/bash
"""
SLURM Job Submission Script for AI Inference Energy Profiling.

This script submits AI inference energy profiling experiments to a SLURM-managed
HPC cluster. It configures the necessary environment and launches the profiling
experiments using the launch.sh script.

SLURM Configuration:
    - Job name: AI_INFERENCE_ENERGY_PROFILING
    - Nodes: 1
    - Tasks per node: 16
    - GPUs per node: 1
    - Partition: toreador

Environment Setup:
    - Loads required modules (GCC, CUDA, cuDNN)
    - Activates conda environment for AI frameworks
    - Sets up GPU profiling tools

Usage:
    sbatch submit_job.sh

Author: AI Inference Energy Research Team
"""

#SBATCH --job-name=AI_INFERENCE_ENERGY_PROFILING
#SBATCH --output=%x.%j.out
#SBATCH --error=%x.%j.err
#SBATCH --partition=toreador
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=16
#SBATCH --gpus-per-node=1
#SBATCH --reservation=ghazanfar
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=mert.side@ttu.edu

# Enable strict error handling
set -euo pipefail

# ============================================================================
# Configuration
# ============================================================================

readonly SCRIPT_NAME="$(basename "$0")"
readonly JOB_START_TIME="$(date '+%Y-%m-%d %H:%M:%S')"

# Environment configuration
readonly CONDA_ENV="tensorflow"
readonly LAUNCH_SCRIPT="./launch.sh"

# ============================================================================
# Logging Functions
# ============================================================================

log_info() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] [INFO] $*"
}

log_error() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] [ERROR] $*" >&2
}

log_warning() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] [WARNING] $*" >&2
}

# ============================================================================
# Environment Setup Functions
# ============================================================================

# Function to display job information
display_job_info() {
    log_info "=== SLURM Job Information ==="
    log_info "Job ID: ${SLURM_JOB_ID:-'N/A'}"
    log_info "Job Name: ${SLURM_JOB_NAME:-'N/A'}"
    log_info "Node: ${SLURM_NODELIST:-'N/A'}"
    log_info "Partition: ${SLURM_JOB_PARTITION:-'N/A'}"
    log_info "CPUs per task: ${SLURM_CPUS_PER_TASK:-'N/A'}"
    log_info "GPUs per node: ${SLURM_GPUS_PER_NODE:-'N/A'}"
    log_info "Start time: $JOB_START_TIME"
    log_info "Working directory: $(pwd)"
    log_info "==============================="
}

# Function to load required modules
load_modules() {
    log_info "Loading required modules..."
    
    # Load GCC compiler
    if ! module load gcc; then
        log_error "Failed to load GCC module"
        return 1
    fi
    log_info "Loaded GCC module"
    
    # Load CUDA toolkit
    if ! module load cuda; then
        log_error "Failed to load CUDA module"
        return 1
    fi
    log_info "Loaded CUDA module"
    
    # Load cuDNN library
    if ! module load cudnn; then
        log_error "Failed to load cuDNN module"
        return 1
    fi
    log_info "Loaded cuDNN module"
    
    log_info "All modules loaded successfully"
    return 0
}

# Function to activate conda environment
activate_conda() {
    log_info "Setting up conda environment..."
    
    # Source conda initialization script
    if [[ -f "$HOME/conda/etc/profile.d/conda.sh" ]]; then
        # shellcheck source=/dev/null
        source "$HOME/conda/etc/profile.d/conda.sh"
        log_info "Sourced conda initialization script"
    else
        log_error "Conda initialization script not found: $HOME/conda/etc/profile.d/conda.sh"
        return 1
    fi
    
    # Activate the conda environment
    if conda activate "$CONDA_ENV"; then
        log_info "Activated conda environment: $CONDA_ENV"
    else
        log_error "Failed to activate conda environment: $CONDA_ENV"
        return 1
    fi
    
    # Display Python and package information
    log_info "Python version: $(python --version)"
    log_info "Python executable: $(which python)"
    
    # Check key packages
    local packages=("torch" "transformers" "diffusers")
    for package in "${packages[@]}"; do
        if python -c "import $package; print(f'$package: {$package.__version__}')" 2>/dev/null; then
            log_info "$package is available"
        else
            log_warning "$package is not available or not importable"
        fi
    done
    
    return 0
}

# Function to verify GPU availability
verify_gpu() {
    log_info "Verifying GPU availability..."
    
    # Check nvidia-smi
    if command -v nvidia-smi &> /dev/null; then
        log_info "NVIDIA GPU information:"
        nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader,nounits
    else
        log_error "nvidia-smi not available"
        return 1
    fi
    
    # Check DCGMI
    if command -v dcgmi &> /dev/null; then
        log_info "DCGMI version:"
        dcgmi --version
    else
        log_error "dcgmi not available"
        return 1
    fi
    
    # Check PyTorch CUDA
    if python -c "import torch; print(f'PyTorch CUDA available: {torch.cuda.is_available()}'); print(f'CUDA device count: {torch.cuda.device_count()}')"; then
        log_info "PyTorch CUDA verification completed"
    else
        log_warning "PyTorch CUDA verification failed"
    fi
    
    return 0
}

# Function to check script dependencies
check_dependencies() {
    log_info "Checking script dependencies..."
    
    # Check if launch script exists and is executable
    if [[ ! -f "$LAUNCH_SCRIPT" ]]; then
        log_error "Launch script not found: $LAUNCH_SCRIPT"
        return 1
    fi
    
    if [[ ! -x "$LAUNCH_SCRIPT" ]]; then
        log_error "Launch script is not executable: $LAUNCH_SCRIPT"
        return 1
    fi
    
    log_info "Launch script found and executable: $LAUNCH_SCRIPT"
    
    # Check other required scripts
    local required_scripts=("./profile" "./control" "./clean.sh")
    for script in "${required_scripts[@]}"; do
        if [[ -f "$script" && -x "$script" ]]; then
            log_info "Found: $script"
        else
            log_warning "Missing or not executable: $script"
        fi
    done
    
    return 0
}

# ============================================================================
# Main Execution Functions
# ============================================================================

# Function to run the profiling experiment
run_experiment() {
    log_info "Starting AI inference energy profiling experiment..."
    
    # Create results directory if it doesn't exist
    if [[ ! -d "results" ]]; then
        mkdir -p results
        log_info "Created results directory"
    fi
    
    # Record experiment start time
    local experiment_start
    experiment_start=$(date +%s)
    
    # Execute the launch script
    if "$LAUNCH_SCRIPT"; then
        local experiment_end
        experiment_end=$(date +%s)
        local experiment_duration=$((experiment_end - experiment_start))
        
        log_info "Experiment completed successfully"
        log_info "Total experiment duration: ${experiment_duration}s"
        
        # Display results summary
        if [[ -d "results" ]]; then
            local result_count
            result_count=$(find results -type f | wc -l)
            log_info "Generated $result_count result files"
        fi
        
        return 0
    else
        log_error "Experiment failed"
        return 1
    fi
}

# Function to cleanup on job completion
cleanup_job() {
    local exit_code=$?
    local job_end_time
    job_end_time="$(date '+%Y-%m-%d %H:%M:%S')"
    
    log_info "=== Job Completion Summary ==="
    log_info "Job ID: ${SLURM_JOB_ID:-'N/A'}"
    log_info "Start time: $JOB_START_TIME"
    log_info "End time: $job_end_time"
    log_info "Exit code: $exit_code"
    
    if (( exit_code == 0 )); then
        log_info "Job completed successfully"
    else
        log_error "Job failed with exit code: $exit_code"
    fi
    
    log_info "=============================="
    
    exit "$exit_code"
}

# ============================================================================
# Main Function
# ============================================================================

main() {
    # Set up signal handlers
    trap cleanup_job EXIT
    trap 'log_error "Job interrupted"; exit 130' INT TERM
    
    log_info "Starting SLURM job for AI inference energy profiling"
    
    # Display job information
    display_job_info
    
    # Load required modules
    if ! load_modules; then
        log_error "Failed to load required modules"
        exit 1
    fi
    
    # Activate conda environment
    if ! activate_conda; then
        log_error "Failed to activate conda environment"
        exit 1
    fi
    
    # Verify GPU availability
    if ! verify_gpu; then
        log_error "GPU verification failed"
        exit 1
    fi
    
    # Check script dependencies
    if ! check_dependencies; then
        log_error "Dependency check failed"
        exit 1
    fi
    
    # Run the profiling experiment
    if ! run_experiment; then
        log_error "Profiling experiment failed"
        exit 1
    fi
    
    log_info "SLURM job completed successfully"
}

# Execute main function
main "$@"
