#!/bin/bash
#
# SLURM Job Submission Script for AI Inference Energy Profiling.
#
# This script submits AI inference energy profiling experiments to a SLURM-managed
# HPC cluster. It configures the necessary environment and launches the profiling
# experiments using the enhanced launch.sh script with command-line arguments.
#
# SLURM Configuration:
#     - Job name: AI_INFERENCE_ENERGY_PROFILING
#     - Nodes: 1
#     - Tasks per node: 16
#     - GPUs per node: 1
#     - Partition: toreador
#
# Environment Setup:
#     - Loads required modules (GCC, CUDA, cuDNN)
#     - Activates conda environment for AI frameworks
#     - Sets up GPU profiling tools
#
# Usage:
#     # Submit with default configuration
#     sbatch submit_job.sh
#     
#     # Submit with custom configuration (edit LAUNCH_ARGS below)
#     sbatch submit_job.sh
#
# Configuration:
#     Edit the LAUNCH_ARGS variable below to customize the experiment:
#     - GPU type (A100/V100)
#     - Profiling tool (dcgmi/nvidia-smi)
#     - Profiling mode (dvfs/baseline)
#     - Application details
#     - Number of runs and intervals
#
# Author: AI Inference Energy Research Team
#

#SBATCH --job-name=LSTM_A100_DATA
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
# Launch Script Configuration
# ============================================================================
# 
# Configure the launch.sh arguments here to customize your experiment.
# Comment/uncomment and modify as needed for your specific experiment.
#

# Example configurations (uncomment ONE set of arguments):

# Default configuration - LSTM on A100 with DCGMI and DVFS
LAUNCH_ARGS=""

# V100 baseline profiling
# LAUNCH_ARGS="--gpu-type V100 --profiling-mode baseline --num-runs 2"

# A100 with custom application
# LAUNCH_ARGS="--gpu-type A100 --app-name StableDiffusion --app-executable stable_diffusion --app-params '--prompt test > results/SD_output.log'"

# Quick test configuration (fewer runs)
# LAUNCH_ARGS="--profiling-mode baseline --num-runs 1 --sleep-interval 0"

# Comprehensive DVFS experiment
# LAUNCH_ARGS="--gpu-type A100 --profiling-mode dvfs --num-runs 5 --sleep-interval 2"

# nvidia-smi profiling alternative
# LAUNCH_ARGS="--profiling-tool nvidia-smi --profiling-mode baseline"

# Custom application example
# LAUNCH_ARGS="--app-name MyApp --app-executable my_inference_script --app-params '--model large --batch-size 8 > results/custom_output.log'"

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
    log_info "Verifying GPU availability and profiling tools..."
    
    # Check nvidia-smi
    if command -v nvidia-smi &> /dev/null; then
        log_info "✓ nvidia-smi is available"
        log_info "NVIDIA GPU information:"
        nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader,nounits
    else
        log_error "✗ nvidia-smi not available"
        return 1
    fi
    
    # Check DCGMI
    if command -v dcgmi &> /dev/null; then
        log_info "✓ DCGMI is available"
        log_info "DCGMI version:"
        dcgmi --version 2>/dev/null || log_warning "DCGMI version check failed"
    else
        log_warning "✗ DCGMI not available (nvidia-smi profiling can be used as alternative)"
    fi
    
    # Check PyTorch CUDA
    log_info "Checking PyTorch CUDA support..."
    if python -c "import torch; print(f'PyTorch CUDA available: {torch.cuda.is_available()}'); print(f'CUDA device count: {torch.cuda.device_count()}')"; then
        log_info "✓ PyTorch CUDA verification completed"
    else
        log_warning "✗ PyTorch CUDA verification failed"
    fi
    
    # Check available profiling configurations
    log_info "Profiling tool availability summary:"
    if command -v dcgmi &> /dev/null; then
        log_info "  - DCGMI profiling: Available (recommended)"
    else
        log_info "  - DCGMI profiling: Not available"
    fi
    
    if command -v nvidia-smi &> /dev/null; then
        log_info "  - nvidia-smi profiling: Available (use --profiling-tool nvidia-smi)"
    else
        log_info "  - nvidia-smi profiling: Not available"
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
    
    # Check profiling scripts (these are now checked by launch.sh itself)
    local profiling_scripts=("./profile.py" "./control.sh")
    for script in "${profiling_scripts[@]}"; do
        if [[ -f "$script" && -x "$script" ]]; then
            log_info "Found: $script"
        else
            log_warning "Missing or not executable: $script (will be checked by launch.sh)"
        fi
    done
    
    # Check alternative profiling scripts
    local alt_scripts=("./profile_smi.py" "./control_smi.sh")
    for script in "${alt_scripts[@]}"; do
        if [[ -f "$script" && -x "$script" ]]; then
            log_info "Found alternative: $script"
        else
            log_info "Alternative script not found: $script (optional)"
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
    
    # Display launch configuration
    if [[ -n "$LAUNCH_ARGS" ]]; then
        log_info "Launch script arguments: $LAUNCH_ARGS"
        log_info "Command: $LAUNCH_SCRIPT $LAUNCH_ARGS"
    else
        log_info "Using default launch script configuration"
        log_info "Command: $LAUNCH_SCRIPT"
    fi
    
    # Create results directory if it doesn't exist
    if [[ ! -d "results" ]]; then
        mkdir -p results
        log_info "Created results directory"
    fi
    
    # Record experiment start time
    local experiment_start
    experiment_start=$(date +%s)
    
    # Execute the launch script with arguments
    if [[ -n "$LAUNCH_ARGS" ]]; then
        # Use eval to properly handle quoted arguments
        if eval "$LAUNCH_SCRIPT $LAUNCH_ARGS"; then
            local experiment_end
            experiment_end=$(date +%s)
            local experiment_duration=$((experiment_end - experiment_start))
            
            log_info "Experiment completed successfully"
            log_info "Total experiment duration: ${experiment_duration}s"
            
            # Display results summary
            display_results_summary
            
            return 0
        else
            log_error "Experiment failed"
            return 1
        fi
    else
        # Run with default arguments
        if "$LAUNCH_SCRIPT"; then
            local experiment_end
            experiment_end=$(date +%s)
            local experiment_duration=$((experiment_end - experiment_start))
            
            log_info "Experiment completed successfully"
            log_info "Total experiment duration: ${experiment_duration}s"
            
            # Display results summary
            display_results_summary
            
            return 0
        else
            log_error "Experiment failed"
            return 1
        fi
    fi
}

# Function to display results summary
display_results_summary() {
    if [[ -d "results" ]]; then
        local result_count
        result_count=$(find results -type f | wc -l)
        log_info "Generated $result_count result files in results/ directory"
        
        # List some example result files
        log_info "Sample result files:"
        find results -type f -name "*.csv" | head -3 | while read -r file; do
            log_info "  - $file"
        done
        
        find results -type f ! -name "*.csv" | head -3 | while read -r file; do
            log_info "  - $file"
        done
        
        if (( result_count > 6 )); then
            log_info "  ... and $((result_count - 6)) more files"
        fi
    else
        log_warning "Results directory not found"
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
    log_info "Submit script version: Enhanced with launch.sh CLI support"
    
    # Display job information
    display_job_info
    
    # Show configuration information
    log_info "=== Experiment Configuration ==="
    if [[ -n "$LAUNCH_ARGS" ]]; then
        log_info "Custom configuration: $LAUNCH_ARGS"
    else
        log_info "Using default configuration (LSTM on A100 with DCGMI)"
        log_info "To customize: Edit LAUNCH_ARGS variable in this script"
    fi
    log_info "Launch script: $LAUNCH_SCRIPT"
    log_info "Conda environment: $CONDA_ENV"
    log_info "==============================="
    
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
