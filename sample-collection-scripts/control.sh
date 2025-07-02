#!/bin/bash
"""
GPU Frequency Control Script for AI Inference Energy Profiling.

This script controls GPU memory and core frequencies using DCGMI (Data Center GPU 
Manager Interface) for energy profiling experiments. It sets the specified 
frequencies and provides a foundation for DVFS (Dynamic Voltage and Frequency 
Scaling) research on AI inference workloads.

Usage:
    ./control.sh <memory_freq> <core_freq>

Arguments:
    memory_freq: Memory frequency in MHz
    core_freq: Core frequency in MHz

Requirements:
    - NVIDIA GPU with DCGMI support
    - DCGMI tools installed and accessible
    - Appropriate permissions to modify GPU frequencies

Author: Mert Side
"""

set -euo pipefail  # Exit on error, undefined variables, and pipe failures

# Configuration
readonly SCRIPT_NAME="$(basename "$0")"
readonly SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Default values
readonly DEFAULT_SLEEP_TIME=2

# Logging functions
log_info() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] [INFO] $*" >&2
}

log_error() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] [ERROR] $*" >&2
}

log_warning() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] [WARNING] $*" >&2
}

# Function to display usage information
usage() {
    cat << EOF
Usage: $SCRIPT_NAME <memory_freq> <core_freq>

GPU Frequency Control Script for AI Inference Energy Profiling

Arguments:
    memory_freq    Memory frequency in MHz (e.g., 1215 for A100, 877 for V100, 1593 for H100)
    core_freq      Core frequency in MHz (e.g., 1410 for A100, 1380 for V100, 1755 for H100)

Examples:
    $SCRIPT_NAME 1215 1410    # Set A100 to default frequencies
    $SCRIPT_NAME 877 1380     # Set V100 to default frequencies
    $SCRIPT_NAME 1593 1755    # Set H100 to default frequencies

Requirements:
    - NVIDIA GPU with DCGMI support
    - DCGMI tools installed and accessible
    - Appropriate permissions to modify GPU frequencies

This script is part of the AI inference energy profiling framework for 
studying GPU DVFS effects on modern AI workloads.
EOF
}

# Function to validate input arguments
validate_arguments() {
    local memory_freq="$1"
    local core_freq="$2"
    
    # Check if arguments are numeric
    if ! [[ "$memory_freq" =~ ^[0-9]+$ ]]; then
        log_error "Memory frequency must be a positive integer: $memory_freq"
        return 1
    fi
    
    if ! [[ "$core_freq" =~ ^[0-9]+$ ]]; then
        log_error "Core frequency must be a positive integer: $core_freq"
        return 1
    fi
    
    # Check reasonable frequency ranges (basic validation)
    if (( memory_freq < 100 || memory_freq > 2000 )); then
        log_warning "Memory frequency seems unusual: ${memory_freq}MHz"
    fi
    
    if (( core_freq < 100 || core_freq > 2000 )); then
        log_warning "Core frequency seems unusual: ${core_freq}MHz"
    fi
    
    return 0
}

# Function to check if DCGMI is available
check_dcgmi() {
    if ! command -v dcgmi &> /dev/null; then
        log_error "dcgmi command not found. Please install NVIDIA DCGMI tools."
        return 1
    fi
    
    # Test if dcgmi can communicate with GPUs
    if ! dcgmi discovery --list &> /dev/null; then
        log_error "dcgmi cannot communicate with GPUs. Check permissions and GPU status."
        return 1
    fi
    
    return 0
}

# Function to set GPU frequencies using DCGMI
set_gpu_frequencies() {
    local memory_freq="$1"
    local core_freq="$2"
    
    log_info "Setting GPU frequencies: Memory=${memory_freq}MHz, Core=${core_freq}MHz"
    
    # Set GPU frequencies using dcgmi config
    if dcgmi config --set -a "${memory_freq},${core_freq}"; then
        log_info "Successfully set GPU frequencies"
    else
        log_error "Failed to set GPU frequencies"
        return 1
    fi
    
    # Wait for settings to take effect
    log_info "Waiting ${DEFAULT_SLEEP_TIME} seconds for frequency changes to take effect..."
    sleep "$DEFAULT_SLEEP_TIME"
    
    return 0
}

# Function to verify current GPU frequencies
verify_frequencies() {
    log_info "Verifying current GPU frequencies..."
    
    # Query current frequencies (this will show current state)
    if dcgmi dmon -e 210,211 -c 1 2>/dev/null | tail -n +2; then
        log_info "Current GPU frequencies displayed above"
    else
        log_warning "Could not verify current frequencies"
    fi
}

# Function to clean up on exit
cleanup() {
    local exit_code=$?
    if (( exit_code != 0 )); then
        log_error "Script exited with error code: $exit_code"
    fi
    exit "$exit_code"
}

# Main function
main() {
    # Set up signal handlers for cleanup
    trap cleanup EXIT
    trap 'log_error "Interrupted by user"; exit 130' INT TERM
    
    log_info "Starting GPU frequency control"
    log_info "Script: $SCRIPT_NAME"
    log_info "Working directory: $(pwd)"
    
    # Check arguments
    if (( $# != 2 )); then
        log_error "Invalid number of arguments. Expected 2, got $#"
        usage
        exit 1
    fi
    
    local memory_freq="$1"
    local core_freq="$2"
    
    log_info "Requested frequencies: Memory=${memory_freq}MHz, Core=${core_freq}MHz"
    
    # Validate arguments
    if ! validate_arguments "$memory_freq" "$core_freq"; then
        exit 1
    fi
    
    # Check DCGMI availability
    if ! check_dcgmi; then
        exit 1
    fi
    
    # Set GPU frequencies
    if ! set_gpu_frequencies "$memory_freq" "$core_freq"; then
        exit 1
    fi
    
    # Verify the changes
    verify_frequencies
    
    log_info "GPU frequency control completed successfully"
}

# Execute main function with all arguments
main "$@"

# #***Resets the GPU clocks to the default values***
# if [ "$freq" != "P0" ]
# then
# #sudo nvidia-smi -i 0 -rgc
# sudo nvidia-smi -i 0 -rac
# fi

# #*** disable PM ***
# sudo nvidia-smi -pm 0

echo "***Exiting Control***"
