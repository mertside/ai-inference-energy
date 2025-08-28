#!/bin/bash
#
# GPU Frequency Control Script using nvidia-smi (Alternative to DCGMI).
#
# This script controls GPU memory and core frequencies using nvidia-smi commands
# as an alternative to DCGMI for energy profiling experiments. It provides similar
# functionality for DVFS (Dynamic Voltage and Frequency Scaling) research on AI
# inference workloads.
#
# Usage:
#     ./control_smi.sh <memory_freq> <core_freq>
#
# Arguments:
#     memory_freq: Memory frequency in MHz
#     core_freq: Core frequency in MHz
#
# Requirements:
#     - NVIDIA GPU with frequency control support
#     - nvidia-smi tool (part of NVIDIA drivers)
#     - Appropriate permissions to modify GPU frequencies
#     - sudo access may be required for frequency modifications
#
# Note:
#     This is an alternative to the DCGMI-based control.sh script.
#     nvidia-smi may have different capabilities and limitations compared to DCGMI.
#
# Author: Mert Side
#

set -euo pipefail  # Exit on error, undefined variables, and pipe failures

# Configuration
readonly SCRIPT_NAME="$(basename "$0")"
readonly SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Default values
readonly DEFAULT_SLEEP_TIME=2
readonly DEFAULT_POWER_LIMIT=250  # Watts
readonly GPU_ID=0  # Default GPU ID

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

GPU Frequency Control Script using nvidia-smi

Arguments:
    memory_freq    Memory frequency in MHz (e.g., 1215 for A100, 877 for V100, 2619 for H100)
    core_freq      Core frequency in MHz (e.g., 1410 for A100, 1380 for V100, 1785 for H100)

Examples:
    $SCRIPT_NAME 1215 1410    # Set A100 to default frequencies
    $SCRIPT_NAME 877 1380     # Set V100 to default frequencies
    $SCRIPT_NAME 2619 1785    # Set H100 to default frequencies
    $SCRIPT_NAME 1215 1200    # Set A100 with reduced core frequency

Requirements:
    - NVIDIA GPU with frequency control support
    - nvidia-smi tool (part of NVIDIA drivers)
    - Appropriate permissions (may require sudo)

This script is an alternative to the DCGMI-based control.sh script,
using nvidia-smi commands for GPU frequency control.

Notes:
    - nvidia-smi frequency control may be limited on some GPU models
    - Some operations may require administrator privileges
    - Frequency ranges and capabilities vary by GPU architecture
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

# Function to check if nvidia-smi is available
check_nvidia_smi() {
    if ! command -v nvidia-smi &> /dev/null; then
        log_error "nvidia-smi command not found. Please install NVIDIA drivers."
        return 1
    fi

    # Test if nvidia-smi can communicate with GPUs
    if ! nvidia-smi -i "$GPU_ID" --query-gpu=name --format=csv,noheader &> /dev/null; then
        log_error "nvidia-smi cannot communicate with GPU $GPU_ID. Check GPU status and permissions."
        return 1
    fi

    return 0
}

# Function to enable persistence mode
enable_persistence_mode() {
    log_info "Enabling persistence mode..."

    if nvidia-smi -pm 1; then
        log_info "Successfully enabled persistence mode"
    else
        log_warning "Failed to enable persistence mode (may require sudo)"
        log_warning "This may affect frequency control stability"
    fi
}

# Function to set GPU frequencies using nvidia-smi
set_gpu_frequencies() {
    local memory_freq="$1"
    local core_freq="$2"

    log_info "Setting GPU frequencies using nvidia-smi: Memory=${memory_freq}MHz, Core=${core_freq}MHz"

    # Enable persistence mode for stable frequency control
    enable_persistence_mode

    # Set application clocks (memory and graphics clocks)
    log_info "Setting application clocks..."
    if nvidia-smi -i "$GPU_ID" -ac "${memory_freq},${core_freq}"; then
        log_info "Successfully set GPU frequencies"
    else
        log_error "Failed to set GPU frequencies using nvidia-smi"
        log_error "This may require:"
        log_error "  1. Administrator privileges (sudo)"
        log_error "  2. GPU with frequency control support"
        log_error "  3. Proper driver installation"
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

    # Query current frequencies
    local gpu_info
    if gpu_info=$(nvidia-smi -i "$GPU_ID" --query-gpu=name,clocks.applications.memory,clocks.applications.graphics --format=csv,noheader 2>/dev/null); then
        log_info "Current GPU status:"
        echo "  $gpu_info" | while IFS=',' read -r name mem_freq core_freq; do
            log_info "  GPU: ${name// /}"
            log_info "  Memory Clock: ${mem_freq// /}"
            log_info "  Graphics Clock: ${core_freq// /}"
        done
    else
        log_warning "Could not verify current frequencies"
    fi
}

# Function to reset GPU frequencies to default
reset_frequencies() {
    log_info "Resetting GPU frequencies to default..."

    if nvidia-smi -i "$GPU_ID" -rac; then
        log_info "Successfully reset GPU frequencies to default"
    else
        log_warning "Failed to reset GPU frequencies to default"
    fi
}

# Function to clean up on exit
cleanup() {
    local exit_code=$?
    if (( exit_code != 0 )); then
        log_error "Script exited with error code: $exit_code"
        log_info "You may want to reset GPU frequencies manually:"
        log_info "  nvidia-smi -i $GPU_ID -rac"
    fi
    exit "$exit_code"
}

# Main function
main() {
    # Set up signal handlers for cleanup
    trap cleanup EXIT
    trap 'log_error "Interrupted by user"; exit 130' INT TERM

    log_info "Starting GPU frequency control using nvidia-smi"
    log_info "Script: $SCRIPT_NAME"
    log_info "Working directory: $(pwd)"
    log_info "GPU ID: $GPU_ID"

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

    # Check nvidia-smi availability
    if ! check_nvidia_smi; then
        exit 1
    fi

    # Set GPU frequencies
    if ! set_gpu_frequencies "$memory_freq" "$core_freq"; then
        exit 1
    fi

    # Verify the changes
    verify_frequencies

    log_info "GPU frequency control completed successfully using nvidia-smi"
}

# Execute main function with all arguments
main "$@"
