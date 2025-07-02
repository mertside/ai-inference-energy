#!/bin/bash
#
# A100 Interactive Session Helper Script
#
# This script provides easy access to A100 interactive sessions on HPCC
# and includes helpful commands for testing the profiling framework.
#
# Usage:
#   ./a100_interactive.sh          # Start interactive session
#   ./a100_interactive.sh test     # Run quick framework test
#   ./a100_interactive.sh help     # Show usage information
#

set -euo pipefail

# Configuration for HPCC A100 nodes
readonly A100_PARTITION="toreador"
readonly A100_GPU_COUNT="1"
readonly A100_RESERVATION="ghazanfar"

# Colors for output
readonly RED='\033[0;31m'
readonly GREEN='\033[0;32m'
readonly YELLOW='\033[1;33m'
readonly BLUE='\033[0;34m'
readonly NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${GREEN}[INFO]${NC} $*"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $*"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $*"
}

log_header() {
    echo -e "${BLUE}=== $* ===${NC}"
}

# Function to show usage
show_usage() {
    cat << EOF
A100 Interactive Session Helper for HPCC

Usage:
    $0                # Start interactive A100 session
    $0 test          # Run quick framework test
    $0 status        # Check A100 node status
    $0 help          # Show this help

Interactive Session Command:
    srun --partition=${A100_PARTITION} --gpus-per-node=${A100_GPU_COUNT} --reservation=${A100_RESERVATION} --pty bash

Quick Test Commands (run in interactive session):
    # Test A100 detection
    ./launch.sh --gpu-type A100 --profiling-mode baseline --num-runs 1

    # Check GPU info
    nvidia-smi

    # Test frequency discovery
    nvidia-smi -q -d SUPPORTED_CLOCKS

Examples:
    $0                   # Start interactive session
    $0 test             # Test framework (after starting session)
    $0 status           # Check if A100 nodes are available

EOF
}

# Function to check A100 node status
check_a100_status() {
    log_header "A100 Node Status Check"
    log_info "Checking HPCC A100 nodes on partition: ${A100_PARTITION}"
    
    if command -v sinfo &> /dev/null; then
        log_info "Partition status:"
        sinfo -p "${A100_PARTITION}" || log_warning "Could not query partition ${A100_PARTITION}"
        
        log_info "GPU node information:"
        sinfo -p "${A100_PARTITION}" -o "%n %t %G %C" || log_warning "Could not query GPU info"
        
        log_info "Current queue:"
        squeue -p "${A100_PARTITION}" || log_warning "Could not query queue for ${A100_PARTITION}"
    else
        log_warning "SLURM commands not available - make sure you're on a SLURM cluster"
    fi
}

# Function to start interactive session
start_interactive() {
    log_header "Starting A100 Interactive Session"
    log_info "HPCC Configuration:"
    log_info "  Partition: ${A100_PARTITION}"
    log_info "  GPUs: ${A100_GPU_COUNT}"
    log_info "  Reservation: ${A100_RESERVATION}"
    
    log_info "Command to run:"
    echo "srun --partition=${A100_PARTITION} --gpus-per-node=${A100_GPU_COUNT} --reservation=${A100_RESERVATION} --pty bash"
    
    log_warning "Note: This will start an interactive session. Exit with 'exit' when done."
    log_info "Starting interactive session..."
    
    # Execute the interactive command
    exec srun --partition="${A100_PARTITION}" --gpus-per-node="${A100_GPU_COUNT}" --reservation="${A100_RESERVATION}" --pty bash
}

# Function to run quick test
run_quick_test() {
    log_header "A100 Framework Quick Test"
    
    # Check if we're in an interactive session or on compute node
    if [[ -z "${SLURM_JOB_ID:-}" ]]; then
        log_error "This test should be run in an interactive SLURM session"
        log_info "First run: $0   # to start interactive session"
        log_info "Then run: $0 test   # to run this test"
        exit 1
    fi
    
    log_info "Running in SLURM job: ${SLURM_JOB_ID}"
    log_info "Node: ${SLURM_NODELIST:-$(hostname)}"
    
    # Test GPU detection
    log_info "1. Testing GPU detection..."
    if nvidia-smi --query-gpu=name --format=csv,noheader,nounits; then
        log_info "✓ GPU detected successfully"
    else
        log_error "✗ GPU detection failed"
        return 1
    fi
    
    # Test framework configuration
    log_info "2. Testing framework A100 configuration..."
    if ./launch.sh --gpu-type A100 --help &> /dev/null; then
        log_info "✓ Framework supports A100"
    else
        log_error "✗ Framework A100 support issue"
        return 1
    fi
    
    # Test quick run (dry run)
    log_info "3. Testing quick A100 baseline configuration..."
    log_info "Running: ./launch.sh --gpu-type A100 --profiling-mode baseline --num-runs 1"
    log_warning "This is a real test - it will run for a few minutes"
    
    if ./launch.sh --gpu-type A100 --profiling-mode baseline --num-runs 1; then
        log_info "✓ A100 baseline test completed successfully"
        log_info "✓ All tests passed!"
    else
        log_error "✗ A100 baseline test failed"
        return 1
    fi
}

# Main function
main() {
    local command="${1:-}"
    
    case "$command" in
        "")
            start_interactive
            ;;
        "test")
            run_quick_test
            ;;
        "status")
            check_a100_status
            ;;
        "help"|"-h"|"--help")
            show_usage
            ;;
        *)
            log_error "Unknown command: $command"
            show_usage
            exit 1
            ;;
    esac
}

# Execute main function
main "$@"
