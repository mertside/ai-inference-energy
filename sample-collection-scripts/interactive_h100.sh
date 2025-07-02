#!/bin/bash
#
# H100 Interactive Session Helper Script
#
# This script provides easy access to H100 interactive sessions on REPACSS
# and includes helpful commands for testing the profiling framework.
#
# Usage:
#   ./interactive_h100.sh          # Start interactive session
#   ./interactive_h100.sh test     # Run quick framework test
#   ./interactive_h100.sh help     # Show usage information
#

set -euo pipefail

# Configuration for REPACSS H100 node
readonly H100_PARTITION="h100-build"
readonly H100_NODE="rpg-93-9"
readonly H100_GPU_COUNT="1"

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
H100 Interactive Session Helper for REPACSS

Usage:
    $0                # Start interactive H100 session
    $0 test          # Run quick framework test
    $0 status        # Check H100 node status
    $0 help          # Show this help

Interactive Session Command:
    interactive -p ${H100_PARTITION} -g ${H100_GPU_COUNT} -w ${H100_NODE}

Quick Test Commands (run in interactive session):
    # Test H100 detection
    ./launch.sh --gpu-type H100 --profiling-mode baseline --num-runs 1

    # Check GPU info
    nvidia-smi

    # Test frequency discovery
    nvidia-smi -q -d SUPPORTED_CLOCKS

Examples:
    $0                   # Start interactive session
    $0 test             # Test framework (after starting session)
    $0 status           # Check if H100 node is available

EOF
}

# Function to check H100 node status
check_h100_status() {
    log_header "H100 Node Status Check"
    log_info "Checking REPACSS H100 node: ${H100_NODE}"
    
    if command -v sinfo &> /dev/null; then
        log_info "Partition status:"
        sinfo -p "${H100_PARTITION}" || log_warning "Could not query partition ${H100_PARTITION}"
        
        log_info "Node-specific status:"
        sinfo -n "${H100_NODE}" || log_warning "Could not query node ${H100_NODE}"
        
        log_info "Current queue:"
        squeue -p "${H100_PARTITION}" || log_warning "Could not query queue for ${H100_PARTITION}"
    else
        log_warning "SLURM commands not available - make sure you're on a SLURM cluster"
    fi
}

# Function to start interactive session
start_interactive() {
    log_header "Starting H100 Interactive Session"
    log_info "REPACSS Configuration:"
    log_info "  Partition: ${H100_PARTITION}"
    log_info "  Node: ${H100_NODE}"
    log_info "  GPUs: ${H100_GPU_COUNT}"
    
    log_info "Command to run:"
    echo "interactive -p ${H100_PARTITION} -g ${H100_GPU_COUNT} -w ${H100_NODE}"
    
    log_warning "Note: This will start an interactive session. Exit with 'exit' when done."
    log_info "Starting interactive session..."
    
    # Execute the interactive command
    exec interactive -p "${H100_PARTITION}" -g "${H100_GPU_COUNT}" -w "${H100_NODE}"
}

# Function to run quick test
run_quick_test() {
    log_header "H100 Framework Quick Test"
    
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
    log_info "2. Testing framework H100 configuration..."
    if ./launch.sh --gpu-type H100 --help &> /dev/null; then
        log_info "✓ Framework supports H100"
    else
        log_error "✗ Framework H100 support issue"
        return 1
    fi
    
    # Test quick run (dry run)
    log_info "3. Testing quick H100 baseline configuration..."
    log_info "Running: ./launch.sh --gpu-type H100 --profiling-mode baseline --num-runs 1"
    log_warning "This is a real test - it will run for a few minutes"
    
    if ./launch.sh --gpu-type H100 --profiling-mode baseline --num-runs 1; then
        log_info "✓ H100 baseline test completed successfully"
        log_info "✓ All tests passed!"
    else
        log_error "✗ H100 baseline test failed"
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
            check_h100_status
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
