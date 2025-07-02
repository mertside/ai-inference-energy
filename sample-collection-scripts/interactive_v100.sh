#!/bin/bash
#
# V100 Interactive Session Helper Script
#
# This script provides easy access to V100 interactive sessions on HPCC
# and includes helpful commands for testing the profiling framework.
#
# Usage:
#   ./interactive_v100.sh          # Start interactive session
#   ./interactive_v100.sh test     # Run quick framework test
#   ./interactive_v100.sh help     # Show usage information
#

set -euo pipefail

# Configuration for HPCC V100 nodes
readonly V100_PARTITION="matador"
readonly V100_GPU_COUNT="1"
readonly V100_GPU_TYPE="v100"

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
V100 Interactive Session Helper for HPCC

Usage:
    $0                # Start interactive V100 session
    $0 test          # Run quick framework test
    $0 status        # Check V100 node status
    $0 help          # Show this help

Interactive Session Command:
    srun --partition=${V100_PARTITION} --gres=gpu:${V100_GPU_TYPE}:${V100_GPU_COUNT} --ntasks=40 --pty bash

Quick Test Commands (run in interactive session):
    # Test V100 detection
    ./launch.sh --gpu-type V100 --profiling-mode baseline --num-runs 1

    # Check GPU info
    nvidia-smi

    # Test frequency discovery
    nvidia-smi -q -d SUPPORTED_CLOCKS

Examples:
    $0                   # Start interactive session
    $0 test             # Test framework (after starting session)
    $0 status           # Check if V100 nodes are available

Note: V100 nodes on HPCC use the 'matador' partition with specific GPU resource allocation.

EOF
}

# Function to check V100 node status
check_v100_status() {
    log_header "V100 Node Status Check"
    log_info "Checking HPCC V100 nodes on partition: ${V100_PARTITION}"
    
    if command -v sinfo &> /dev/null; then
        log_info "Partition status:"
        sinfo -p "${V100_PARTITION}" || log_warning "Could not query partition ${V100_PARTITION}"
        
        log_info "GPU node information:"
        sinfo -p "${V100_PARTITION}" -o "%n %t %G %C" || log_warning "Could not query GPU info"
        
        log_info "Current queue:"
        squeue -p "${V100_PARTITION}" || log_warning "Could not query queue for ${V100_PARTITION}"
        
        log_info "V100 GPU availability:"
        sinfo -p "${V100_PARTITION}" --format="%.10P %.5a %.10l %.6D %.6t %.8z %.15C %.8G %.15N" | grep -i v100 || log_warning "No V100 GPU information found"
    else
        log_warning "SLURM commands not available - make sure you're on a SLURM cluster"
    fi
}

# Function to start interactive session
start_interactive() {
    log_header "Starting V100 Interactive Session"
    log_info "HPCC Configuration:"
    log_info "  Partition: ${V100_PARTITION}"
    log_info "  GPU Type: ${V100_GPU_TYPE}"
    log_info "  GPU Count: ${V100_GPU_COUNT}"
    log_info "  CPU Tasks: 40"
    
    log_info "Command to run:"
    echo "srun --partition=${V100_PARTITION} --gres=gpu:${V100_GPU_TYPE}:${V100_GPU_COUNT} --ntasks=40 --pty bash"
    
    log_warning "Note: This will start an interactive session. Exit with 'exit' when done."
    log_info "V100 nodes may have limited availability - check status first if needed."
    log_info "Starting interactive session..."
    
    # Execute the interactive command
    exec srun --partition="${V100_PARTITION}" --gres=gpu:"${V100_GPU_TYPE}":"${V100_GPU_COUNT}" --ntasks=40 --pty bash
}

# Function to run quick test
run_quick_test() {
    log_header "V100 Framework Quick Test"
    
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
    if gpu_info=$(nvidia-smi --query-gpu=name --format=csv,noheader,nounits); then
        log_info "✓ GPU detected: $gpu_info"
        if echo "$gpu_info" | grep -qi "v100"; then
            log_info "✓ V100 GPU confirmed"
        else
            log_warning "⚠ GPU detected but may not be V100: $gpu_info"
        fi
    else
        log_error "✗ GPU detection failed"
        return 1
    fi
    
    # Test GPU memory and specifications
    log_info "2. Testing GPU specifications..."
    if gpu_memory=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits); then
        log_info "✓ GPU Memory: ${gpu_memory} MB"
    fi
    
    # Test framework configuration
    log_info "3. Testing framework V100 configuration..."
    if ./launch.sh --gpu-type V100 --help &> /dev/null; then
        log_info "✓ Framework supports V100"
    else
        log_error "✗ Framework V100 support issue"
        return 1
    fi
    
    # Test frequency support
    log_info "4. Testing V100 frequency support..."
    if nvidia-smi -q -d SUPPORTED_CLOCKS &> /dev/null; then
        log_info "✓ Frequency control appears available"
        freq_count=$(nvidia-smi -q -d SUPPORTED_CLOCKS | grep -c "Graphics" || echo "0")
        log_info "  Found $freq_count graphics frequency options"
    else
        log_warning "⚠ Frequency control may not be available"
        log_info "  Framework will fallback to nvidia-smi monitoring"
    fi
    
    # Test quick run (dry run)
    log_info "5. Testing quick V100 baseline configuration..."
    log_info "Running: ./launch.sh --gpu-type V100 --profiling-mode baseline --num-runs 1"
    log_warning "This is a real test - it will run for a few minutes"
    
    if ./launch.sh --gpu-type V100 --profiling-mode baseline --num-runs 1; then
        log_info "✓ V100 baseline test completed successfully"
        log_info "✓ All tests passed!"
        
        # Show any output files
        if ls *.csv *.json *.log &>/dev/null; then
            log_info "Generated output files:"
            ls -la *.csv *.json *.log 2>/dev/null | head -5
        fi
    else
        log_error "✗ V100 baseline test failed"
        return 1
    fi
}

# Function to show V100 specific information
show_v100_info() {
    log_header "V100 GPU Information"
    log_info "Texas Tech HPCC V100 Configuration:"
    log_info "  GPU Type: NVIDIA Tesla V100"
    log_info "  Memory: 32GB HBM2"
    log_info "  Partition: ${V100_PARTITION}"
    log_info "  Resource Spec: --gres=gpu:${V100_GPU_TYPE}:${V100_GPU_COUNT}"
    log_info "  Typical CPU Allocation: 40 tasks"
    echo
    log_info "V100 Frequency Ranges (from framework config):"
    log_info "  Graphics: 405-1380 MHz (137 frequency steps)"
    log_info "  Memory: 877 MHz (fixed)"
    log_info "  Default Graphics: 1380 MHz"
    echo
    log_info "Profiling Modes Available:"
    log_info "  baseline       - Default frequency only (~15-20 min)"
    log_info "  comprehensive  - All 137 frequencies (~6-8 hours)"
    log_info "  custom         - Selected frequencies (variable time)"
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
            check_v100_status
            ;;
        "info")
            show_v100_info
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
