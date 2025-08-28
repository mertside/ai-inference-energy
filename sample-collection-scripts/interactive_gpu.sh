#!/bin/bash
#
# Unified GPU Interactive Session Helper Script
#
# This script provides easy access to interactive sessions for all GPU types on HPCC/REPACSS
# and includes helpful commands for testing the profiling framework.
#
# Usage:
#   ./interactive_gpu.sh v100          # Start V100 interactive session
#   ./interactive_gpu.sh a100          # Start A100 interactive session
#   ./interactive_gpu.sh h100          # Start H100 interactive session
#   ./interactive_gpu.sh <gpu> test    # Run quick framework test
#   ./interactive_gpu.sh <gpu> status  # Check GPU node status
#   ./interactive_gpu.sh help          # Show usage information
#

set -euo pipefail

# GPU Configurations
declare -A GPU_CONFIGS=(
    # V100 Configuration (HPCC)
    ["v100_partition"]="matador"
    ["v100_gpu_type"]="v100"
    ["v100_gpu_count"]="1"
    ["v100_cpu_cores"]="8"
    ["v100_memory"]="32GB HBM2"
    ["v100_freq_range"]="405-1380 MHz (137 steps)"
    ["v100_system"]="HPCC"

    # A100 Configuration (HPCC)
    ["a100_partition"]="toreador"
    ["a100_gpu_count"]="1"
    ["a100_reservation"]="ghazanfar"
    ["a100_memory"]="40GB HBM2"
    ["a100_freq_range"]="210-1410 MHz (61 steps)"
    ["a100_system"]="HPCC"

    # H100 Configuration (REPACSS)
    ["h100_partition"]="h100"
    ["h100_node"]="rpg-93-1"
    ["h100_gpu_count"]="1"
    ["h100_memory"]="94GB HBM3"
    ["h100_freq_range"]="210-2520 MHz (86 steps)"
    ["h100_system"]="REPACSS"
)

# Colors for output
readonly RED='\033[0;31m'
readonly GREEN='\033[0;32m'
readonly YELLOW='\033[1;33m'
readonly BLUE='\033[0;34m'
readonly PURPLE='\033[0;35m'
readonly CYAN='\033[0;36m'
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

log_gpu_header() {
    local gpu_type="$1"
    echo -e "${PURPLE}=== ${gpu_type^^} GPU SESSION ===${NC}"
}

# Function to show usage
show_usage() {
    echo -e "${CYAN}Unified GPU Interactive Session Helper${NC}"
    echo
    echo -e "${BLUE}Usage:${NC}"
    echo "    $0 v100                 # Start V100 interactive session (HPCC)"
    echo "    $0 a100                 # Start A100 interactive session (HPCC)"
    echo "    $0 h100                 # Start H100 interactive session (REPACSS)"
    echo "    $0 <gpu> test          # Run quick framework test for GPU type"
    echo "    $0 <gpu> status        # Check GPU node status"
    echo "    $0 <gpu> info          # Show detailed GPU information"
    echo "    $0 help                # Show this help"
    echo
    echo -e "${BLUE}Supported GPU Types:${NC}"
    echo -e "    ${GREEN}v100${NC}    - Tesla V100 (32GB) on HPCC matador partition"
    echo -e "    ${GREEN}a100${NC}    - Tesla A100 (40GB) on HPCC toreador partition"
    echo -e "    ${GREEN}h100${NC}    - H100 (94GB) on REPACSS h100 partition"
    echo
    echo -e "${BLUE}Examples:${NC}"
    echo "    $0 v100                 # Start V100 session"
    echo "    $0 a100 test           # Test A100 framework"
    echo "    $0 h100 status         # Check H100 availability"
    echo "    $0 v100 info           # Show V100 details"
    echo
    echo -e "${BLUE}Interactive Commands (run in session):${NC}"
    echo "    # Test GPU detection"
    echo "    ./launch.sh --gpu-type <GPU> --profiling-mode baseline --num-runs 1"
    echo "    "
    echo "    # Check GPU info"
    echo "    nvidia-smi"
    echo "    "
    echo "    # Test frequency discovery"
    echo "    nvidia-smi -q -d SUPPORTED_CLOCKS"
    echo
    echo -e "${YELLOW}Note:${NC} Each GPU type uses different SLURM configurations and may require"
    echo "different access permissions or reservations."
    echo
}

# Function to validate GPU type
validate_gpu_type() {
    local gpu_type="$1"
    case "${gpu_type,,}" in
        v100|a100|h100)
            echo "${gpu_type,,}"
            return 0
            ;;
        *)
            log_error "Invalid GPU type: $gpu_type"
            log_info "Valid options: v100, a100, h100"
            return 1
            ;;
    esac
}

# Function to get GPU configuration value
get_gpu_config() {
    local gpu_type="$1"
    local config_key="$2"
    local key="${gpu_type}_${config_key}"
    echo "${GPU_CONFIGS[$key]:-}"
}

# Function to check GPU node status
check_gpu_status() {
    local gpu_type="$1"
    local partition=$(get_gpu_config "$gpu_type" "partition")
    local system=$(get_gpu_config "$gpu_type" "system")

    log_header "${gpu_type^^} Node Status Check"
    log_info "Checking $system ${gpu_type^^} nodes on partition: $partition"

    if ! command -v sinfo &> /dev/null; then
        log_warning "SLURM commands not available - make sure you're on a SLURM cluster"
        return 1
    fi

    log_info "Partition status:"
    if ! sinfo -p "$partition" 2>/dev/null; then
        log_warning "Could not query partition $partition"
    fi

    log_info "GPU node information:"
    if ! sinfo -p "$partition" -o "%n %t %G %C" 2>/dev/null; then
        log_warning "Could not query GPU info for $partition"
    fi

    log_info "Current queue:"
    if ! squeue -p "$partition" 2>/dev/null; then
        log_warning "Could not query queue for $partition"
    fi

    # Special handling for H100 (single node)
    if [[ "$gpu_type" == "h100" ]]; then
        local node=$(get_gpu_config "$gpu_type" "node")
        log_info "H100 node-specific status:"
        if ! sinfo -n "$node" 2>/dev/null; then
            log_warning "Could not query node $node"
        fi
    fi

    # Try to find GPU-specific information
    if [[ "$gpu_type" != "h100" ]]; then
        log_info "${gpu_type^^} GPU availability:"
        sinfo -p "$partition" --format="%.10P %.5a %.10l %.6D %.6t %.8z %.15C %.8G %.15N" | grep -i "$gpu_type" || log_warning "No ${gpu_type^^} GPU information found"
    fi
}

# Function to start interactive session
start_interactive() {
    local gpu_type="$1"
    local partition=$(get_gpu_config "$gpu_type" "partition")
    local system=$(get_gpu_config "$gpu_type" "system")

    log_gpu_header "$gpu_type"
    log_info "$system Configuration:"
    log_info "  GPU Type: ${gpu_type^^}"
    log_info "  Partition: $partition"
    log_info "  Memory: $(get_gpu_config "$gpu_type" "memory")"
    log_info "  System: $system"

    case "$gpu_type" in
        "v100")
            local gpu_count=$(get_gpu_config "$gpu_type" "gpu_count")
            local cpu_cores=$(get_gpu_config "$gpu_type" "cpu_cores")
            local gpu_type_slurm=$(get_gpu_config "$gpu_type" "gpu_type")

            log_info "  GPU Count: $gpu_count"
            log_info "  CPU Cores: $cpu_cores"

            local cmd="srun --partition=$partition --gres=gpu:$gpu_type_slurm:$gpu_count --nodes=1 --ntasks=1 --cpus-per-task=$cpu_cores --pty bash"
            log_info "Command: $cmd"
            log_warning "Starting V100 interactive session..."
            exec $cmd
            ;;

        "a100")
            local gpu_count=$(get_gpu_config "$gpu_type" "gpu_count")
            local reservation=$(get_gpu_config "$gpu_type" "reservation")

            log_info "  GPU Count: $gpu_count"
            log_info "  Reservation: $reservation"

            local cmd="srun --partition=$partition --gpus-per-node=$gpu_count --reservation=$reservation --pty bash"
            log_info "Command: $cmd"
            log_warning "Starting A100 interactive session..."
            exec $cmd
            ;;

        "h100")
            local gpu_count=$(get_gpu_config "$gpu_type" "gpu_count")
            local node=$(get_gpu_config "$gpu_type" "node")

            log_info "  GPU Count: $gpu_count"
            log_info "  Node: $node"

            local cmd="interactive -p $partition -g $gpu_count -w $node"
            log_info "Command: $cmd"
            log_warning "Starting H100 interactive session..."
            exec $cmd
            ;;
    esac
}

# Function to run quick test
run_quick_test() {
    local gpu_type="$1"

    log_header "${gpu_type^^} Framework Quick Test"

    # Check if we're in an interactive session or on compute node
    if [[ -z "${SLURM_JOB_ID:-}" ]]; then
        log_error "This test should be run in an interactive SLURM session"
        log_info "First run: $0 $gpu_type   # to start interactive session"
        log_info "Then run: $0 $gpu_type test   # to run this test"
        exit 1
    fi

    log_info "Running in SLURM job: ${SLURM_JOB_ID}"
    log_info "Node: ${SLURM_NODELIST:-$(hostname)}"

    # Test GPU detection
    log_info "1. Testing GPU detection..."
    if gpu_info=$(nvidia-smi --query-gpu=name --format=csv,noheader,nounits 2>/dev/null); then
        log_info "✓ GPU detected: $gpu_info"
        if echo "$gpu_info" | grep -qi "$gpu_type"; then
            log_info "✓ ${gpu_type^^} GPU confirmed"
        else
            log_warning "⚠ GPU detected but may not be ${gpu_type^^}: $gpu_info"
        fi
    else
        log_error "✗ GPU detection failed"
        return 1
    fi

    # Test GPU memory and specifications
    log_info "2. Testing GPU specifications..."
    if gpu_memory=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null); then
        log_info "✓ GPU Memory: ${gpu_memory} MB"
    fi

    # Test framework configuration
    log_info "3. Testing framework ${gpu_type^^} configuration..."
    if ./launch.sh --gpu-type "${gpu_type^^}" --help &> /dev/null; then
        log_info "✓ Framework supports ${gpu_type^^}"
    else
        log_error "✗ Framework ${gpu_type^^} support issue"
        return 1
    fi

    # Test frequency support (V100 only has detailed frequency testing)
    if [[ "$gpu_type" == "v100" ]]; then
        log_info "4. Testing ${gpu_type^^} frequency support..."
        if nvidia-smi -q -d SUPPORTED_CLOCKS &> /dev/null; then
            log_info "✓ Frequency control appears available"
            freq_count=$(nvidia-smi -q -d SUPPORTED_CLOCKS | grep -c "Graphics" || echo "0")
            log_info "  Found $freq_count graphics frequency options"
        else
            log_warning "⚠ Frequency control may not be available"
            log_info "  Framework will fallback to nvidia-smi monitoring"
        fi
    fi

    # Test quick run
    log_info "5. Testing quick ${gpu_type^^} baseline configuration..."
    log_info "Running: ./launch.sh --gpu-type ${gpu_type^^} --profiling-mode baseline --num-runs 1"
    log_warning "This is a real test - it will run for a few minutes"

    if ./launch.sh --gpu-type "${gpu_type^^}" --profiling-mode baseline --num-runs 1; then
        log_info "✓ ${gpu_type^^} baseline test completed successfully"
        log_info "✓ All tests passed!"

        # Show any output files
        if ls *.csv *.json *.log &>/dev/null 2>&1; then
            log_info "Generated output files:"
            ls -la *.csv *.json *.log 2>/dev/null | head -5
        fi
    else
        log_error "✗ ${gpu_type^^} baseline test failed"
        return 1
    fi
}

# Function to show GPU specific information
show_gpu_info() {
    local gpu_type="$1"
    local system=$(get_gpu_config "$gpu_type" "system")
    local partition=$(get_gpu_config "$gpu_type" "partition")
    local memory=$(get_gpu_config "$gpu_type" "memory")
    local freq_range=$(get_gpu_config "$gpu_type" "freq_range")

    log_header "${gpu_type^^} GPU Information"
    log_info "$system ${gpu_type^^} Configuration:"
    log_info "  GPU Type: NVIDIA Tesla ${gpu_type^^}"
    log_info "  Memory: $memory"
    log_info "  System: $system"
    log_info "  Partition: $partition"

    case "$gpu_type" in
        "v100")
            local gpu_type_slurm=$(get_gpu_config "$gpu_type" "gpu_type")
            local gpu_count=$(get_gpu_config "$gpu_type" "gpu_count")
            log_info "  Resource Spec: --gres=gpu:$gpu_type_slurm:$gpu_count"
            log_info "  CPU Allocation: $(get_gpu_config "$gpu_type" "cpu_cores") cores per task"
            ;;
        "a100")
            local reservation=$(get_gpu_config "$gpu_type" "reservation")
            log_info "  Reservation: $reservation"
            log_info "  Resource Spec: --gpus-per-node=$(get_gpu_config "$gpu_type" "gpu_count")"
            ;;
        "h100")
            local node=$(get_gpu_config "$gpu_type" "node")
            log_info "  Specific Node: $node"
            log_info "  Resource Spec: -g $(get_gpu_config "$gpu_type" "gpu_count") -w $node"
            ;;
    esac

    echo
    log_info "${gpu_type^^} Frequency Ranges (from framework config):"
    log_info "  Graphics: $freq_range"
    echo
    log_info "Profiling Modes Available:"
    case "$gpu_type" in
        "v100")
            log_info "  baseline       - Default frequency only (~15-20 min)"
            log_info "  comprehensive  - All 137 frequencies (~6-8 hours)"
            ;;
        "a100")
            log_info "  baseline       - Default frequency only (~10-15 min)"
            log_info "  comprehensive  - All 61 frequencies (~3-4 hours)"
            ;;
        "h100")
            log_info "  baseline       - Default frequency only (~8-12 min)"
            log_info "  comprehensive  - All 86 frequencies (~4-5 hours)"
            ;;
    esac
    log_info "  custom         - Selected frequencies (variable time)"
}

# Main function
main() {
    local gpu_type="${1:-}"
    local command="${2:-}"

    # Handle help first
    if [[ "$gpu_type" == "help" || "$gpu_type" == "-h" || "$gpu_type" == "--help" ]]; then
        show_usage
        exit 0
    fi

    # Validate GPU type
    if [[ -z "$gpu_type" ]]; then
        log_error "GPU type required"
        show_usage
        exit 1
    fi

    if ! gpu_type=$(validate_gpu_type "$gpu_type"); then
        show_usage
        exit 1
    fi

    # Handle commands
    case "$command" in
        "")
            start_interactive "$gpu_type"
            ;;
        "test")
            run_quick_test "$gpu_type"
            ;;
        "status")
            check_gpu_status "$gpu_type"
            ;;
        "info")
            show_gpu_info "$gpu_type"
            ;;
        "help"|"-h"|"--help")
            show_usage
            ;;
        *)
            log_error "Unknown command: $command"
            log_info "Valid commands: test, status, info, help"
            exit 1
            ;;
    esac
}

# Execute main function
main "$@"
