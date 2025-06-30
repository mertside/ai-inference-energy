#!/bin/bash
#
# =============================================================================
# QUICK CONFIGURATION GUIDE
# =============================================================================
#
# To switch between GPU types, edit these variables in the Configuration section:
#
# For NVIDIA A100:
#   GPU_TYPE="A100"           # Automatically sets GA100, 1215MHz mem, 1410MHz core
#
# For NVIDIA V100:
#   GPU_TYPE="V100"           # Automatically sets GV100, 877MHz mem, 1380MHz core
#
# To switch between profiling tools:
#
# For DCGMI profiling:
#   PROFILING_TOOL="dcgmi"    # Uses ./profile.py and ./control.sh
#
# For nvidia-smi profiling:
#   PROFILING_TOOL="nvidia-smi" # Uses ./profile_smi.py and ./control_smi.sh
#
# Required scripts for each configuration:
#   DCGMI:      profile.py, control.sh (existing)
#   nvidia-smi: profile_smi.py, control_smi.sh (new alternatives)
#
# =============================================================================
#
"""
AI Inference Energy Profiling Launch Script.

This script orchestrates energy profiling experiments for AI inference workloads
across different GPU frequencies. It systematically tests various core frequencies
while monitoring power consumption and performance metrics.

The script:
1. Supports both NVIDIA A100 and V100 GPUs with optimized frequency ranges
2. Allows switching between DCGMI and nvidia-smi profiling tools
3. Iterates through GPU-specific core frequency ranges
4. Runs each AI application multiple times per frequency
5. Collects power and performance data using selected profiling tools
6. Saves results for analysis with consistent naming conventions

Features:
- GPU Type Selection: Easy switching between A100 and V100 configurations
- Profiling Tool Selection: Support for both DCGMI and nvidia-smi
- Automatic Configuration: GPU-specific frequencies and parameters
- Comprehensive Logging: Detailed progress and error reporting
- Robust Error Handling: Graceful failure recovery and cleanup

Usage:
    ./launch.sh

Configuration:
    Edit the variables in the Configuration section below to customize:
    - GPU_TYPE: "A100" or "V100" for automatic configuration
    - PROFILING_TOOL: "dcgmi" or "nvidia-smi" for tool selection
    - Number of runs per frequency
    - Applications to test
    - Output directories

Requirements:
    - NVIDIA GPU (A100 or V100)
    - GPU profiling tools (DCGMI or nvidia-smi)
    - AI inference applications (LLaMA, Stable Diffusion, LSTM, etc.)
    - Corresponding profiling and control scripts
    - Bash 4.0+ for associative arrays

Author: AI Inference Energy Research Team
"""

set -euo pipefail  # Exit on error, undefined variables, and pipe failures

# ============================================================================
# Configuration Section
# ============================================================================

# GPU Architecture Selection (A100 or V100)
# Set to "A100" for NVIDIA A100 GPUs or "V100" for NVIDIA V100 GPUs
readonly GPU_TYPE="A100"  # Options: "A100" or "V100"

# Profiling Tool Selection (DCGMI or NVIDIA-SMI)
# Set to "dcgmi" for DCGMI tools or "nvidia-smi" for nvidia-smi equivalent
readonly PROFILING_TOOL="dcgmi"  # Options: "dcgmi" or "nvidia-smi"

# Experiment configuration
readonly NUM_RUNS=2
readonly PROFILING_MODE="dvfs"
readonly SLEEP_INTERVAL=1  # seconds between runs

# GPU-specific configuration based on architecture
if [[ "$GPU_TYPE" == "A100" ]]; then
    readonly GPU_ARCH="GA100"
    readonly MEMORY_FREQ=1215  # A100 memory frequency (MHz)
    readonly DEFAULT_CORE_FREQ=1410  # A100 default core frequency (MHz)
    
    # A100 core frequencies for testing (MHz)
    readonly CORE_FREQUENCIES=(
        1410 1395 1380 1365 1350 1335 1320 1305 1290 1275
        1260 1245 1230 1215 1200 1185 1170 1155 1140 1125
        1110 1095 1080 1065 1050 1035 1020 1005 990 975
        960 945 930 915 900 885 870 855 840 825
        810 795 780 765 750 735 720 705 690 675
        660 645 630 615 600 585 570 555 540 525 510
    )
elif [[ "$GPU_TYPE" == "V100" ]]; then
    readonly GPU_ARCH="GV100"
    readonly MEMORY_FREQ=877  # V100 memory frequency (MHz)
    readonly DEFAULT_CORE_FREQ=1380  # V100 default core frequency (MHz)
    
    # V100 core frequencies for testing (MHz)
    readonly CORE_FREQUENCIES=(
        1380 1372 1365 1357 1350 1342 1335 1327 1320 1312 1305 1297 1290 1282 1275 1267 
        1260 1252 1245 1237 1230 1222 1215 1207 1200 1192 1185 1177 1170 1162 1155 1147 
        1140 1132 1125 1117 1110 1102 1095 1087 1080 1072 1065 1057 1050 1042 1035 1027 
        1020 1012 1005 997 990 982 975 967 960 952 945 937 930 922 915 907 900 892 885 877 
        870 862 855 847 840 832 825 817 810 802 795 787 780 772 765 757 750 742 735 727 
        720 712 705 697 690 682 675 667 660 652 645 637 630 622 615 607 600 592 585 577 
        570 562 555 547 540 532 525 517 510 502 495 487 480 472 465 457 450 442 435 427 
        420 412 405
    )
else
    log_error "Unsupported GPU type: $GPU_TYPE. Supported types: A100, V100"
    exit 1
fi

# Application configuration
declare -A APPLICATIONS=(
    ["LSTM"]="lstm"  # Display name -> executable name
    # Add more applications here as needed
    # ["StableDiffusion"]="stable_diffusion"
    # ["LLaMA"]="llama"
)

declare -A APP_PARAMS=(
    ["LSTM"]=" > results/LSTM_RUN_OUT"
    # Add more application parameters here
)

# File and directory configuration
readonly TEMP_OUTPUT_FILE="changeme"
readonly RESULTS_DIR="results"

# Profiling tool configuration
if [[ "$PROFILING_TOOL" == "dcgmi" ]]; then
    readonly PROFILE_SCRIPT="./profile.py"  # Python-based DCGMI profiler
    readonly CONTROL_SCRIPT="./control.sh"  # DCGMI-based frequency control
elif [[ "$PROFILING_TOOL" == "nvidia-smi" ]]; then
    readonly PROFILE_SCRIPT="./profile_smi.py"  # nvidia-smi based profiler (if exists)
    readonly CONTROL_SCRIPT="./control_smi.sh"  # nvidia-smi based frequency control (if exists)
else
    log_error "Unsupported profiling tool: $PROFILING_TOOL. Supported tools: dcgmi, nvidia-smi"
    exit 1
fi

# ============================================================================
# Utility Functions
# ============================================================================

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

# Function to display script usage
usage() {
    cat << EOF
Usage: $(basename "$0")

AI Inference Energy Profiling Launch Script

This script runs comprehensive energy profiling experiments across different
GPU frequencies for AI inference workloads.

Configuration:
    Edit the Configuration section in this script to customize:
    
    GPU Configuration:
    - GPU Type: $GPU_TYPE (Supported: A100, V100)
    - GPU Architecture: $GPU_ARCH
    - Profiling Tool: $PROFILING_TOOL (Supported: dcgmi, nvidia-smi)
    
    Experiment Configuration:
    - Number of runs per frequency: $NUM_RUNS
    - Memory frequency: ${MEMORY_FREQ}MHz
    - Core frequencies: ${#CORE_FREQUENCIES[@]} frequencies from ${CORE_FREQUENCIES[0]} to ${CORE_FREQUENCIES[-1]}MHz
    - Applications: ${!APPLICATIONS[*]}

GPU Type Selection:
    To switch between GPU types, edit the GPU_TYPE variable:
    - For A100: GPU_TYPE="A100"
    - For V100: GPU_TYPE="V100"
    
    This automatically configures:
    - Architecture identifier (GA100/GV100)
    - Memory frequencies (1215MHz/877MHz)
    - Default core frequencies (1410MHz/1380MHz)
    - Available frequency ranges

Profiling Tool Selection:
    To switch between profiling tools, edit the PROFILING_TOOL variable:
    - For DCGMI: PROFILING_TOOL="dcgmi"
    - For nvidia-smi: PROFILING_TOOL="nvidia-smi"
    
    This automatically selects the appropriate scripts:
    - DCGMI: ./profile.py and ./control.sh
    - nvidia-smi: ./profile_smi.py and ./control_smi.sh

Output:
    Results are saved to the '$RESULTS_DIR' directory with the naming convention:
    \$ARCH-\$MODE-\$APP-\$FREQ-\$ITERATION

Requirements:
    - NVIDIA GPU (A100 or V100)
    - GPU profiling tools (DCGMI or nvidia-smi)
    - GPU profiling and control scripts
    - AI inference applications
    - Sufficient disk space for results

EOF
}

# Function to check prerequisites
check_prerequisites() {
    log_info "Checking prerequisites..."
    log_info "GPU Type: $GPU_TYPE ($GPU_ARCH)"
    log_info "Profiling Tool: $PROFILING_TOOL"
    
    # Check required scripts
    local required_scripts=("$PROFILE_SCRIPT" "$CONTROL_SCRIPT")
    for script in "${required_scripts[@]}"; do
        if [[ ! -x "$script" ]]; then
            log_error "Required script not found or not executable: $script"
            log_info "Make sure the script exists and is executable: chmod +x $script"
            return 1
        fi
    done
    
    # Check profiling tool availability
    if [[ "$PROFILING_TOOL" == "dcgmi" ]]; then
        if ! command -v dcgmi &> /dev/null; then
            log_error "dcgmi command not found. Please install NVIDIA DCGMI tools."
            log_info "Install DCGMI: https://developer.nvidia.com/dcgm"
            return 1
        fi
        log_info "✓ DCGMI available"
    elif [[ "$PROFILING_TOOL" == "nvidia-smi" ]]; then
        if ! command -v nvidia-smi &> /dev/null; then
            log_error "nvidia-smi command not found. Please install NVIDIA drivers."
            return 1
        fi
        log_info "✓ nvidia-smi available"
    fi
    
    # Check GPU type compatibility
    if command -v nvidia-smi &> /dev/null; then
        local gpu_name
        gpu_name=$(nvidia-smi --query-gpu=name --format=csv,noheader,nounits | head -n1)
        log_info "Detected GPU: $gpu_name"
        
        if [[ "$GPU_TYPE" == "A100" && ! "$gpu_name" =~ A100 ]]; then
            log_warning "GPU type set to A100 but detected GPU doesn't appear to be A100: $gpu_name"
            log_warning "Consider changing GPU_TYPE in the configuration section"
        elif [[ "$GPU_TYPE" == "V100" && ! "$gpu_name" =~ V100 ]]; then
            log_warning "GPU type set to V100 but detected GPU doesn't appear to be V100: $gpu_name"
            log_warning "Consider changing GPU_TYPE in the configuration section"
        fi
    fi
    
    # Create results directory
    if [[ ! -d "$RESULTS_DIR" ]]; then
        log_info "Creating results directory: $RESULTS_DIR"
        mkdir -p "$RESULTS_DIR"
    fi
    
    log_info "Prerequisites check completed successfully"
    return 0
}

# Function to set GPU frequency using the control script
set_gpu_frequency() {
    local core_freq="$1"
    
    log_info "Setting GPU frequency: Memory=${MEMORY_FREQ}MHz, Core=${core_freq}MHz"
    
    if ! "$CONTROL_SCRIPT" "$MEMORY_FREQ" "$core_freq"; then
        log_error "Failed to set GPU frequency to ${core_freq}MHz"
        return 1
    fi
    
    return 0
}

# Function to run a single application with profiling
run_application() {
    local app_name="$1"
    local app_executable="$2"
    local app_params="$3"
    local frequency="$4"
    local iteration="$5"
    
    local app_command="python ${app_executable}.py${app_params}"
    local output_file="${RESULTS_DIR}/${GPU_ARCH}-${PROFILING_MODE}-${app_name}-${frequency}-${iteration}"
    
    log_info "Running $app_name at ${frequency}MHz (iteration $iteration)"
    log_info "Command: $app_command"
    log_info "Output file: $output_file"
    
    # Record start time
    local start_time
    start_time=$(date +%s)
    
    # Run application with profiling
    if ! "$PROFILE_SCRIPT" $app_command; then
        log_error "Failed to run application: $app_name"
        return 1
    fi
    
    # Calculate execution time
    local end_time
    end_time=$(date +%s)
    local duration=$((end_time - start_time))
    
    # Move profiling output to results file
    if [[ -f "$TEMP_OUTPUT_FILE" ]]; then
        mv "$TEMP_OUTPUT_FILE" "$output_file"
        log_info "Profiling data saved to: $output_file"
    else
        log_warning "Profiling output file not found: $TEMP_OUTPUT_FILE"
    fi
    
    # Extract and save performance metrics for LSTM
    if [[ "$app_name" == "LSTM" ]]; then
        extract_lstm_performance "$frequency" "$duration"
    fi
    
    log_info "Application completed in ${duration}s"
    return 0
}

# Function to extract LSTM performance metrics
extract_lstm_performance() {
    local frequency="$1"
    local wall_time="$2"
    local performance_file="${RESULTS_DIR}/${GPU_ARCH}-dvfs-lstm-perf.csv"
    
    # Try to extract execution time from LSTM output
    local lstm_output_file="${RESULTS_DIR}/LSTM_RUN_OUT"
    if [[ -f "$lstm_output_file" ]]; then
        # Extract execution time from line 20 (as in original script)
        if (( $(wc -l < "$lstm_output_file") >= 20 )); then
            local line_20
            line_20=$(sed -n '20p' "$lstm_output_file")
            
            # Extract first token (execution time)
            local exec_time
            exec_time=$(echo "$line_20" | awk '{print $1}')
            
            # Clean up the execution time
            exec_time=$(echo "$exec_time" | tr -d '\t\r\n ')
            
            # Save to performance file
            printf '%s,%s\n' "$frequency" "$exec_time" >> "$performance_file"
            log_info "LSTM performance saved: ${frequency}MHz -> ${exec_time}s"
        else
            log_warning "LSTM output file has fewer than 20 lines"
        fi
    else
        log_warning "LSTM output file not found: $lstm_output_file"
    fi
}

# Function to run experiments for all frequencies
run_frequency_sweep() {
    local total_runs=0
    local completed_runs=0
    
    # Calculate total number of runs
    total_runs=$((${#CORE_FREQUENCIES[@]} * NUM_RUNS * ${#APPLICATIONS[@]}))
    log_info "Starting frequency sweep: $total_runs total runs"
    
    # Iterate through core frequencies
    for core_freq in "${CORE_FREQUENCIES[@]}"; do
        log_info "Processing frequency: ${core_freq}MHz"
        
        # Set GPU frequency
        if ! set_gpu_frequency "$core_freq"; then
            log_error "Skipping frequency ${core_freq}MHz due to frequency setting failure"
            continue
        fi
        
        # Run multiple iterations for this frequency
        for iteration in $(seq 0 $((NUM_RUNS - 1))); do
            log_info "Iteration $(( iteration + 1 ))/$NUM_RUNS for ${core_freq}MHz"
            
            # Run each application
            for app_name in "${!APPLICATIONS[@]}"; do
                local app_executable="${APPLICATIONS[$app_name]}"
                local app_params="${APP_PARAMS[$app_name]:-}"
                
                if ! run_application "$app_name" "$app_executable" "$app_params" "$core_freq" "$iteration"; then
                    log_error "Failed to run $app_name at ${core_freq}MHz (iteration $iteration)"
                    continue
                fi
                
                ((completed_runs++))
                log_info "Progress: $completed_runs/$total_runs runs completed"
                
                # Sleep between runs
                if (( SLEEP_INTERVAL > 0 )); then
                    sleep "$SLEEP_INTERVAL"
                fi
            done
        done
    done
    
    log_info "Frequency sweep completed: $completed_runs/$total_runs runs successful"
    return 0
}

# Function to restore default GPU frequency
restore_default_frequency() {
    log_info "Restoring default GPU frequency: ${DEFAULT_CORE_FREQ}MHz"
    
    if ! set_gpu_frequency "$DEFAULT_CORE_FREQ"; then
        log_warning "Failed to restore default GPU frequency"
    else
        log_info "Default GPU frequency restored successfully"
    fi
}

# Function to cleanup on exit
cleanup() {
    local exit_code=$?
    
    log_info "Cleaning up..."
    
    # Remove temporary files
    if [[ -f "$TEMP_OUTPUT_FILE" ]]; then
        rm -f "$TEMP_OUTPUT_FILE"
    fi
    
    # Restore default frequency
    restore_default_frequency
    
    if (( exit_code == 0 )); then
        log_info "Experiment completed successfully"
    else
        log_error "Experiment terminated with errors (exit code: $exit_code)"
    fi
    
    exit "$exit_code"
}

# ============================================================================
# Main Function
# ============================================================================

main() {
    # Set up signal handlers
    trap cleanup EXIT
    trap 'log_error "Interrupted by user"; exit 130' INT TERM
    
    log_info "Starting AI inference energy profiling experiment"
    log_info "Configuration:"
    log_info "  GPU Type: $GPU_TYPE"
    log_info "  GPU Architecture: $GPU_ARCH"
    log_info "  Profiling Tool: $PROFILING_TOOL"
    log_info "  Profiling Mode: $PROFILING_MODE"
    log_info "  Runs per frequency: $NUM_RUNS"
    log_info "  Memory frequency: ${MEMORY_FREQ}MHz"
    log_info "  Default core frequency: ${DEFAULT_CORE_FREQ}MHz"
    log_info "  Core frequencies: ${#CORE_FREQUENCIES[@]} frequencies (${CORE_FREQUENCIES[0]}-${CORE_FREQUENCIES[-1]}MHz)"
    log_info "  Applications: ${!APPLICATIONS[*]}"
    log_info "  Results directory: $RESULTS_DIR"
    log_info "  Profile script: $PROFILE_SCRIPT"
    log_info "  Control script: $CONTROL_SCRIPT"
    
    # Check command line arguments
    if (( $# > 0 )); then
        if [[ "$1" == "-h" || "$1" == "--help" ]]; then
            usage
            exit 0
        else
            log_error "Unknown argument: $1"
            usage
            exit 1
        fi
    fi
    
    # Check prerequisites
    if ! check_prerequisites; then
        exit 1
    fi
    
    # Run the frequency sweep experiment
    if ! run_frequency_sweep; then
        exit 1
    fi
    
    log_info "All experiments completed successfully!"
}

# Execute main function with all arguments
main "$@"
