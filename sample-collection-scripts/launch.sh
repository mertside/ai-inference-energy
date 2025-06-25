#!/bin/bash
"""
AI Inference Energy Profiling Launch Script.

This script orchestrates energy profiling experiments for AI inference workloads
across different GPU frequencies. It systematically tests various core frequencies
while monitoring power consumption and performance metrics.

The script:
1. Iterates through a range of GPU core frequencies
2. Runs each AI application multiple times per frequency
3. Collects power and performance data using GPU profiling tools
4. Saves results for analysis

Usage:
    ./launch.sh

Configuration:
    Edit the variables in the Configuration section below to customize:
    - Number of runs per frequency
    - GPU architecture and frequencies
    - Applications to test
    - Output directories

Requirements:
    - NVIDIA GPU with DCGMI support
    - AI inference applications (LLaMA, Stable Diffusion, etc.)
    - GPU profiling tools (dcgmi, profile script)
    - Bash 4.0+ for associative arrays

Author: AI Inference Energy Research Team
"""

set -euo pipefail  # Exit on error, undefined variables, and pipe failures

# ============================================================================
# Configuration Section
# ============================================================================

# Experiment configuration
readonly NUM_RUNS=2
readonly PROFILING_MODE="dvfs"
readonly GPU_ARCH="GA100"  # A100 architecture
readonly SLEEP_INTERVAL=1  # seconds between runs

# GPU frequency configuration
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
readonly PROFILE_SCRIPT="./profile"
readonly CONTROL_SCRIPT="./control"

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
    - Number of runs per frequency: $NUM_RUNS
    - GPU architecture: $GPU_ARCH
    - Memory frequency: ${MEMORY_FREQ}MHz
    - Core frequencies: ${#CORE_FREQUENCIES[@]} frequencies from ${CORE_FREQUENCIES[0]} to ${CORE_FREQUENCIES[-1]}MHz
    - Applications: ${!APPLICATIONS[*]}

Output:
    Results are saved to the '$RESULTS_DIR' directory with the naming convention:
    \$ARCH-\$MODE-\$APP-\$FREQ-\$ITERATION

Requirements:
    - NVIDIA GPU with DCGMI support
    - GPU profiling and control scripts (profile, control)
    - AI inference applications
    - Sufficient disk space for results

EOF
}

# Function to check prerequisites
check_prerequisites() {
    log_info "Checking prerequisites..."
    
    # Check required scripts
    local required_scripts=("$PROFILE_SCRIPT" "$CONTROL_SCRIPT")
    for script in "${required_scripts[@]}"; do
        if [[ ! -x "$script" ]]; then
            log_error "Required script not found or not executable: $script"
            return 1
        fi
    done
    
    # Check DCGMI availability
    if ! command -v dcgmi &> /dev/null; then
        log_error "dcgmi command not found. Please install NVIDIA DCGMI tools."
        return 1
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
    log_info "  GPU Architecture: $GPU_ARCH"
    log_info "  Profiling Mode: $PROFILING_MODE"
    log_info "  Runs per frequency: $NUM_RUNS"
    log_info "  Memory frequency: ${MEMORY_FREQ}MHz"
    log_info "  Core frequencies: ${#CORE_FREQUENCIES[@]} frequencies"
    log_info "  Applications: ${!APPLICATIONS[*]}"
    log_info "  Results directory: $RESULTS_DIR"
    
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
