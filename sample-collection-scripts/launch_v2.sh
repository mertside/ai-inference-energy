#!/bin/bash
#
# AI Inference Energy Profiling Framework - Main Launch Script
#
# This is the main entry point for the AI inference energy profiling framework.
# It orchestrates energy profiling experiments for AI inference workloads across
# different GPU frequencies, supporting NVIDIA A100, V100, and H100 GPUs.
#
# The script provides:
# - Complete command-line interface with comprehensive options
# - Automatic GPU detection and configuration
# - Support for DCGMI and nvidia-smi profiling tools
# - DVFS (full frequency sweep) and baseline profiling modes
# - Robust error handling and progress reporting
# - Modular architecture for easy maintenance
#
# Author: Mert Side
# Version: 2.0.1 (Enhanced)
#

set -eo pipefail  # Exit on error and pipe failures

# =============================================================================
# Script Setup and Library Loading
# =============================================================================

# Get script directory and setup paths
readonly SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly LIB_DIR="${SCRIPT_DIR}/lib"
readonly CONFIG_DIR="${SCRIPT_DIR}/config"

# Verify library directory exists
if [[ ! -d "$LIB_DIR" ]]; then
    echo "ERROR: Library directory not found: $LIB_DIR" >&2
    echo "Please ensure the lib/ directory exists with required libraries." >&2
    exit 1
fi

# Load core libraries in dependency order
source "${LIB_DIR}/common.sh"         # Common utilities and logging
source "${LIB_DIR}/gpu_config.sh"     # GPU configuration and detection
source "${LIB_DIR}/profiling.sh"      # Profiling orchestration
source "${LIB_DIR}/args_parser.sh"    # Argument parsing and validation

# =============================================================================
# Main Script Functions
# =============================================================================

# Initialize the profiling environment
initialize_environment() {
    log_info "Initializing AI Inference Energy Profiling Framework v${FRAMEWORK_VERSION}"
    
    # Setup signal handlers for cleanup
    setup_signal_handlers
    
    # Create necessary directories
    ensure_directory "$PARSED_OUTPUT_DIR"
    ensure_directory "${SCRIPT_DIR}/logs"
    ensure_directory "${SCRIPT_DIR}/tmp"
    
    # Configure output redirection
    local log_file="${SCRIPT_DIR}/logs/launch_$(date +%Y%m%d_%H%M%S).log"
    
    # Enable debug logging if requested
    if $PARSED_DEBUG_MODE; then
        log_info "Debug mode enabled - detailed logging active"
        exec 19> >(tee -a "$log_file" >&2)
        export BASH_XTRACEFD=19
        set -x
    fi
    
    log_info "Environment initialized successfully"
    log_debug "Log file: $log_file"
}

# Validate system prerequisites
check_system_prerequisites() {
    log_info "Checking system prerequisites..."
    
    local missing_tools=()
    local warnings=()
    
    # Check for essential tools
    if ! command_exists "nvidia-smi"; then
        missing_tools+=("nvidia-smi (NVIDIA drivers)")
    fi
    
    if ! command_exists "python3"; then
        missing_tools+=("python3")
    fi
    
    # Check for profiling tools
    local has_profiling_tool=false
    if command_exists "dcgmi"; then
        log_debug "DCGMI available"
        has_profiling_tool=true
    fi
    
    if command_exists "nvidia-smi"; then
        log_debug "nvidia-smi available"
        has_profiling_tool=true
    fi
    
    if ! $has_profiling_tool; then
        missing_tools+=("profiling tools (dcgmi or nvidia-smi)")
    fi
    
    # Check for optional tools
    if ! command_exists "conda"; then
        warnings+=("conda not available - using system Python")
    fi
    
    # Report missing tools
    if [[ ${#missing_tools[@]} -gt 0 ]]; then
        log_error "Missing required tools:"
        printf "  - %s\n" "${missing_tools[@]}" >&2
        die "Please install missing tools before running the profiling framework"
    fi
    
    # Report warnings
    if [[ ${#warnings[@]} -gt 0 ]]; then
        for warning in "${warnings[@]}"; do
            log_warning "$warning"
        done
    fi
    
    # Check GPU availability
    if ! is_gpu_available; then
        die "No GPU detected. Please ensure NVIDIA GPU is available and drivers are installed."
    fi
    
    local gpu_count
    gpu_count=$(get_gpu_count)
    log_info "System check passed - $gpu_count GPU(s) detected"
}

# Validate application and dependencies
validate_application() {
    log_info "Validating application: $PARSED_APP_EXECUTABLE"
    
    # Try to resolve application path
    local resolution_result
    if ! resolution_result=$(resolve_application_path "$PARSED_APP_EXECUTABLE"); then
        die "Failed to resolve application path: $PARSED_APP_EXECUTABLE"
    fi
    
    local app_path app_dir
    IFS='|' read -r app_path app_dir <<< "$resolution_result"
    
    log_success "Application found: $app_path"
    log_debug "Application directory: $app_dir"
    
    # Validate conda environment
    local conda_env
    conda_env=$(determine_conda_env "$PARSED_APP_EXECUTABLE" "$app_dir")
    
    if command_exists "conda"; then
        if validate_conda_env "$conda_env"; then
            log_info "Conda environment validated: $conda_env"
        else
            log_warning "Conda environment '$conda_env' not found - using system Python"
        fi
    fi
    
    log_success "Application validation completed"
}

# Run the main profiling experiment
run_profiling_experiment() {
    log_info "Starting profiling experiment"
    log_info "Mode: $PARSED_PROFILING_MODE, Tool: $PARSED_PROFILING_TOOL, GPU: $PARSED_GPU_TYPE"
    
    # Parse application parameters into array
    local app_params_array=()
    if [[ -n "$PARSED_APP_PARAMS" ]]; then
        # Use bash read to properly parse quoted parameters
        while IFS= read -r -d '' param; do
            app_params_array+=("$param")
        done < <(printf '%s\0' $PARSED_APP_PARAMS)
    fi
    
    log_debug "Application parameters: ${#app_params_array[@]} items"
    for i in "${!app_params_array[@]}"; do
        log_debug "  [$i]: ${app_params_array[$i]}"
    done
    
    # Run experiment based on mode
    case "$PARSED_PROFILING_MODE" in
        dvfs)
            run_dvfs_experiment \
                "$PARSED_PROFILING_TOOL" \
                "$PARSED_GPU_TYPE" \
                "$PARSED_APP_EXECUTABLE" \
                "$PARSED_OUTPUT_DIR" \
                "$PARSED_NUM_RUNS" \
                "$PARSED_SLEEP_INTERVAL" \
                "${app_params_array[@]}"
            ;;
        custom)
            run_custom_experiment \
                "$PARSED_PROFILING_TOOL" \
                "$PARSED_GPU_TYPE" \
                "$PARSED_APP_EXECUTABLE" \
                "$PARSED_OUTPUT_DIR" \
                "$PARSED_NUM_RUNS" \
                "$PARSED_SLEEP_INTERVAL" \
                "$PARSED_CUSTOM_FREQUENCIES" \
                "${app_params_array[@]}"
            ;;
        baseline)
            run_baseline_experiment \
                "$PARSED_PROFILING_TOOL" \
                "$PARSED_GPU_TYPE" \
                "$PARSED_APP_EXECUTABLE" \
                "$PARSED_OUTPUT_DIR" \
                "$PARSED_NUM_RUNS" \
                "$PARSED_SLEEP_INTERVAL" \
                "${app_params_array[@]}"
            ;;
        *)
            die "Unknown profiling mode: $PARSED_PROFILING_MODE"
            ;;
    esac
}

# Generate experiment summary
generate_experiment_summary() {
    local summary_file="${PARSED_OUTPUT_DIR}/experiment_summary.log"
    local end_time
    end_time=$(date '+%Y-%m-%d %H:%M:%S')
    
    log_info "Generating experiment summary: $summary_file"
    
    # Create basic summary with error handling
    {
        cat << EOF
AI Inference Energy Profiling Experiment Summary
================================================

Experiment Details:
  Framework Version: ${FRAMEWORK_VERSION}
  Timestamp: ${end_time}
  Mode: ${PARSED_PROFILING_MODE}
  
GPU Configuration:
  Type: ${PARSED_GPU_TYPE}
  Architecture: $(get_gpu_architecture "$PARSED_GPU_TYPE")
  Memory Frequency: $(get_gpu_memory_freq "$PARSED_GPU_TYPE") MHz
  Core Frequency Range: $(get_gpu_core_freq_min "$PARSED_GPU_TYPE")-$(get_gpu_core_freq_max "$PARSED_GPU_TYPE") MHz

Profiling Configuration:
  Tool: ${PARSED_PROFILING_TOOL}
  Runs per frequency: ${PARSED_NUM_RUNS}
  Sleep interval: ${PARSED_SLEEP_INTERVAL}s$(
    if [[ "$PARSED_PROFILING_MODE" == "custom" ]]; then
        echo ""
        echo "  Custom frequencies: ${PARSED_CUSTOM_FREQUENCIES} MHz"
    fi
)

Application Configuration:
  Name: ${PARSED_APP_NAME}
  Executable: ${PARSED_APP_EXECUTABLE}
  Parameters: ${PARSED_APP_PARAMS:-"(none)"}

Output Files:
EOF
    } > "$summary_file" || {
        log_error "Failed to create basic summary file"
        return 1
    }
    
    # List generated output files with error handling
    if [[ -d "$PARSED_OUTPUT_DIR" ]]; then
        find "$PARSED_OUTPUT_DIR" -name "*.csv" -o -name "*.out" -o -name "*.err" -o -name "*.log" 2>/dev/null | \
            sort | sed 's/^/  /' >> "$summary_file" || true
    fi
    
    # Add timing summary if available - with robust error handling
    local timing_file="${PARSED_OUTPUT_DIR}/timing_summary.log"
    if [[ -f "$timing_file" ]]; then
        {
            echo ""
            echo "Run Timing Summary:"
            echo "==================="
        } >> "$summary_file" || {
            log_warning "Failed to write timing summary header"
        }
        
        # Process timing file with maximum error tolerance
        generate_timing_statistics "$timing_file" "$summary_file" || {
            log_warning "Failed to generate detailed timing statistics, adding basic completion note"
            echo "  Timing details available in: $timing_file" >> "$summary_file" || true
        }
    fi
    
    # Add completion timestamp with error handling
    {
        echo ""
        echo "Experiment completed at: ${end_time}"
    } >> "$summary_file" || true
    
    log_success "Experiment summary saved to: $summary_file"
}

# Helper function to generate timing statistics with error handling
generate_timing_statistics() {
    local timing_file="$1"
    local summary_file="$2"
    
    local total_runs=0
    local total_duration=0
    local successful_runs=0
    local failed_runs=0
    local min_duration=999999
    local max_duration=0
    
    # Process each line with individual error handling
    while IFS=',' read -r run_id freq duration exit_code status || [[ -n "$run_id" ]]; do
        # Skip header and comment lines
        [[ "$run_id" =~ ^#.*$ ]] && continue
        [[ -z "$run_id" ]] && continue
        
        # Validate and process numeric fields
        if [[ "$duration" =~ ^[0-9]+$ ]] && [[ "$freq" =~ ^[0-9]+$ ]]; then
            ((total_runs++)) || true
            total_duration=$((total_duration + duration)) || true
            
            if [[ "$status" == "success" ]]; then
                ((successful_runs++))
            else
                ((failed_runs++))
            fi
            
            # Update min/max with safe comparisons
            [[ $duration -lt $min_duration ]] && min_duration=$duration
            [[ $duration -gt $max_duration ]] && max_duration=$duration
            
            # Write individual run details
            printf "  Run %-15s: %3ds (freq: %4dMHz, status: %s)\n" \
                "$run_id" "$duration" "$freq" "$status" >> "$summary_file" || true
        fi
    done < "$timing_file"
    
    # Add summary statistics if we have valid data
    if [[ $total_runs -gt 0 ]] && [[ $total_duration -gt 0 ]]; then
        local avg_duration=$((total_duration / total_runs))
        {
            echo ""
            echo "Timing Statistics:"
            printf "  Total runs:       %d\n" "$total_runs"
            printf "  Successful runs:  %d\n" "$successful_runs"
            printf "  Failed runs:      %d\n" "$failed_runs"
            printf "  Total duration:   %ds\n" "$total_duration"
            printf "  Average duration: %ds\n" "$avg_duration"
            printf "  Min duration:     %ds\n" "$min_duration"
            printf "  Max duration:     %ds\n" "$max_duration"
        } >> "$summary_file" || true
    fi
}

# Cleanup function
cleanup_experiment() {
    log_debug "Running experiment cleanup..."
    
    # Reset GPU frequencies to default (if not baseline mode)
    if [[ "$PARSED_PROFILING_MODE" != "baseline" ]]; then
        log_info "Resetting GPU frequencies to defaults..."
        reset_gpu_frequency "$PARSED_PROFILING_TOOL" "$PARSED_GPU_TYPE" || true
    fi
    
    # Clean up temporary files
    local temp_files
    temp_files=$(find "${SCRIPT_DIR}/tmp" -name "run_app_*.py" 2>/dev/null || true)
    if [[ -n "$temp_files" ]]; then
        log_debug "Cleaning up temporary files..."
        echo "$temp_files" | xargs rm -f
    fi
    
    log_debug "Cleanup completed"
}

# =============================================================================
# Main Function
# =============================================================================

main() {
    local start_time end_time duration
    start_time=$(date +%s)
    
    # Process command-line arguments
    process_arguments "$@"
    
    # Initialize environment
    initialize_environment
    
    # Register cleanup function
    register_cleanup cleanup_experiment
    
    # System validation
    check_system_prerequisites
    validate_application
    
    # Show GPU configuration summary
    show_gpu_config "$PARSED_GPU_TYPE"
    
    # Confirm before starting (unless in non-interactive mode)
    if [[ -t 0 ]] && [[ "${SKIP_CONFIRMATION:-false}" != "true" ]]; then
        echo -n "Press Enter to start the experiment, or Ctrl+C to cancel... "
        read -r
    fi
    
    # Run the main experiment
    run_profiling_experiment
    
    # Generate summary with error tolerance
    generate_experiment_summary || {
        log_warning "Summary generation encountered issues, but experiment data was collected successfully"
        log_info "All profiling data is available in: $PARSED_OUTPUT_DIR"
    }
    
    # Calculate and display duration
    end_time=$(date +%s)
    duration=$((end_time - start_time))
    
    log_success "Profiling experiment completed successfully!"
    log_info "Total duration: $(printf '%02d:%02d:%02d' $((duration/3600)) $((duration%3600/60)) $((duration%60)))"
    log_info "Results saved to: $PARSED_OUTPUT_DIR"
    
    return 0
}

# =============================================================================
# Script Entry Point
# =============================================================================

# Only run main if script is executed directly (not sourced)
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi
