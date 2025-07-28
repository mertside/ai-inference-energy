#!/bin/bash
#
# Argument Parsing Library for AI Inference Energy Profiling
#
# This library provides robust command-line argument parsing with validation,
# help generation, and configuration management.
#
# Author: Mert Side
#

# Prevent multiple inclusions
if [[ "${ARGS_LIB_LOADED:-}" == "true" ]]; then
    return 0
fi
readonly ARGS_LIB_LOADED="true"

# Load dependencies
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
source "$(dirname "${BASH_SOURCE[0]}")/gpu_config.sh"
source "$(dirname "${BASH_SOURCE[0]}")/../config/defaults.sh"

# =============================================================================
# Argument Parsing Library
# =============================================================================

readonly ARGS_LIB_VERSION="1.0.0"

# Global variables for parsed arguments
declare -g PARSED_GPU_TYPE=""
declare -g PARSED_PROFILING_TOOL=""
declare -g PARSED_PROFILING_MODE=""
declare -g PARSED_CUSTOM_FREQUENCIES=""
declare -g PARSED_NUM_RUNS=""
declare -g PARSED_SLEEP_INTERVAL=""
declare -g PARSED_APP_NAME=""
declare -g PARSED_APP_EXECUTABLE=""
declare -g PARSED_APP_PARAMS=""
declare -g PARSED_OUTPUT_DIR=""
declare -g PARSED_HELP_REQUESTED=false
declare -g PARSED_VERSION_REQUESTED=false
declare -g PARSED_DEBUG_MODE=false

# Setup colors for this output context
setup_output_colors() {
    if should_use_colors; then
        COLOR_RED='\033[0;31m'
        COLOR_GREEN='\033[0;32m'
        COLOR_YELLOW='\033[1;33m'
        COLOR_BLUE='\033[0;34m'
        COLOR_PURPLE='\033[0;35m'
        COLOR_CYAN='\033[0;36m'
        COLOR_NC='\033[0m'
    else
        COLOR_RED=''
        COLOR_GREEN=''
        COLOR_YELLOW=''
        COLOR_BLUE=''
        COLOR_PURPLE=''
        COLOR_CYAN=''
        COLOR_NC=''
    fi
}

# Display detailed usage information
show_usage() {
    setup_output_colors
    
    printf "%sAI Inference Energy Profiling Framework v%s%s\n\n" "$COLOR_GREEN" "$FRAMEWORK_VERSION" "$COLOR_NC"
    printf "%sUSAGE:%s\n" "$COLOR_BLUE" "$COLOR_NC"
    printf "    %s$(basename "$0")%s [OPTIONS]\n\n" "$COLOR_CYAN" "$COLOR_NC"
    printf "%sDESCRIPTION:%s\n" "$COLOR_BLUE" "$COLOR_NC"
    printf "    Orchestrates energy profiling experiments for AI inference workloads across\n"
    printf "    different GPU frequencies. Supports NVIDIA A100, V100, and H100 GPUs with\n"
    printf "    both DCGMI and nvidia-smi profiling tools.\n\n"
    
    printf "%sOPTIONS:%s\n" "$COLOR_BLUE" "$COLOR_NC"
    printf "    %sGPU Configuration:%s\n" "$COLOR_YELLOW" "$COLOR_NC"
    printf "    --gpu-type TYPE          GPU type: A100, V100, or H100\n"
    printf "                             (default: %s, auto-detected when possible)\n" "$DEFAULT_GPU_TYPE"
    printf "    \n"
    printf "    %sProfiling Configuration:%s\n" "$COLOR_YELLOW" "$COLOR_NC"
    printf "    --profiling-tool TOOL    Profiling tool: dcgmi or nvidia-smi\n"
    printf "                             (default: %s, auto-fallback enabled)\n" "$DEFAULT_PROFILING_TOOL"
    printf "    --profiling-mode MODE    Profiling mode: dvfs, custom, or baseline\n"
    printf "                             (default: %s)\n" "$DEFAULT_PROFILING_MODE"
    printf "                             dvfs: Full frequency sweep (comprehensive)\n"
    printf "                             custom: Selected frequencies (efficient)\n"
    printf "                             baseline: Single frequency (quick validation)\n"
    printf "    --custom-frequencies FREQS Comma-separated list of frequencies (MHz)\n"
    printf "                             Required when profiling-mode is 'custom'\n"
    printf "                             Example: '405,892,1380' for low/mid/high\n"
    printf "    \n"
    printf "    %sExperiment Configuration:%s\n" "$COLOR_YELLOW" "$COLOR_NC"
    printf "    --num-runs NUM           Number of runs per frequency\n"
    printf "                             (default: %s)\n" "$DEFAULT_NUM_RUNS"
    printf "    --sleep-interval SEC     Sleep between runs in seconds\n"
    printf "                             (default: %s)\n" "$DEFAULT_SLEEP_INTERVAL"
    printf "    \n"
    printf "    %sApplication Configuration:%s\n" "$COLOR_YELLOW" "$COLOR_NC"
    printf "    --app-name NAME          Application display name\n"
    printf "                             (default: %s)\n" "$DEFAULT_APP_NAME"
    printf "    --app-executable PATH    Application executable path\n"
    printf "                             (default: %s)\n" "$DEFAULT_APP_EXECUTABLE"
    printf "    --app-params \"PARAMS\"    Application parameters (quoted string)\n"
    printf "                             (default: \"%s\")\n" "$DEFAULT_APP_PARAMS"
    printf "    \n"
    printf "    %sOutput Configuration:%s\n" "$COLOR_YELLOW" "$COLOR_NC"
    printf "    --output-dir DIR         Output directory for results\n"
    printf "                             (default: auto-generated as results_GPU_APP_job_ID)\n"
    printf "                             Job ID automatically appended in SLURM environments\n"
    printf "                             (fallback: %s)\n" "$DEFAULT_OUTPUT_DIR"
    printf "    \n"
    printf "    %sGeneral Options:%s\n" "$COLOR_YELLOW" "$COLOR_NC"
    printf "    --debug                  Enable debug output\n"
    printf "    --version                Show version information\n"
    printf "    -h, --help               Show this help message\n\n"
    
    printf "%sEXAMPLES:%s\n" "$COLOR_BLUE" "$COLOR_NC"
    printf "    %s# Default A100 DVFS experiment%s\n" "$COLOR_CYAN" "$COLOR_NC"
    printf "    $(basename "$0")\n"
    printf "    \n"
    printf "    %s# Quick V100 baseline test%s\n" "$COLOR_CYAN" "$COLOR_NC"
    printf "    $(basename "$0") --gpu-type V100 --profiling-mode baseline --num-runs 1\n"
    printf "    \n"
    printf "    %s# Custom frequency analysis (V100 low/mid/high)%s\n" "$COLOR_CYAN" "$COLOR_NC"
    printf "    $(basename "$0") --gpu-type V100 --profiling-mode custom \\\\\n"
    printf "        --custom-frequencies \"405,892,1380\" --num-runs 5\n"
    printf "    \n"
    printf "    %s# Custom Stable Diffusion profiling%s\n" "$COLOR_CYAN" "$COLOR_NC"
    printf "    $(basename "$0") \\\\\n"
    printf "        --app-name \"StableDiffusion\" \\\\\n"
    printf "        --app-executable \"StableDiffusionViaHF.py\" \\\\\n"
    printf "        --app-params \"--prompt 'A beautiful landscape' --steps 20\" \\\\\n"
    printf "        --num-runs 5\n"
    printf "    \n"
    printf "    %s# Whisper speech recognition benchmark%s\n" "$COLOR_CYAN" "$COLOR_NC"
    printf "    $(basename "$0") \\\\\n"
    printf "        --app-name \"Whisper\" \\\\\n"
    printf "        --app-executable \"WhisperViaHF.py\" \\\\\n"
    printf "        --app-params \"--benchmark --model base --num-samples 3 --quiet\" \\\\\n"
    printf "        --num-runs 3\n"
    printf "    \n"
    printf "    %s# LLaMA text generation benchmark%s\n" "$COLOR_CYAN" "$COLOR_NC"
    printf "    $(basename "$0") \\\\\n"
    printf "        --app-name \"LLaMA\" \\\\\n"
    printf "        --app-executable \"LlamaViaHF.py\" \\\\\n"
    printf "        --app-params \"--benchmark --num-generations 3 --quiet\" \\\\\n"
    printf "        --num-runs 3\n"
    printf "    \n"
    printf "    %s# Vision Transformer image classification%s\n" "$COLOR_CYAN" "$COLOR_NC"
    printf "    $(basename "$0") \\\\\n"
    printf "        --app-name \"ViT\" \\\\\n"
    printf "        --app-executable \"ViTViaHF.py\" \\\\\n"
    printf "        --app-params \"--benchmark-mode\" \\\\\n"
    printf "        --num-runs 3\n"
    printf "    \n"
    printf "    %s# H100 with nvidia-smi fallback%s\n" "$COLOR_CYAN" "$COLOR_NC"
    printf "    $(basename "$0") --gpu-type H100 --profiling-tool nvidia-smi\n\n"
    
    printf "%sGPU CONFIGURATIONS:%s\n" "$COLOR_BLUE" "$COLOR_NC"

    # Show supported GPU configurations
    for gpu_type in A100 V100 H100; do
        local architecture memory_freq min_freq max_freq partition cluster
        architecture=$(get_gpu_architecture "$gpu_type")
        memory_freq=$(get_gpu_memory_freq "$gpu_type")
        min_freq=$(get_gpu_core_freq_min "$gpu_type")
        max_freq=$(get_gpu_core_freq_max "$gpu_type")
        partition=$(get_slurm_partition "$gpu_type")
        cluster=$(get_cluster_name "$gpu_type")
        
        printf "    %s%s:%s %s, Memory: %sMHz, Core: %s-%sMHz, Partition: %s (%s)\n" \
            "$COLOR_YELLOW" "$gpu_type" "$COLOR_NC" "$architecture" "$memory_freq" "$min_freq" "$max_freq" "$partition" "$cluster"
    done

    printf "\n%sPROFILING MODES:%s\n" "$COLOR_BLUE" "$COLOR_NC"
    printf "    %sdvfs:%s     Full frequency sweep across all supported frequencies\n" "$COLOR_YELLOW" "$COLOR_NC"
    printf "                Comprehensive energy analysis, longer execution time\n"
    printf "    %scustom:%s   Selected frequencies for targeted analysis\n" "$COLOR_YELLOW" "$COLOR_NC"
    printf "                Efficient frequency sampling, moderate execution time\n"
    printf "    %sbaseline:%s Single frequency at default GPU settings\n" "$COLOR_YELLOW" "$COLOR_NC"
    printf "                Quick profiling for testing and validation\n\n"
    
    printf "%sPROFILING TOOLS:%s\n" "$COLOR_BLUE" "$COLOR_NC"
    printf "    %sdcgmi:%s      NVIDIA Data Center GPU Manager Interface\n" "$COLOR_YELLOW" "$COLOR_NC"
    printf "                Comprehensive GPU metrics, requires DCGMI installation\n"
    printf "    %snvidia-smi:%s NVIDIA System Management Interface\n" "$COLOR_YELLOW" "$COLOR_NC"
    printf "                Standard GPU monitoring, fallback option\n\n"
    
    printf "%sOUTPUT:%s\n" "$COLOR_BLUE" "$COLOR_NC"
    printf "    Results are saved to an auto-generated directory named results_GPU_APP_job_ID.\n"
    printf "    Examples:\n"
    printf "      • results_h100_stablediffusion_job_12345/  (SLURM environment)\n"
    printf "      • results_h100_stablediffusion/            (non-SLURM environment)\n"
    printf "    \n"
    printf "    Directory structure:\n"
    printf "    %s/\n" "${PARSED_OUTPUT_DIR:-results_GPU_APP_job_ID}"
    printf "    ├── run_XX_freq_YYYY_app.out      # Application output\n"
    printf "    ├── run_XX_freq_YYYY_app.err      # Application errors\n"
    printf "    ├── run_XX_freq_YYYY_profile.csv  # GPU profiling data\n"
    printf "    └── experiment_summary.log        # Overall experiment log\n\n"
    
    printf "%sREQUIREMENTS:%s\n" "$COLOR_BLUE" "$COLOR_NC"
    printf "    - NVIDIA GPU (A100, V100, or H100)\n"
    printf "    - CUDA drivers and toolkit\n"
    printf "    - Profiling tools (DCGMI or nvidia-smi)\n"
    printf "    - Python environment with required dependencies\n"
    printf "    - Sufficient disk space for results\n\n"
    
    printf "For more information, see the documentation in the docs/ directory.\n\n"
}

# Show version information
show_version_info() {
    cat << EOF
AI Inference Energy Profiling Framework
Version: ${FRAMEWORK_VERSION}
Argument Parser: v${ARGS_LIB_VERSION}
Common Library: v${COMMON_LIB_VERSION}
GPU Config Library: v${GPU_CONFIG_VERSION}

Supported GPU Types: ${!GPU_ARCHITECTURES[*]}
Supported Profiling Tools: dcgmi, nvidia-smi
Supported Profiling Modes: dvfs, custom, baseline

Build Information:
  Built on: $(date)
  Platform: $(uname -s) $(uname -m)
  Shell: ${BASH_VERSION}
EOF
}

# =============================================================================
# Argument Parsing Functions
# =============================================================================

# Parse command-line arguments
parse_arguments() {
    # Initialize with defaults
    PARSED_GPU_TYPE="$DEFAULT_GPU_TYPE"
    PARSED_PROFILING_TOOL="$DEFAULT_PROFILING_TOOL"
    PARSED_PROFILING_MODE="$DEFAULT_PROFILING_MODE"
    PARSED_NUM_RUNS="$DEFAULT_NUM_RUNS"
    PARSED_SLEEP_INTERVAL="$DEFAULT_SLEEP_INTERVAL"
    PARSED_APP_NAME="$DEFAULT_APP_NAME"
    PARSED_APP_EXECUTABLE="$DEFAULT_APP_EXECUTABLE"
    PARSED_APP_PARAMS="$DEFAULT_APP_PARAMS"
    PARSED_OUTPUT_DIR="$DEFAULT_OUTPUT_DIR"
    
    # Parse arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
            --gpu-type)
                shift
                if [[ $# -eq 0 ]]; then
                    die "Option --gpu-type requires an argument"
                fi
                PARSED_GPU_TYPE="$1"
                ;;
            --profiling-tool)
                shift
                if [[ $# -eq 0 ]]; then
                    die "Option --profiling-tool requires an argument"
                fi
                PARSED_PROFILING_TOOL="$1"
                ;;
            --profiling-mode)
                shift
                if [[ $# -eq 0 ]]; then
                    die "Option --profiling-mode requires an argument"
                fi
                PARSED_PROFILING_MODE="$1"
                ;;
            --custom-frequencies)
                shift
                if [[ $# -eq 0 ]]; then
                    die "Option --custom-frequencies requires an argument"
                fi
                PARSED_CUSTOM_FREQUENCIES="$1"
                ;;
            --num-runs)
                shift
                if [[ $# -eq 0 ]]; then
                    die "Option --num-runs requires an argument"
                fi
                PARSED_NUM_RUNS="$1"
                ;;
            --sleep-interval)
                shift
                if [[ $# -eq 0 ]]; then
                    die "Option --sleep-interval requires an argument"
                fi
                PARSED_SLEEP_INTERVAL="$1"
                ;;
            --app-name)
                shift
                if [[ $# -eq 0 ]]; then
                    die "Option --app-name requires an argument"
                fi
                PARSED_APP_NAME="$1"
                ;;
            --app-executable)
                shift
                if [[ $# -eq 0 ]]; then
                    die "Option --app-executable requires an argument"
                fi
                PARSED_APP_EXECUTABLE="$1"
                ;;
            --app-params)
                shift
                if [[ $# -eq 0 ]]; then
                    die "Option --app-params requires an argument"
                fi
                PARSED_APP_PARAMS="$1"
                ;;
            --output-dir)
                shift
                if [[ $# -eq 0 ]]; then
                    die "Option --output-dir requires an argument"
                fi
                PARSED_OUTPUT_DIR="$1"
                ;;
            --debug)
                PARSED_DEBUG_MODE=true
                export DEBUG=1
                ;;
            --debug-colors)
                debug_color_detection
                exit 0
                ;;
            --version)
                PARSED_VERSION_REQUESTED=true
                ;;
            -h|--help)
                PARSED_HELP_REQUESTED=true
                ;;
            -*)
                die "Unknown option: $1"
                ;;
            *)
                die "Unexpected argument: $1"
                ;;
        esac
        shift
    done
    
    # Handle help and version requests
    if $PARSED_HELP_REQUESTED; then
        show_usage
        exit 0
    fi
    
    if $PARSED_VERSION_REQUESTED; then
        show_version_info
        exit 0
    fi
    
    # Validate parsed arguments
    validate_arguments
}

# Validate parsed arguments
validate_arguments() {
    local errors=()
    
    # Validate GPU type
    if ! is_valid_gpu_type "$PARSED_GPU_TYPE"; then
        errors+=("Invalid GPU type: $PARSED_GPU_TYPE (supported: ${!GPU_ARCHITECTURES[*]})")
    fi
    
    # Validate profiling tool
    if ! is_valid_profiling_tool "$PARSED_PROFILING_TOOL"; then
        errors+=("Invalid profiling tool: $PARSED_PROFILING_TOOL (supported: dcgmi, nvidia-smi)")
    fi
    
    # Validate profiling mode
    if ! is_valid_profiling_mode "$PARSED_PROFILING_MODE"; then
        errors+=("Invalid profiling mode: $PARSED_PROFILING_MODE (supported: dvfs, custom, baseline)")
    fi
    
    # Validate custom frequencies when in custom mode
    if [[ "$PARSED_PROFILING_MODE" == "custom" ]]; then
        if [[ -z "$PARSED_CUSTOM_FREQUENCIES" ]]; then
            errors+=("Custom frequencies required when profiling-mode is 'custom'")
        else
            # Validate frequency format and values
            if ! validate_custom_frequencies "$PARSED_CUSTOM_FREQUENCIES" "$PARSED_GPU_TYPE"; then
                errors+=("Invalid custom frequencies: $PARSED_CUSTOM_FREQUENCIES")
            fi
        fi
    elif [[ -n "$PARSED_CUSTOM_FREQUENCIES" ]]; then
        errors+=("Custom frequencies can only be used with profiling-mode 'custom'")
    fi
    
    # Validate numeric arguments
    if ! is_positive_integer "$PARSED_NUM_RUNS"; then
        errors+=("Invalid number of runs: $PARSED_NUM_RUNS (must be positive integer)")
    fi
    
    if ! [[ "$PARSED_SLEEP_INTERVAL" =~ ^[0-9]+$ ]]; then
        errors+=("Invalid sleep interval: $PARSED_SLEEP_INTERVAL (must be non-negative integer)")
    fi
    
    # Validate application executable
    if [[ -z "$PARSED_APP_EXECUTABLE" ]]; then
        errors+=("Application executable cannot be empty")
    fi
    
    # Validate output directory
    if [[ -z "$PARSED_OUTPUT_DIR" ]]; then
        errors+=("Output directory cannot be empty")
    fi
    
    # Report validation errors
    if [[ ${#errors[@]} -gt 0 ]]; then
        log_error "Argument validation failed:"
        for error in "${errors[@]}"; do
            log_error "  - $error"
        done
        echo >&2
        log_info "Use --help for usage information"
        exit 1
    fi
    
    log_debug "All arguments validated successfully"
}

# =============================================================================
# Configuration Functions
# =============================================================================

# Apply auto-detection and intelligent defaults
apply_intelligent_defaults() {
    # Auto-detect GPU type if possible
    if [[ "$PARSED_GPU_TYPE" == "$DEFAULT_GPU_TYPE" ]]; then
        local detected_gpu
        if detected_gpu=$(detect_gpu_type); then
            log_info "Auto-detected GPU type: $detected_gpu (overriding default: $DEFAULT_GPU_TYPE)"
            PARSED_GPU_TYPE="$detected_gpu"
        else
            log_warning "Could not auto-detect GPU type, using default: $DEFAULT_GPU_TYPE"
        fi
    fi
    
    # Validate and potentially fallback profiling tool
    local validated_tool
    if validated_tool=$(validate_profiling_tool "$PARSED_PROFILING_TOOL"); then
        if [[ "$validated_tool" != "$PARSED_PROFILING_TOOL" ]]; then
            log_info "Profiling tool changed: $PARSED_PROFILING_TOOL -> $validated_tool"
            PARSED_PROFILING_TOOL="$validated_tool"
        fi
    else
        die "No valid profiling tool available"
    fi
    
    # Apply GPU-specific recommended run counts if using defaults
    if [[ "$PARSED_NUM_RUNS" == "$DEFAULT_NUM_RUNS" ]]; then
        local recommended_runs="${GPU_RECOMMENDED_RUNS[$PARSED_GPU_TYPE]:-$DEFAULT_NUM_RUNS}"
        if [[ "$recommended_runs" != "$DEFAULT_NUM_RUNS" ]]; then
            log_info "Using GPU-specific recommended runs: $recommended_runs (was: $DEFAULT_NUM_RUNS)"
            PARSED_NUM_RUNS="$recommended_runs"
        fi
    fi
    
    # Set up GPU environment
    set_gpu_environment "$PARSED_GPU_TYPE"
    
    # Auto-generate results directory name if using default
    if [[ "$PARSED_OUTPUT_DIR" == "$DEFAULT_OUTPUT_DIR" ]]; then
        local gpu_name=$(echo "$PARSED_GPU_TYPE" | tr '[:upper:]' '[:lower:]')
        local app_name=$(echo "$PARSED_APP_NAME" | tr '[:upper:]' '[:lower:]' | sed 's/[^a-z0-9]//g')
        local base_output_dir="results_${gpu_name}_${app_name}"
        
        # Append job ID if available (for SLURM environments)
        if [[ -n "${SLURM_JOB_ID:-}" ]]; then
            local new_output_dir="${base_output_dir}_job_${SLURM_JOB_ID}"
            log_info "Auto-generating results directory with job ID: $new_output_dir (was: $DEFAULT_OUTPUT_DIR)"
        else
            local new_output_dir="$base_output_dir"
            log_info "Auto-generating results directory: $new_output_dir (was: $DEFAULT_OUTPUT_DIR)"
        fi
        
        PARSED_OUTPUT_DIR="$new_output_dir"
    elif [[ -n "${SLURM_JOB_ID:-}" && "$PARSED_OUTPUT_DIR" != *"job_"* ]]; then
        # Append job ID to custom output directory if not already present
        local new_output_dir="${PARSED_OUTPUT_DIR}_job_${SLURM_JOB_ID}"
        log_info "Appending job ID to custom output directory: $new_output_dir (was: $PARSED_OUTPUT_DIR)"
        PARSED_OUTPUT_DIR="$new_output_dir"
    fi
    
    log_debug "Intelligent defaults applied successfully"
}

# Show final configuration summary
show_configuration_summary() {
    setup_output_colors
    
    printf "\n%sConfiguration Summary%s\n" "$COLOR_GREEN" "$COLOR_NC"
    printf "════════════════════════════════════════\n"
    printf "%sGPU Configuration:%s\n" "$COLOR_BLUE" "$COLOR_NC"
    printf "  Type:           %s\n" "$PARSED_GPU_TYPE"
    printf "  Architecture:   %s\n" "$(get_gpu_architecture "$PARSED_GPU_TYPE")"
    printf "  Memory Freq:    %s MHz\n" "$(get_gpu_memory_freq "$PARSED_GPU_TYPE")"
    printf "  Core Freq:      %s-%s MHz\n" "$(get_gpu_core_freq_min "$PARSED_GPU_TYPE")" "$(get_gpu_core_freq_max "$PARSED_GPU_TYPE")"
    printf "\n"
    printf "%sProfiling Configuration:%s\n" "$COLOR_BLUE" "$COLOR_NC"
    printf "  Tool:           %s\n" "$PARSED_PROFILING_TOOL"
    printf "  Mode:           %s\n" "$PARSED_PROFILING_MODE"
    printf "  Runs per freq:  %s\n" "$PARSED_NUM_RUNS"
    printf "  Sleep interval: %ss\n" "$PARSED_SLEEP_INTERVAL"
    printf "\n"
    printf "%sApplication Configuration:%s\n" "$COLOR_BLUE" "$COLOR_NC"
    printf "  Name:           %s\n" "$PARSED_APP_NAME"
    printf "  Executable:     %s\n" "$PARSED_APP_EXECUTABLE"
    printf "  Parameters:     %s\n" "${PARSED_APP_PARAMS:-"(none)"}"
    printf "\n"
    printf "%sOutput Configuration:%s\n" "$COLOR_BLUE" "$COLOR_NC"
    printf "  Directory:      %s\n" "$PARSED_OUTPUT_DIR"
    printf "  Debug mode:     %s\n" "$PARSED_DEBUG_MODE"
    printf "\n"

    if [[ "$PARSED_PROFILING_MODE" == "dvfs" ]]; then
        local freq_count
        freq_count=$(get_frequency_count "$PARSED_GPU_TYPE")
        local total_runs=$((freq_count * PARSED_NUM_RUNS))
        local estimated_time=$((total_runs * 30 / 60))  # Rough estimate: 30s per run
        
        printf "%sDVFS Experiment Details:%s\n" "$COLOR_YELLOW" "$COLOR_NC"
        printf "  Frequencies:    %s\n" "$freq_count"
        printf "  Total runs:     %s\n" "$total_runs"
        printf "  Estimated time: ~%s minutes\n" "$estimated_time"
        printf "\n"
    fi
}

# Export parsed configuration to environment
export_configuration() {
    export PROFILING_GPU_TYPE="$PARSED_GPU_TYPE"
    export PROFILING_TOOL="$PARSED_PROFILING_TOOL"
    export PROFILING_MODE="$PARSED_PROFILING_MODE"
    export PROFILING_NUM_RUNS="$PARSED_NUM_RUNS"
    export PROFILING_SLEEP_INTERVAL="$PARSED_SLEEP_INTERVAL"
    export PROFILING_APP_NAME="$PARSED_APP_NAME"
    export PROFILING_APP_EXECUTABLE="$PARSED_APP_EXECUTABLE"
    export PROFILING_APP_PARAMS="$PARSED_APP_PARAMS"
    export PROFILING_OUTPUT_DIR="$PARSED_OUTPUT_DIR"
    export PROFILING_DEBUG_MODE="$PARSED_DEBUG_MODE"
    
    log_debug "Configuration exported to environment variables"
}

# =============================================================================
# Main Argument Processing Function
# =============================================================================

# Process all arguments and configuration
process_arguments() {
    local args=("$@")
    
    log_debug "Processing ${#args[@]} command-line arguments"
    
    # Parse arguments
    parse_arguments "${args[@]}"
    
    # Apply intelligent defaults and auto-detection
    apply_intelligent_defaults
    
    # Show configuration if not in quiet mode
    if $PARSED_DEBUG_MODE || [[ "${QUIET:-false}" != "true" ]]; then
        show_configuration_summary
    fi
    
    # Export configuration
    export_configuration
    
    log_success "Argument processing completed successfully"
}

# Export functions for use in other scripts
export -f parse_arguments validate_arguments apply_intelligent_defaults
export -f show_configuration_summary export_configuration process_arguments
export -f show_usage show_version_info

log_debug "Argument parsing library v${ARGS_LIB_VERSION} loaded successfully"
