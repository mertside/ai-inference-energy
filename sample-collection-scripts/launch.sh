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
# For NVIDIA H100:
#   GPU_TYPE="H100"           # Automatically sets GH100, 2619MHz mem, 1785MHz core
#
# To switch between profiling tools:
#
# For DCGMI profiling:
#   PROFILING_TOOL="dcgmi"    # Uses ./profile.py and ./control.sh
#
# For nvidia-smi profiling:
#   PROFILING_TOOL="nvidia-smi" # Uses ./profile_smi.py and ./control_smi.sh
#
# To switch between profiling modes:
#
# For DVFS experiments:
#   PROFILING_MODE="dvfs"     # Sweeps through all GPU frequencies
#
# For baseline measurements:
#   PROFILING_MODE="baseline" # Runs at default frequency only, no control scripts
#
# Required scripts for each configuration:
#   DCGMI:      profile.py, control.sh (existing)
#   nvidia-smi: profile_smi.py, control_smi.sh (new alternatives)
#
# =============================================================================
#
# AI Inference Energy Profiling Launch Script.
#
# This script orchestrates energy profiling experiments for AI inference workloads
# across different GPU frequencies. It systematically tests various core frequencies
# while monitoring power consumption and performance metrics.
#
# The script:
# 1. Supports NVIDIA A100, V100, and H100 GPUs with optimized frequency ranges
# 2. Allows switching between DCGMI and nvidia-smi profiling tools
# 3. Iterates through GPU-specific core frequency ranges
# 4. Runs each AI application multiple times per frequency
# 5. Collects power and performance data using selected profiling tools
# 6. Saves results for analysis with consistent naming conventions
#
# Features:
# - GPU Type Selection: Easy switching between A100, V100, and H100 configurations
# - Profiling Tool Selection: Support for both DCGMI and nvidia-smi
# - Automatic Configuration: GPU-specific frequencies and parameters
# - Comprehensive Logging: Detailed progress and error reporting
# - Robust Error Handling: Graceful failure recovery and cleanup
#
# Usage:
#     ./launch.sh [OPTIONS]
#
# Configuration:
#     Use command-line options to customize:
#     - GPU_TYPE: "A100", "V100", or "H100" for automatic configuration
#     - PROFILING_TOOL: "dcgmi" or "nvidia-smi" for tool selection
#     - Number of runs per frequency
#     - Applications to test
#     - Output directories
#
# Requirements:
#     - NVIDIA GPU (A100, V100, or H100)
#     - GPU profiling tools (DCGMI or nvidia-smi)
#     - AI inference applications (LLaMA, Stable Diffusion, LSTM, etc.)
#     - Corresponding profiling and control scripts
#     - Bash 4.0+ for associative arrays
#
# Author: Mert Side
#

set -euo pipefail  # Exit on error, undefined variables, and pipe failures

# ============================================================================
# Configuration Section - Default Values (can be overridden by parameters)
# ============================================================================

# Default GPU Architecture Selection (A100, V100, or H100)
DEFAULT_GPU_TYPE="A100"  # Options: "A100", "V100", or "H100"

# Default Profiling Tool Selection (DCGMI or NVIDIA-SMI)
DEFAULT_PROFILING_TOOL="dcgmi"  # Options: "dcgmi" or "nvidia-smi"

# Default Experiment configuration
DEFAULT_NUM_RUNS=3  # Number of runs per frequency
DEFAULT_PROFILING_MODE="dvfs"  # Options: "dvfs" or "baseline"
DEFAULT_SLEEP_INTERVAL=1  # seconds between runs

# Default Application configuration (used when no app parameters provided)
DEFAULT_APP_NAME="LSTM"
DEFAULT_APP_EXECUTABLE="../app-lstm/lstm"
DEFAULT_APP_PARAMS=""  # Output redirection will be added dynamically

# Initialize configuration variables (will be set by parse_arguments and configure_gpu_settings)
GPU_TYPE=""
PROFILING_TOOL=""
NUM_RUNS=""
PROFILING_MODE=""
SLEEP_INTERVAL=""
APP_NAME=""
APP_EXECUTABLE=""
APP_PARAMS=""

# GPU-specific variables (will be set by configure_gpu_settings)
GPU_ARCH=""
MEMORY_FREQ=""
DEFAULT_CORE_FREQ=""
CORE_FREQUENCIES=()
EFFECTIVE_FREQUENCIES=()
PROFILE_SCRIPT=""
CONTROL_SCRIPT=""

# Function to configure dynamic output redirection after GPU settings are available
configure_output_redirection() {
    # Ensure output redirection for application parameters using dynamic naming
    if [[ ! "$APP_PARAMS" =~ \> ]]; then
        log_info "No output redirection detected in app parameters. Adding dynamic output redirection."
        local output_file="results/${GPU_ARCH}-${PROFILING_MODE}-${APP_NAME}-RUN-OUT"
        APP_PARAMS="${APP_PARAMS} > ${output_file}"
        log_info "Dynamic output file: $output_file"
    else
        log_info "Output redirection already specified in app parameters"
    fi
    
    return 0
}

# Function to parse command line arguments
parse_arguments() {
    # Set defaults
    GPU_TYPE="$DEFAULT_GPU_TYPE"
    PROFILING_TOOL="$DEFAULT_PROFILING_TOOL"
    NUM_RUNS="$DEFAULT_NUM_RUNS"
    PROFILING_MODE="$DEFAULT_PROFILING_MODE"
    SLEEP_INTERVAL="$DEFAULT_SLEEP_INTERVAL"
    APP_NAME="$DEFAULT_APP_NAME"
    APP_EXECUTABLE="$DEFAULT_APP_EXECUTABLE"
    APP_PARAMS="$DEFAULT_APP_PARAMS"
    
    while (( $# > 0 )); do
        case "$1" in
            --gpu-type)
                if [[ -z "${2:-}" ]]; then
                    log_error "Option --gpu-type requires a value (A100, V100, or H100)"
                    return 1
                fi
                GPU_TYPE="$2"
                shift 2
                ;;
            --profiling-tool)
                if [[ -z "${2:-}" ]]; then
                    log_error "Option --profiling-tool requires a value (dcgmi or nvidia-smi)"
                    return 1
                fi
                PROFILING_TOOL="$2"
                shift 2
                ;;
            --profiling-mode)
                if [[ -z "${2:-}" ]]; then
                    log_error "Option --profiling-mode requires a value (dvfs or baseline)"
                    return 1
                fi
                PROFILING_MODE="$2"
                shift 2
                ;;
            --num-runs)
                if [[ -z "${2:-}" ]]; then
                    log_error "Option --num-runs requires a value"
                    return 1
                fi
                if ! [[ "$2" =~ ^[1-9][0-9]*$ ]]; then
                    log_error "Option --num-runs must be a positive integer"
                    return 1
                fi
                NUM_RUNS="$2"
                shift 2
                ;;
            --sleep-interval)
                if [[ -z "${2:-}" ]]; then
                    log_error "Option --sleep-interval requires a value"
                    return 1
                fi
                if ! [[ "$2" =~ ^[0-9]+$ ]]; then
                    log_error "Option --sleep-interval must be a non-negative integer"
                    return 1
                fi
                SLEEP_INTERVAL="$2"
                shift 2
                ;;
            --app-name)
                if [[ -z "${2:-}" ]]; then
                    log_error "Option --app-name requires a value"
                    return 1
                fi
                APP_NAME="$2"
                shift 2
                ;;
            --app-executable)
                if [[ -z "${2:-}" ]]; then
                    log_error "Option --app-executable requires a value"
                    return 1
                fi
                APP_EXECUTABLE="$2"
                shift 2
                ;;
            --app-params)
                # Allow empty app-params since dynamic output redirection will be added
                APP_PARAMS="${2:-}"
                shift 2
                ;;
            -h|--help)
                usage
                exit 0
                ;;
            -*)
                log_error "Unknown option: $1"
                usage
                return 1
                ;;
            *)
                log_error "Unexpected argument: $1"
                usage
                return 1
                ;;
        esac
    done
    
    # Validate GPU type
    if [[ "$GPU_TYPE" != "A100" && "$GPU_TYPE" != "V100" && "$GPU_TYPE" != "H100" ]]; then
        log_error "Invalid GPU type: $GPU_TYPE. Supported types: A100, V100, H100"
        return 1
    fi
    
    # Validate profiling tool
    if [[ "$PROFILING_TOOL" != "dcgmi" && "$PROFILING_TOOL" != "nvidia-smi" ]]; then
        log_error "Invalid profiling tool: $PROFILING_TOOL. Supported tools: dcgmi, nvidia-smi"
        return 1
    fi
    
    # Validate profiling mode
    if [[ "$PROFILING_MODE" != "dvfs" && "$PROFILING_MODE" != "baseline" ]]; then
        log_error "Invalid profiling mode: $PROFILING_MODE. Supported modes: dvfs, baseline"
        return 1
    fi
    
    return 0
}

# Function to auto-detect GPU type using nvidia-smi
detect_gpu_type() {
    if ! command -v nvidia-smi &> /dev/null; then
        log_warning "nvidia-smi not found - skipping GPU auto-detection"
        return 0
    fi

    local gpu_name detected_type=""
    gpu_name=$(nvidia-smi --query-gpu=name --format=csv,noheader,nounits | head -n1 || true)
    if [[ -z "$gpu_name" ]]; then
        log_warning "Unable to detect GPU name"
        return 0
    fi

    if [[ "$gpu_name" =~ A100 ]]; then
        detected_type="A100"
    elif [[ "$gpu_name" =~ V100 ]]; then
        detected_type="V100"
    elif [[ "$gpu_name" =~ H100 ]]; then
        detected_type="H100"
    fi

    if [[ -n "$detected_type" && "$detected_type" != "$GPU_TYPE" ]]; then
        log_info "Auto-detected GPU type '$detected_type' (was '$GPU_TYPE'). Updating configuration."
        GPU_TYPE="$detected_type"
    fi

    return 0
}

# Function to configure GPU-specific settings after argument parsing
configure_gpu_settings() {
    # GPU-specific configuration based on architecture
    if [[ "$GPU_TYPE" == "A100" ]]; then
        GPU_ARCH="GA100"
        MEMORY_FREQ=1215  # A100 memory frequency (MHz)
        DEFAULT_CORE_FREQ=1410  # A100 default core frequency (MHz)
        
        # A100 core frequencies for testing (MHz)
        CORE_FREQUENCIES=(
            1410 1395 1380 1365 1350 1335 1320 1305 1290 1275
            1260 1245 1230 1215 1200 1185 1170 1155 1140 1125
            1110 1095 1080 1065 1050 1035 1020 1005 990 975
            960 945 930 915 900 885 870 855 840 825
            810 795 780 765 750 735 720 705 690 675
            660 645 630 615 600 585 570 555 540 525 510
        )
    elif [[ "$GPU_TYPE" == "V100" ]]; then
        GPU_ARCH="GV100"
        MEMORY_FREQ=877  # V100 memory frequency (MHz)
        DEFAULT_CORE_FREQ=1380  # V100 default core frequency (MHz)
        
        # V100 core frequencies for testing (MHz) - updated with actual nvidia-smi data ≥510 MHz
        CORE_FREQUENCIES=(
            1380 1372 1365 1357 1350 1342 1335 1327 1320 1312 1305 1297 1290 1282 1275 1267 
            1260 1252 1245 1237 1230 1222 1215 1207 1200 1192 1185 1177 1170 1162 1155 1147 
            1140 1132 1125 1117 1110 1102 1095 1087 1080 1072 1065 1057 1050 1042 1035 1027 
            1020 1012 1005 997 990 982 975 967 960 952 945 937 930 922 915 907 900 892 885 877 
            870 862 855 847 840 832 825 817 810 802 795 787 780 772 765 757 750 742 735 727 
            720 712 705 697 690 682 675 667 660 652 645 637 630 622 615 607 600 592 585 577 
            570 562 555 547 540 532 525 517 510
        )
    elif [[ "$GPU_TYPE" == "H100" ]]; then
        GPU_ARCH="GH100"
        MEMORY_FREQ=2619  # H100 memory frequency (MHz)
        DEFAULT_CORE_FREQ=1785  # H100 default core frequency (MHz)
        
        # H100 core frequencies for testing (MHz) - updated with actual nvidia-smi data ≥510 MHzw
        CORE_FREQUENCIES=(
            1785 1770 1755 1740 1725 1710 1695 1680 1665 1650
            1635 1620 1605 1590 1575 1560 1545 1530 1515 1500
            1485 1470 1455 1440 1425 1410 1395 1380 1365 1350
            1335 1320 1305 1290 1275 1260 1245 1230 1215 1200
            1185 1170 1155 1140 1125 1110 1095 1080 1065 1050
            1035 1020 1005 990 975 960 945 930 915 900
            885 870 855 840 825 810 795 780 765 750
            735 720 705 690 675 660 645 630 615 600
            585 570 555 540 525 510
        )
    else
        log_error "Unsupported GPU type: $GPU_TYPE. Supported types: A100, V100, H100"
        return 1
    fi
    
    # Set frequency range based on profiling mode
    if [[ "$PROFILING_MODE" == "baseline" ]]; then
        # For baseline mode, only use default frequency
        EFFECTIVE_FREQUENCIES=("$DEFAULT_CORE_FREQ")
    elif [[ "$PROFILING_MODE" == "dvfs" ]]; then
        # For DVFS mode, use full frequency range
        EFFECTIVE_FREQUENCIES=("${CORE_FREQUENCIES[@]}")
    fi
    
    # Profiling tool configuration
    if [[ "$PROFILING_TOOL" == "dcgmi" ]]; then
        PROFILE_SCRIPT="./profile.py"  # Python-based DCGMI profiler
        if [[ "$PROFILING_MODE" == "dvfs" ]]; then
            CONTROL_SCRIPT="./control.sh"  # DCGMI-based frequency control
        fi
    elif [[ "$PROFILING_TOOL" == "nvidia-smi" ]]; then
        PROFILE_SCRIPT="./profile_smi.py"  # nvidia-smi based profiler (if exists)
        if [[ "$PROFILING_MODE" == "dvfs" ]]; then
            CONTROL_SCRIPT="./control_smi.sh"  # nvidia-smi based frequency control (if exists)
        fi
    fi
    
    return 0
}

# File and directory configuration
readonly TEMP_OUTPUT_FILE="changeme"
readonly RESULTS_DIR="results"

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
Usage: $(basename "$0") [OPTIONS]

AI Inference Energy Profiling Launch Script

This script runs comprehensive energy profiling experiments across different
GPU frequencies for AI inference workloads.

OPTIONS:
    --gpu-type TYPE           GPU type: A100, V100, or H100 (default: $DEFAULT_GPU_TYPE)
    --profiling-tool TOOL     Profiling tool: dcgmi or nvidia-smi (default: $DEFAULT_PROFILING_TOOL)
    --profiling-mode MODE     Profiling mode: dvfs or baseline (default: $DEFAULT_PROFILING_MODE)
    --num-runs NUM            Number of runs per frequency (default: $DEFAULT_NUM_RUNS)
    --sleep-interval SEC      Sleep interval between runs in seconds (default: $DEFAULT_SLEEP_INTERVAL)
    --app-name NAME           Application display name (default: $DEFAULT_APP_NAME)
    --app-executable PATH     Path to application executable (default: $DEFAULT_APP_EXECUTABLE)
    --app-params "PARAMS"     Application parameters with output redirection (default: "$DEFAULT_APP_PARAMS")
    -h, --help                Show this help message

EXAMPLES:
    # Use defaults (LSTM on A100 with DCGMI and DVFS)
    ./launch.sh
    
    # Run baseline profiling on V100
    ./launch.sh --gpu-type V100 --profiling-mode baseline
    
    # Run baseline profiling on H100
    ./launch.sh --gpu-type H100 --profiling-mode baseline
    
    # Run LSTM applications from app-lstm directory
    ./launch.sh --app-name "LSTM" --app-executable "../app-lstm/lstm"
    
    # Run Stable Diffusion from app-stable-diffusion directory
    ./launch.sh --app-name "StableDiffusion" --app-executable "../app-stable-diffusion/StableDiffusionViaHF" --app-params "--prompt 'test image' > results/SD_output.log"
    
    # Custom application with specific parameters
    ./launch.sh --app-name "MyApp" --app-executable "../my-app/my_script" --app-params "--input data.txt > results/MyApp_output.log"
    
    # Full custom configuration
    ./launch.sh --gpu-type A100 --profiling-tool dcgmi --profiling-mode dvfs --num-runs 5 --app-name "StableDiffusion" --app-executable "../app-stable-diffusion/StableDiffusionViaHF" --app-params "--prompt 'test image' > results/SD_output.log"

GPU TYPE SELECTION:
    A100: Automatically configures for NVIDIA A100 GPUs
        - Architecture: GA100
        - Memory frequency: 1215MHz
        - Default core frequency: 1410MHz
        - Core frequency range: 1410-510MHz (61 frequencies)
    
    V100: Automatically configures for NVIDIA V100 GPUs
        - Architecture: GV100
        - Memory frequency: 877MHz
        - Default core frequency: 1380MHz
        - Core frequency range: 1380-510MHz (117 frequencies)
    
    H100: Automatically configures for NVIDIA H100 GPUs
        - Architecture: GH100
        - Memory frequency: 2619MHz
        - Default core frequency: 1785MHz
        - Core frequency range: 1785-510MHz (86 frequencies)

PROFILING TOOL SELECTION:
    dcgmi: Uses DCGMI tools for profiling (recommended)
        - Profile script: ./profile.py
        - Control script: ./control.sh (DVFS mode only)
        - Auto-fallback: If dcgmi is not available, automatically switches to nvidia-smi
    
    nvidia-smi: Uses nvidia-smi for profiling
        - Profile script: ./profile_smi.py
        - Control script: ./control_smi.sh (DVFS mode only)
        - Fallback option: Used automatically when dcgmi is not available

PROFILING MODE SELECTION:
    dvfs: Full frequency sweep experiments
        - Tests all available frequencies for the GPU type
        - Requires control scripts for frequency management
        - Comprehensive energy vs performance analysis
    
    baseline: Single frequency measurements
        - Runs only at default GPU frequency
        - No frequency control scripts needed
        - Faster execution, useful for reference measurements

APPLICATION PARAMETERS:
    The --app-executable parameter should specify the path to the Python script
    relative to the sample-collection-scripts directory, without the .py extension.
    
    Application Resolution:
    - If the path contains '/', it's treated as a relative/absolute path
    - Otherwise, the script attempts to map known app names to their directories:
      - "lstm" → "../app-lstm/lstm.py"
      - "lstm_modern" → "../app-lstm/lstm_modern.py" 
      - "StableDiffusionViaHF" → "../app-stable-diffusion/StableDiffusionViaHF.py"
      - "LlamaViaHF" → "../app-llama/LlamaViaHF.py"
    
    The --app-params option can include output redirection to capture results.
    If no output redirection is specified, it will be automatically added using
    dynamic naming: results/\$ARCH-\$MODE-\$APP-RUN-OUT
    
    Examples:
        --app-executable "../app-lstm/lstm"
        --app-executable "../app-stable-diffusion/StableDiffusionViaHF"
        --app-params "> results/custom_output.log"
        --app-params "--epochs 10 --batch-size 32 > results/training.log"
        --app-params "--model bert-base --input data.txt > results/inference.out"
        --app-params ""  # Will auto-generate: results/GA100-dvfs-LSTM-RUN-OUT

OUTPUT:
    Results are saved to the 'results' directory with naming convention:
    
    Performance data: \$ARCH-\$MODE-\$APP-\$FREQ-\$ITERATION
    Application output: \$ARCH-\$MODE-\$APP-RUN-OUT (when auto-generated)
    
    Examples:
        Performance: GA100-dvfs-LSTM-1410-0
        App output: GA100-dvfs-LSTM-RUN-OUT

REQUIREMENTS:
    - NVIDIA GPU (A100 or V100)
    - GPU profiling tools (DCGMI or nvidia-smi)
    - Profiling and control scripts (./profile.py, ./control.sh, etc.)
    - AI inference applications
    - Sufficient disk space for results

EOF
}

# Function to check prerequisites
check_prerequisites() {
    log_info "Checking prerequisites..."
    log_info "GPU Type: $GPU_TYPE ($GPU_ARCH)"
    log_info "Profiling Tool: $PROFILING_TOOL"
    log_info "Profiling Mode: $PROFILING_MODE"
    
    # Check required scripts (after potential profiling tool fallback)
    local required_scripts=("$PROFILE_SCRIPT")
    
    # Only check control script for DVFS mode
    if [[ "$PROFILING_MODE" == "dvfs" ]]; then
        required_scripts+=("$CONTROL_SCRIPT")
    fi
    
    log_info "Checking required scripts for $PROFILING_TOOL profiling..."
    for script in "${required_scripts[@]}"; do
        if [[ ! -x "$script" ]]; then
            log_error "Required script not found or not executable: $script"
            log_info "Make sure the script exists and is executable: chmod +x $script"
            
            # Provide helpful suggestions based on profiling tool
            if [[ "$PROFILING_TOOL" == "nvidia-smi" ]]; then
                log_info "For nvidia-smi profiling, you need:"
                log_info "  - ./profile_smi.py (nvidia-smi based profiler)"
                if [[ "$PROFILING_MODE" == "dvfs" ]]; then
                    log_info "  - ./control_smi.sh (nvidia-smi based frequency control)"
                fi
            elif [[ "$PROFILING_TOOL" == "dcgmi" ]]; then
                log_info "For DCGMI profiling, you need:"
                log_info "  - ./profile.py (DCGMI based profiler)"
                if [[ "$PROFILING_MODE" == "dvfs" ]]; then
                    log_info "  - ./control.sh (DCGMI based frequency control)"
                fi
            fi
            return 1
        fi
    done
    
    log_info "✓ All required scripts found and executable"
    
    # Check if the application executable exists
    log_info "Checking application: $APP_NAME ($APP_EXECUTABLE)"
    if ! resolve_app_path "$APP_EXECUTABLE"; then
        log_error "Application validation failed"
        return 1
    fi
    log_info "✓ Application found: $RESOLVED_APP_DIR/$RESOLVED_APP_SCRIPT.py"
    
    # Check profiling tool availability with automatic fallback
    if [[ "$PROFILING_TOOL" == "dcgmi" ]]; then
        if ! command -v dcgmi &> /dev/null; then
            log_warning "DCGMI command not found. Attempting automatic fallback to nvidia-smi..."
            
            # Check if nvidia-smi is available for fallback
            if command -v nvidia-smi &> /dev/null; then
                log_info "✓ nvidia-smi available - switching to nvidia-smi profiling"
                log_info "Reconfiguring profiling tool from dcgmi to nvidia-smi"
                
                # Switch profiling tool
                PROFILING_TOOL="nvidia-smi"
                
                # Reconfigure scripts for nvidia-smi
                PROFILE_SCRIPT="./profile_smi.py"
                if [[ "$PROFILING_MODE" == "dvfs" ]]; then
                    CONTROL_SCRIPT="./control_smi.sh"
                fi
                
                log_info "Updated configuration:"
                log_info "  - Profile script: $PROFILE_SCRIPT"
                if [[ "$PROFILING_MODE" == "dvfs" ]]; then
                    log_info "  - Control script: $CONTROL_SCRIPT"
                fi
            else
                log_error "Neither dcgmi nor nvidia-smi commands are available."
                log_error "Please install either:"
                log_error "  - NVIDIA DCGMI tools: https://developer.nvidia.com/dcgm"
                log_error "  - NVIDIA drivers (includes nvidia-smi)"
                return 1
            fi
        else
            log_info "✓ DCGMI available"
        fi
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
        elif [[ "$GPU_TYPE" == "H100" && ! "$gpu_name" =~ H100 ]]; then
            log_warning "GPU type set to H100 but detected GPU doesn't appear to be H100: $gpu_name"
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

# Function to resolve application paths and directories
resolve_app_path() {
    local app_executable="$1"
    local resolved_dir=""
    local resolved_script=""
    
    # Check if the path is already absolute or relative with directory
    if [[ "$app_executable" = /* ]] || [[ "$app_executable" = */* ]]; then
        # Extract directory and script name
        resolved_dir=$(dirname "$app_executable")
        resolved_script=$(basename "$app_executable")
        
        # Remove .py extension if present
        resolved_script=${resolved_script%.py}
    else
        # Try to map application names to their directories
        case "$app_executable" in
            "lstm")
                resolved_dir="../app-lstm"
                resolved_script="lstm"
                ;;
            "StableDiffusionViaHF")
                resolved_dir="../app-stable-diffusion"
                resolved_script="StableDiffusionViaHF"
                ;;
            "LlamaViaHF")
                resolved_dir="../app-llama"
                resolved_script="LlamaViaHF"
                ;;
            *)
                # Default: assume it's in the current directory
                resolved_dir="."
                resolved_script="$app_executable"
                ;;
        esac
    fi
    
    # Validate that the script exists
    if [[ ! -f "$resolved_dir/$resolved_script.py" ]]; then
        log_error "Application script not found: $resolved_dir/$resolved_script.py"
        return 1
    fi
    
    # Export the resolved paths for use in other functions
    export RESOLVED_APP_DIR="$resolved_dir"
    export RESOLVED_APP_SCRIPT="$resolved_script"
    
    log_info "Resolved application: $resolved_dir/$resolved_script.py"
    return 0
}

# Function to set GPU frequency using the control script
set_gpu_frequency() {
    local core_freq="$1"
    
    # Skip frequency setting in baseline mode
    if [[ "$PROFILING_MODE" == "baseline" ]]; then
        log_info "Baseline mode: Using default GPU frequency (no frequency control)"
        return 0
    fi
    
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
    
    # Resolve application path and directory
    if ! resolve_app_path "$app_executable"; then
        log_error "Failed to resolve application path: $app_executable"
        return 1
    fi
    
    local app_command="cd '$RESOLVED_APP_DIR' && python '$RESOLVED_APP_SCRIPT.py'${app_params}"
    local output_file="${RESULTS_DIR}/${GPU_ARCH}-${PROFILING_MODE}-${app_name}-${frequency}-${iteration}"
    
    log_info "Running $app_name at ${frequency}MHz (iteration $iteration)"
    log_info "Application directory: $RESOLVED_APP_DIR"
    log_info "Application script: $RESOLVED_APP_SCRIPT.py"
    log_info "Command: $app_command"
    log_info "Output file: $output_file"
    
    # Record start time
    local start_time
    start_time=$(date +%s)
    
    # Run application with profiling (use bash -c to handle the cd command)
    if ! "$PROFILE_SCRIPT" bash -c "$app_command"; then
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
    
    # Extract and save performance metrics (for known applications)
    extract_performance_metrics "$app_name" "$frequency" "$duration"
    
    log_info "Application completed in ${duration}s"
    return 0
}

# Function to extract performance metrics for known applications
extract_performance_metrics() {
    local app_name="$1"
    local frequency="$2"
    local wall_time="$3"
    
    case "$app_name" in
        "LSTM")
            extract_lstm_performance "$frequency" "$wall_time"
            ;;
        "StableDiffusion"|"SD")
            extract_stable_diffusion_performance "$frequency" "$wall_time"
            ;;
        "LLaMA"|"Llama")
            extract_llama_performance "$frequency" "$wall_time"
            ;;
        *)
            log_info "No specific performance extraction for application: $app_name"
            # For unknown applications, just log the wall time
            local performance_file="${RESULTS_DIR}/${GPU_ARCH}-${PROFILING_MODE}-${app_name}-perf.csv"
            printf '%s,%s\n' "$frequency" "$wall_time" >> "$performance_file"
            log_info "Wall time saved: ${frequency}MHz -> ${wall_time}s"
            ;;
    esac
}

# Function to extract LSTM performance metrics
extract_lstm_performance() {
    local frequency="$1"
    local wall_time="$2"
    local performance_file="${RESULTS_DIR}/${GPU_ARCH}-${PROFILING_MODE}-lstm-perf.csv"
    
    # Try to extract execution time from LSTM output using dynamic naming
    local lstm_output_file="${RESULTS_DIR}/${GPU_ARCH}-${PROFILING_MODE}-${APP_NAME}-RUN-OUT"
    if [[ -f "$lstm_output_file" ]]; then
        # Extract execution time from last line (LSTM script prints timing at the end)
        local last_line
        last_line=$(tail -n 1 "$lstm_output_file")
        
        # Look for timing information in the last line
        if [[ "$last_line" =~ ([0-9]+\.?[0-9]*)[[:space:]]*seconds ]]; then
            local exec_time="${BASH_REMATCH[1]}"
            printf '%s,%s\n' "$frequency" "$exec_time" >> "$performance_file"
            log_info "LSTM performance saved: ${frequency}MHz -> ${exec_time}s"
        else
            # Fallback to wall time if no timing found in output
            printf '%s,%s\n' "$frequency" "$wall_time" >> "$performance_file"
            log_info "LSTM wall time saved: ${frequency}MHz -> ${wall_time}s"
        fi
    else
        log_warning "LSTM output file not found: $lstm_output_file"
        # Use wall time as fallback
        printf '%s,%s\n' "$frequency" "$wall_time" >> "$performance_file"
        log_info "LSTM wall time saved: ${frequency}MHz -> ${wall_time}s"
    fi
}

# Function to extract Stable Diffusion performance metrics
extract_stable_diffusion_performance() {
    local frequency="$1"
    local wall_time="$2"
    local performance_file="${RESULTS_DIR}/${GPU_ARCH}-${PROFILING_MODE}-stable-diffusion-perf.csv"
    
    # For Stable Diffusion, use wall time (image generation time is the key metric)
    printf '%s,%s\n' "$frequency" "$wall_time" >> "$performance_file"
    log_info "Stable Diffusion performance saved: ${frequency}MHz -> ${wall_time}s"
}

# Function to extract LLaMA performance metrics
extract_llama_performance() {
    local frequency="$1"
    local wall_time="$2"
    local performance_file="${RESULTS_DIR}/${GPU_ARCH}-${PROFILING_MODE}-llama-perf.csv"
    
    # For LLaMA, use wall time (text generation time is the key metric)
    printf '%s,%s\n' "$frequency" "$wall_time" >> "$performance_file"
    log_info "LLaMA performance saved: ${frequency}MHz -> ${wall_time}s"
}

# Function to run experiments for all frequencies
run_frequency_sweep() {
    local total_runs=0
    local completed_runs=0
    
    # Calculate total number of runs (single application now)
    total_runs=$((${#EFFECTIVE_FREQUENCIES[@]} * NUM_RUNS))
    
    if [[ "$PROFILING_MODE" == "baseline" ]]; then
        log_info "Starting baseline profiling: $total_runs total runs at default frequency"
    else
        log_info "Starting frequency sweep: $total_runs total runs across ${#EFFECTIVE_FREQUENCIES[@]} frequencies"
    fi
    
    # Iterate through frequencies (or just default frequency for baseline)
    for core_freq in "${EFFECTIVE_FREQUENCIES[@]}"; do
        if [[ "$PROFILING_MODE" == "baseline" ]]; then
            log_info "Running baseline measurements at default frequency: ${core_freq}MHz"
        else
            log_info "Processing frequency: ${core_freq}MHz"
        fi
        
        # Set GPU frequency (no-op for baseline mode)
        if ! set_gpu_frequency "$core_freq"; then
            log_error "Skipping frequency ${core_freq}MHz due to frequency setting failure"
            continue
        fi
        
        # Run multiple iterations for this frequency
        for iteration in $(seq 0 $((NUM_RUNS - 1))); do
            log_info "Iteration $(( iteration + 1 ))/$NUM_RUNS for ${core_freq}MHz"
            
            # Run the specified application
            if ! run_application "$APP_NAME" "$APP_EXECUTABLE" "$APP_PARAMS" "$core_freq" "$iteration"; then
                log_error "Failed to run $APP_NAME at ${core_freq}MHz (iteration $iteration)"
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
    
    log_info "Frequency sweep completed: $completed_runs/$total_runs runs successful"
    return 0
}

# Function to restore default GPU frequency
restore_default_frequency() {
    # Skip frequency restoration if variables are not set (e.g., help mode)
    if [[ -z "${PROFILING_MODE:-}" || -z "${DEFAULT_CORE_FREQ:-}" ]]; then
        return 0
    fi
    
    # Skip frequency restoration in baseline mode
    if [[ "$PROFILING_MODE" == "baseline" ]]; then
        log_info "Baseline mode: No frequency restoration needed"
        return 0
    fi
    
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
    
    # Parse command line arguments first
    if ! parse_arguments "$@"; then
        exit 1
    fi

    # Auto-detect GPU type and adjust configuration if needed
    detect_gpu_type

    # Configure GPU settings based on parsed arguments
    if ! configure_gpu_settings; then
        exit 1
    fi
    
    # Configure output redirection for application parameters
    if ! configure_output_redirection; then
        exit 1
    fi
    
    log_info "Starting AI inference energy profiling experiment"
    log_info "Final Configuration (after any automatic adjustments):"
    log_info "  GPU Type: $GPU_TYPE"
    log_info "  GPU Architecture: $GPU_ARCH"
    log_info "  Profiling Tool: $PROFILING_TOOL"
    log_info "  Profiling Mode: $PROFILING_MODE"
    log_info "  Runs per frequency: $NUM_RUNS"
    log_info "  Sleep interval: ${SLEEP_INTERVAL}s"
    log_info "  Memory frequency: ${MEMORY_FREQ}MHz"
    log_info "  Default core frequency: ${DEFAULT_CORE_FREQ}MHz"
    if [[ "$PROFILING_MODE" == "baseline" ]]; then
        log_info "  Running in baseline mode: ${#EFFECTIVE_FREQUENCIES[@]} frequency (${EFFECTIVE_FREQUENCIES[0]}MHz)"
    else
        log_info "  Core frequencies: ${#EFFECTIVE_FREQUENCIES[@]} frequencies (${EFFECTIVE_FREQUENCIES[0]}-${EFFECTIVE_FREQUENCIES[-1]}MHz)"
    fi
    log_info "  Application: $APP_NAME"
    log_info "  Executable: $APP_EXECUTABLE"
    log_info "  Parameters: $APP_PARAMS"
    log_info "  Results directory: $RESULTS_DIR"
    log_info "  Profile script: $PROFILE_SCRIPT"
    if [[ "$PROFILING_MODE" == "dvfs" ]]; then
        log_info "  Control script: $CONTROL_SCRIPT"
    else
        log_info "  Control script: Not used (baseline mode)"
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
