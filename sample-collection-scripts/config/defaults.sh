#!/bin/bash
#
# Default Configuration for AI Inference Energy Profiling
#
# This file contains default configuration values for the profiling framework.
# Values can be overridden by command-line arguments or environment variables.
#
# Author: Mert Side
#

# =============================================================================
# Core Configuration
# =============================================================================

# Framework version
readonly FRAMEWORK_VERSION="2.0.0"

# Default GPU and profiling settings
readonly DEFAULT_GPU_TYPE="A100"
readonly DEFAULT_PROFILING_TOOL="dcgmi"
readonly DEFAULT_PROFILING_MODE="dvfs"
readonly DEFAULT_NUM_RUNS=3
readonly DEFAULT_SLEEP_INTERVAL=1

# Default application settings
readonly DEFAULT_APP_NAME="LSTM"
readonly DEFAULT_APP_EXECUTABLE="lstm.py"
readonly DEFAULT_APP_PARAMS=""

# =============================================================================
# Directory Configuration
# =============================================================================

# Base directories (relative to scripts directory)
readonly DEFAULT_OUTPUT_DIR="results"
readonly DEFAULT_BACKUP_DIR="backups"
readonly DEFAULT_LOGS_DIR="logs"
readonly DEFAULT_TEMP_DIR="tmp"

# =============================================================================
# Profiling Configuration
# =============================================================================

# Profiling timeouts (seconds)
readonly PROFILE_TIMEOUT=300
readonly CONTROL_TIMEOUT=30
readonly APPLICATION_TIMEOUT=600

# Profiling intervals
readonly PROFILE_INTERVAL=0.1
readonly STATUS_UPDATE_INTERVAL=5

# Output file settings
readonly OUTPUT_FILE_EXTENSION=".csv"
readonly LOG_FILE_EXTENSION=".log"
readonly BACKUP_TIMESTAMP_FORMAT="%Y%m%d_%H%M%S"

# =============================================================================
# SLURM Configuration
# =============================================================================

# Default SLURM job settings
readonly DEFAULT_SLURM_TIME="04:00:00"
readonly DEFAULT_SLURM_MEMORY="32G"
readonly DEFAULT_SLURM_CPUS="8"
readonly DEFAULT_SLURM_GPUS="1"

# Job naming
readonly SLURM_JOB_PREFIX="ai_profiling"

# =============================================================================
# Application-Specific Defaults
# =============================================================================

# LSTM application
readonly LSTM_DEFAULT_PARAMS=""
readonly LSTM_CONDA_ENV="tensorflow"

# Stable Diffusion application  
readonly STABLE_DIFFUSION_DEFAULT_PARAMS="--prompt 'A beautiful landscape' --steps 20"
readonly STABLE_DIFFUSION_CONDA_ENV="stable-diffusion-gpu"

# LLaMA application
readonly LLAMA_DEFAULT_PARAMS=""
readonly LLAMA_CONDA_ENV="pytorch-gpu"

# =============================================================================
# Hardware-Specific Configuration
# =============================================================================

# Memory requirements by application type
declare -A APP_MEMORY_REQUIREMENTS=(
    ["lstm"]="8G"
    ["stable_diffusion"]="16G"
    ["llama"]="32G"
    ["default"]="16G"
)

# Recommended run counts by GPU type
declare -A GPU_RECOMMENDED_RUNS=(
    ["A100"]=3
    ["V100"]=3
    ["H100"]=3
)

# =============================================================================
# Logging Configuration
# =============================================================================

# Log levels: DEBUG, INFO, WARNING, ERROR
readonly DEFAULT_LOG_LEVEL="INFO"

# Log rotation settings
readonly MAX_LOG_SIZE="100M"
readonly MAX_LOG_FILES=10

# =============================================================================
# Safety and Validation Configuration
# =============================================================================

# Minimum disk space required (in GB)
readonly MIN_DISK_SPACE=10

# Maximum concurrent jobs
readonly MAX_CONCURRENT_JOBS=1

# Frequency validation ranges (MHz)
readonly MIN_CORE_FREQUENCY=300
readonly MAX_CORE_FREQUENCY=2000
readonly MIN_MEMORY_FREQUENCY=500
readonly MAX_MEMORY_FREQUENCY=3000

# =============================================================================
# Network and Cluster Configuration
# =============================================================================

# Cluster-specific settings
declare -A CLUSTER_SETTINGS=(
    ["HPCC_MODULES"]="CUDA/11.8 GCC/11.3.0"
    ["REPACSS_MODULES"]="cuda/11.8 gcc/11.3.0"
)

# =============================================================================
# Feature Flags
# =============================================================================

# Enable/disable features
readonly ENABLE_AUTO_GPU_DETECTION=true
readonly ENABLE_PROFILING_TOOL_FALLBACK=true
readonly ENABLE_AUTOMATIC_CLEANUP=true
readonly ENABLE_PROGRESS_INDICATORS=true
readonly ENABLE_COLOR_OUTPUT=true

# =============================================================================
# Error Handling Configuration
# =============================================================================

# Retry settings
readonly MAX_RETRIES=3
readonly RETRY_DELAY=5

# Error recovery options
readonly CONTINUE_ON_ERROR=false
readonly SAVE_FAILED_RUNS=true

# =============================================================================
# Performance Configuration
# =============================================================================

# Parallel execution settings
readonly MAX_PARALLEL_PROFILES=1
readonly MAX_PARALLEL_APPS=1

# Optimization flags
readonly ENABLE_FAST_MODE=false
readonly ENABLE_DETAILED_LOGGING=true

# =============================================================================
# Configuration Validation
# =============================================================================

# Validate configuration values
validate_config() {
    local errors=()
    
    # Validate numeric values
    if ! [[ "$DEFAULT_NUM_RUNS" =~ ^[1-9][0-9]*$ ]]; then
        errors+=("DEFAULT_NUM_RUNS must be a positive integer")
    fi
    
    if ! [[ "$DEFAULT_SLEEP_INTERVAL" =~ ^[0-9]+$ ]]; then
        errors+=("DEFAULT_SLEEP_INTERVAL must be a non-negative integer")
    fi
    
    # Validate timeouts
    if [[ "$PROFILE_TIMEOUT" -le 0 ]]; then
        errors+=("PROFILE_TIMEOUT must be positive")
    fi
    
    # Report validation errors
    if [[ ${#errors[@]} -gt 0 ]]; then
        echo "Configuration validation errors:" >&2
        printf "  - %s\n" "${errors[@]}" >&2
        return 1
    fi
    
    return 0
}

# Load user configuration overrides if available
load_user_config() {
    local user_config="${SCRIPTS_ROOT_DIR}/config/user_config.sh"
    if [[ -f "$user_config" ]]; then
        source "$user_config"
        echo "Loaded user configuration: $user_config" >&2
    fi
}

# Load environment-specific configuration
load_environment_config() {
    local env_name="${ENVIRONMENT:-development}"
    local env_config="${SCRIPTS_ROOT_DIR}/config/${env_name}_config.sh"
    if [[ -f "$env_config" ]]; then
        source "$env_config"
        echo "Loaded environment configuration: $env_config" >&2
    fi
}

# Initialize configuration
init_config() {
    validate_config || return 1
    load_user_config
    load_environment_config
    echo "Configuration initialized successfully" >&2
}

# Export configuration functions
export -f validate_config load_user_config load_environment_config init_config

# Auto-initialize if not disabled
if [[ "${AUTO_INIT_CONFIG:-true}" == "true" ]]; then
    init_config
fi
