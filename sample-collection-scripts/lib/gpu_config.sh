#!/bin/bash
#
# GPU Configuration Library for AI Inference Energy Profiling
#
# This library provides GPU-specific configurations, detection, and management
# functions for different GPU architectures (A100, V100, H100).
#
# Author: Mert Side
#

# Prevent multiple inclusions
if [[ "${GPU_CONFIG_LIB_LOADED:-}" == "true" ]]; then
    return 0
fi
readonly GPU_CONFIG_LIB_LOADED="true"

# Load common library
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

# =============================================================================
# GPU Configuration Constants
# =============================================================================

readonly GPU_CONFIG_VERSION="1.0.0"

# GPU Architecture Definitions
declare -A GPU_ARCHITECTURES=(
    ["A100"]="GA100"
    ["V100"]="GV100"
    ["H100"]="GH100"
)

# Memory Frequency Settings (MHz)
declare -A GPU_MEMORY_FREQ=(
    ["A100"]=1215
    ["V100"]=877
    ["H100"]=2619
)

# Core Frequency Ranges (MHz)
declare -A GPU_CORE_FREQ_MAX=(
    ["A100"]=1410
    ["V100"]=1380
    ["H100"]=1785
)

declare -A GPU_CORE_FREQ_MIN=(
    ["A100"]=510
    ["V100"]=510
    ["H100"]=510
)

declare -A GPU_CORE_FREQ_STEP=(
    ["A100"]=15
    ["V100"]=15
    ["H100"]=15
)

# SLURM Partition Mappings
declare -A GPU_SLURM_PARTITIONS=(
    ["A100"]="toreador"
    ["V100"]="matador"
    ["H100"]="h100-build"
)

# Cluster Information
declare -A GPU_CLUSTERS=(
    ["A100"]="HPCC"
    ["V100"]="HPCC"
    ["H100"]="REPACSS"
)

# Node Specifications (optional, for specific cluster configurations)
declare -A GPU_NODES=(
    ["H100"]="rpg-93-9"
)

# =============================================================================
# GPU Detection Functions
# =============================================================================

# Detect GPU type using nvidia-smi
detect_gpu_type() {
    local detected_gpu=""
    
    if ! command_exists "nvidia-smi"; then
        log_warning "nvidia-smi not available for GPU detection"
        return 1
    fi
    
    log_debug "Detecting GPU type using nvidia-smi..."
    local gpu_name
    gpu_name=$(nvidia-smi --query-gpu=name --format=csv,noheader,nounits 2>/dev/null | head -n1)
    
    if [[ -z "$gpu_name" ]]; then
        log_warning "Could not detect GPU name"
        return 1
    fi
    
    log_debug "Detected GPU name: $gpu_name"
    
    # Match GPU name to our supported types
    case "$gpu_name" in
        *"A100"*) detected_gpu="A100" ;;
        *"V100"*) detected_gpu="V100" ;;
        *"H100"*) detected_gpu="H100" ;;
        *)
            log_warning "Unsupported GPU detected: $gpu_name"
            return 1
            ;;
    esac
    
    log_info "Auto-detected GPU type: $detected_gpu"
    echo "$detected_gpu"
    return 0
}

# Get GPU count
get_gpu_count() {
    if command_exists "nvidia-smi"; then
        nvidia-smi --query-gpu=count --format=csv,noheader,nounits 2>/dev/null | head -n1
    else
        echo "0"
    fi
}

# Check if GPU is available
is_gpu_available() {
    local gpu_count
    gpu_count=$(get_gpu_count)
    [[ "$gpu_count" -gt 0 ]]
}

# Get GPU memory info
get_gpu_memory_info() {
    if command_exists "nvidia-smi"; then
        nvidia-smi --query-gpu=memory.total,memory.free,memory.used --format=csv,noheader,nounits 2>/dev/null
    fi
}

# =============================================================================
# GPU Configuration Functions
# =============================================================================

# Get GPU architecture for given type
get_gpu_architecture() {
    local gpu_type="$1"
    echo "${GPU_ARCHITECTURES[$gpu_type]:-}"
}

# Get memory frequency for GPU type
get_gpu_memory_freq() {
    local gpu_type="$1"
    echo "${GPU_MEMORY_FREQ[$gpu_type]:-}"
}

# Get core frequency range for GPU type
get_gpu_core_freq_max() {
    local gpu_type="$1"
    echo "${GPU_CORE_FREQ_MAX[$gpu_type]:-}"
}

get_gpu_core_freq_min() {
    local gpu_type="$1"
    echo "${GPU_CORE_FREQ_MIN[$gpu_type]:-}"
}

get_gpu_core_freq_step() {
    local gpu_type="$1"
    echo "${GPU_CORE_FREQ_STEP[$gpu_type]:-}"
}

# Generate frequency range for GPU type
generate_frequency_range() {
    local gpu_type="$1"
    local min_freq max_freq step
    
    min_freq=$(get_gpu_core_freq_min "$gpu_type")
    max_freq=$(get_gpu_core_freq_max "$gpu_type")
    step=$(get_gpu_core_freq_step "$gpu_type")
    
    if [[ -z "$min_freq" || -z "$max_freq" || -z "$step" ]]; then
        log_error "Invalid GPU type or missing frequency configuration: $gpu_type"
        return 1
    fi
    
    local frequencies=()
    for ((freq = max_freq; freq >= min_freq; freq -= step)); do
        frequencies+=("$freq")
    done
    
    echo "${frequencies[@]}"
}

# Get frequency count for GPU type
get_frequency_count() {
    local gpu_type="$1"
    local frequencies
    frequencies=($(generate_frequency_range "$gpu_type"))
    echo "${#frequencies[@]}"
}

# =============================================================================
# SLURM Integration Functions
# =============================================================================

# Get SLURM partition for GPU type
get_slurm_partition() {
    local gpu_type="$1"
    echo "${GPU_SLURM_PARTITIONS[$gpu_type]:-}"
}

# Get cluster name for GPU type
get_cluster_name() {
    local gpu_type="$1"
    echo "${GPU_CLUSTERS[$gpu_type]:-}"
}

# Get specific node for GPU type (if applicable)
get_gpu_node() {
    local gpu_type="$1"
    echo "${GPU_NODES[$gpu_type]:-}"
}

# Generate SLURM job parameters for GPU type
get_slurm_job_params() {
    local gpu_type="$1"
    local partition node
    
    partition=$(get_slurm_partition "$gpu_type")
    node=$(get_gpu_node "$gpu_type")
    
    local params="--partition=$partition"
    if [[ -n "$node" ]]; then
        params="$params --nodelist=$node"
    fi
    
    echo "$params"
}

# =============================================================================
# GPU Validation Functions
# =============================================================================

# Validate GPU type is supported
validate_gpu_type() {
    local gpu_type="$1"
    
    if ! is_valid_gpu_type "$gpu_type"; then
        log_error "Invalid GPU type: $gpu_type"
        log_error "Supported types: ${!GPU_ARCHITECTURES[*]}"
        return 1
    fi
    
    local architecture
    architecture=$(get_gpu_architecture "$gpu_type")
    if [[ -z "$architecture" ]]; then
        log_error "No architecture found for GPU type: $gpu_type"
        return 1
    fi
    
    log_debug "GPU type $gpu_type validated (architecture: $architecture)"
    return 0
}

# Validate frequency for GPU type
validate_frequency() {
    local gpu_type="$1"
    local frequency="$2"
    
    if ! is_valid_frequency "$frequency"; then
        log_error "Invalid frequency format: $frequency"
        return 1
    fi
    
    local min_freq max_freq
    min_freq=$(get_gpu_core_freq_min "$gpu_type")
    max_freq=$(get_gpu_core_freq_max "$gpu_type")
    
    if [[ "$frequency" -lt "$min_freq" || "$frequency" -gt "$max_freq" ]]; then
        log_error "Frequency $frequency MHz out of range for $gpu_type (${min_freq}-${max_freq} MHz)"
        return 1
    fi
    
    log_debug "Frequency $frequency MHz validated for $gpu_type"
    return 0
}

# =============================================================================
# GPU Information Display Functions
# =============================================================================

# Show GPU configuration summary
show_gpu_config() {
    local gpu_type="$1"
    
    if ! validate_gpu_type "$gpu_type"; then
        return 1
    fi
    
    local architecture memory_freq min_freq max_freq step partition cluster
    architecture=$(get_gpu_architecture "$gpu_type")
    memory_freq=$(get_gpu_memory_freq "$gpu_type")
    min_freq=$(get_gpu_core_freq_min "$gpu_type")
    max_freq=$(get_gpu_core_freq_max "$gpu_type")
    step=$(get_gpu_core_freq_step "$gpu_type")
    partition=$(get_slurm_partition "$gpu_type")
    cluster=$(get_cluster_name "$gpu_type")
    
    local freq_count
    freq_count=$(get_frequency_count "$gpu_type")
    
    cat << EOF

${COLOR_BLUE}GPU Configuration Summary${COLOR_NC}
────────────────────────────────────────
GPU Type:          $gpu_type
Architecture:      $architecture
Memory Frequency:  $memory_freq MHz
Core Frequency:    $min_freq - $max_freq MHz (step: $step MHz)
Frequency Count:   $freq_count frequencies
SLURM Partition:   $partition
Cluster:          $cluster

EOF
}

# Show all supported GPU types
show_supported_gpus() {
    cat << EOF

${COLOR_GREEN}Supported GPU Types${COLOR_NC}
────────────────────────────────────────
EOF
    
    for gpu_type in "${!GPU_ARCHITECTURES[@]}"; do
        local architecture cluster
        architecture=$(get_gpu_architecture "$gpu_type")
        cluster=$(get_cluster_name "$gpu_type")
        printf "%-6s %-8s %s\n" "$gpu_type" "$architecture" "$cluster"
    done
    echo
}

# =============================================================================
# Environment Configuration Functions
# =============================================================================

# Set GPU environment variables
set_gpu_environment() {
    local gpu_type="$1"
    
    export GPU_TYPE="$gpu_type"
    export GPU_ARCHITECTURE="$(get_gpu_architecture "$gpu_type")"
    export GPU_MEMORY_FREQ="$(get_gpu_memory_freq "$gpu_type")"
    export GPU_CORE_FREQ_MIN="$(get_gpu_core_freq_min "$gpu_type")"
    export GPU_CORE_FREQ_MAX="$(get_gpu_core_freq_max "$gpu_type")"
    export GPU_CORE_FREQ_STEP="$(get_gpu_core_freq_step "$gpu_type")"
    export GPU_SLURM_PARTITION="$(get_slurm_partition "$gpu_type")"
    export GPU_CLUSTER="$(get_cluster_name "$gpu_type")"
    
    local node
    node=$(get_gpu_node "$gpu_type")
    if [[ -n "$node" ]]; then
        export GPU_NODE="$node"
    fi
    
    log_debug "GPU environment variables set for $gpu_type"
}

# =============================================================================
# Conda Environment Management
# =============================================================================

# Determine appropriate conda environment for application
determine_conda_env() {
    local app_executable="$1"
    local app_dir="${2:-}"
    
    # Default environment
    local conda_env="base"
    
    # Application-specific environment mapping
    case "$(to_lower "$(basename "$app_executable")")" in
        *stable*|*diffusion*|*sd*)
            conda_env="stable-diffusion-gpu"
            ;;
        *llama*|*llm*)
            conda_env="llama"
            ;;
        *whisper*)
            conda_env="whisper"
            ;;
        *vit*|*vision*|*transformer*)
            conda_env="vit"
            ;;
        *lstm*|lstm.py)
            conda_env="tensorflow"
            ;;
        *)
            # Try to detect from application directory
            if [[ -n "$app_dir" ]]; then
                case "$(to_lower "$(basename "$app_dir")")" in
                    *stable*|*diffusion*)
                        conda_env="stable-diffusion-gpu"
                        ;;
                    *llama*|*llm*)
                        conda_env="llama"
                        ;;
                    *whisper*)
                        conda_env="whisper"
                        ;;
                    *vision*|*transformer*|*vit*)
                        conda_env="vit"
                        ;;
                    *lstm*)
                        conda_env="tensorflow"
                        ;;
                esac
            fi
            ;;
    esac
    
    log_debug "Determined conda environment '$conda_env' for application '$app_executable'"
    echo "$conda_env"
}

# Validate conda environment exists
validate_conda_env() {
    local env_name="$1"
    
    if ! command_exists "conda"; then
        log_warning "Conda not available, cannot validate environment: $env_name"
        return 1
    fi
    
    if conda env list | grep -q "^${env_name} "; then
        log_debug "Conda environment '$env_name' validated"
        return 0
    else
        log_warning "Conda environment '$env_name' not found"
        return 1
    fi
}

# Export functions for use in other scripts
export -f detect_gpu_type get_gpu_count is_gpu_available
export -f get_gpu_architecture get_gpu_memory_freq
export -f get_gpu_core_freq_max get_gpu_core_freq_min get_gpu_core_freq_step
export -f generate_frequency_range get_frequency_count
export -f get_slurm_partition get_cluster_name get_gpu_node
export -f validate_gpu_type validate_frequency
export -f show_gpu_config show_supported_gpus
export -f set_gpu_environment determine_conda_env validate_conda_env

log_debug "GPU configuration library v${GPU_CONFIG_VERSION} loaded successfully"
