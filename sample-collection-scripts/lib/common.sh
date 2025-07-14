#!/bin/bash
#
# Common Library Functions for AI Inference Energy Profiling
#
# This library provides shared functionality used across the profiling framework,
# including logging, error handling, utility functions, and common patterns.
#
# Author: Mert Side
#

# Prevent multiple inclusions
if [[ "${COMMON_LIB_LOADED:-}" == "true" ]]; then
    return 0
fi
readonly COMMON_LIB_LOADED="true"

# =============================================================================
# Constants and Global Variables
# =============================================================================

# Script identification
readonly COMMON_LIB_VERSION="1.0.0"
readonly COMMON_LIB_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly SCRIPTS_ROOT_DIR="$(dirname "${COMMON_LIB_DIR}")"

# Helper function: Check if colors are explicitly disabled
is_color_disabled() {
    [[ "${NO_COLOR:-}" == "1" ]] || [[ "${DISABLE_COLORS:-}" == "1" ]] || [[ "${FORCE_COLOR:-}" == "0" ]]
}

# Helper function: Check if colors are explicitly enabled
is_color_forced() {
    [[ "${FORCE_COLOR:-}" == "1" ]]
}

# Helper function: Check if terminal supports colors
is_terminal_valid() {
    [[ -t 1 ]] && [[ "${TERM:-}" != "dumb" ]] && [[ "${TERM:-}" != "" ]]
}

# Color detection function - checks at runtime
should_use_colors() {
    # Explicit disable flags take priority
    if is_color_disabled; then
        return 1
    fi
    
    # Force enable if explicitly requested
    if is_color_forced; then
        return 0
    fi
    
    # Enable colors if output is a TTY and TERM is valid
    if is_terminal_valid; then
        return 0
    fi
    
    # Default to NO colors
    return 1
}

# Color codes (defined for reference, actual usage controlled by should_use_colors)
readonly _COLOR_RED='\033[0;31m'
readonly _COLOR_GREEN='\033[0;32m'
readonly _COLOR_YELLOW='\033[1;33m'
readonly _COLOR_BLUE='\033[0;34m'
readonly _COLOR_PURPLE='\033[0;35m'
readonly _COLOR_CYAN='\033[0;36m'
readonly _COLOR_NC='\033[0m' # No Color

# Initialize color variables as empty by default
COLOR_RED=''
COLOR_GREEN=''
COLOR_YELLOW=''
COLOR_BLUE=''
COLOR_PURPLE=''
COLOR_CYAN=''
COLOR_NC=''

# =============================================================================
# Logging Functions
# =============================================================================

# Get current timestamp for logging
get_timestamp() {
    date '+%Y-%m-%d %H:%M:%S'
}

# Log info message
log_info() {
    echo -e "[$(get_timestamp)] [${COLOR_GREEN}INFO${COLOR_NC}] $*" >&2
}

# Log error message
log_error() {
    echo -e "[$(get_timestamp)] [${COLOR_RED}ERROR${COLOR_NC}] $*" >&2
}

# Log warning message
log_warning() {
    echo -e "[$(get_timestamp)] [${COLOR_YELLOW}WARNING${COLOR_NC}] $*" >&2
}

# Log debug message (only if DEBUG=1)
log_debug() {
    if [[ "${DEBUG:-0}" == "1" ]]; then
        echo -e "[$(get_timestamp)] [${COLOR_CYAN}DEBUG${COLOR_NC}] $*" >&2
    fi
}

# Log success message
log_success() {
    echo -e "[$(get_timestamp)] [${COLOR_GREEN}SUCCESS${COLOR_NC}] $*" >&2
}

# =============================================================================
# Error Handling Functions
# =============================================================================

# Exit with error message
die() {
    log_error "$@"
    exit 1
}

# Check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Check if file exists and is readable
file_readable() {
    [[ -f "$1" && -r "$1" ]]
}

# Check if directory exists and is accessible
dir_accessible() {
    [[ -d "$1" && -x "$1" ]]
}

# Validate required environment variables
check_required_vars() {
    local missing_vars=()
    
    for var in "$@"; do
        if [[ -z "${!var:-}" ]]; then
            missing_vars+=("$var")
        fi
    done
    
    if [[ ${#missing_vars[@]} -gt 0 ]]; then
        die "Missing required environment variables: ${missing_vars[*]}"
    fi
}

# =============================================================================
# File and Path Utilities
# =============================================================================

# Get absolute path
get_absolute_path() {
    local path="$1"
    if [[ "$path" = /* ]]; then
        echo "$path"
    else
        echo "$(pwd)/$path"
    fi
}

# Create directory if it doesn't exist
ensure_directory() {
    local dir="$1"
    if [[ ! -d "$dir" ]]; then
        log_debug "Creating directory: $dir"
        mkdir -p "$dir" || die "Failed to create directory: $dir"
    fi
}

# Backup file with timestamp
backup_file() {
    local file="$1"
    if [[ -f "$file" ]]; then
        local backup="${file}.backup.$(date +%Y%m%d_%H%M%S)"
        cp "$file" "$backup" || die "Failed to backup file: $file"
        log_debug "Backed up $file to $backup"
    fi
}

# =============================================================================
# String Utilities
# =============================================================================

# Trim whitespace from string
trim() {
    local var="$*"
    # Remove leading whitespace
    var="${var#"${var%%[![:space:]]*}"}"
    # Remove trailing whitespace
    var="${var%"${var##*[![:space:]]}"}"
    echo "$var"
}

# Convert string to lowercase
to_lower() {
    echo "$*" | tr '[:upper:]' '[:lower:]'
}

# Convert string to uppercase
to_upper() {
    echo "$*" | tr '[:lower:]' '[:upper:]'
}

# Check if string contains substring
contains() {
    local string="$1"
    local substring="$2"
    [[ "$string" == *"$substring"* ]]
}

# =============================================================================
# Process and System Utilities
# =============================================================================

# Check if process is running by PID
process_running() {
    local pid="$1"
    kill -0 "$pid" 2>/dev/null
}

# Wait for process to finish with timeout
wait_for_process() {
    local pid="$1"
    local timeout="${2:-30}"
    local count=0
    
    while process_running "$pid" && [[ $count -lt $timeout ]]; do
        sleep 1
        ((count++))
    done
    
    if process_running "$pid"; then
        log_warning "Process $pid still running after ${timeout}s timeout"
        return 1
    fi
    return 0
}

# Get available memory in GB
get_available_memory() {
    awk '/MemAvailable:/ {printf "%.1f", $2/1024/1024}' /proc/meminfo
}

# Get CPU count
get_cpu_count() {
    nproc
}

# =============================================================================
# Validation Functions
# =============================================================================

# Validate positive integer
is_positive_integer() {
    [[ "$1" =~ ^[1-9][0-9]*$ ]]
}

# Validate frequency value (positive integer)
is_valid_frequency() {
    is_positive_integer "$1"
}

# Validate GPU type
is_valid_gpu_type() {
    local gpu_type="$1"
    case "$(to_upper "$gpu_type")" in
        A100|V100|H100) return 0 ;;
        *) return 1 ;;
    esac
}

# Validate profiling tool
is_valid_profiling_tool() {
    local tool="$1"
    case "$(to_lower "$tool")" in
        dcgmi|nvidia-smi) return 0 ;;
        *) return 1 ;;
    esac
}

# Validate profiling mode
is_valid_profiling_mode() {
    local mode="$1"
    case "$(to_lower "$mode")" in
        dvfs|baseline) return 0 ;;
        *) return 1 ;;
    esac
}

# =============================================================================
# Array Utilities
# =============================================================================

# Join array elements with delimiter
join_array() {
    local delimiter="$1"
    shift
    local first="$1"
    shift
    printf "%s" "$first" "${@/#/$delimiter}"
}

# Check if array contains element
array_contains() {
    local element="$1"
    shift
    local array=("$@")
    
    for item in "${array[@]}"; do
        if [[ "$item" == "$element" ]]; then
            return 0
        fi
    done
    return 1
}

# =============================================================================
# Progress and Status Functions
# =============================================================================

# Show progress spinner
show_spinner() {
    local pid="$1"
    local message="${2:-Working...}"
    local spinner='|/-\'
    local i=0
    
    while process_running "$pid"; do
        printf "\r%s [%c]" "$message" "${spinner:i++%${#spinner}:1}"
        sleep 0.1
    done
    printf "\r%s [Done]\n" "$message"
}

# Show progress bar
show_progress() {
    local current="$1"
    local total="$2"
    local message="${3:-Progress}"
    local width=50
    
    local percent=$((current * 100 / total))
    local filled=$((current * width / total))
    local empty=$((width - filled))
    
    printf "\r%s: [" "$message"
    printf "%*s" "$filled" | tr ' ' '='
    printf "%*s" "$empty"
    printf "] %d%% (%d/%d)" "$percent" "$current" "$total"
    
    if [[ $current -eq $total ]]; then
        printf "\n"
    fi
}

# =============================================================================
# Cleanup and Signal Handling
# =============================================================================

# Global cleanup functions array
declare -a CLEANUP_FUNCTIONS=()

# Register cleanup function
register_cleanup() {
    CLEANUP_FUNCTIONS+=("$1")
}

# Execute all cleanup functions
execute_cleanup() {
    log_debug "Executing cleanup functions..."
    for cleanup_func in "${CLEANUP_FUNCTIONS[@]}"; do
        if declare -f "$cleanup_func" > /dev/null; then
            log_debug "Running cleanup function: $cleanup_func"
            "$cleanup_func" || log_warning "Cleanup function failed: $cleanup_func"
        fi
    done
}

# Setup signal handlers for cleanup
setup_signal_handlers() {
    trap 'execute_cleanup; exit 130' INT  # Ctrl+C
    trap 'execute_cleanup; exit 143' TERM # Termination
    trap 'execute_cleanup' EXIT           # Normal exit
}

# =============================================================================
# Version and Help Functions
# =============================================================================

# Show library version
show_version() {
    echo "Common Library v${COMMON_LIB_VERSION}"
}

# Debug function for color detection (can be called with --debug-colors)
debug_color_detection() {
    echo "=== Color Detection Debug ==="
    echo "TTY stdout (-t 1): $(test -t 1 && echo true || echo false)"
    echo "TTY stderr (-t 2): $(test -t 2 && echo true || echo false)"
    echo "TERM: ${TERM:-not_set}"
    echo "NO_COLOR: ${NO_COLOR:-not_set}"
    echo "DISABLE_COLORS: ${DISABLE_COLORS:-not_set}"
    echo "FORCE_COLOR: ${FORCE_COLOR:-not_set}"
    echo "should_use_colors(): $(should_use_colors && echo true || echo false)"
    if should_use_colors; then
        echo "Color example: ${_COLOR_GREEN}GREEN${_COLOR_NC} ${_COLOR_RED}RED${_COLOR_NC}"
    else
        echo "Colors disabled - no color example shown"
    fi
    echo "=========================="
}

# Export commonly used functions
export -f log_info log_error log_warning log_debug log_success
export -f die command_exists file_readable dir_accessible
export -f get_absolute_path ensure_directory trim to_lower to_upper
export -f is_positive_integer is_valid_frequency is_valid_gpu_type
export -f is_valid_profiling_tool is_valid_profiling_mode

log_debug "Common library v${COMMON_LIB_VERSION} loaded successfully"
