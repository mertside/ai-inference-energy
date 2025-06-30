#!/bin/bash
"""
Clean-up Script for AI Inference Energy Profiling.

This script cleans up previous experiment results and log files,
preparing the workspace for new profiling experiments.

Usage:
    ./clean.sh [OPTIONS]

Options:
    -h, --help     Show this help message
    -v, --verbose  Enable verbose output
    -f, --force    Force removal without confirmation prompts

The script removes:
    - results/ directory and all contents
    - log.* files
    - Temporary profiling files

Author: AI Inference Energy Research Team
"""

set -euo pipefail  # Exit on error, undefined variables, and pipe failures

# Configuration
readonly SCRIPT_NAME="$(basename "$0")"
readonly RESULTS_DIR="results"
readonly LOG_PATTERN="log.*"
readonly TEMP_FILES=("changeme" "*.tmp" "*.temp")

# Default options
VERBOSE=false
FORCE=false

# Logging functions
log_info() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] [INFO] $*" >&2
}

log_error() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] [ERROR] $*" >&2
}

log_verbose() {
    if [[ "$VERBOSE" == true ]]; then
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] [VERBOSE] $*" >&2
    fi
}

# Function to display usage information
usage() {
    cat << EOF
Usage: $SCRIPT_NAME [OPTIONS]

Clean-up Script for AI Inference Energy Profiling

This script prepares the workspace for new profiling experiments by removing
previous results, logs, and temporary files.

Options:
    -h, --help     Show this help message and exit
    -v, --verbose  Enable verbose output showing detailed operations
    -f, --force    Force removal without confirmation prompts

What gets cleaned:
    - $RESULTS_DIR/ directory and all contents
    - Log files matching pattern: $LOG_PATTERN
    - Temporary files: ${TEMP_FILES[*]}

Examples:
    $SCRIPT_NAME           # Interactive cleanup with confirmations
    $SCRIPT_NAME -f        # Force cleanup without prompts
    $SCRIPT_NAME -v        # Verbose cleanup with detailed output
    $SCRIPT_NAME -f -v     # Force verbose cleanup

Safety:
    - By default, the script asks for confirmation before removing data
    - Use --force to skip confirmations (useful for automation)
    - The script will not remove files outside the current directory

EOF
}

# Function to parse command line arguments
parse_arguments() {
    while (( $# > 0 )); do
        case "$1" in
            -h|--help)
                usage
                exit 0
                ;;
            -v|--verbose)
                VERBOSE=true
                log_verbose "Verbose mode enabled"
                shift
                ;;
            -f|--force)
                FORCE=true
                log_verbose "Force mode enabled"
                shift
                ;;
            -*)
                log_error "Unknown option: $1"
                usage
                exit 1
                ;;
            *)
                log_error "Unexpected argument: $1"
                usage
                exit 1
                ;;
        esac
    done
}

# Function to confirm action with user
confirm_action() {
    local message="$1"
    
    if [[ "$FORCE" == true ]]; then
        log_verbose "Force mode: skipping confirmation for '$message'"
        return 0
    fi
    
    local response
    echo -n "$message (y/N): "
    read -r response
    
    case "$response" in
        [yY]|[yY][eE][sS])
            return 0
            ;;
        *)
            return 1
            ;;
    esac
}

# Function to safely remove directory
remove_directory() {
    local dir_path="$1"
    local description="$2"
    
    if [[ ! -e "$dir_path" ]]; then
        log_verbose "$description does not exist: $dir_path"
        return 0
    fi
    
    if [[ ! -d "$dir_path" ]]; then
        log_error "$description is not a directory: $dir_path"
        return 1
    fi
    
    # Count files for information
    local file_count
    file_count=$(find "$dir_path" -type f | wc -l)
    
    if ! confirm_action "Remove $description with $file_count files? ($dir_path)"; then
        log_info "Skipped removal of $description"
        return 0
    fi
    
    log_info "Removing $description: $dir_path"
    
    if rm -rf "$dir_path"; then
        log_info "Successfully removed $description"
        log_verbose "Creating new empty $description"
        
        if mkdir -p "$dir_path"; then
            log_info "Created new empty $description"
        else
            log_error "Failed to create new $description"
            return 1
        fi
    else
        log_error "Failed to remove $description"
        return 1
    fi
    
    return 0
}

# Function to remove files matching pattern
remove_files_by_pattern() {
    local pattern="$1"
    local description="$2"
    
    # Find files matching pattern
    local matching_files
    mapfile -t matching_files < <(find . -maxdepth 1 -name "$pattern" -type f 2>/dev/null || true)
    
    if (( ${#matching_files[@]} == 0 )); then
        log_verbose "No $description found matching pattern: $pattern"
        return 0
    fi
    
    log_info "Found ${#matching_files[@]} $description matching pattern: $pattern"
    
    if [[ "$VERBOSE" == true ]]; then
        for file in "${matching_files[@]}"; do
            log_verbose "  - $file"
        done
    fi
    
    if ! confirm_action "Remove ${#matching_files[@]} $description?"; then
        log_info "Skipped removal of $description"
        return 0
    fi
    
    log_info "Removing $description..."
    
    local removed_count=0
    for file in "${matching_files[@]}"; do
        if rm -f "$file"; then
            log_verbose "Removed: $file"
            ((removed_count++))
        else
            log_error "Failed to remove: $file"
        fi
    done
    
    log_info "Successfully removed $removed_count/$((${#matching_files[@]})) $description"
    return 0
}

# Function to remove temporary files
remove_temp_files() {
    log_info "Cleaning temporary files..."
    
    for pattern in "${TEMP_FILES[@]}"; do
        remove_files_by_pattern "$pattern" "temporary files"
    done
}

# Function to display cleanup summary
display_summary() {
    log_info "Cleanup summary:"
    
    # Check what exists after cleanup
    if [[ -d "$RESULTS_DIR" ]]; then
        local result_count
        result_count=$(find "$RESULTS_DIR" -type f | wc -l)
        log_info "  - Results directory: $result_count files"
    else
        log_info "  - Results directory: does not exist"
    fi
    
    # Check for remaining log files
    local log_count
    log_count=$(find . -maxdepth 1 -name "$LOG_PATTERN" -type f | wc -l)
    log_info "  - Log files: $log_count files"
    
    # Check for remaining temp files
    local temp_count=0
    for pattern in "${TEMP_FILES[@]}"; do
        local pattern_count
        pattern_count=$(find . -maxdepth 1 -name "$pattern" -type f | wc -l)
        temp_count=$((temp_count + pattern_count))
    done
    log_info "  - Temporary files: $temp_count files"
}

# Main cleanup function
main() {
    log_info "Starting workspace cleanup for AI inference energy profiling"
    log_info "Working directory: $(pwd)"
    
    # Parse command line arguments
    parse_arguments "$@"
    
    if [[ "$FORCE" == true ]]; then
        log_info "Running in force mode - no confirmation prompts"
    fi
    
    # Perform cleanup operations
    local cleanup_failed=false
    
    # Remove results directory
    if ! remove_directory "$RESULTS_DIR" "results directory"; then
        cleanup_failed=true
    fi
    
    # Remove log files
    if ! remove_files_by_pattern "$LOG_PATTERN" "log files"; then
        cleanup_failed=true
    fi
    
    # Remove temporary files
    if ! remove_temp_files; then
        cleanup_failed=true
    fi
    
    # Display summary
    display_summary
    
    if [[ "$cleanup_failed" == true ]]; then
        log_error "Cleanup completed with some errors"
        exit 1
    else
        log_info "Cleanup completed successfully"
        log_info "Workspace is ready for new profiling experiments"
    fi
}

# Execute main function with all arguments
main "$@"
