#!/bin/bash
#
# Clean-up Script for AI Inference Energy Profiling.
#
# This script cleans up previous experiment results and log files,
# preparing the workspace for new profiling experiments.
#
# Usage:
#     ./clean.sh [OPTIONS]
#
# Options:
#     -h, --help            Show this help message
#     -v, --verbose         Enable verbose output
#     -f, --force           Force removal without confirmation prompts
#     -n, --dry-run         Show what would be cleaned without actually removing
#     -b, --backup          Create backup archive before cleaning
#     -s, --selective       Interactive mode to select what to clean
#     -o, --older-than DAYS Only clean results older than specified days
#     -g, --gpu-type TYPE   Only clean results for specific GPU type (H100, A100, V100)
#     -a, --app-name APP    Only clean results for specific application
#
# The script removes:
#     - results_*/ directories (new naming convention)
#     - results/ directory (legacy)
#     - PROFILING_*.out and PROFILING_*.err files (SLURM output)
#     - log.* files
#     - Temporary profiling files
#
# Author: Mert Side
#

set -euo pipefail  # Exit on error, undefined variables, and pipe failures

# Configuration
readonly SCRIPT_NAME="$(basename "$0")"
readonly RESULTS_PATTERN="results*"
readonly LEGACY_RESULTS_DIR="results"
readonly LOG_PATTERN="log.*"
readonly SLURM_PATTERN="PROFILING_*"
readonly TEMP_FILES=("changeme" "*.tmp" "*.temp" "*.pyc" "core.*")

# Default options
VERBOSE=false
FORCE=false
SELECTIVE=false
BACKUP=false
ARCHIVE_DIR="archive_$(date +%Y%m%d_%H%M%S)"
OLDER_THAN=""
GPU_TYPE=""
APP_NAME=""
DRY_RUN=false

# Cleanup flags (initialize all to false for selective mode)
CLEAN_RESULTS=false
CLEAN_SLURM=false
CLEAN_LOGS=false
CLEAN_TEMP=false

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
    - results_*/ directories (e.g., results_h100_lstm, results_a100_sd)
    - $LEGACY_RESULTS_DIR/ directory (legacy naming)
    - SLURM output files: $SLURM_PATTERN
    - Log files matching pattern: $LOG_PATTERN
    - Temporary files: ${TEMP_FILES[*]}

Options:
    -h, --help            Show this help message and exit
    -v, --verbose         Enable verbose output showing detailed operations
    -f, --force           Force removal without confirmation prompts
    -n, --dry-run         Show what would be cleaned without actually removing
    -b, --backup          Create backup archive before cleaning
    -s, --selective       Interactive mode to select specific items to clean
    -o, --older-than DAYS Only clean results older than specified days
    -g, --gpu-type TYPE   Only clean results for specific GPU (H100, A100, V100)
    -a, --app-name APP    Only clean results for specific application

Examples:
    $SCRIPT_NAME                    # Interactive cleanup with confirmations
    $SCRIPT_NAME -f                 # Force cleanup without prompts
    $SCRIPT_NAME -v                 # Verbose cleanup with detailed output
    $SCRIPT_NAME -n                 # Dry run - show what would be cleaned
    $SCRIPT_NAME -b -f              # Backup then force cleanup
    $SCRIPT_NAME -g H100            # Clean only H100 results
    $SCRIPT_NAME -a lstm            # Clean only LSTM results
    $SCRIPT_NAME -o 7               # Clean results older than 7 days
    $SCRIPT_NAME -s                 # Selective interactive cleanup

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
            -n|--dry-run)
                DRY_RUN=true
                log_verbose "Dry run mode enabled"
                shift
                ;;
            -b|--backup)
                BACKUP=true
                log_verbose "Backup mode enabled"
                shift
                ;;
            -s|--selective)
                SELECTIVE=true
                log_verbose "Selective mode enabled"
                shift
                ;;
            -o|--older-than)
                if [[ -z "${2:-}" ]] || [[ "$2" =~ ^- ]]; then
                    log_error "Option --older-than requires a value (number of days)"
                    exit 1
                fi
                OLDER_THAN="$2"
                log_verbose "Only cleaning files older than $OLDER_THAN days"
                shift 2
                ;;
            -g|--gpu-type)
                if [[ -z "${2:-}" ]] || [[ "$2" =~ ^- ]]; then
                    log_error "Option --gpu-type requires a value (H100, A100, V100)"
                    exit 1
                fi
                GPU_TYPE="$2"
                log_verbose "Only cleaning results for GPU type: $GPU_TYPE"
                shift 2
                ;;
            -a|--app-name)
                if [[ -z "${2:-}" ]] || [[ "$2" =~ ^- ]]; then
                    log_error "Option --app-name requires a value"
                    exit 1
                fi
                APP_NAME="$2"
                log_verbose "Only cleaning results for application: $APP_NAME"
                shift 2
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

    # Apply filters
    if ! matches_filters "$(basename "$dir_path")" || ! is_older_than "$dir_path" "$OLDER_THAN"; then
        log_verbose "Skipping $description (filtered): $dir_path"
        return 0
    fi

    # Count files and get size for information
    local file_count size
    file_count=$(find "$dir_path" -type f | wc -l)
    size=$(get_dir_size "$dir_path")

    if [[ "$DRY_RUN" == true ]]; then
        log_info "[DRY RUN] Would remove $description: $dir_path ($size, $file_count files)"
        return 0
    fi

    if ! confirm_action "Remove $description with $file_count files ($size)? ($dir_path)"; then
        log_info "Skipped removal of $description"
        return 0
    fi

    log_info "Removing $description: $dir_path ($size)"

    if rm -rf "$dir_path"; then
        log_info "Successfully removed $description ($size freed)"
    else
        log_error "Failed to remove $description"
        return 1
    fi

    return 0
}

# Function to remove results directories
remove_results_directories() {
    log_info "Cleaning results directories..."

    # Find all results directories (new naming convention)
    local results_dirs
    mapfile -t results_dirs < <(find . -maxdepth 1 -name "$RESULTS_PATTERN" -type d 2>/dev/null || true)

    if (( ${#results_dirs[@]} > 0 )); then
        log_info "Found ${#results_dirs[@]} results directories"
        for dir in "${results_dirs[@]}"; do
            remove_directory "$dir" "results directory"
        done
    else
        log_verbose "No results directories found"
    fi

    # Also check for legacy results directory
    if [[ -d "$LEGACY_RESULTS_DIR" ]]; then
        remove_directory "$LEGACY_RESULTS_DIR" "legacy results directory"
    fi
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

    # Apply age filter
    local filtered_files=()
    for file in "${matching_files[@]}"; do
        if is_older_than "$file" "$OLDER_THAN"; then
            filtered_files+=("$file")
        fi
    done

    if (( ${#filtered_files[@]} == 0 )); then
        log_verbose "No $description match age filter"
        return 0
    fi

    log_info "Found ${#filtered_files[@]} $description matching pattern: $pattern"

    if [[ "$VERBOSE" == true ]]; then
        for file in "${filtered_files[@]}"; do
            local size
            size=$(du -sh "$file" 2>/dev/null | cut -f1)
            local age
            age=$(stat -c %y "$file" 2>/dev/null | cut -d' ' -f1)
            log_verbose "  - $file ($size, $age)"
        done
    fi

    if [[ "$DRY_RUN" == true ]]; then
        log_info "[DRY RUN] Would remove ${#filtered_files[@]} $description"
        return 0
    fi

    if ! confirm_action "Remove ${#filtered_files[@]} $description?"; then
        log_info "Skipped removal of $description"
        return 0
    fi

    log_info "Removing $description..."

    local removed_count=0
    for file in "${filtered_files[@]}"; do
        if rm -f "$file"; then
            log_verbose "Removed: $file"
            removed_count=$((removed_count + 1))
        else
            log_error "Failed to remove: $file"
        fi
    done

    log_info "Successfully removed $removed_count/${#filtered_files[@]} $description"
    return 0
}

# Function to remove temporary files
remove_temp_files() {
    log_info "Cleaning temporary files..."

    for pattern in "${TEMP_FILES[@]}"; do
        remove_files_by_pattern "$pattern" "temporary files"
    done
}

# Function to get directory size in human readable format
get_dir_size() {
    local dir_path="$1"
    if [[ -d "$dir_path" ]]; then
        du -sh "$dir_path" 2>/dev/null | cut -f1
    else
        echo "0B"
    fi
}

# Function to check if directory matches filters
matches_filters() {
    local dir_name="$1"

    # Apply GPU type filter
    if [[ -n "$GPU_TYPE" ]]; then
        if [[ ! "$dir_name" =~ results_${GPU_TYPE,,}_ ]]; then
            return 1
        fi
    fi

    # Apply application name filter
    if [[ -n "$APP_NAME" ]]; then
        if [[ ! "$dir_name" =~ _${APP_NAME,,} ]] && [[ ! "$dir_name" =~ _${APP_NAME}$ ]]; then
            return 1
        fi
    fi

    return 0
}

# Function to check if file/directory is older than specified days
is_older_than() {
    local path="$1"
    local days="$2"

    if [[ -z "$days" ]]; then
        return 0  # No age filter
    fi

    local file_time
    file_time=$(stat -c %Y "$path" 2>/dev/null || echo 0)
    local current_time
    current_time=$(date +%s)
    local age_seconds=$(( (current_time - file_time) ))
    local age_days=$(( age_seconds / 86400 ))

    [[ $age_days -gt $days ]]
}

# Function to create backup archive
create_backup() {
    if [[ "$BACKUP" != true ]]; then
        return 0
    fi

    log_info "Creating backup archive: $ARCHIVE_DIR"

    if [[ "$DRY_RUN" == true ]]; then
        log_info "[DRY RUN] Would create backup archive"
        return 0
    fi

    mkdir -p "$ARCHIVE_DIR"

    # Find all results directories
    local results_dirs
    mapfile -t results_dirs < <(find . -maxdepth 1 -name "$RESULTS_PATTERN" -type d 2>/dev/null || true)

    if (( ${#results_dirs[@]} > 0 )); then
        log_info "Backing up ${#results_dirs[@]} results directories..."
        for dir in "${results_dirs[@]}"; do
            if matches_filters "$(basename "$dir")" && is_older_than "$dir" "$OLDER_THAN"; then
                log_verbose "Backing up: $dir"
                cp -r "$dir" "$ARCHIVE_DIR/" 2>/dev/null || log_error "Failed to backup $dir"
            fi
        done
    fi

    # Backup SLURM files
    local slurm_files
    mapfile -t slurm_files < <(find . -maxdepth 1 -name "$SLURM_PATTERN" -type f 2>/dev/null || true)

    if (( ${#slurm_files[@]} > 0 )); then
        log_info "Backing up ${#slurm_files[@]} SLURM output files..."
        for file in "${slurm_files[@]}"; do
            if is_older_than "$file" "$OLDER_THAN"; then
                cp "$file" "$ARCHIVE_DIR/" 2>/dev/null || log_error "Failed to backup $file"
            fi
        done
    fi

    log_info "Backup completed in: $ARCHIVE_DIR"
}

# Function to display cleanup preview
show_cleanup_preview() {
    log_info "Cleanup Preview"
    log_info "==============="

    local total_size=0
    local item_count=0

    # Check results directories
    local results_dirs
    mapfile -t results_dirs < <(find . -maxdepth 1 -name "$RESULTS_PATTERN" -type d 2>/dev/null || true)

    for dir in "${results_dirs[@]}"; do
        if matches_filters "$(basename "$dir")" && is_older_than "$dir" "$OLDER_THAN"; then
            local size
            size=$(get_dir_size "$dir")
            log_info "  📁 $dir ($size)"
            item_count=$((item_count + 1))
        fi
    done

    # Check legacy results directory
    if [[ -d "$LEGACY_RESULTS_DIR" ]] && is_older_than "$LEGACY_RESULTS_DIR" "$OLDER_THAN"; then
        local size
        size=$(get_dir_size "$LEGACY_RESULTS_DIR")
        log_info "  📁 $LEGACY_RESULTS_DIR ($size) [legacy]"
        item_count=$((item_count + 1))
    fi

    # Check SLURM files
    local slurm_files
    mapfile -t slurm_files < <(find . -maxdepth 1 -name "$SLURM_PATTERN" -type f 2>/dev/null || true)

    local slurm_count=0
    for file in "${slurm_files[@]}"; do
        if is_older_than "$file" "$OLDER_THAN"; then
            slurm_count=$((slurm_count + 1))
        fi
    done

    if (( slurm_count > 0 )); then
        log_info "  📄 $slurm_count SLURM output files"
        item_count=$((item_count + 1))
    fi

    # Check log files
    local log_files
    mapfile -t log_files < <(find . -maxdepth 1 -name "$LOG_PATTERN" -type f 2>/dev/null || true)

    local log_count=0
    for file in "${log_files[@]}"; do
        if is_older_than "$file" "$OLDER_THAN"; then
            log_count=$((log_count + 1))
        fi
    done

    if (( log_count > 0 )); then
        log_info "  📄 $log_count log files"
        item_count=$((item_count + 1))
    fi

    # Check temp files
    local temp_count=0
    for pattern in "${TEMP_FILES[@]}"; do
        local pattern_files
        mapfile -t pattern_files < <(find . -maxdepth 1 -name "$pattern" -type f 2>/dev/null || true)
        for file in "${pattern_files[@]}"; do
            if is_older_than "$file" "$OLDER_THAN"; then
                temp_count=$((temp_count + 1))
            fi
        done
    done

    if (( temp_count > 0 )); then
        log_info "  🗑️  $temp_count temporary files"
        item_count=$((item_count + 1))
    fi

    log_info ""
    log_info "Total items to clean: $item_count"

    if [[ -n "$GPU_TYPE" ]]; then
        log_info "Filter: GPU type = $GPU_TYPE"
    fi
    if [[ -n "$APP_NAME" ]]; then
        log_info "Filter: Application = $APP_NAME"
    fi
    if [[ -n "$OLDER_THAN" ]]; then
        log_info "Filter: Older than $OLDER_THAN days"
    fi
}

# Function to display cleanup summary
display_summary() {
    log_info "Cleanup summary:"

    # Check results directories
    local results_count=0
    local results_dirs
    mapfile -t results_dirs < <(find . -maxdepth 1 -name "$RESULTS_PATTERN" -type d 2>/dev/null || true)

    for dir in "${results_dirs[@]}"; do
        if [[ -d "$dir" ]]; then
            local file_count size
            file_count=$(find "$dir" -type f | wc -l)
            size=$(get_dir_size "$dir")
            log_info "  - $(basename "$dir"): $file_count files ($size)"
            results_count=$((results_count + 1))
        fi
    done

    if [[ -d "$LEGACY_RESULTS_DIR" ]]; then
        local file_count size
        file_count=$(find "$LEGACY_RESULTS_DIR" -type f | wc -l)
        size=$(get_dir_size "$LEGACY_RESULTS_DIR")
        log_info "  - $LEGACY_RESULTS_DIR (legacy): $file_count files ($size)"
        results_count=$((results_count + 1))
    fi

    log_info "  - Results directories: $results_count"

    # Check for remaining SLURM files
    local slurm_count
    slurm_count=$(find . -maxdepth 1 -name "$SLURM_PATTERN" -type f | wc -l)
    log_info "  - SLURM output files: $slurm_count files"

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

    if [[ "$BACKUP" == true ]] && [[ -d "$ARCHIVE_DIR" ]]; then
        local archive_size
        archive_size=$(get_dir_size "$ARCHIVE_DIR")
        log_info "  - Backup archive: $ARCHIVE_DIR ($archive_size)"
    fi
}

# Main cleanup function
main() {
    log_info "Starting workspace cleanup for AI inference energy profiling"
    log_info "Working directory: $(pwd)"

    # Parse command line arguments
    parse_arguments "$@"

    # Show mode information
    if [[ "$DRY_RUN" == true ]]; then
        log_info "Running in DRY RUN mode - no files will be removed"
    fi

    if [[ "$FORCE" == true ]]; then
        log_info "Running in force mode - no confirmation prompts"
    fi

    if [[ "$BACKUP" == true ]]; then
        log_info "Backup mode enabled - creating archive before cleanup"
    fi

    # Show cleanup preview
    show_cleanup_preview

    # Handle selective mode
    if [[ "$SELECTIVE" == true ]]; then
        log_info "Selective mode - choose what to clean:"
        echo "1) Results directories"
        echo "2) SLURM output files"
        echo "3) Log files"
        echo "4) Temporary files"
        echo "5) All of the above"
        echo -n "Select option (1-5): "
        read -r selection

        case "$selection" in
            1) CLEAN_RESULTS=true ;;
            2) CLEAN_SLURM=true ;;
            3) CLEAN_LOGS=true ;;
            4) CLEAN_TEMP=true ;;
            5) CLEAN_RESULTS=true; CLEAN_SLURM=true; CLEAN_LOGS=true; CLEAN_TEMP=true ;;
            *) log_error "Invalid selection"; exit 1 ;;
        esac
    else
        # Default: clean everything
        CLEAN_RESULTS=true
        CLEAN_SLURM=true
        CLEAN_LOGS=true
        CLEAN_TEMP=true
    fi

    # Exit early if dry run
    if [[ "$DRY_RUN" == true ]]; then
        log_info "Dry run completed. Use without -n/--dry-run to perform actual cleanup."
        exit 0
    fi

    # Ask for confirmation to proceed (unless force mode)
    if [[ "$FORCE" != true ]]; then
        echo ""
        if ! confirm_action "Proceed with cleanup"; then
            log_info "Cleanup cancelled by user"
            exit 0
        fi
    fi

    # Create backup if requested
    create_backup

    # Perform cleanup operations
    local cleanup_failed=false

    # Remove results directories
    if [[ "$CLEAN_RESULTS" == true ]]; then
        if ! remove_results_directories; then
            cleanup_failed=true
        fi
    fi

    # Remove SLURM output files
    if [[ "$CLEAN_SLURM" == true ]]; then
        if ! remove_files_by_pattern "$SLURM_PATTERN" "SLURM output files"; then
            cleanup_failed=true
        fi
    fi

    # Remove log files
    if [[ "$CLEAN_LOGS" == true ]]; then
        if ! remove_files_by_pattern "$LOG_PATTERN" "log files"; then
            cleanup_failed=true
        fi
    fi

    # Remove temporary files
    if [[ "$CLEAN_TEMP" == true ]]; then
        if ! remove_temp_files; then
            cleanup_failed=true
        fi
    fi

    # Display summary
    display_summary

    if [[ "$cleanup_failed" == true ]]; then
        log_error "Cleanup completed with some errors"
        exit 1
    else
        log_info "Cleanup completed successfully"
        log_info "Workspace is ready for new profiling experiments"

        if [[ "$BACKUP" == true ]] && [[ -d "$ARCHIVE_DIR" ]]; then
            log_info "Backup archive available at: $ARCHIVE_DIR"
        fi
    fi
}

# Execute main function with all arguments
main "$@"
