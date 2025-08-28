#!/bin/bash
#
# 🎨 Stable Diffusion Image Management Script
#
# Smart cleanup and archiving for Stable Diffusion images generated during
# DVFS frequency sweeps. Prevents disk space issues while preserving important data.
#
# Usage:
#     ./cleanup_sd_images.sh [OPTIONS]
#
# Options:
#     -h, --help              Show this help message
#     -v, --verbose           Enable verbose output
#     -n, --dry-run          Show what would be cleaned without removing
#     -a, --archive          Archive images before cleanup
#     -c, --compress         Compress archives (requires --archive)
#     -k, --keep-samples N   Keep N sample images per experiment
#     -o, --older-than DAYS  Only process images older than N days
#     -s, --size-limit SIZE  Clean when images exceed SIZE (e.g., 1G, 500M)
#     -f, --force            Skip confirmation prompts
#
# Examples:
#     ./cleanup_sd_images.sh --archive --compress  # Archive all, then clean
#     ./cleanup_sd_images.sh --keep-samples 5      # Keep 5 representative images
#     ./cleanup_sd_images.sh --size-limit 1G       # Clean if over 1GB
#     ./cleanup_sd_images.sh --older-than 7        # Clean images older than 7 days
#
# Author: Mert Side
# Date: July 2025

set -euo pipefail

# Configuration
readonly SCRIPT_NAME="$(basename "$0")"
readonly SD_IMAGES_DIR="../app-stable-diffusion/images"
readonly ARCHIVE_DIR="sd_archives"
readonly MAX_ARCHIVE_SIZE="10G"

# Default options
VERBOSE=false
DRY_RUN=false
ARCHIVE=false
COMPRESS=false
KEEP_SAMPLES=0
JOB_ID=""
OLDER_THAN=""
SIZE_LIMIT=""
FORCE=false

# Color codes
readonly RED='\033[0;31m'
readonly GREEN='\033[0;32m'
readonly YELLOW='\033[1;33m'
readonly BLUE='\033[0;34m'
readonly NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${GREEN}[INFO]${NC} $*"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $*"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $*" >&2
}

log_debug() {
    [[ "$VERBOSE" == "true" ]] && echo -e "${BLUE}[DEBUG]${NC} $*"
}

# Show usage information
show_help() {
    cat << EOF
🎨 Stable Diffusion Image Management Script

DESCRIPTION:
    Smart cleanup and archiving for Stable Diffusion images generated during
    DVFS frequency sweeps. Prevents disk space issues while preserving data.

USAGE:
    $SCRIPT_NAME [OPTIONS]

OPTIONS:
    -h, --help              Show this help message
    -v, --verbose           Enable verbose output
    -n, --dry-run          Show what would be cleaned without removing
    -a, --archive          Archive images before cleanup
    -c, --compress         Compress archives (requires --archive)
    -k, --keep-samples N   Keep N sample images per experiment
    -j, --job-id ID        Only process images from specific job ID
    -o, --older-than DAYS  Only process images older than N days
    -s, --size-limit SIZE  Clean when images exceed SIZE (e.g., 1G, 500M)
    -f, --force            Skip confirmation prompts

EXAMPLES:
    # Archive all images, compress, then clean
    $SCRIPT_NAME --archive --compress

    # Keep 5 representative images per experiment
    $SCRIPT_NAME --keep-samples 5

    # Clean if images exceed 1GB total
    $SCRIPT_NAME --size-limit 1G

    # Clean images older than 7 days
    $SCRIPT_NAME --older-than 7

    # Research mode: clean all but keep metrics
    $SCRIPT_NAME --keep-samples 0 --force

STORAGE ESTIMATES:
    H100 DVFS (86 freq): ~400MB - 1.2GB
    A100 DVFS (61 freq): ~300MB - 900MB
    V100 DVFS (117 freq): ~500MB - 1.5GB

AUTHOR:
    Mert Side - AI Inference Energy Research Framework
EOF
}

# Parse command line arguments
parse_args() {
    while [[ $# -gt 0 ]]; do
        case $1 in
            -h|--help)
                show_help
                exit 0
                ;;
            -v|--verbose)
                VERBOSE=true
                shift
                ;;
            -n|--dry-run)
                DRY_RUN=true
                shift
                ;;
            -a|--archive)
                ARCHIVE=true
                shift
                ;;
            -c|--compress)
                COMPRESS=true
                shift
                ;;
            -k|--keep-samples)
                KEEP_SAMPLES="$2"
                shift 2
                ;;
            -j|--job-id)
                JOB_ID="$2"
                shift 2
                ;;
            -o|--older-than)
                OLDER_THAN="$2"
                shift 2
                ;;
            -s|--size-limit)
                SIZE_LIMIT="$2"
                shift 2
                ;;
            -f|--force)
                FORCE=true
                shift
                ;;
            *)
                log_error "Unknown option: $1"
                show_help
                exit 1
                ;;
        esac
    done
}

# Convert human-readable size to bytes
size_to_bytes() {
    local size="$1"
    case "$size" in
        *K|*k) echo $((${size%?} * 1024)) ;;
        *M|*m) echo $((${size%?} * 1024 * 1024)) ;;
        *G|*g) echo $((${size%?} * 1024 * 1024 * 1024)) ;;
        *) echo "$size" ;;
    esac
}

# Get total size of images directory
get_images_size() {
    if [[ -d "$SD_IMAGES_DIR" ]]; then
        du -sb "$SD_IMAGES_DIR" 2>/dev/null | cut -f1 || echo "0"
    else
        echo "0"
    fi
}

# Check if size limit is exceeded
check_size_limit() {
    [[ -z "$SIZE_LIMIT" ]] && return 1

    local current_size
    current_size=$(get_images_size)
    local limit_bytes
    limit_bytes=$(size_to_bytes "$SIZE_LIMIT")

    [[ $current_size -gt $limit_bytes ]]
}

# Find PNG files based on criteria
find_images() {
    local search_dir="$SD_IMAGES_DIR"

    # If job ID is specified, search only in that job's folder
    if [[ -n "$JOB_ID" ]]; then
        search_dir="$SD_IMAGES_DIR/job_$JOB_ID"
        if [[ ! -d "$search_dir" ]]; then
            log_warning "Job folder not found: $search_dir"
            return 0
        fi
    fi

    local find_cmd="find \"$search_dir\" -name \"*.png\" -type f"

    # Add age filter if specified
    if [[ -n "$OLDER_THAN" ]]; then
        find_cmd+=" -mtime +$OLDER_THAN"
    fi

    eval "$find_cmd" 2>/dev/null || true
}

# Create archive of images
create_archive() {
    local images
    mapfile -t images < <(find_images)

    [[ ${#images[@]} -eq 0 ]] && {
        log_info "No images found to archive"
        return 0
    }

    local timestamp
    timestamp=$(date +%Y%m%d_%H%M%S)
    local archive_name="sd_images_${timestamp}"

    # Create archive directory
    mkdir -p "$ARCHIVE_DIR"

    if [[ "$COMPRESS" == "true" ]]; then
        archive_name+=".tar.gz"
        local archive_path="$ARCHIVE_DIR/$archive_name"

        log_info "Creating compressed archive: $archive_path"
        if [[ "$DRY_RUN" == "false" ]]; then
            tar -czf "$archive_path" -C "$SD_IMAGES_DIR" . 2>/dev/null || {
                log_error "Failed to create archive"
                return 1
            }
            log_info "Archive created: $(du -h "$archive_path" | cut -f1)"
        fi
    else
        local archive_path="$ARCHIVE_DIR/$archive_name"
        log_info "Creating uncompressed archive: $archive_path"
        if [[ "$DRY_RUN" == "false" ]]; then
            cp -r "$SD_IMAGES_DIR" "$archive_path" 2>/dev/null || {
                log_error "Failed to create archive"
                return 1
            }
            log_info "Archive created: $(du -sh "$archive_path" | cut -f1)"
        fi
    fi
}

# Keep sample images (newest files)
keep_samples() {
    [[ $KEEP_SAMPLES -eq 0 ]] && return 0

    local images
    mapfile -t images < <(find_images | sort -t_ -k3 -r | head -n "$KEEP_SAMPLES")

    [[ ${#images[@]} -eq 0 ]] && return 0

    log_info "Keeping $KEEP_SAMPLES sample images:"
    printf '%s\n' "${images[@]}" | while read -r img; do
        log_debug "  Keeping: $(basename "$img")"
    done
}

# Remove images
remove_images() {
    local images_to_remove
    if [[ $KEEP_SAMPLES -gt 0 ]]; then
        # Keep newest N images, remove the rest
        mapfile -t images_to_remove < <(find_images | sort -t_ -k3 -r | tail -n +$((KEEP_SAMPLES + 1)))
    else
        # Remove all matching images
        mapfile -t images_to_remove < <(find_images)
    fi

    [[ ${#images_to_remove[@]} -eq 0 ]] && {
        log_info "No images to remove"
        return 0
    }

    log_info "Removing ${#images_to_remove[@]} images..."

    if [[ "$DRY_RUN" == "true" ]]; then
        printf '%s\n' "${images_to_remove[@]}" | while read -r img; do
            echo "  Would remove: $(basename "$img")"
        done
    else
        printf '%s\n' "${images_to_remove[@]}" | while read -r img; do
            log_debug "Removing: $(basename "$img")"
            rm -f "$img"
        done
        log_info "Removed ${#images_to_remove[@]} images"
    fi
}

# Show current status
show_status() {
    log_info "🎨 Stable Diffusion Image Status"
    echo "─────────────────────────────────────"

    if [[ ! -d "$SD_IMAGES_DIR" ]]; then
        log_info "Images directory doesn't exist: $SD_IMAGES_DIR"
        return 0
    fi

    local total_images
    total_images=$(find "$SD_IMAGES_DIR" -name "*.png" -type f 2>/dev/null | wc -l)

    local total_size_human
    total_size_human=$(du -sh "$SD_IMAGES_DIR" 2>/dev/null | cut -f1 || echo "0B")

    echo "📁 Directory: $SD_IMAGES_DIR"
    echo "🖼️  Images: $total_images PNG files"
    echo "💾 Size: $total_size_human"

    # Show job-specific folders if they exist
    local job_folders
    job_folders=$(find "$SD_IMAGES_DIR" -type d -name "job_*" 2>/dev/null | sort)
    if [[ -n "$job_folders" ]]; then
        echo "📁 Job-specific folders:"
        while IFS= read -r folder; do
            if [[ -n "$folder" ]]; then
                local job_id=$(basename "$folder" | sed 's/job_//')
                local job_images=$(find "$folder" -name "*.png" -type f 2>/dev/null | wc -l)
                local job_size=$(du -sh "$folder" 2>/dev/null | cut -f1 || echo "0B")
                echo "   📂 Job $job_id: $job_images images ($job_size)"
            fi
        done <<< "$job_folders"
    fi

    if [[ -n "$SIZE_LIMIT" ]]; then
        if check_size_limit; then
            echo "⚠️  Size limit exceeded: $SIZE_LIMIT"
        else
            echo "✅ Within size limit: $SIZE_LIMIT"
        fi
    fi

    # Show newest and oldest images
    if [[ $total_images -gt 0 ]]; then
        local newest oldest
        newest=$(find "$SD_IMAGES_DIR" -name "*.png" -type f -printf '%T@ %p\n' 2>/dev/null | sort -n | tail -1 | cut -d' ' -f2- | xargs basename 2>/dev/null || echo "unknown")
        oldest=$(find "$SD_IMAGES_DIR" -name "*.png" -type f -printf '%T@ %p\n' 2>/dev/null | sort -n | head -1 | cut -d' ' -f2- | xargs basename 2>/dev/null || echo "unknown")
        echo "🆕 Newest: $newest"
        echo "⏰ Oldest: $oldest"
    fi
    echo "─────────────────────────────────────"
}

# Confirm action
confirm_action() {
    [[ "$FORCE" == "true" ]] && return 0
    [[ "$DRY_RUN" == "true" ]] && return 0

    echo
    read -p "Proceed with image management? [y/N] " -n 1 -r
    echo
    [[ $REPLY =~ ^[Yy]$ ]]
}

# Main function
main() {
    parse_args "$@"

    log_info "🎨 Stable Diffusion Image Management"
    echo

    # Show current status
    show_status
    echo

    # Check if action is needed
    local action_needed=false

    if [[ -n "$SIZE_LIMIT" ]] && check_size_limit; then
        log_warning "Size limit ($SIZE_LIMIT) exceeded - cleanup recommended"
        action_needed=true
    fi

    if [[ -n "$OLDER_THAN" ]]; then
        local old_images
        old_images=$(find_images | wc -l)
        if [[ $old_images -gt 0 ]]; then
            log_warning "Found $old_images images older than $OLDER_THAN days"
            action_needed=true
        fi
    fi

    if [[ "$ARCHIVE" == "true" ]] || [[ $KEEP_SAMPLES -ge 0 ]]; then
        action_needed=true
    fi

    if [[ "$action_needed" == "false" ]]; then
        log_info "No action needed - all criteria met"
        exit 0
    fi

    # Confirm action
    if ! confirm_action; then
        log_info "Operation cancelled"
        exit 0
    fi

    # Create archive if requested
    if [[ "$ARCHIVE" == "true" ]]; then
        create_archive
    fi

    # Remove images (keeping samples if specified)
    remove_images

    # Show final status
    echo
    show_status

    log_info "✅ Image management completed"
}

# Run main function
main "$@"
