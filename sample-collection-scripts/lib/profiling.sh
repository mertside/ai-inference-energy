#!/bin/bash
#
# Profiling Library for AI Inference Energy Profiling
#
# This library provides profiling tool management, control script execution,
# and application running functionality for the energy profiling framework.
#
# Author: Mert Side
#

# Prevent multiple inclusions
if [[ "${PROFILING_LIB_LOADED:-}" == "true" ]]; then
    return 0
fi
readonly PROFILING_LIB_LOADED="true"

# Load dependencies
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
source "$(dirname "${BASH_SOURCE[0]}")/gpu_config.sh"

# =============================================================================
# Profiling Configuration Constants
# =============================================================================

readonly PROFILING_LIB_VERSION="1.0.0"

# Profiling tool configurations
declare -A PROFILING_TOOLS=(
    ["dcgmi"]="profile.py"
    ["nvidia-smi"]="profile_smi.py"
)

declare -A CONTROL_SCRIPTS=(
    ["dcgmi"]="control.sh"
    ["nvidia-smi"]="control_smi.sh"
)

# Default profiling parameters
readonly DEFAULT_PROFILING_INTERVAL=0.1  # seconds
readonly DEFAULT_PROFILE_TIMEOUT=300     # seconds
readonly DEFAULT_CONTROL_TIMEOUT=30      # seconds

# =============================================================================
# Profiling Tool Validation
# =============================================================================

# Check if profiling tool is available
is_profiling_tool_available() {
    local tool="$1"
    
    case "$(to_lower "$tool")" in
        dcgmi)
            command_exists "dcgmi" || return 1
            ;;
        nvidia-smi)
            command_exists "nvidia-smi" || return 1
            ;;
        *)
            log_error "Unknown profiling tool: $tool"
            return 1
            ;;
    esac
}

# Validate profiling tool and fallback if needed
validate_profiling_tool() {
    local requested_tool="$1"
    local validated_tool="$requested_tool"
    
    if ! is_valid_profiling_tool "$requested_tool"; then
        log_error "Invalid profiling tool: $requested_tool"
        return 1
    fi
    
    # Check if requested tool is available
    if is_profiling_tool_available "$requested_tool"; then
        log_debug "Profiling tool '$requested_tool' is available"
    else
        log_warning "Profiling tool '$requested_tool' not available"
        
        # Attempt fallback to nvidia-smi if dcgmi was requested
        if [[ "$(to_lower "$requested_tool")" == "dcgmi" ]]; then
            if is_profiling_tool_available "nvidia-smi"; then
                validated_tool="nvidia-smi"
                log_info "Falling back to nvidia-smi profiling"
            else
                log_error "No profiling tools available (dcgmi and nvidia-smi both unavailable)"
                return 1
            fi
        else
            log_error "Profiling tool '$requested_tool' not available and no fallback possible"
            return 1
        fi
    fi
    
    echo "$validated_tool"
    return 0
}

# Get profiling script for tool
get_profiling_script() {
    local tool="$1"
    local script="${PROFILING_TOOLS[$(to_lower "$tool")]:-}"
    
    if [[ -z "$script" ]]; then
        log_error "No profiling script configured for tool: $tool"
        return 1
    fi
    
    local script_path="${SCRIPTS_ROOT_DIR}/${script}"
    if ! file_readable "$script_path"; then
        log_error "Profiling script not found or not readable: $script_path"
        return 1
    fi
    
    echo "$script_path"
}

# Get control script for tool
get_control_script() {
    local tool="$1"
    local script="${CONTROL_SCRIPTS[$(to_lower "$tool")]:-}"
    
    if [[ -z "$script" ]]; then
        log_error "No control script configured for tool: $tool"
        return 1
    fi
    
    local script_path="${SCRIPTS_ROOT_DIR}/${script}"
    if ! file_readable "$script_path"; then
        log_error "Control script not found or not readable: $script_path"
        return 1
    fi
    
    echo "$script_path"
}

# =============================================================================
# GPU Frequency Control
# =============================================================================

# Set GPU frequency using appropriate control script
set_gpu_frequency() {
    local profiling_tool="$1"
    local memory_freq="$2"
    local core_freq="$3"
    local gpu_type="${4:-A100}"
    
    # Validate inputs
    if ! is_valid_profiling_tool "$profiling_tool"; then
        log_error "Invalid profiling tool: $profiling_tool"
        return 1
    fi
    
    if ! validate_frequency "$gpu_type" "$core_freq"; then
        return 1
    fi
    
    # Skip frequency control in baseline mode
    if [[ "${PROFILING_MODE:-dvfs}" == "baseline" ]]; then
        log_info "Baseline mode: Skipping frequency control (using default frequencies)"
        return 0
    fi
    
    # Get control script
    local control_script
    control_script=$(get_control_script "$profiling_tool")
    if [[ $? -ne 0 ]]; then
        return 1
    fi
    
    log_info "Setting GPU frequencies: memory=${memory_freq}MHz, core=${core_freq}MHz"
    log_debug "Using control script: $control_script"
    
    # Execute control script with timeout
    local start_time
    start_time=$(date +%s)
    
    if timeout "$DEFAULT_CONTROL_TIMEOUT" "$control_script" "$memory_freq" "$core_freq"; then
        local end_time duration
        end_time=$(date +%s)
        duration=$((end_time - start_time))
        log_success "GPU frequencies set successfully (took ${duration}s)"
        return 0
    else
        local exit_code=$?
        log_error "Failed to set GPU frequencies (exit code: $exit_code)"
        return 1
    fi
}

# Reset GPU frequencies to default
reset_gpu_frequency() {
    local profiling_tool="$1"
    local gpu_type="${2:-A100}"
    
    local memory_freq core_freq
    memory_freq=$(get_gpu_memory_freq "$gpu_type")
    core_freq=$(get_gpu_core_freq_max "$gpu_type")
    
    log_info "Resetting GPU frequencies to defaults"
    set_gpu_frequency "$profiling_tool" "$memory_freq" "$core_freq" "$gpu_type"
}

# =============================================================================
# Application Execution
# =============================================================================

# Resolve application path and validate environment
resolve_application_path() {
    local app_executable="$1"
    local base_dir="${SCRIPTS_ROOT_DIR}/.."
    local resolved_path=""
    local app_dir=""
    
    # Try different path resolutions
    local search_paths=(
        "$app_executable"                           # Absolute or relative path
        "${base_dir}/${app_executable}"            # Relative to project root
    )
    
    # Only add app directory patterns if app_executable doesn't contain path separators
    if [[ "$app_executable" != */* ]]; then
        search_paths+=(
            "${base_dir}/app-*/${app_executable}"      # In app directories
            "${base_dir}/app-*/${app_executable}.py"   # Python files in app directories
        )
    else
        # If it contains path separators, try adding .py extension
        search_paths+=("${app_executable}.py")
        search_paths+=("${base_dir}/${app_executable}.py")
    fi
    
    for path_pattern in "${search_paths[@]}"; do
        # Handle glob patterns
        if [[ "$path_pattern" == *"*"* ]]; then
            for expanded_path in $path_pattern; do
                if [[ -f "$expanded_path" && -x "$expanded_path" ]]; then
                    resolved_path="$expanded_path"
                    app_dir="$(dirname "$expanded_path")"
                    break 2
                fi
            done
        else
            if [[ -f "$path_pattern" && -x "$path_pattern" ]]; then
                resolved_path="$path_pattern"
                app_dir="$(dirname "$path_pattern")"
                break
            fi
        fi
    done
    
    if [[ -z "$resolved_path" ]]; then
        log_error "Application executable not found: $app_executable"
        log_error "Searched paths: ${search_paths[*]}"
        return 1
    fi
    
    # Convert to absolute path
    resolved_path="$(get_absolute_path "$resolved_path")"
    app_dir="$(get_absolute_path "$app_dir")"
    
    log_debug "Resolved application path: $resolved_path"
    log_debug "Application directory: $app_dir"
    
    # Export for use by calling functions
    echo "$resolved_path|$app_dir"
    return 0
}

# Create temporary Python script for robust argument handling
create_temp_python_script() {
    local conda_env="$1"
    local app_path="$2"
    local app_dir="$3"
    shift 3
    local app_args=("$@")
    
    local temp_script
    temp_script=$(mktemp "${TMPDIR:-/tmp}/run_app_XXXXXX.py")
    
    cat > "$temp_script" << 'EOF'
#!/usr/bin/env python3
import os
import sys
import subprocess
from pathlib import Path

def main():
    # Get arguments from command line
    if len(sys.argv) < 4:
        print("Usage: script.py <conda_env> <app_path> <app_dir> [app_args...]", file=sys.stderr)
        sys.exit(1)
    
    conda_env = sys.argv[1]
    app_path = sys.argv[2]
    app_dir = sys.argv[3]
    app_args = sys.argv[4:]  # Preserve all remaining arguments
    
    # Change to application directory
    os.chdir(app_dir)
    print(f"Changed directory to: {app_dir}", file=sys.stderr)
    
    # Find conda Python executable
    conda_python = None
    if 'CONDA_PREFIX' in os.environ:
        # If in a conda environment, use current Python
        conda_python = sys.executable
    else:
        # Try to find conda and activate environment
        import shutil
        conda_cmd = shutil.which('conda')
        if conda_cmd:
            # Get conda info to find environment path
            try:
                result = subprocess.run([conda_cmd, 'env', 'list'], 
                                      capture_output=True, text=True, timeout=10)
                if result.returncode == 0:
                    for line in result.stdout.splitlines():
                        if line.strip().startswith(conda_env + ' '):
                            env_path = line.split()[-1]
                            conda_python = os.path.join(env_path, 'bin', 'python')
                            break
            except (subprocess.TimeoutExpired, subprocess.SubprocessError):
                pass
    
    # Fallback to system python
    if not conda_python or not os.path.exists(conda_python):
        conda_python = sys.executable
    
    print(f"Using Python: {conda_python}", file=sys.stderr)
    print(f"Running: {app_path} with args: {app_args}", file=sys.stderr)
    
    # Execute the application
    cmd = [conda_python, app_path] + app_args
    try:
        result = subprocess.run(cmd, timeout=300)  # 5 minute timeout
        sys.exit(result.returncode)
    except subprocess.TimeoutExpired:
        print("Application timed out after 5 minutes", file=sys.stderr)
        sys.exit(124)
    except Exception as e:
        print(f"Error running application: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
EOF
    
    chmod +x "$temp_script"
    
    # Add cleanup for temp script
    register_cleanup "rm -f '$temp_script'"
    
    echo "$temp_script"
}

# Run application with profiling
run_application_with_profiling() {
    local profiling_tool="$1"
    local app_executable="$2"
    local output_dir="$3"
    local run_id="$4"
    local frequency="$5"
    shift 5
    local app_params=("$@")
    
    # Resolve application path
    local resolution_result app_path app_dir
    resolution_result=$(resolve_application_path "$app_executable")
    if [[ $? -ne 0 ]]; then
        return 1
    fi
    
    IFS='|' read -r app_path app_dir <<< "$resolution_result"
    
    # Determine conda environment
    local conda_env
    conda_env=$(determine_conda_env "$app_executable" "$app_dir")
    
    # Get profiling script
    local profiling_script
    profiling_script=$(get_profiling_script "$profiling_tool")
    if [[ $? -ne 0 ]]; then
        return 1
    fi
    
    # Create output files
    ensure_directory "$output_dir"
    local output_prefix="${output_dir}/run_${run_id}_freq_${frequency}"
    local app_output="${output_prefix}_app.out"
    local app_error="${output_prefix}_app.err"
    local profile_output="${output_prefix}_profile.csv"
    
    log_info "Running application: $(basename "$app_executable") (run $run_id, frequency ${frequency}MHz)"
    log_debug "Application path: $app_path"
    log_debug "Application directory: $app_dir"
    log_debug "Conda environment: $conda_env"
    log_debug "Output files: $output_prefix.*"
    
    # Create temporary Python script for robust execution
    local temp_script
    temp_script=$(create_temp_python_script "$conda_env" "$app_path" "$app_dir" "${app_params[@]}")
    
    # Build profiling command - profile.py expects: profile.py [options] [command ...]
    # Use -- to separate profile.py options from command arguments
    local profile_cmd=(
        "$profiling_script"
        --output "$profile_output"
        --
        # Command and all arguments after -- are treated as positional
        "python3" "$temp_script" "$conda_env" "$app_path" "$app_dir"
    )
    
    # Add application parameters after the -- separator
    for param in "${app_params[@]}"; do
        profile_cmd+=("$param")
    done
    
    log_debug "Profiling command: ${profile_cmd[*]}"
    
    # Create timing log file if it doesn't exist
    local timing_file="${output_dir}/timing_summary.log"
    if [[ ! -f "$timing_file" ]]; then
        echo "# Run Timing Summary" > "$timing_file"
        echo "# Format: run_id,frequency_mhz,duration_seconds,exit_code,status" >> "$timing_file"
    fi
    
    # Execute profiling with timeout
    local start_time end_time duration exit_code=0
    start_time=$(date +%s)
    
    if timeout "$DEFAULT_PROFILE_TIMEOUT" "${profile_cmd[@]}" > "$app_output" 2> "$app_error"; then
        end_time=$(date +%s)
        duration=$((end_time - start_time))
        log_success "Application completed successfully (took ${duration}s)"
        
        # Record timing information
        echo "${run_id},${frequency},${duration},0,success" >> "$timing_file"
    else
        exit_code=$?
        end_time=$(date +%s)
        duration=$((end_time - start_time))
        
        if [[ $exit_code -eq 124 ]]; then
            log_error "Application timed out after ${DEFAULT_PROFILE_TIMEOUT}s"
            echo "${run_id},${frequency},${duration},${exit_code},timeout" >> "$timing_file"
        else
            log_error "Application failed (exit code: $exit_code, duration: ${duration}s)"
            echo "${run_id},${frequency},${duration},${exit_code},failed" >> "$timing_file"
        fi
        
        # Log error details
        if [[ -s "$app_error" ]]; then
            log_debug "Application errors:"
            head -n 10 "$app_error" | while read -r line; do
                log_debug "  $line"
            done
        fi
    fi
    
    # Validate output files
    local success=true
    if [[ ! -f "$profile_output" || ! -s "$profile_output" ]]; then
        log_warning "Profile output file missing or empty: $profile_output"
        success=false
    fi
    
    if [[ ! -f "$app_output" ]]; then
        log_warning "Application output file missing: $app_output"
        success=false
    fi
    
    if $success; then
        log_debug "All output files created successfully"
        # Log application output size for verification
        local app_size profile_size
        app_size=$(stat -c%s "$app_output" 2>/dev/null || echo "0")
        profile_size=$(stat -c%s "$profile_output" 2>/dev/null || echo "0")
        log_debug "Output sizes: app=${app_size}B, profile=${profile_size}B"
    fi
    
    return $exit_code
}

# =============================================================================
# Experiment Orchestration
# =============================================================================

# Run DVFS experiment (full frequency sweep)
run_dvfs_experiment() {
    local profiling_tool="$1"
    local gpu_type="$2"
    local app_executable="$3"
    local output_dir="$4"
    local num_runs="$5"
    local sleep_interval="${6:-1}"
    shift 6
    local app_params=("$@")
    
    log_info "Starting DVFS experiment"
    log_info "GPU: $gpu_type, Tool: $profiling_tool, Runs: $num_runs"
    
    # Get frequency range
    local frequencies memory_freq
    frequencies=($(generate_frequency_range "$gpu_type"))
    memory_freq=$(get_gpu_memory_freq "$gpu_type")
    
    local total_runs=$((${#frequencies[@]} * num_runs))
    local current_run=0
    
    log_info "Testing ${#frequencies[@]} frequencies with $num_runs runs each ($total_runs total runs)"
    
    # Iterate through frequencies
    for frequency in "${frequencies[@]}"; do
        log_info "Testing frequency: ${frequency}MHz (${current_run}/${total_runs} runs completed)"
        
        # Set GPU frequency
        if ! set_gpu_frequency "$profiling_tool" "$memory_freq" "$frequency" "$gpu_type"; then
            log_error "Failed to set frequency ${frequency}MHz, skipping..."
            continue
        fi
        
        # Run multiple times at this frequency
        for ((run = 1; run <= num_runs; run++)); do
            ((current_run++))
            show_progress "$current_run" "$total_runs" "DVFS Progress"
            
            local run_id="${current_run}_$(printf "%02d" "$run")"
            
            if ! run_application_with_profiling "$profiling_tool" "$app_executable" \
                "$output_dir" "$run_id" "$frequency" "${app_params[@]}"; then
                log_warning "Run $run_id failed at frequency ${frequency}MHz"
            fi
            
            # Sleep between runs
            if [[ $run -lt $num_runs ]] && [[ $sleep_interval -gt 0 ]]; then
                log_debug "Sleeping ${sleep_interval}s between runs..."
                sleep "$sleep_interval"
            fi
        done
    done
    
    # Reset to default frequency
    reset_gpu_frequency "$profiling_tool" "$gpu_type"
    
    log_success "DVFS experiment completed ($total_runs runs)"
}

# Run baseline experiment (single frequency)
run_baseline_experiment() {
    local profiling_tool="$1"
    local gpu_type="$2"
    local app_executable="$3"
    local output_dir="$4"
    local num_runs="$5"
    local sleep_interval="${6:-1}"
    shift 6
    local app_params=("$@")
    
    log_info "Starting baseline experiment"
    log_info "GPU: $gpu_type, Tool: $profiling_tool, Runs: $num_runs"
    
    # Use default (maximum) frequency
    local frequency
    frequency=$(get_gpu_core_freq_max "$gpu_type")
    
    log_info "Using default frequency: ${frequency}MHz"
    
    # Run multiple times at default frequency
    for ((run = 1; run <= num_runs; run++)); do
        show_progress "$run" "$num_runs" "Baseline Progress"
        
        local run_id="baseline_$(printf "%02d" "$run")"
        
        if ! run_application_with_profiling "$profiling_tool" "$app_executable" \
            "$output_dir" "$run_id" "$frequency" "${app_params[@]}"; then
            log_warning "Baseline run $run failed"
        fi
        
        # Sleep between runs
        if [[ $run -lt $num_runs ]] && [[ $sleep_interval -gt 0 ]]; then
            log_debug "Sleeping ${sleep_interval}s between runs..."
            sleep "$sleep_interval"
        fi
    done
    
    log_success "Baseline experiment completed ($num_runs runs)"
}

# Export functions for use in other scripts
export -f is_profiling_tool_available validate_profiling_tool
export -f get_profiling_script get_control_script
export -f set_gpu_frequency reset_gpu_frequency
export -f resolve_application_path run_application_with_profiling
export -f run_dvfs_experiment run_baseline_experiment

log_debug "Profiling library v${PROFILING_LIB_VERSION} loaded successfully"
