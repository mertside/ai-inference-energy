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
readonly DEFAULT_PROFILE_TIMEOUT=1800    # seconds (30 minutes for low-frequency research)  
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
    
    # Method 1: If already in conda environment, use current Python
    if 'CONDA_PREFIX' in os.environ and conda_env in os.environ.get('CONDA_DEFAULT_ENV', ''):
        conda_python = sys.executable
        print(f"Using current conda environment Python: {conda_python}", file=sys.stderr)
    else:
        # Method 2: Try to find conda environment path directly
        import shutil
        conda_cmd = shutil.which('conda')
        if conda_cmd:
            try:
                # Use shorter timeout and more robust parsing
                result = subprocess.run([conda_cmd, 'info', '--envs'], 
                                      capture_output=True, text=True, timeout=5)
                if result.returncode == 0:
                    for line in result.stdout.splitlines():
                        # Look for environment name at start of line
                        parts = line.strip().split()
                        if len(parts) >= 2 and parts[0] == conda_env:
                            env_path = parts[-1]  # Last part is the path
                            potential_python = os.path.join(env_path, 'bin', 'python')
                            if os.path.exists(potential_python):
                                conda_python = potential_python
                                print(f"Found conda environment Python: {conda_python}", file=sys.stderr)
                                break
            except (subprocess.TimeoutExpired, subprocess.SubprocessError, Exception) as e:
                print(f"Conda detection failed: {e}, using fallback", file=sys.stderr)
    
    # Fallback to system python if conda detection failed
    if not conda_python or not os.path.exists(conda_python):
        conda_python = sys.executable
        print(f"Using fallback system Python: {conda_python}", file=sys.stderr)
    
    print(f"Final Python executable: {conda_python}", file=sys.stderr)
    print(f"Application path: {app_path}", file=sys.stderr)
    print(f"Application directory: {app_dir}", file=sys.stderr)
    print(f"Application arguments: {app_args}", file=sys.stderr)
    
    # Verify application file exists
    if not os.path.exists(app_path):
        print(f"ERROR: Application file not found: {app_path}", file=sys.stderr)
        sys.exit(1)
        
    # Verify application is executable
    if not os.access(app_path, os.X_OK):
        print(f"ERROR: Application file not executable: {app_path}", file=sys.stderr)
        sys.exit(1)
    
    # Execute the application
    cmd = [conda_python, app_path] + app_args
    try:
        result = subprocess.run(cmd, timeout=1800)  # 30 minute timeout for low-frequency research
        sys.exit(result.returncode)
    except subprocess.TimeoutExpired:
        print("Application timed out after 30 minutes", file=sys.stderr)
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
    
    log_debug "Starting run_application_with_profiling: tool=$profiling_tool, app=$app_executable, run_id=$run_id, freq=${frequency}MHz"
    
    # Resolve application path
    local resolution_result app_path app_dir
    log_debug "Resolving application path for: $app_executable"
    resolution_result=$(resolve_application_path "$app_executable")
    if [[ $? -ne 0 ]]; then
        log_error "Failed to resolve application path for: $app_executable"
        return 1
    fi
    
    IFS='|' read -r app_path app_dir <<< "$resolution_result"
    log_debug "Resolved paths - app_path: $app_path, app_dir: $app_dir"
    
    # Determine conda environment
    local conda_env
    log_debug "Determining conda environment for: $app_executable"
    conda_env=$(determine_conda_env "$app_executable" "$app_dir")
    log_debug "Selected conda environment: $conda_env"
    
    # Get profiling script
    local profiling_script
    log_debug "Getting profiling script for tool: $profiling_tool"
    profiling_script=$(get_profiling_script "$profiling_tool")
    if [[ $? -ne 0 ]]; then
        log_error "Failed to get profiling script for tool: $profiling_tool"
        return 1
    fi
    log_debug "Profiling script path: $profiling_script"
    
    # Create output files
    log_debug "Setting up output directory: $output_dir"
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
    log_debug "Creating temporary Python script..."
    temp_script=$(create_temp_python_script "$conda_env" "$app_path" "$app_dir" "${app_params[@]}")
    if [[ $? -ne 0 || -z "$temp_script" ]]; then
        log_error "Failed to create temporary Python script"
        return 1
    fi
    log_debug "Temporary script created: $temp_script"
    
    # Build profiling command - profile.py expects: profile.py [options] [command ...]
    # Use -- to separate profile.py options from command arguments
    local profile_cmd=(
        "$profiling_script"
        --output "$profile_output"
        --interval "${PARSED_SAMPLING_INTERVAL:-50}"
    )
    
    # Add all-gpus flag if requested
    if [[ "${PARSED_ALL_GPUS:-false}" == "true" ]]; then
        profile_cmd+=(--all-gpus)
    fi
    
    # Add command separator and actual command
    profile_cmd+=(
        --
        # Command and all arguments after -- are treated as positional
        "python3" "$temp_script" "$conda_env" "$app_path" "$app_dir"
    )
    
    # Add application parameters after the -- separator
    for param in "${app_params[@]}"; do
        profile_cmd+=("$param")
    done
    
    log_debug "Profiling command: ${profile_cmd[*]}"
    
    # Verify profiling script exists and is executable
    if [[ ! -f "$profiling_script" ]]; then
        log_error "Profiling script not found: $profiling_script"
        return 1
    fi
    
    if [[ ! -x "$profiling_script" ]]; then
        log_error "Profiling script not executable: $profiling_script"
        return 1
    fi
    
    # Verify temporary script exists
    if [[ ! -f "$temp_script" ]]; then
        log_error "Temporary script not found: $temp_script"
        return 1
    fi
    
    # Create timing log file if it doesn't exist
    local timing_file="${output_dir}/timing_summary.log"
    if [[ ! -f "$timing_file" ]]; then
        echo "# Run Timing Summary" > "$timing_file"
        echo "# Format: run_id,frequency_mhz,duration_seconds,exit_code,status" >> "$timing_file"
    fi
    
    # Execute profiling with timeout
    local start_time end_time duration exit_code=0
    start_time=$(date +%s)
    
    log_debug "Starting profiling command execution..."
    log_debug "Command: ${profile_cmd[*]}"
    log_debug "App output: $app_output"
    log_debug "App error: $app_error"
    log_debug "Profile output: $profile_output"
    
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
        
        log_error "Profiling command failed with exit code: $exit_code"
        log_error "Command was: ${profile_cmd[*]}"
        
        # Show last few lines of error output for debugging
        if [[ -f "$app_error" && -s "$app_error" ]]; then
            log_error "Last few lines of application error output:"
            tail -5 "$app_error" | while IFS= read -r line; do
                log_error "  $line"
            done
        fi
        
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
        log_debug "Setting GPU frequency to ${frequency}MHz..."
        if ! set_gpu_frequency "$profiling_tool" "$memory_freq" "$frequency" "$gpu_type"; then
            log_error "Failed to set frequency ${frequency}MHz, skipping..."
            continue
        fi
        log_debug "GPU frequency set successfully to ${frequency}MHz"
        
        # Run multiple times at this frequency
        log_debug "Starting inner loop: run=1 to num_runs=$num_runs"
        for ((run = 1; run <= num_runs; run++)); do
            log_debug "Inner loop iteration: run=$run, num_runs=$num_runs"
            current_run=$((current_run + 1))
            log_debug "Incremented current_run to $current_run"
            show_progress "$current_run" "$total_runs" "DVFS Progress"
            log_debug "show_progress completed successfully"
            
            local run_id="${current_run}_$(printf "%02d" "$run")"
            log_debug "Starting run $run_id at frequency ${frequency}MHz"
            
            # Use || true to prevent DVFS from failing on single application failures
            if ! run_application_with_profiling "$profiling_tool" "$app_executable" \
                "$output_dir" "$run_id" "$frequency" "${app_params[@]}"; then
                log_warning "Run $run_id failed at frequency ${frequency}MHz - continuing with next run"
                # Don't exit the entire DVFS experiment for a single failed run
            else
                log_debug "Run $run_id completed successfully at frequency ${frequency}MHz"
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

# Run custom frequency experiment (selected frequencies)
run_custom_experiment() {
    local profiling_tool="$1"
    local gpu_type="$2"
    local app_executable="$3"
    local output_dir="$4"
    local num_runs="$5"
    local sleep_interval="$6"
    local custom_frequencies="$7"
    shift 7
    local app_params=("$@")
    
    log_info "Starting custom frequency experiment"
    log_info "Custom frequencies: $custom_frequencies"
    
    # Parse frequencies into array
    local freq_array
    IFS=',' read -ra freq_array <<< "$custom_frequencies"
    
    local total_frequencies=${#freq_array[@]}
    local total_runs=$((total_frequencies * num_runs))
    local current_run=0
    
    log_info "Testing $total_frequencies frequencies with $num_runs runs each ($total_runs total runs)"
    
    # Iterate through custom frequencies
    for frequency in "${freq_array[@]}"; do
        # Remove whitespace
        frequency=$(echo "$frequency" | tr -d ' ')
        log_info "Testing frequency: ${frequency} MHz"
        
        # Set GPU frequency
        local memory_freq
        memory_freq=$(get_gpu_memory_freq "$gpu_type")
        if ! set_gpu_frequency "$profiling_tool" "$memory_freq" "$frequency" "$gpu_type"; then
            log_warning "Failed to set GPU frequency to $frequency MHz, skipping"
            continue
        fi
        
        # Run multiple tests at this frequency
        for ((run=1; run<=num_runs; run++)); do
            current_run=$((current_run + 1))
            show_progress "$current_run" "$total_runs" "Custom Frequency Progress"
            
            local run_id="$(printf "%02d" "$run")"
            
            if ! run_application_with_profiling \
                "$profiling_tool" "$app_executable" \
                "$output_dir" "$run_id" "$frequency" "${app_params[@]}"; then
                log_warning "Custom frequency run $current_run failed (freq: $frequency MHz, run: $run)"
            fi
            
            # Sleep between runs if configured
            if [[ "$sleep_interval" -gt 0 ]]; then
                sleep "$sleep_interval"
            fi
        done
    done
    
    # Reset GPU frequency
    reset_gpu_frequency "$profiling_tool" "$gpu_type"
    
    log_success "Custom frequency experiment completed ($total_runs runs across $total_frequencies frequencies)"
}

# Export functions for use in other scripts
export -f is_profiling_tool_available validate_profiling_tool
export -f get_profiling_script get_control_script
export -f set_gpu_frequency reset_gpu_frequency
export -f resolve_application_path run_application_with_profiling
export -f run_dvfs_experiment run_baseline_experiment run_custom_experiment

log_debug "Profiling library v${PROFILING_LIB_VERSION} loaded successfully"
