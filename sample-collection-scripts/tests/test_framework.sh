#!/bin/bash
#
# Test Suite for AI Inference Energy Profiling Framework v2.0
#
# This script runs comprehensive tests to validate the refactored framework
# functionality, ensuring all components work correctly before migration.
#
# Author: Mert Side
#

set -euo pipefail

# Get script directory
readonly TEST_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly SCRIPTS_DIR="$(dirname "$TEST_DIR")"
readonly LIB_DIR="${SCRIPTS_DIR}/lib"

# Test configuration
readonly TEST_OUTPUT_DIR="${TEST_DIR}/test_results"
readonly TEST_LOG="${TEST_OUTPUT_DIR}/test_suite.log"

# Test counters
TESTS_PASSED=0
TESTS_FAILED=0
TESTS_TOTAL=0

# Colors for output (same logic as common.sh)
if [[ -t 1 ]] && [[ "${TERM:-}" != "dumb" ]] && [[ "${NO_COLOR:-}" != "1" ]] && [[ "${DISABLE_COLORS:-}" != "1" ]]; then
    readonly COLOR_RED='\033[0;31m'
    readonly COLOR_GREEN='\033[0;32m'
    readonly COLOR_YELLOW='\033[1;33m'
    readonly COLOR_BLUE='\033[0;34m'
    readonly COLOR_NC='\033[0m'
else
    readonly COLOR_RED=''
    readonly COLOR_GREEN=''
    readonly COLOR_YELLOW=''
    readonly COLOR_BLUE=''
    readonly COLOR_NC=''
fi

# Logging functions
log_test() {
    echo -e "[$(date '+%H:%M:%S')] [${COLOR_BLUE}TEST${COLOR_NC}] $*" | tee -a "$TEST_LOG"
}

log_pass() {
    echo -e "[$(date '+%H:%M:%S')] [${COLOR_GREEN}PASS${COLOR_NC}] $*" | tee -a "$TEST_LOG"
    ((TESTS_PASSED++))
}

log_fail() {
    echo -e "[$(date '+%H:%M:%S')] [${COLOR_RED}FAIL${COLOR_NC}] $*" | tee -a "$TEST_LOG"
    ((TESTS_FAILED++))
}

log_info() {
    echo -e "[$(date '+%H:%M:%S')] [${COLOR_YELLOW}INFO${COLOR_NC}] $*" | tee -a "$TEST_LOG"
}

# Test execution wrapper
run_test() {
    local test_name="$1"
    local test_function="$2"
    
    ((TESTS_TOTAL++))
    log_test "Running: $test_name"
    
    if $test_function 2>&1; then
        log_pass "$test_name"
        return 0
    else
        local exit_code=$?
        log_fail "$test_name (exit code: $exit_code)"
        return 1
    fi
}

# =============================================================================
# Library Tests
# =============================================================================

test_common_library() {
    log_info "Testing common library..."
    
    # Test library loading
    if ! source "${LIB_DIR}/common.sh"; then
        echo "Failed to load common.sh"
        return 1
    fi
    
    # Test logging functions
    if ! command -v log_info >/dev/null 2>&1; then
        echo "log_info function not available"
        return 1
    fi
    
    # Test utility functions
    if ! command -v trim >/dev/null 2>&1; then
        echo "trim function not available"
        return 1
    fi
    
    # Test validation functions
    if ! is_positive_integer "123"; then
        echo "is_positive_integer validation failed"
        return 1
    fi
    
    if is_positive_integer "-123"; then
        echo "is_positive_integer should reject negative numbers"
        return 1
    fi
    
    echo "Common library tests passed"
    return 0
}

test_gpu_config_library() {
    log_info "Testing GPU config library..."
    
    # Test library loading (depends on common.sh)
    if ! source "${LIB_DIR}/common.sh"; then
        echo "Failed to load common.sh dependency"
        return 1
    fi
    
    if ! source "${LIB_DIR}/gpu_config.sh"; then
        echo "Failed to load gpu_config.sh"
        return 1
    fi
    
    # Test GPU type validation
    if ! is_valid_gpu_type "A100"; then
        echo "A100 should be valid GPU type"
        return 1
    fi
    
    if is_valid_gpu_type "INVALID"; then
        echo "INVALID should not be valid GPU type"
        return 1
    fi
    
    # Test GPU configuration functions
    local arch
    arch=$(get_gpu_architecture "A100")
    if [[ "$arch" != "GA100" ]]; then
        echo "Expected GA100 for A100, got: $arch"
        return 1
    fi
    
    # Test frequency generation
    local frequencies
    frequencies=($(generate_frequency_range "A100"))
    if [[ ${#frequencies[@]} -eq 0 ]]; then
        echo "No frequencies generated for A100"
        return 1
    fi
    
    echo "GPU config library tests passed"
    return 0
}

test_profiling_library() {
    log_info "Testing profiling library..."
    
    # Load dependencies
    if ! source "${LIB_DIR}/common.sh"; then
        echo "Failed to load common.sh dependency"
        return 1
    fi
    
    if ! source "${LIB_DIR}/gpu_config.sh"; then
        echo "Failed to load gpu_config.sh dependency"
        return 1
    fi
    
    if ! source "${LIB_DIR}/profiling.sh"; then
        echo "Failed to load profiling.sh"
        return 1
    fi
    
    # Test profiling tool validation
    if ! is_valid_profiling_tool "dcgmi"; then
        echo "dcgmi should be valid profiling tool"
        return 1
    fi
    
    if is_valid_profiling_tool "invalid"; then
        echo "invalid should not be valid profiling tool"
        return 1
    fi
    
    # Test application path resolution (should handle non-existent files gracefully)
    if resolve_application_path "nonexistent_app.py" >/dev/null 2>&1; then
        echo "Should fail to resolve non-existent application"
        return 1
    fi
    
    echo "Profiling library tests passed"
    return 0
}

test_args_parser_library() {
    log_info "Testing args parser library..."
    
    # Load dependencies
    if ! source "${LIB_DIR}/common.sh"; then
        echo "Failed to load common.sh dependency"
        return 1
    fi
    
    if ! source "${LIB_DIR}/gpu_config.sh"; then
        echo "Failed to load gpu_config.sh dependency"
        return 1
    fi
    
    # Load configuration
    if ! source "${SCRIPTS_DIR}/config/defaults.sh"; then
        echo "Failed to load defaults.sh dependency"
        return 1
    fi
    
    if ! source "${LIB_DIR}/args_parser.sh"; then
        echo "Failed to load args_parser.sh"
        return 1
    fi
    
    # Test validation functions
    if ! is_valid_gpu_type "V100"; then
        echo "V100 should be valid GPU type"
        return 1
    fi
    
    if ! is_valid_profiling_mode "dvfs"; then
        echo "dvfs should be valid profiling mode"
        return 1
    fi
    
    if ! is_valid_profiling_mode "baseline"; then
        echo "baseline should be valid profiling mode"
        return 1
    fi
    
    echo "Args parser library tests passed"
    return 0
}

# =============================================================================
# Integration Tests
# =============================================================================

test_launch_script_help() {
    log_info "Testing launch script help system..."
    
    cd "$SCRIPTS_DIR"
    
    # Test help flag
    if ! ./launch_v2.sh --help >/dev/null 2>&1; then
        echo "Help command failed"
        return 1
    fi
    
    # Test version flag
    if ! ./launch_v2.sh --version >/dev/null 2>&1; then
        echo "Version command failed"
        return 1
    fi
    
    echo "Launch script help tests passed"
    return 0
}

test_launch_script_validation() {
    log_info "Testing launch script argument validation..."
    
    cd "$SCRIPTS_DIR"
    
    # Test invalid GPU type (should fail)
    if ./launch_v2.sh --gpu-type INVALID >/dev/null 2>&1; then
        echo "Should reject invalid GPU type"
        return 1
    fi
    
    # Test invalid profiling tool (should fail)
    if ./launch_v2.sh --profiling-tool invalid >/dev/null 2>&1; then
        echo "Should reject invalid profiling tool"
        return 1
    fi
    
    # Test invalid number of runs (should fail)
    if ./launch_v2.sh --num-runs -1 >/dev/null 2>&1; then
        echo "Should reject negative number of runs"
        return 1
    fi
    
    echo "Launch script validation tests passed"
    return 0
}

test_configuration_loading() {
    log_info "Testing configuration loading..."
    
    # Test defaults loading
    if ! source "${SCRIPTS_DIR}/config/defaults.sh"; then
        echo "Failed to load default configuration"
        return 1
    fi
    
    # Test that key variables are set
    if [[ -z "$DEFAULT_GPU_TYPE" ]]; then
        echo "DEFAULT_GPU_TYPE not set"
        return 1
    fi
    
    if [[ -z "$DEFAULT_PROFILING_TOOL" ]]; then
        echo "DEFAULT_PROFILING_TOOL not set"
        return 1
    fi
    
    echo "Configuration loading tests passed"
    return 0
}

# =============================================================================
# File Structure Tests
# =============================================================================

test_file_structure() {
    log_info "Testing file structure..."
    
    # Check required directories
    local required_dirs=("lib" "config")
    for dir in "${required_dirs[@]}"; do
        if [[ ! -d "${SCRIPTS_DIR}/${dir}" ]]; then
            echo "Required directory missing: $dir"
            return 1
        fi
    done
    
    # Check required library files
    local required_libs=("common.sh" "gpu_config.sh" "profiling.sh" "args_parser.sh")
    for lib in "${required_libs[@]}"; do
        if [[ ! -f "${LIB_DIR}/${lib}" ]]; then
            echo "Required library missing: $lib"
            return 1
        fi
        
        if [[ ! -r "${LIB_DIR}/${lib}" ]]; then
            echo "Library not readable: $lib"
            return 1
        fi
    done
    
    # Check main script
    if [[ ! -f "${SCRIPTS_DIR}/launch_v2.sh" ]]; then
        echo "Main script missing: launch_v2.sh"
        return 1
    fi
    
    if [[ ! -x "${SCRIPTS_DIR}/launch_v2.sh" ]]; then
        echo "Main script not executable: launch_v2.sh"
        return 1
    fi
    
    echo "File structure tests passed"
    return 0
}

# =============================================================================
# Performance Tests
# =============================================================================

test_library_load_performance() {
    log_info "Testing library loading performance..."
    
    local start_time end_time duration
    start_time=$(date +%s.%N)
    
    # Load all libraries
    source "${LIB_DIR}/common.sh" >/dev/null 2>&1
    source "${LIB_DIR}/gpu_config.sh" >/dev/null 2>&1
    source "${SCRIPTS_DIR}/config/defaults.sh" >/dev/null 2>&1
    source "${LIB_DIR}/profiling.sh" >/dev/null 2>&1
    source "${LIB_DIR}/args_parser.sh" >/dev/null 2>&1
    
    end_time=$(date +%s.%N)
    duration=$(echo "$end_time - $start_time" | bc -l 2>/dev/null || echo "0.1")
    
    # Should load in reasonable time (< 1 second)
    if (( $(echo "$duration > 1.0" | bc -l 2>/dev/null || echo "0") )); then
        echo "Library loading too slow: ${duration}s"
        return 1
    fi
    
    echo "Library loading performance: ${duration}s"
    return 0
}

# =============================================================================
# Main Test Execution
# =============================================================================

main() {
    log_info "Starting AI Inference Energy Profiling Framework Test Suite"
    log_info "Framework Version: 2.0.0"
    log_info "Test Suite Started: $(date)"
    
    # Create test output directory
    mkdir -p "$TEST_OUTPUT_DIR"
    
    # Initialize log file
    echo "AI Inference Energy Profiling Framework Test Suite" > "$TEST_LOG"
    echo "Started: $(date)" >> "$TEST_LOG"
    echo "======================================================" >> "$TEST_LOG"
    
    # Run tests
    echo -e "\n${COLOR_BLUE}=== File Structure Tests ===${COLOR_NC}"
    run_test "File Structure Validation" test_file_structure
    
    echo -e "\n${COLOR_BLUE}=== Library Tests ===${COLOR_NC}"
    run_test "Common Library" test_common_library
    run_test "GPU Config Library" test_gpu_config_library
    run_test "Profiling Library" test_profiling_library
    run_test "Args Parser Library" test_args_parser_library
    
    echo -e "\n${COLOR_BLUE}=== Configuration Tests ===${COLOR_NC}"
    run_test "Configuration Loading" test_configuration_loading
    
    echo -e "\n${COLOR_BLUE}=== Integration Tests ===${COLOR_NC}"
    run_test "Launch Script Help System" test_launch_script_help
    run_test "Launch Script Validation" test_launch_script_validation
    
    echo -e "\n${COLOR_BLUE}=== Performance Tests ===${COLOR_NC}"
    run_test "Library Load Performance" test_library_load_performance
    
    # Summary
    echo -e "\n${COLOR_BLUE}=== Test Summary ===${COLOR_NC}"
    echo -e "Total Tests: $TESTS_TOTAL"
    echo -e "Passed: ${COLOR_GREEN}$TESTS_PASSED${COLOR_NC}"
    echo -e "Failed: ${COLOR_RED}$TESTS_FAILED${COLOR_NC}"
    echo -e "Success Rate: $(( TESTS_PASSED * 100 / TESTS_TOTAL ))%"
    
    # Write summary to log
    {
        echo "======================================================"
        echo "Test Summary:"
        echo "  Total: $TESTS_TOTAL"
        echo "  Passed: $TESTS_PASSED"
        echo "  Failed: $TESTS_FAILED"
        echo "  Success Rate: $(( TESTS_PASSED * 100 / TESTS_TOTAL ))%"
        echo "Completed: $(date)"
    } >> "$TEST_LOG"
    
    log_info "Test results saved to: $TEST_LOG"
    
    # Exit with appropriate code
    if [[ $TESTS_FAILED -eq 0 ]]; then
        log_info "All tests passed! Framework is ready for use."
        return 0
    else
        log_fail "$TESTS_FAILED test(s) failed. Please review and fix issues."
        return 1
    fi
}

# Create test directory and run tests
mkdir -p "$(dirname "$TEST_DIR")/tests"
cd "$(dirname "$TEST_DIR")/tests"

# Only run main if script is executed directly
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi
