#!/bin/bash
#
# Example Job Submission Helper for A100
#
# This script helps you quickly submit different types of profiling jobs
# on A100 GPUs with the correct configuration.
#

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Print colored output
print_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to display usage
usage() {
    cat << EOF
Usage: $0 [OPTION]

Submit example profiling jobs for A100 GPUs on HPCC

Options:
    lstm-baseline        Submit LSTM baseline profiling job
    lstm-comprehensive   Submit LSTM comprehensive profiling job  
    lstm-custom          Submit LSTM custom frequency profiling job
    sd-baseline          Submit Stable Diffusion baseline profiling job
    custom               Submit custom application profiling job (template)
    list                 List all available example scripts
    status               Check status of your running jobs
    help                 Show this help message

Examples:
    $0 lstm-baseline     # Quick LSTM test (~20 minutes)
    $0 lstm-comprehensive # Full LSTM analysis (~4 hours)
    $0 status            # Check job status

EOF
}

# Function to check if we're in the right directory
check_directory() {
    if [[ ! -f "submit_lstm_a100_baseline.sh" ]]; then
        print_error "Example scripts not found in current directory"
        print_info "Please run this script from the examples/ directory"
        print_info "Current directory: $(pwd)"
        exit 1
    fi
}

# Function to validate script before submission
validate_script() {
    local script="$1"
    
    if [[ ! -f "$script" ]]; then
        print_error "Script not found: $script"
        return 1
    fi
    
    if [[ ! -x "$script" ]]; then
        print_warning "Script not executable, making it executable..."
        chmod +x "$script"
    fi
    
    # Check for placeholder email
    if grep -q "your_email@ttu.edu" "$script"; then
        print_warning "Please update the email address in $script"
        print_info "Edit line: #SBATCH --mail-user=your_email@ttu.edu"
    fi
    
    return 0
}

# Function to submit job and provide feedback
submit_job() {
    local script="$1"
    local description="$2"
    
    print_info "Submitting $description..."
    print_info "Script: $script"
    
    if validate_script "$script"; then
        if output=$(sbatch "$script" 2>&1); then
            job_id=$(echo "$output" | grep -o '[0-9]\+' | tail -1)
            print_success "Job submitted successfully!"
            print_info "Job ID: $job_id"
            print_info "Monitor with: squeue -u \$USER"
            print_info "View output: tail -f *.${job_id}.out"
        else
            print_error "Failed to submit job:"
            echo "$output"
            return 1
        fi
    else
        return 1
    fi
}

# Function to list available scripts
list_scripts() {
    print_info "Available example scripts:"
    echo
    find . -name "submit_*a100*.sh" -type f | while read -r script; do
        if [[ -x "$script" ]]; then
            status="${GREEN}✓${NC}"
        else
            status="${RED}✗${NC}"
        fi
        echo -e "  $status $(basename "$script")"
    done
    echo
    print_info "Legend: ${GREEN}✓${NC} = Executable, ${RED}✗${NC} = Not executable"
}

# Function to check job status
check_status() {
    print_info "Your current jobs:"
    if squeue -u "$USER" --format="%.10i %.20j %.8T %.10M %.6D %.20V" 2>/dev/null; then
        echo
        print_info "Use 'scancel JOBID' to cancel a job"
        print_info "Use 'scontrol show job JOBID' for detailed job info"
    else
        print_warning "No jobs found or squeue command failed"
    fi
}

# Main script logic
main() {
    case "${1:-help}" in
        "lstm-baseline")
            check_directory
            submit_job "submit_lstm_a100_baseline.sh" "LSTM baseline profiling"
            ;;
        "lstm-comprehensive")
            check_directory
            submit_job "submit_lstm_a100_comprehensive.sh" "LSTM comprehensive profiling"
            ;;
        "lstm-custom")
            check_directory
            submit_job "submit_lstm_a100_custom.sh" "LSTM custom frequency profiling"
            ;;
        "sd-baseline")
            check_directory
            submit_job "submit_stablediffusion_a100_baseline.sh" "Stable Diffusion baseline profiling"
            ;;
        "custom")
            check_directory
            print_warning "Custom application template requires configuration!"
            print_info "Please edit submit_custom_app_a100_template.sh first:"
            print_info "1. Update CONDA_ENV to your environment"
            print_info "2. Update APP_PATH to your application"
            print_info "3. Update email address"
            print_info "Then run: sbatch submit_custom_app_a100_template.sh"
            ;;
        "list")
            check_directory
            list_scripts
            ;;
        "status")
            check_status
            ;;
        "help"|"-h"|"--help")
            usage
            ;;
        *)
            print_error "Unknown option: $1"
            echo
            usage
            exit 1
            ;;
    esac
}

# Execute main function
main "$@"
