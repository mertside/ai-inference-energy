#!/bin/bash

# =============================================================================
# GPU Profiling Batch Plot Generator
# =============================================================================
# This script generates comprehensive profiling plots for all applications
# on a specified GPU using standardized frequency sets and key metrics.
#
# Usage:
#   ./batch_plot_gpu.sh V100
#   ./batch_plot_gpu.sh A100  
#   ./batch_plot_gpu.sh H100
#
# Output: Creates plots in plots/ directory with systematic naming
# =============================================================================

# Don't exit on errors - we want to continue with other plots
set +e

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PLOT_SCRIPT="$SCRIPT_DIR/plot_metric_vs_time.py"

# Applications to analyze
APPLICATIONS=("LLAMA" "VIT" "STABLEDIFFUSION" "WHISPER" "LSTM")

# Key metrics for comprehensive analysis
METRICS=("DRAMA" "GPUTL" "POWER" "MCUTL" "TMPTR")

# Default run number
RUN_NUMBER=3

# Function to display usage
usage() {
    echo -e "${BLUE}GPU Profiling Batch Plot Generator${NC}"
    echo ""
    echo "Usage: $0 <GPU_TYPE>"
    echo ""
    echo "Arguments:"
    echo "  GPU_TYPE    GPU to analyze (V100, A100, or H100)"
    echo ""
    echo "Examples:"
    echo "  $0 V100     # Generate all plots for V100"
    echo "  $0 A100     # Generate all plots for A100"
    echo "  $0 H100     # Generate all plots for H100"
    echo ""
    echo "Output:"
    echo "  - Creates plots/ directory if it doesn't exist"
    echo "  - Generates 20 plots total (5 apps × 4 metrics)"
    echo "  - Uses GPU-specific optimized frequency sets"
    echo ""
    exit 1
}

# Function to check if plot script exists
check_plot_script() {
    if [[ ! -f "$PLOT_SCRIPT" ]]; then
        echo -e "${RED}Error: Plot script not found at $PLOT_SCRIPT${NC}"
        echo "Please ensure you're running this script from the correct directory."
        exit 1
    fi
}

# Function to validate GPU type
validate_gpu() {
    local gpu="$1"
    case "${gpu^^}" in
        V100|A100|H100)
            return 0
            ;;
        *)
            echo -e "${RED}Error: Unsupported GPU type '$gpu'${NC}"
            echo "Supported GPUs: V100, A100, H100"
            exit 1
            ;;
    esac
}

# Function to get GPU frequency information
get_gpu_frequencies() {
    local gpu="$1"
    case "${gpu^^}" in
        V100)
            echo "510,750,960,1200,1380"
            ;;
        A100)
            echo "510,750,960,1200,1410"
            ;;
        H100)
            echo "510,750,960,1200,1830"
            ;;
    esac
}

# Function to generate a single plot
generate_plot() {
    local gpu="$1"
    local app="$2"
    local metric="$3"
    local frequencies="$4"
    
    echo -e "${BLUE}📊 Generating: ${gpu} + ${app} + ${metric}${NC}"
    
    # Debug: Show the exact command that will be run
    echo -e "${YELLOW}Command: python $PLOT_SCRIPT --gpu $gpu --app $app --metric $metric --frequencies $frequencies --run $RUN_NUMBER --no-show${NC}"
    
    # Run the plotting script with error capture
    local output
    local exit_code
    
    output=$(python "$PLOT_SCRIPT" \
        --gpu "$gpu" \
        --app "$app" \
        --metric "$metric" \
        --frequencies "$frequencies" \
        --run "$RUN_NUMBER" \
        --no-show 2>&1)
    exit_code=$?
    
    if [[ $exit_code -eq 0 ]]; then
        echo -e "${GREEN}✅ Success: ${gpu}_${app,,}_${metric,,}${NC}"
        return 0
    else
        echo -e "${RED}❌ Failed: ${gpu}_${app,,}_${metric,,}${NC}"
        echo -e "${RED}Error output:${NC}"
        echo "$output" | head -10
        return 1
    fi
}

# Function to display summary statistics
display_summary() {
    local gpu="$1"
    local total_plots="$2"
    local successful_plots="$3"
    local failed_plots="$4"
    
    echo ""
    echo -e "${BLUE}═══════════════════════════════════════════════════════════${NC}"
    echo -e "${BLUE}                    BATCH PLOTTING SUMMARY                   ${NC}"
    echo -e "${BLUE}═══════════════════════════════════════════════════════════${NC}"
    echo -e "${YELLOW}GPU:${NC} $gpu"
    echo -e "${YELLOW}Frequency Set:${NC} $(get_gpu_frequencies "$gpu") MHz"
    echo -e "${YELLOW}Applications:${NC} ${APPLICATIONS[*]}"
    echo -e "${YELLOW}Metrics:${NC} ${METRICS[*]}"
    echo ""
    echo -e "${YELLOW}Results:${NC}"
    echo -e "  ${GREEN}✅ Successful plots: $successful_plots${NC}"
    if [[ $failed_plots -gt 0 ]]; then
        echo -e "  ${RED}❌ Failed plots: $failed_plots${NC}"
    fi
    echo -e "  📊 Total attempts: $total_plots"
    echo ""
    
    # Show plots directory contents
    if [[ -d "plots" ]]; then
        echo -e "${YELLOW}Generated plots in plots/ directory:${NC}"
        ls -la plots/*${gpu,,}* 2>/dev/null | head -10 || echo "  No plots found for $gpu"
        
        local plot_count=$(ls plots/*${gpu,,}* 2>/dev/null | wc -l)
        if [[ $plot_count -gt 10 ]]; then
            echo "  ... and $((plot_count - 10)) more files"
        fi
    fi
    echo ""
}

# Main execution function
main() {
    local gpu="$1"
    
    # Input validation
    if [[ $# -ne 1 ]]; then
        usage
    fi
    
    validate_gpu "$gpu"
    check_plot_script
    
    # Convert to uppercase for consistency
    gpu="${gpu^^}"
    
    # Get GPU-specific frequencies
    local frequencies
    frequencies=$(get_gpu_frequencies "$gpu")
    
    echo -e "${BLUE}🚀 Starting Batch Plot Generation${NC}"
    echo -e "${YELLOW}GPU:${NC} $gpu"
    echo -e "${YELLOW}Frequencies:${NC} $frequencies MHz"
    echo -e "${YELLOW}Applications:${NC} ${APPLICATIONS[*]}"
    echo -e "${YELLOW}Metrics:${NC} ${METRICS[*]}"
    echo -e "${YELLOW}Run Number:${NC} $RUN_NUMBER"
    echo ""
    
    # Create plots directory if it doesn't exist
    mkdir -p plots
    
    # Initialize counters
    local total_plots=0
    local successful_plots=0
    local failed_plots=0
    
    # Generate plots for each combination
    for app in "${APPLICATIONS[@]}"; do
        echo -e "${YELLOW}📱 Processing application: $app${NC}"
        
        for metric in "${METRICS[@]}"; do
            ((total_plots++))
            
            if generate_plot "$gpu" "$app" "$metric" "$frequencies"; then
                ((successful_plots++))
            else
                ((failed_plots++))
                echo -e "${YELLOW}⚠️  Continuing with next plot...${NC}"
            fi
            
            # Small delay to avoid overwhelming the system
            sleep 0.5
        done
        
        echo ""
    done
    
    # Display summary
    display_summary "$gpu" "$total_plots" "$successful_plots" "$failed_plots"
    
    # Exit with appropriate code
    if [[ $failed_plots -eq 0 ]]; then
        echo -e "${GREEN}🎉 All plots generated successfully!${NC}"
        exit 0
    else
        echo -e "${YELLOW}⚠️  Some plots failed. Check the output above for details.${NC}"
        exit 1
    fi
}

# Handle script execution
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi
