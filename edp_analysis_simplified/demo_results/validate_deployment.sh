#!/bin/bash
# GPU Frequency Deployment Validation Script

echo 'GPU Frequency Deployment Validation'
echo '==================================='

# Check if nvidia-smi is available
if ! command -v nvidia-smi &> /dev/null; then
    echo "❌ nvidia-smi not found. Please install NVIDIA drivers."
    exit 1
fi

# Check current GPU frequencies
echo "Current GPU Status:"
nvidia-smi --query-gpu=index,name,clocks.gr,clocks.mem,power.draw,temperature.gpu --format=csv

# Check if any GPUs are available
GPU_COUNT=$(nvidia-smi --query-gpu=count --format=csv,noheader,nounits)
if [ $GPU_COUNT -eq 0 ]; then
    echo "❌ No GPUs detected"
    exit 1
fi

echo "Available Optimal Configurations:"
echo "  A100+LLAMA: 825MHz (Expected: 29.3% energy savings)"
echo "  A100+STABLEDIFFUSION: 675MHz (Expected: 29.3% energy savings)"

echo ""
echo "To deploy a configuration, use:"
echo "  ./deploy_frequencies.sh <CONFIG> deploy"
echo ""
echo "To check status:"
echo "  ./deploy_frequencies.sh <CONFIG> status"
echo ""
echo "To reset to defaults:"
echo "  ./deploy_frequencies.sh <CONFIG> reset"
