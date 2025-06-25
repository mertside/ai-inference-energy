#!/bin/bash
"""
Test Script for MPI Applications (Template).

This is a template SLURM submission script for testing MPI applications.
It demonstrates the basic structure for running parallel applications
on HPC clusters.

Note: This script is included as a reference template and is not directly
related to the AI inference energy profiling experiments.

Usage:
    sbatch test.sh

Requirements:
    - MPI application executable: ./my_mpi
    - GCC and OpenMPI modules
    - Access to the specified partition

Author: AI Inference Energy Research Team
"""

#SBATCH --job-name=MPI_test
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=128
#SBATCH --output=%x.%j.out
#SBATCH --error=%x.%j.err
#SBATCH --partition=nocona
#SBATCH --time=01:00:00

# Enable strict error handling
set -euo pipefail

# Logging function
log_info() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] [INFO] $*"
}

log_error() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] [ERROR] $*" >&2
}

# Display job information
log_info "=== MPI Test Job Information ==="
log_info "Job ID: ${SLURM_JOB_ID:-'N/A'}"
log_info "Job Name: ${SLURM_JOB_NAME:-'N/A'}"
log_info "Nodes: ${SLURM_JOB_NUM_NODES:-'N/A'}"
log_info "Tasks per node: ${SLURM_NTASKS_PER_NODE:-'N/A'}"
log_info "Total tasks: ${SLURM_NTASKS:-'N/A'}"
log_info "Partition: ${SLURM_JOB_PARTITION:-'N/A'}"
log_info "Node list: ${SLURM_JOB_NODELIST:-'N/A'}"
log_info "================================"

# Load required modules
log_info "Loading required modules..."

if ! module load gcc/10.1.0; then
    log_error "Failed to load GCC module"
    exit 1
fi
log_info "Loaded GCC 10.1.0"

if ! module load openmpi/3.1.6; then
    log_error "Failed to load OpenMPI module"
    exit 1
fi
log_info "Loaded OpenMPI 3.1.6"

# Verify MPI setup
log_info "Verifying MPI setup..."
log_info "MPI compiler: $(which mpicc)"
log_info "MPI runtime: $(which mpirun)"

# Check if executable exists
if [[ ! -f "./my_mpi" ]]; then
    log_error "MPI executable not found: ./my_mpi"
    log_error "Please compile your MPI application before running this script"
    exit 1
fi

if [[ ! -x "./my_mpi" ]]; then
    log_error "MPI executable is not executable: ./my_mpi"
    exit 1
fi

log_info "MPI executable found and executable: ./my_mpi"

# Run the MPI application
log_info "Starting MPI application..."
log_info "Command: mpirun ./my_mpi"

if mpirun ./my_mpi; then
    log_info "MPI application completed successfully"
else
    log_error "MPI application failed"
    exit 1
fi

log_info "MPI test job completed"
