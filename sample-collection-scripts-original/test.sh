#!/bin/bash
#SBATCH –J MPI_test
#SBATCH –N 2
#SBATCH –ntasks-per-node=128 
#SBATCH –o %x.%j.out 
#SBATCH –e %x.%j.err 
#SBATCH –p nocona

module load gcc/10.1.0 openmpi/3.1.6 
mpirun ./my_mpi
