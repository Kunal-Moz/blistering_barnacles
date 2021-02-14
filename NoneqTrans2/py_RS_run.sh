#!/bin/bash

#SBATCH --time=72:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
##SBATCH --cpus-per-task=1
#SBATCH --mem=16000
##SBATCH --clusters=mae
#SBATCH --mail-user=kunalmoz@buffalo.edu
#SBATCH --mail-type=ALL
#SBATCH --job-name='rst_dist_40_ts2'
#SBATCH --output=rs_glt-%j.dat
#SBATCH --error=rs_glt-%j.err
#SBATCH --partition=general-compute --qos=general-compute
##SBATCH --partition=debug --qos=debug
#SBATCH --requeue
#SBATCH --exclusive

module load intel/17.0
module load mkl/2018.3
module load gcc/7.3.0
module load intel-mpi/2017.0.1


ulimit -s unlimited

## export HOME=$SLURMTMPDIR
## export TMP=$SLURMTMPDIR

## export MKL_NUM_THREADS=8
## export MKL_DOMAIN_NUM_THREADS="MKL_ALL=8, MKL_BLAS=8"
## export MKL_DYNAMIC=FALSE
## export OMP_NUM_THREADS=8


srun python glt.py > rs_tb_0.85.dat ;

echo "Okay!! Run python 3 for 64x64"


