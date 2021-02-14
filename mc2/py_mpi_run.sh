#!/bin/bash

#SBATCH --time=70:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
##SBATCH --cpus-per-task=1
#SBATCH --mem=16000
##SBATCH --clusters=mae
#SBATCH --mail-user=kunalmoz@buffalo.edu
#SBATCH --mail-type=ALL
#SBATCH --job-name='RS_ts_ns1'
#SBATCH --output=rs_glt-%j.dat
#SBATCH --error=rs_glt-%j.err
#SBATCH --partition=general-compute --qos=general-compute
##SBATCH --partition=debug --qos=debug
#SBATCH --requeue
#SBATCH --exclusive

# load modules
module load python/anaconda
module load intel/17.0
module load mkl/2018.3
module load gcc/7.3.0
module load intel-mpi


ulimit -s unlimited

# enable mpi4py module
export PYTHONPATH=/util/academic/python/mpi4py/v2.0.0/lib/python2.7/site-packages:$PYTHONPATH

# launch app
export I_MPI_FABRICS_LIST=tcp
export I_MPI_DEBUG=4
export I_MPI_PMI_LIBRARY=/usr/lib64/libpmi.so

srun -n 2 python glt_mpi.py


##echo "Okay!! Run python 3 for 64x64"


