#!/bin/bash -l
#PBS -A SCSG0001
#PBS -N python_cfd_perf
#PBS -j oe
#PBS -k oed
#PBS -q casper
#PBS -l walltime=10:00:00
#PBS -l select=1:ncpus=18:ompthreads=18:mem=16GB:ngpus=1
#PBS -l gpu_type=v100
#PBS -o /glade/u/home/bneuman/codes/12_CuPyAndLegate/code/cfd/cupy/pbs_bm.o

export TMPDIR=/glade/scratch/${USER}/temp
mkdir -p $TMPDIR
module load nvhpc cuda
module load conda/latest
conda activate pgpu

### Run
python step11_perf_datamovement.py > results_bm_100ts_small.txt

### Store job stats
#qstat -f $PBS_JOBID