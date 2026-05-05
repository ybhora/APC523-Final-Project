#!/bin/bash
#SBATCH --job-name=qsolver
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=00:30:00
#SBATCH --mem-per-cpu=4G
#SBATCH --mail-type=begin,end,fail
#SBATCH --mail-user=yb9406@princeton.edu

module purge
module load anaconda3/2025.6
python solver.py
