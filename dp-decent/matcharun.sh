#!/usr/bin/env bash

# Lines that begin with #SBATCH specify commands to be used by SLURM for scheduling

#SBATCH --job-name=pdsgd0.05     # sets the job name if not set from environment
#SBATCH --time=03:30:00                  # how long you think your job will take to complete; format=hh:mm:ss
#SBATCH --account=scavenger                  # set QOS, this will determine what resources can be requested
#SBATCH --qos=scavenger                                         # set QOS, this will determine what resources can be requested
#SBATCH --partition=scavenger
#SBATCH --gres=gpu:2
#SBATCH --ntasks=8
#SBATCH --mem 64gb                                              # memory required by job; if unit is not specified MB will be assumed
#SBATCH --nice=0
#SBATCH --mail-type=END                 # Valid type values are NONE, BEGIN, END, FAIL, REQUEUE,
#SBATCH --error Logs/%x_%A_%a.log

#module load python
module load openmpi

mpirun -np 8 python matcha-train.py --description pdsgd0.05 --resSize 50 --randomSeed 9001 --datasetRoot ./data --budget 0.05 --outputFolder pdsgd0.05 --bs 64 --epoch 200 --name pdsgd0.05
