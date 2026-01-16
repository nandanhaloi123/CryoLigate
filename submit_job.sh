#!/bin/bash

# specify resources
#SBATCH -G 1
#SBATCH -N 1
#SBATCH -e slurm-%j.log
#SBATCH -o slurm-%j.log

#SBATCH -t 48:00:00
#SBATCH -J Train

# queue
#SBATCH -p lindahl4

python src/04_train.py 
