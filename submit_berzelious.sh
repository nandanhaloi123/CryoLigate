#!/bin/bash
# SLURM batch job script for Berzelius

#SBATCH --gpus=4                     # Request 4 GPUs
#SBATCH -t 3-00:00:00                 # Wall time: 3 days (72h)


# Execute your code
#python src/03_precompute_molformer.py
python src/04_train.py
