#!/bin/bash

# specify resources
#SBATCH -G 1
#SBATCH -N 1
#SBATCH -e slurm-%j.log
#SBATCH -o slurm-%j.log
#SBATCH --array=0-9
#SBATCH --mem=32G
#SBATCH -t 24:00:00
#SBATCH -J Train

# queue
#SBATCH -p lindahl5

echo "Starting Job Array ID: $SLURM_ARRAY_TASK_ID"
echo "Running on node: $(hostname)"

# 4. Run the Python Script
# $SLURM_ARRAY_TASK_ID passes the numbers 0, 1, 2... automatically
# We set --total_nodes 10 because --array=0-9 creates 10 jobs.
python src/03_build_processed_dataset_multinode.py --node_id $SLURM_ARRAY_TASK_ID --total_nodes 10

