#!/bin/bash
#SBATCH --job-name=audio_fingerprint
#SBATCH --output=pipeline_%j.log
#SBATCH --error=pipeline_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32          # Request 32 CPU cores
#SBATCH --mem=64gb                   # 64GB RAM
#SBATCH --time=02:00:00              # 2 hour time limit
#SBATCH --partition=hpg-default      # Or your preferred partition
#SBATCH --account=YOUR_ACCOUNT       # Replace with your HiperGator account

# Load conda
module load conda

# Activate your environment
conda activate diy-shazam

# Print job info
echo "Job started at: $(date)"
echo "Running on node: $(hostname)"
echo "Number of CPUs: $SLURM_CPUS_PER_TASK"

# Run pipeline with parallel processing
# The script will automatically use all available CPUs
python -m database.pipeline --num_workers $SLURM_CPUS_PER_TASK

echo "Job finished at: $(date)"

