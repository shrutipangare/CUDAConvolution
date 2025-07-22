#!/bin/bash
#SBATCH --job-name=convolution
#SBATCH --output=%x.out
#SBATCH --account=ece_gy_9143-2025sp
#SBATCH --partition=c12m85-a100-1
#SBATCH --cpus-per-task=4
#SBATCH --time=10:00:00
#SBATCH --mem=10GB
#SBATCH --gres=gpu:1

cd /scratch/stp8232/PartC

# Load singularity
module load singularity

# Container path
CONTAINER=/scratch/work/public/singularity/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif

# Run compilation inside the container
singularity exec --nv $CONTAINER bash -c "cd /scratch/stp8232/PartC && nvcc -o convolve convolve.cu -lcudnn"

# Run the program inside the container
singularity exec --nv $CONTAINER bash -c "cd /scratch/stp8232/PartC && ./convolve"