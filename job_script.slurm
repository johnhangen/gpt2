#!/bin/bash
#SBATCH --job-name=gpt2_training         # Job name
#SBATCH --output=output_%j.txt           # Output file
#SBATCH --error=error_%j.txt             # Error file
#SBATCH --time=1:00:00                   # Time limit
#SBATCH --partition=coc-gpu              # Specify GPU partition
#SBATCH --gres=gpu:V100:1                # Request 1 GPU (V100)
#SBATCH --mem=16G                        # Allocate 16GB of memory
#SBATCH --cpus-per-task=4                # Number of CPU cores per task

# Load necessary modules
module spider anaconda

# Run the Python script
python3 train_gpt2.py
