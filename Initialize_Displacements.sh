#!/bin/bash
#SBATCH --account=***
#SBATCH --partition=gputest
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=00:05:00
#SBATCH --gres=gpu:v100:1

module load python-data
module load pytorch

srun python3 Displacement_Production.py --update_iterations 30 --image_resolution 1000 --result_file_name demoing_iceraft
