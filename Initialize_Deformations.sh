#!/bin/bash
#SBATCH --account=project_***
#SBATCH --partition=small
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=12G
#SBATCH --time=06:00:00

module load python-data
source /projappl/project_***/my-venv/bin/activate

srun python3 Trajectories_Strains_Deformation.py --source_file demo_data/displacements/demoing_iceraft.npy --saved_quantities all --pixel_to_metric 10 --resolution_ratio 1
