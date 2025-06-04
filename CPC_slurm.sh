#!/bin/bash

#SBATCH --job-name=v2esurreal     # create a short name for your job
#SBATCH --nodes=1                # node count
#SBATCH --ntasks=1               # total number of tasks across all nodes
#SBATCH --cpus-per-task=8        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem-per-cpu=32G         # memory per cpu-core (4G per cpu-core is default)
#SBATCH --time=24:00:00          # total run time limit (HH:MM:SS)
#SBATCH --gres=gpu:1             # number of gpus per node
#SBATCH --mail-type=begin        # send email when job begins
#SBATCH --mail-type=end          # send email when job ends
#SBATCH --mail-user=martin.barry@hevs.ch
ulimit -n 65535
apptainer exec --nv --bind /home/martin.barry/datasets/processed_surreal:/home/martin.barry/projects/SurrealEvent/dataset/ /home/martin.barry/datasets/Surreal.sif python src/main_CPC.py
