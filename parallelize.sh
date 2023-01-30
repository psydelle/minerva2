#!/bin/bash #SBATCH -p CDT_GPU
#SBATCH -J *JOB_NAME*
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --output="./logs/%A.out"
#SBATCH --error="./logs/%A.err"