#!/bin/bash
#SBATCH --job-name=MINERVA_hopfield_sweep
#SBATCH --partition=ILCC-CDT,ILCC-Standard
#SBATCH --nodelist=arnold,duflo
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16000
#SBATCH --gres=gpu:rtx_2080_ti:4
#SBATCH --time=48:00:00
#SBATCH --output="/home/%u/minerva2/slurm_logs/%x_%A-%a.out"
#dummy-SBATCH --error="/home/%u/minerva2/slurm_logs/%x_%A-%a.err"

#SBATCH --signal=B:TERM@120 # tells the controller
                            # to send SIGTERM to the job 60 secs
                            # before its time ends to give it a
                            # chance for better cleanup.


module load miniconda/3

nvidia-smi

# =====================
# Logging information
# =====================

# slurm info - more at https://slurm.schedmd.com/sbatch.html#lbAJ
echo "Job running on ${SLURM_JOB_NODELIST}"

dt=$(date '+%d/%m/%Y %H:%M:%S')
echo "Job started: $dt"


# ===================
# Environment setup
# ===================

echo "Setting up bash enviroment"

# Make available all commands on $PATH as on headnode
source ~/.bashrc

# Make script bail out after first error
set -e

# Make your own folder on the node's scratch disk
# N.B. disk could be at /disk/scratch_big, or /disk/scratch_fast. Check
# yourself using an interactive session, or check the docs:
#     http://computing.help.inf.ed.ac.uk/cluster-computing
# SCRATCH_DISK=/disk/scratch2
# SCRATCH_DISK=/disk/scratch
# SCRATCH_HOME=${SCRATCH_DISK}/${USER}
# SCRATCH_PROJECT_DIR=${SCRATCH_HOME}/minerva2/wandb
# SCRATCH_DATA_DIR=${SCRATCH_PROJECT_DIR}/data
# mkdir -p ${SCRATCH_DATA_DIR}

# Activate your conda environment
CONDA_ENV_NAME=minerva
echo "Activating conda environment: ${CONDA_ENV_NAME}"
conda activate ${CONDA_ENV_NAME}
conda list
echo "Environment Activated!"

# Install trap for the signals INT and TERM to
 # the main BATCH script here.
 # Send SIGTERM using kill to the internal script's
 # process and wait for it to close gracefully.

 # Note: Most python scripts don't install handler
 # for SIGTERM and hence might die a quick painful death
 # on recieveing SIGTERM (kill -15).
 # To avoid this, you can send SIGINT,
 # i.e., KeyboardInterrupt using (kill -2).
trap 'echo signal received in BATCH!; kill -15 "${PID}"; wait "${PID}";' SIGINT SIGTERM

# start script in background and get its PID
# Note: $* traps all arguments passed to the script

# if directory /disk/scratch_fast exists, use it
if [ -d /disk/scratch_fast ]; then
    echo "Using /disk/scratch_fast for wandb"
    export WANDB_DIR=/disk/scratch_fast/s2191163/minerva_wandb
elif [ -d /disk/scratch ]; then
    echo "Using /disk/scratch for wandb"
    export WANDB_DIR=/disk/scratch/s2191163/minerva_wandb
elif [ -d /disk/scratch2 ]; then
    echo "Using /disk/scratch2 for wandb"
    export WANDB_DIR=/disk/scratch2/s2191163/minerva_wandb
else
    # break if no scratch disk is available
    echo "No scratch disk available, exiting."
    exit 1
fi
mkdir -p $WANDB_DIR
python run_hopfield_experiments.py $*

# Set the PID var so that the trap can use it
PID="$!"
wait "${PID}"

echo "The end!"
# =========================
# Post experiment logging
# =========================
echo ""
echo "============"
echo "job finished successfully"