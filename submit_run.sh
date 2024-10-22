#!/bin/bash
#SBATCH --job-name=RL-test
#SBATCH --nodes=34
#SBATCH --ntasks-per-node=128
#SBATCH --cpus-per-task=1
#SBATCH --time=72:00:00

#SBATCH --account=e01-ICL-Laizet
#SBATCH --partition=standard
#SBATCH --qos=long


# load required modules
module swap PrgEnv-cray PrgEnv-gnu
module load cray-python

# load python environment
source /work/e01/e01/amole/smartsim_venv/bin/activate

# Set environment variables
# export OMP_NUM_THREADS=1
export SMARTSIM_LOG_LEVEL=DEBUG
# export SR_LOG_LEVEL=DEBUG
export SR_LOG_LEVEL=INFO
export SR_LOG_FILE=./log.sr
export SR_SOCKET_TIMEOUT=300000
export D4RL_DATASET_DIR=./.cache/torchrl/data/d4rl/datasets

# Launch script
python launch_run.py
