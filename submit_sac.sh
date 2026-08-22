#!/bin/bash
#SBATCH --job-name=RL3-sac-test
#SBATCH --nodes=34
#SBATCH --ntasks-per-node=128
#SBATCH --cpus-per-task=1
#SBATCH --time=12:00:00

#SBATCH --account=e01-ICL-Laizet
#SBATCH --partition=standard
#SBATCH --qos=standard


# load required modules
module swap PrgEnv-cray PrgEnv-gnu
module load cray-python

# load python environment
source /work/e01/e01/klop1/wind/venv-windrl-gym/bin/activate

# Set environment variables
# export OMP_NUM_THREADS=1
export GIT_PYTHON_GIT_EXECUTABLE=$(which git || echo /usr/bin/git)
export PATH="$(dirname "$GIT_PYTHON_GIT_EXECUTABLE"):$PATH"
export SMARTSIM_LOG_LEVEL=DEBUG
export SMARTSIM_WLM_TRIALS=50  # default 10 stopped working  
# export SR_LOG_LEVEL=DEBUG
export SR_LOG_LEVEL=INFO
export SR_LOG_FILE=./log.sr
export SR_SOCKET_TIMEOUT=300000
export D4RL_DATASET_DIR=./.cache/torchrl/data/d4rl/datasets

# Launch script
python launch_sac_run.py
