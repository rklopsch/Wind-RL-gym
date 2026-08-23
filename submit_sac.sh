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
# Keep SAC parent thread pools bounded. Env workers set their own 1-thread limits in config.
export SAC_OMP_NUM_THREADS=${SAC_OMP_NUM_THREADS:-8}
export SAC_MKL_NUM_THREADS=${SAC_MKL_NUM_THREADS:-8}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-$SAC_OMP_NUM_THREADS}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-$SAC_MKL_NUM_THREADS}
export SAC_CPUS_PER_TASK=${SAC_CPUS_PER_TASK:-128}
export SAC_TORCH_NUM_THREADS=${SAC_TORCH_NUM_THREADS:-8}
export SAC_TORCH_NUM_INTEROP_THREADS=${SAC_TORCH_NUM_INTEROP_THREADS:-1}
export GIT_PYTHON_GIT_EXECUTABLE=$(which git || echo /usr/bin/git)
export PATH="$(dirname "$GIT_PYTHON_GIT_EXECUTABLE"):$PATH"
export SMARTSIM_LOG_LEVEL=DEBUG
export SMARTSIM_WLM_TRIALS=50  # default 10 stopped working  
# export SR_LOG_LEVEL=DEBUG
export SR_LOG_LEVEL=INFO
export SR_LOG_FILE=./log.sr
export SR_SOCKET_TIMEOUT=300000
export D4RL_DATASET_DIR=./.cache/torchrl/data/d4rl/datasets

# Useful post-launch diagnostics on ARCHER2:
# scontrol show step ${SLURM_JOB_ID}.1
# sstat -j ${SLURM_JOB_ID}.1 --format=JobID,AveCPU,AveRSS,MaxRSS
# srun --ntasks=1 --cpus-per-task=1 bash -lc 'ps -L -o pid,tid,psr,pcpu,comm -p <sac-pid>'

# Launch script
python launch_sac_run.py
