#!/bin/bash
#
# Short test of the open-loop evaluation pipeline: 2 envs, 2 rounds of 20-step episodes.
# Nodes: eval.n_parallel solvers + 1 for open_loop_eval.py + 1 for Redis.
# Control frames per solver: 150 (burn-in) + 3 * 2 * (20 + 150) = 1170.

#SBATCH --job-name=RL3-sac-open-loop-eval-test
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=128
#SBATCH --cpus-per-task=1
#SBATCH --time=00:20:00

#SBATCH --account=e01-ICL-Laizet
#SBATCH --partition=standard
#SBATCH --qos=short

# Load required modules.
module swap PrgEnv-cray PrgEnv-gnu
module load cray-python

# Load the Python environment.
source /work/e01/e01/klop1/wind/venv-windrl-gym/bin/activate

# Keep the evaluation process's thread pools bounded; env workers set their own limits from the config.
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-8}"
# CPUs for the Redis orchestrator step; without this it inherits --cpus-per-task=1.
export SMARTSIM_DB_CPUS="${SMARTSIM_DB_CPUS:-128}"
GIT_PYTHON_GIT_EXECUTABLE="$(command -v git || echo /usr/bin/git)"
export GIT_PYTHON_GIT_EXECUTABLE
PATH="$(dirname "${GIT_PYTHON_GIT_EXECUTABLE}"):${PATH}"
export PATH
export SMARTSIM_LOG_LEVEL=DEBUG
export SMARTSIM_WLM_TRIALS=50
export SR_LOG_LEVEL=INFO
export SR_LOG_FILE=./log.sr
export SR_SOCKET_TIMEOUT=300000

python launch_open_loop_eval_run.py \
  eval.training_name=training_sac \
  eval.run_name=open_loop_eval_test \
  eval.model_id=500000 \
  eval.n_parallel=2 \
  eval.episode_length=20 \
  eval.num_rounds=2 \
  eval.initial_reset_frames=0
