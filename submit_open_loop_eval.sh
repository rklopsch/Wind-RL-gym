#!/bin/bash
#
# Submit an open-loop evaluation of a trained SAC model.
# Nodes: one per parallel env (eval.n_parallel), plus one for open_loop_eval.py and one for Redis.
# Any arguments are forwarded as Hydra overrides for eval/config_open_loop_eval.yaml, e.g.
#   sbatch submit_open_loop_eval.sh eval.training_name=training_sac eval.model_id=515 eval.num_rounds=4

#SBATCH --job-name=RL3-sac-open-loop-eval
#SBATCH --nodes=18
#SBATCH --ntasks-per-node=128
#SBATCH --cpus-per-task=1
#SBATCH --time=12:00:00

#SBATCH --account=e01-ICL-Laizet
#SBATCH --partition=standard
#SBATCH --qos=standard

# Load required modules.
module swap PrgEnv-cray PrgEnv-gnu
module load cray-python

# Load the Python environment.
source /work/e01/e01/klop1/wind/venv-windrl-gym/bin/activate

# Keep the evaluation process's thread pools bounded; env workers set their own limits from the config.
export SAC_OMP_NUM_THREADS="${SAC_OMP_NUM_THREADS:-8}"
export SAC_MKL_NUM_THREADS="${SAC_MKL_NUM_THREADS:-8}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-${SAC_OMP_NUM_THREADS}}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-${SAC_MKL_NUM_THREADS}}"
# CPUs for the Redis orchestrator step; without this it inherits --cpus-per-task=1.
export SMARTSIM_DB_CPUS="${SMARTSIM_DB_CPUS:-128}"
export SAC_TORCH_NUM_THREADS="${SAC_TORCH_NUM_THREADS:-8}"
export SAC_TORCH_NUM_INTEROP_THREADS="${SAC_TORCH_NUM_INTEROP_THREADS:-1}"
GIT_PYTHON_GIT_EXECUTABLE="$(command -v git || echo /usr/bin/git)"
export GIT_PYTHON_GIT_EXECUTABLE
PATH="$(dirname "${GIT_PYTHON_GIT_EXECUTABLE}"):${PATH}"
export PATH
export SMARTSIM_LOG_LEVEL=DEBUG
export SMARTSIM_WLM_TRIALS=50
export SR_LOG_LEVEL=INFO
export SR_LOG_FILE=./log.sr
export SR_SOCKET_TIMEOUT=300000

python launch_open_loop_eval_run.py "$@"
