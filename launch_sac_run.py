import os
import shutil
import numpy as np

git_bin = shutil.which("git") or "/usr/bin/git"
os.environ.setdefault("GIT_PYTHON_GIT_EXECUTABLE", git_bin)
os.environ["PATH"] = os.path.dirname(git_bin) + os.pathsep + os.environ.get("PATH", "")

PYTHON_BIN = "/home/roman/.pyenv/versions/windRL/bin/python"
if not os.path.exists(PYTHON_BIN):
    PYTHON_BIN = shutil.which("python") or "python"

from smartsim import Experiment
from smartsim.status import SmartSimStatus
import time
import math
from smartredis import Client

if shutil.which("srun"):
    LAUNCHER = "auto"
    RUN_COMMAND = "srun"
    SOLVER_TASKS = 128
else:
    LAUNCHER = "local"
    RUN_COMMAND = None
    SOLVER_TASKS = 16

from Solver.ADM_setup import ADMSimulation
from Solver.farm import Farm, Turbine
from Solver.precursor_utils import ensure_precursor
from Solver.probe_layout import probe_params, stage_probe_layout
from Solver.solver_runtime import verified_solver, record_solver
from hydra import initialize, compose
import sys


def launch_database(experiment, port):
    interface = 'lo' if not shutil.which("srun") else ['hsn0', 'hsn1']
    db = experiment.create_database(port=port, db_nodes=1, interface=interface)
    if shutil.which("srun"):
        # SmartSim sets no --cpus-per-task for Slurm DB steps, so srun inherits the job's value (1),
        # pinning Redis and its I/O threads to a single core and saturating it with solver polling.
        db.set_cpus(int(os.environ.get("SMARTSIM_DB_CPUS", 128)))

    # generate directories for output files
    # pass in objects to make dirs for
    experiment.generate(db, overwrite=True)

    # start database on interactive allocation and wait until database is launched
    experiment.start(db, block=True)

    # get status of database
    statuses = experiment.get_status(db)
    print(f"Status of all database nodes: {statuses}.")
    print(f"Database started on {db.get_address()}.")

    os.environ['SR_DB_TYPE'] = "Standalone"  # visible in this process + all children
    os.environ['SSDB'] = db.get_address()[0]  # visible in this process + all children

    return db


def launch_solver(experiment, instance, cfg):
    executable, _ = verified_solver(cfg.env)
    kwargs = {"run_command": RUN_COMMAND} if RUN_COMMAND is not None else {}
    aprun = experiment.create_run_settings(exe=executable, **kwargs)
    aprun.set_tasks(SOLVER_TASKS)
    aprun.set_cpus_per_task(1)
    aprun.set_nodes(1)
    aprun.set_tasks_per_node(SOLVER_TASKS)
    print(aprun.format_run_args())
    producer = experiment.create_model(f"WindFarm_{instance}", aprun)
    files = ["./Solver/ADM/Base"]
    precursor_dir = ensure_precursor(instance, cfg)
    producer.attach_generator_files(to_copy=files, to_symlink=[precursor_dir])
    experiment.generate(producer, overwrite=True)

    # Configure case
    # Smartsims to_configure flag not working so doing manually with ADM_setup function
    farm1 = Farm(cfg.env.turbine_diameter * cfg.env.turbine_spacing * (cfg.env.turbines-1),
                 cfg.env.turbine_diameter * 1 * 1,
                 cfg.env.turbines,
                 Turbine(cfg.env.turbine_diameter, cfg.env.turbine_height, yaw=0),
                 offset=[(cfg.env.turbine_spacing-1)/2*cfg.env.turbine_diameter, (cfg.env.turbine_spacing_z-1)/2*cfg.env.turbine_diameter])
    farm1.grid()
    simulation_steps = (cfg.env.steps_per_frame
                        *((cfg.collector.total_frames//cfg.collector.frames_per_batch)+1)
                        *((cfg.collector.frames_per_batch//cfg.env.n_parallel)+1)
                        *(1 + (cfg.env.reset_frames / cfg.collector.max_episode_length))
                        +cfg.env.steps_per_frame*cfg.env.initial_reset_frames)  # This is horrible | Max: I agree, it truly is. horrendous
    case = ADMSimulation(farm1, timesteps=math.ceil(simulation_steps),
                         control_freq=cfg.env.steps_per_frame,
                         probe_layout=probe_params(cfg.env)["probe_layout"],
                         instance=instance)
    case.setup_case(f"./{experiment.name}/WindFarm_{instance}", precursor_root=cfg.env.precursor_root)
    stage_probe_layout(producer.path, cfg.env)
    record_solver(producer.path, cfg.env)
    # case.setup_precursor(f"./{experiment.name}/WindFarm_{instance}/precursor_{instance}")

    return producer


def launch_sac(experiment, cfg, config_modifiers):
    kwargs = {"run_command": RUN_COMMAND} if RUN_COMMAND is not None else {}
    exe_args = " ".join(["sac.py", *config_modifiers])
    aprun = experiment.create_run_settings(exe=PYTHON_BIN, exe_args=exe_args, **kwargs)
    aprun.set_tasks(1)
    aprun.set_cpus_per_task(128)
    aprun.set_nodes(1)
    aprun.set_tasks_per_node(1)
    producer = experiment.create_model("sac", aprun)

    # create directories for output files and copy
    # scripts to execution location inside newly created dir
    # only necessary if its not an executable (python is executable here)
    file_list = ["./sac/sac.py", "./sac/utils_sac.py", "./sac/config_sac.yaml"]  # sac files
    file_list += ["./Solver/WF_enviroment.py", "./Solver/ADM_setup.py", "./Solver/farm.py"]  # Env and simulator files
    if bool(cfg.checkpoint.load_from_checkpoint):  # copy in checkpointed models if desired
        file_list += [f"{cfg.checkpoint.model_checkpoint_path}/actor_{cfg.checkpoint.model_checkpoint_id}.pkl"]
        file_list += [f"{cfg.checkpoint.model_checkpoint_path}/critic_{cfg.checkpoint.model_checkpoint_id}.pkl"]
        file_list += [f"{cfg.checkpoint.model_checkpoint_path}/observation_{cfg.checkpoint.model_checkpoint_id}.json"]
        file_list += [f"{cfg.checkpoint.model_checkpoint_path}/critic2_{cfg.checkpoint.model_checkpoint_id}.pkl"]
        file_list += [f"{cfg.checkpoint.model_checkpoint_path}/sac_checkpoint_{cfg.checkpoint.model_checkpoint_id}.pt"]
    producer.attach_generator_files(to_copy=file_list)

    experiment.generate(producer, overwrite=True)
    cfg.env.probes_file = stage_probe_layout(producer.path, cfg.env)
    return producer


if __name__ == '__main__':
    # The first command line argument determines the run-directory name. Any
    # remaining arguments are Hydra overrides applied consistently to both the
    # launcher and the SAC process.
    run_name = sys.argv[1] if len(sys.argv) > 1 else "training_sac"
    config_modifiers = list(sys.argv[2:])

    initialize(config_path="./sac/", version_base="1.2")
    cfg = compose(
        config_name="config_sac.yaml",
        overrides=config_modifiers,
    )

    probe_params(cfg.env)
    verified_solver(cfg.env)

    # Set up experiment
    exp = Experiment(run_name, launcher=LAUNCHER)

    # Runtime parameters
    n_environments = cfg.env.n_parallel

    # Start database
    db_port = 6783
    db = launch_database(exp, db_port)

    # Start RL
    rl_app = launch_sac(exp, cfg, config_modifiers)

    # Start simulations
    simulations = []
    for i in range(1, n_environments+1):
        simulation = launch_solver(exp, instance=i, cfg=cfg)
        simulations.append(simulation)

    everything = simulations + [rl_app, db]

    exp.start(rl_app, block=False, summary=False)
    time.sleep(60)
    exp.start(*simulations, block=False, summary=False)

    while True:
        statuses = exp.get_status(*everything)
        ended = [(s == SmartSimStatus.STATUS_COMPLETED or s == SmartSimStatus.STATUS_FAILED) for s in statuses]
        if any(ended):
            print('Something finished/crashed so stopping everything')
            break
        time.sleep(30)

    exp.stop(*everything)  # lol i love the "stop everything" command
    print(exp.summary())
