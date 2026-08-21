import os
import shutil
import numpy as np

git_bin = shutil.which("git") or "/usr/bin/git"
os.environ.setdefault("GIT_PYTHON_GIT_EXECUTABLE", git_bin)
os.environ["PATH"] = os.path.dirname(git_bin) + os.pathsep + os.environ.get("PATH", "")

if shutil.which("srun"):
    LAUNCHER = "auto"
    RUN_COMMAND = "srun"
    SOLVER_TASKS = 128
else:
    LAUNCHER = "local"
    RUN_COMMAND = None
    SOLVER_TASKS = 1

from smartsim import Experiment
from smartsim.status import SmartSimStatus
import time
import math
from smartredis import Client
from Solver.ADM_setup import ADMSimulation
from Solver.farm import Farm, Turbine
from Solver.precursor_utils import ensure_precursor
from hydra import initialize, compose
import sys


def launch_database(experiment, port):
    interface = 'lo' if not shutil.which("srun") else ['hsn0', 'hsn1']
    db = experiment.create_database(port=port, db_nodes=1, interface=interface)

    # generate directories for output files
    # pass in objects to make dirs for
    experiment.generate(db, overwrite=True)

    # start the database on interactive allocation and wait until database is launched
    experiment.start(db, block=True)

    # get the status of the database
    statuses = experiment.get_status(db)
    print(f"Status of all database nodes: {statuses}.")
    print(f"Database started on {db.get_address()}.")

    os.environ['SR_DB_TYPE'] = "Standalone"  # visible in this process + all children
    os.environ['SSDB'] = db.get_address()[0]  # visible in this process + all children

    return db


def launch_solver(experiment, instance, cfg):
    if os.path.isdir(LOCAL_SMARTREDIS_LIB):
        os.environ['LD_LIBRARY_PATH'] = LOCAL_SMARTREDIS_LIB + os.pathsep + os.environ.get('LD_LIBRARY_PATH', "")
    if os.path.isdir(LOCAL_INCOMPACT_BIN):
        os.environ['PATH'] = LOCAL_INCOMPACT_BIN + os.pathsep + os.environ.get('PATH', "")
    # TODO: probably (definitely) want a better way to set these

    aprun = experiment.create_run_settings(exe="xcompact3d")
    if RUN_COMMAND is not None:
        aprun.set_run_command(RUN_COMMAND)
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
                        * (cfg.env.reset_frames*3 + cfg.optim.episode_steps) * cfg.optim.train_steps
                        +cfg.env.steps_per_frame*cfg.env.initial_reset_frames)  # This is horrible | Max: I agree, it truly is. horrendous
    case = ADMSimulation(farm1, timesteps=math.ceil(simulation_steps),
                         control_freq=cfg.env.steps_per_frame,
                         probes_per_turbine=cfg.env.probes_per_turbine,
                         instance=instance)
    case.setup_case(f"./{experiment.name}/WindFarm_{instance}", precursor_root=cfg.env.precursor_root)
    # case.setup_precursor(f"./{experiment.name}/WindFarm_{instance}/precursor_{instance}")

    return producer


def launch_bo(experiment, cfg, config_modifiers):
    aprun = experiment.create_run_settings(exe=PYTHON_BIN, exe_args="bo.py " + " ".join(config_modifiers))
    if RUN_COMMAND is not None:
        aprun.set_run_command(RUN_COMMAND)
    aprun.set_tasks(1)
    aprun.set_cpus_per_task(128)
    aprun.set_nodes(1)
    aprun.set_tasks_per_node(1)
    producer = experiment.create_model("bo", aprun)

    # create directories for the output files and copy
    # scripts to execution location inside newly created dir
    # only necessary if its not an executable (python is executable here) 
    file_list = ["./bo/bo.py", "./bo/utils_bo.py", "./bo/config_bo.yaml"]  # bo files
    file_list += ["./Solver/WF_enviroment.py", "./Solver/ADM_setup.py", "./Solver/farm.py"]  # Env and simulator files
    producer.attach_generator_files(to_copy=file_list)

    experiment.generate(producer, overwrite=True)
    return producer


if __name__ == '__main__':

    # The first command line argument determines the name of the run directory that everything is saved to
    run_name = sys.argv[1] if len(sys.argv) > 1 else "training_bo"

    # Any subsequent arguments (space-separated) are taken to be modifiers for the config file
    # We assume that the arguments are valid modifiers for the config; this is not tested here.
    # An example of a valid modifier is "optim.gamma=0.9"
    config_modifiers = list(sys.argv[2:]) if len(sys.argv) > 1 else [""]

    # Read bo config
    initialize(config_path="./bo/", version_base="1.2")
    cfg = compose(config_name="config_bo.yaml", overrides=config_modifiers)

    # Set up experiment
    exp = Experiment(run_name, launcher=LAUNCHER)

    # Runtime parameters
    n_environments = cfg.env.n_parallel

    # Start database
    db_port = 6783
    db = launch_database(exp, db_port)

    # Start RL
    rl_app = launch_bo(exp, cfg, config_modifiers)

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
