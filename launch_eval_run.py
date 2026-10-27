import os
import shutil
import numpy as np

git_bin = shutil.which("git") or "/usr/bin/git"
os.environ.setdefault("GIT_PYTHON_GIT_EXECUTABLE", git_bin)
os.environ["PATH"] = os.path.dirname(git_bin) + os.pathsep + os.environ.get("PATH", "")

PYTHON_BIN = "/home/roman/.pyenv/versions/windRL/bin/python"
if not os.path.exists(PYTHON_BIN):
    PYTHON_BIN = shutil.which("python") or "python"

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
import sys
from smartredis import Client
from Solver.ADM_setup import ADMSimulation
from Solver.farm import Farm, Turbine
from Solver.precursor_utils import ensure_precursor
from Solver.probe_layout import probe_params, stage_probe_layout
from Solver.solver_runtime import verified_solver, record_solver
from hydra import initialize, compose
from omegaconf import OmegaConf
from datetime import datetime


def launch_database(experiment, port, dummy_update=False):
    interface = ['hsn0', 'hsn1'] if (not dummy_update and shutil.which("srun")) else 'lo'
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
    executable, _ = verified_solver(cfg.env)
    aprun = experiment.create_run_settings(exe=executable)
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
                        * (cfg.env.reset_frames * 3
                           + cfg.eval.initial_reset_frames
                           + cfg.eval.num_rollouts *
                              (cfg.eval.episode_length
                               + cfg.eval.inbetween_reset_frames
                               + cfg.env.reset_frames
                              )
                          )
                       )

    case = ADMSimulation(farm1, timesteps=math.ceil(simulation_steps),
                         control_freq=cfg.env.steps_per_frame,
                         probe_layout=probe_params(cfg.env)["probe_layout"],
                         instance=instance)
    case.setup_case(f"./{experiment.name}/WindFarm_{instance}", save_flow=cfg.eval.save_flow, precursor_root=cfg.env.precursor_root)
    stage_probe_layout(producer.path, cfg.env)
    record_solver(producer.path, cfg.env)
    # case.setup_precursor(f"./{experiment.name}/WindFarm_{instance}/precursor_{instance}")

    return producer


def launch_dummy_solver(experiment):
    aprun = experiment.create_run_settings(exe="python", exe_args="dummy_solver.py")
    aprun.set_tasks(1)
    producer = experiment.create_model("dummy_solver", aprun)

    # create directories for the output files and copy
    # scripts to execution location inside newly created dir
    # only necessary if its not an executable (python is executable here)
    producer.attach_generator_files(to_copy="./Solver/dummy_solver.py")

    experiment.generate(producer, overwrite=True)
    shutil.copy2("./Solver/probe_layout.py", os.path.join(producer.path, "probe_layout.py"))
    return producer


def launch_eval(experiment, cfg, dummy_update=False):
    run_command = "" if dummy_update else RUN_COMMAND
    aprun = experiment.create_run_settings(exe=PYTHON_BIN, exe_args="eval.py")
    if run_command:
        aprun.set_run_command(run_command)
    aprun.set_tasks(1)
    aprun.set_cpus_per_task(128)
    aprun.set_nodes(1)
    aprun.set_tasks_per_node(1)
    producer = experiment.create_model("eval", aprun)

    # create directories for the output files and copy
    # scripts to execution location inside newly created dir
    # only necessary if its not an executable (python is executable here)
    file_list = ["./eval/eval.py"]
    if 'ppo' in cfg.eval.training_name.lower():
        algo = 'ppo'
    elif 'sac' in cfg.eval.training_name.lower():
        algo = 'sac'
    else:
        raise ValueError(f"Could not determine algorithm.")
    
    file_list += [f"./{algo}/utils_{algo}.py"]  # SAC files
    file_list += ["./Solver/WF_enviroment.py", "./Solver/ADM_setup.py", "./Solver/farm.py"]  # Env and simulator
    file_list += [f"{cfg.eval.training_name}/{algo}/checkpoints/actor_{cfg.eval.model_id}.pkl"]
    file_list += [f"{cfg.eval.training_name}/{algo}/checkpoints/critic_{cfg.eval.model_id}.pkl"]
    file_list += [f"{cfg.eval.training_name}/{algo}/checkpoints/observation_{cfg.eval.model_id}.json"]
    producer.attach_generator_files(to_copy=file_list)

    experiment.generate(producer, overwrite=True)
    stage_probe_layout(producer.path, cfg.env)

    if not os.path.exists(f"./{experiment.name}/eval/"):
        os.makedirs(f"./{experiment.name}/eval/")
    OmegaConf.save(cfg, f"./{experiment.name}/eval/config_eval.yaml")

    return producer

if __name__ == '__main__':

    # Any arguments (space-separated) are taken to be modifiers for the config file
    # We assume that the arguments are valid modifiers for the config; this is not tested here.
    # An example of a valid modifier is "optim.gamma=0.9"
    config_modifiers = list(sys.argv[1:]) if len(sys.argv) > 1 else [""]

    with initialize(config_path="eval", version_base="1.2"):
        cfg_eval = compose(config_name="config_eval.yaml", overrides=config_modifiers)

    training_name = cfg_eval.eval.training_name
    # run_name = training_name.replace("training", "eval", 1)  # not good
    run_name = training_name + "_eval_" + datetime.now().strftime("%Y-%m-%d_%H-%M")
    dummy_update = cfg_eval.eval.dummy_update

    print(f"Launching eval run at {run_name}.")

    if 'ppo' in run_name.lower():
        with initialize(config_path=f"{training_name}/ppo/outputs/hydra_logs", version_base="1.2"):
            cfg_train = compose(config_name="config.yaml")
    elif 'sac' in run_name.lower():
        with initialize(config_path=f"{training_name}/sac/outputs/hydra_logs", version_base="1.2"):
            cfg_train = compose(config_name="config.yaml")
    else:
        raise Exception("Can not determine training algorithm. We expect to find the name of the algorithm in the directory name.")

    # Merge configurations
    cfg = {**cfg_eval, **cfg_train}
    cfg = OmegaConf.create(cfg)
    algo = "ppo" if "ppo" in training_name.lower() else "sac"
    saved_probes = os.path.abspath(os.path.join(training_name, algo, "probes.csv"))
    if not os.path.isfile(saved_probes):
        raise FileNotFoundError(f"Training probe snapshot missing: {saved_probes}; migrate legacy runs explicitly")
    cfg.env.probes_file = saved_probes
    probe_params(cfg.env)
    if not dummy_update:
        verified_solver(cfg.env)

    # Set up experiment
    exp = Experiment(run_name, launcher=LAUNCHER)

    # Runtime parameters
    n_environments = cfg.eval.n_parallel

    # Start database
    db_port = 6783
    db = launch_database(exp, db_port, dummy_update=dummy_update)

    # Start RL
    rl_app = launch_eval(exp, cfg, dummy_update=dummy_update)

    # Start simulations
    simulations = []
    if dummy_update:  # this is not how we want to decide whether to dummy update or not...
        simulation = launch_dummy_solver(exp)
        simulations.append(simulation)
    else:
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
