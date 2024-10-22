from smartsim import Experiment
import time
from hydra import initialize, compose
import os


def launch_database(experiment, port):
    db = experiment.create_database(port=port, db_nodes=1, interface='lo')

    # generate directories for output files
    # pass in objects to make dirs for
    experiment.generate(db, overwrite=True)

    # start the database on interactive allocation and wait until database is launched
    experiment.start(db, block=True)

    # get the status of the database
    statuses = experiment.get_status(db)
    print(f"Status of all database nodes: {statuses}.")
    print(f"Database started on {db.get_address()}.")

    return db

def launch_solver(experiment):
    aprun = experiment.create_run_settings(exe="python", exe_args="dummy_solver.py")
    aprun.set_tasks(1)
    producer = experiment.create_model("dummy_solver", aprun)

    # create directories for the output files and copy
    # scripts to execution location inside newly created dir
    # only necessary if its not an executable (python is executable here)
    producer.attach_generator_files(to_copy="./Solver/dummy_solver.py")

    experiment.generate(producer, overwrite=True)
    return producer

def launch_eval(experiment, cfg):
    aprun = experiment.create_run_settings(exe="python", exe_args="eval.py")
    aprun.set_tasks(1)
    producer = experiment.create_model("eval", aprun)

    # Copy relevant files
    file_list = ["./ppo/eval.py", "./ppo/utils_ppo.py", "./ppo/config_ppo.yaml"]  # PPO files
    file_list += ["./Solver/WF_enviroment.py", "./Solver/ADM_setup.py", "./Solver/farm.py"]  # Env and simulator
    file_list += [f"{cfg.eval.model_path}/actor_{cfg.eval.model_id}.pkl"]
    file_list += [f"{cfg.eval.model_path}/critic_{cfg.eval.model_id}.pkl"]
    producer.attach_generator_files(to_copy=file_list)

    experiment.generate(producer, overwrite=True)
    return producer


if __name__ == '__main__':
    # Read PPO config
    initialize(config_path="./ppo/", version_base="1.2")
    cfg = compose(config_name="config_ppo.yaml")

    exp = Experiment("eval_dummy_run", launcher="auto")

    total_runtime = 120  # seconds, without including setup of orchestrator etc.

    db_port = 6782
    db = launch_database(exp, db_port)

    solver_app = launch_solver(exp)
    exp.start(solver_app, block=False, summary=False)

    rl_app = launch_eval(exp, cfg)
    exp.start(rl_app, block=False, summary=False)

    # TO-DO: can we remove the RUNTIME parameter everywhere? it's rly inconvenient...

    # shutdown the database because we don't need it anymore
    time.sleep(total_runtime)
    exp.stop(db)

    print(exp.summary())
