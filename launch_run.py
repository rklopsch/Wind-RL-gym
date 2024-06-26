import numpy as np
from smartsim import Experiment
import time
from smartredis import Client


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
    aprun = experiment.create_run_settings(exe="xcompact3d")
    aprun.set_tasks(1)
    producer = experiment.create_model("xcompact3d", aprun)

    # create directories for the output files and copy
    # scripts to execution location inside newly created dir
    # only necessary if its not an executable (python is executable here)
    producer.attach_generator_files(to_copy="./Solver/ADM/Base/input.i3d")

    experiment.generate(producer, overwrite=True)
    return producer

def launch_ppo(experiment):
    aprun = experiment.create_run_settings(exe="python", exe_args="ppo.py")
    aprun.set_tasks(1)
    producer = experiment.create_model("ppo", aprun)

    # create directories for the output files and copy
    # scripts to execution location inside newly created dir
    # only necessary if its not an executable (python is executable here) 
    producer.attach_generator_files(to_copy=["./ppo/ppo.py", "./ppo/utils_ppo.py", "./ppo/config_ppo.yaml", "./Solver/WF_enviroment.py", "./Solver/ADM_runner.py", "./Solver/farm.py"])

    experiment.generate(producer, overwrite=True)
    return producer


if __name__ == '__main__':
    exp = Experiment("launch_dummy_run", launcher="local")

    total_runtime = 60  # seconds, without including setup of orchestrator etc.

    db_port = 6783
    db = launch_database(exp, db_port)

    print(f"Database setup... waiting now...")
    time.sleep(60)

    solver_app = launch_solver(exp)
    exp.start(solver_app, block=False, summary=False)

    rl_app = launch_ppo(exp)
    exp.start(rl_app, block=False, summary=False)

    # shutdown the database because we don't need it anymore
    time.sleep(total_runtime)
    exp.stop(db)

    print(exp.summary())
