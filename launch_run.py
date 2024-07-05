import numpy as np
from smartsim import Experiment
import time
import os
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
    os.environ['SR_DB_TYPE'] = "Standalone" # visible in this process + all children
    os.environ['SSDB'] = "127.0.0.1:6783" # visible in this process + all children
    os.environ['LD_LIBRARY_PATH'] = "/home/eidf079/eidf079/amole-ai4nz/XcompactSmartRedis/Incompact3d/build/smartredis-build/smartredis/install/lib:" + os.environ.get('LD_LIBRARY_PATH', "")
    os.environ['PATH'] = os.environ['PATH'] + ":/home/eidf079/eidf079/amole-ai4nz/XcompactSmartRedis/Incompact3d/build/bin"

    aprun = experiment.create_run_settings(exe="xcompact3d")
    aprun.set_tasks(1)
    producer = experiment.create_model("WindFarm", aprun)

    # create directories for the output files and copy
    # scripts to execution location inside newly created dir
    # only necessary if its not an executable (python is executable here)
    files = ["./Solver/ADM/Base"]
    producer.attach_generator_files(to_copy=files)

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
    exp = Experiment("launch_run", launcher="local")

    total_runtime = 200  # seconds, without including setup of orchestrator etc.

    db_port = 6783
    db = launch_database(exp, db_port)

    solver_app = launch_solver(exp)
    exp.start(solver_app, block=False, summary=False)

    rl_app = launch_ppo(exp)
    exp.start(rl_app, block=False, summary=False)

    # shutdown the database because we don't need it anymore
    time.sleep(total_runtime)
    exp.stop(solver_app, rl_app, db)
    print(exp.summary())
