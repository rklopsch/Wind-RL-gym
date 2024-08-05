import numpy as np
from smartsim import Experiment
import time
import os
from smartredis import Client
from Solver.ADM_runner import ADM
from Solver.farm import Farm, Turbine


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


def launch_solver(experiment, instance):
    os.environ['SR_DB_TYPE'] = "Standalone"  # visible in this process + all children
    os.environ['SSDB'] = "127.0.0.1:6783"  # visible in this process + all children
    os.environ['LD_LIBRARY_PATH'] = "/home/amole/Documents/Incompact3d/build/smartredis-build/smartredis/install/lib:" + os.environ.get('LD_LIBRARY_PATH', "")
    os.environ['PATH'] = os.environ['PATH'] + ":/home/amole/Documents/Incompact3d/build/bin"
    # TODO: probably (definitely) want a better way to set these

    aprun = experiment.create_run_settings(exe="xcompact3d", run_command="mpirun")
    aprun.set_tasks(2)
    aprun.set_cpus_per_task(1)
    # aprun.set_nodes(4)
    # aprun.set_tasks_per_node(25)
    producer = experiment.create_model(f"WindFarm_{instance}", aprun)
    files = ["./Solver/ADM/Base"]
    producer.attach_generator_files(to_copy=files)
    experiment.generate(producer, overwrite=True)

    # Configure case
    # Smartsims to_configure flag not working so doing manually with ADM_runner function
    # TODO: use the config to set the variables used here
    farm1 = Farm(126*14, 126*4, 3, Turbine(126, 90, yaw=0), offset=[2 * 126, 2*126])
    farm1.grid()
    case = ADM(farm1, 25, instance=instance)
    case.modify_input(f"./launch_run/WindFarm_{instance}")

    return producer


def launch_ppo(experiment):
    aprun = experiment.create_run_settings(exe="python", exe_args="ppo.py")
    aprun.set_tasks(1)
    producer = experiment.create_model("ppo", aprun)

    # create directories for the output files and copy
    # scripts to execution location inside newly created dir
    # only necessary if its not an executable (python is executable here) 
    producer.attach_generator_files(to_copy=["./ppo/ppo.py",
                                             "./ppo/utils_ppo.py",
                                             "./ppo/config_ppo.yaml",
                                             "./Solver/WF_enviroment.py",
                                             "./Solver/ADM_runner.py",
                                             "./Solver/farm.py"])

    experiment.generate(producer, overwrite=True)
    return producer


if __name__ == '__main__':
    exp = Experiment("launch_run", launcher="local")

    total_runtime = 200  # seconds, without including setup of orchestrator etc.
    n_environments = 2

    db_port = 6783
    db = launch_database(exp, db_port)

    # Start Simulations
    simulations = []
    for i in range(1, n_environments+1):
        simulation = launch_solver(exp, instance=i)
        simulations.append(simulation)
        exp.start(simulation, block=False, summary=False)


    # Start RL
    rl_app = launch_ppo(exp)
    exp.start(rl_app, block=False, summary=False)

    # shutdown the database because we don't need it anymore
    time.sleep(total_runtime)
    exp.stop(simulations, rl_app, db)
    print(exp.summary())
