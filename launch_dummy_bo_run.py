from smartsim import Experiment
import time
from hydra import initialize, compose
import sys


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

def launch_bo(experiment, load_params, config_modifiers):
    aprun = experiment.create_run_settings(exe="python", exe_args="bo.py " + " ".join(config_modifiers))
    aprun.set_tasks(1)
    producer = experiment.create_model("bo", aprun)

    # Copy relevant files
    file_list = ["./bo/bo.py", "./bo/utils_bo.py", "./bo/config_bo.yaml"]  # BO files
    file_list += ["./Solver/WF_enviroment.py", "./Solver/ADM_setup.py", "./Solver/farm.py"]  # Env and simulator
    producer.attach_generator_files(to_copy=file_list)

    experiment.generate(producer, overwrite=True)
    return producer


if __name__ == '__main__':
    # Read PPO config
    initialize(config_path="./sac/", version_base="1.2")
    cfg = compose(config_name="config_sac.yaml")

    # Load a checkpointed model?
    load_params = {
        'load_checkpoint': bool(cfg.checkpoint.load_from_checkpoint),
        'checkpoint_id': cfg.checkpoint.model_checkpoint_id,
        'checkpoint_path': cfg.checkpoint.model_checkpoint_path,
    }

    config_modifiers = list(sys.argv[1:]) if len(sys.argv) > 1 else [""]

    print("WARNING: The dummy solver mode is currently broken when using too many parallel envs. Since this is not really a relevant feature, it is recommended to use 5 parallel envs when using the dummy solver mode. This should be sufficient for all testing purposes.\n")

    exp = Experiment("launch_dummy_SAC_run", launcher="auto")

    total_runtime = 240  # seconds, without including setup of orchestrator etc.

    db_port = 6784
    db = launch_database(exp, db_port)

    solver_app = launch_solver(exp)
    exp.start(solver_app, block=False, summary=False)

    bo_app = launch_bo(exp, load_params, config_modifiers)
    exp.start(bo_app, block=False, summary=False)

    # shutdown the database because we don't need it anymore
    time.sleep(total_runtime)
    exp.stop(db)

    print(exp.summary())
