"""Launch an open-loop evaluation of a trained SAC model (see eval/open_loop_eval.py).

Usage: python launch_open_loop_eval_run.py [hydra overrides for eval/config_open_loop_eval.yaml]
Example: python launch_open_loop_eval_run.py eval.training_name=training_sac eval.model_id=515 eval.num_rounds=4
"""

import os
import shutil
import sys
import time
from datetime import datetime

from hydra import compose, initialize
from omegaconf import OmegaConf
from smartsim import Experiment
from smartsim.status import SmartSimStatus

from launch_eval_run import LAUNCHER, PYTHON_BIN, RUN_COMMAND, launch_solver
from launch_sac_run import launch_database
from Solver.probe_layout import probe_params, stage_probe_layout
from Solver.solver_runtime import verified_solver

MODEL_NAME = "open_loop_eval"
DB_PORT = 6783
SOLVER_START_DELAY_S = 60
POLL_INTERVAL_S = 30
TERMINAL_STATUSES = (
    SmartSimStatus.STATUS_COMPLETED,
    SmartSimStatus.STATUS_FAILED,
    SmartSimStatus.STATUS_CANCELLED,
)


def load_config(config_modifiers):
    with initialize(config_path="eval", version_base="1.2"):
        cfg_eval = compose(config_name="config_open_loop_eval.yaml", overrides=config_modifiers)
    training_name = cfg_eval.eval.training_name
    if "sac" not in training_name.lower():
        raise ValueError(f"Open-loop evaluation supports SAC only; '{training_name}' does not look like a SAC run.")
    with initialize(config_path=f"{training_name}/sac/outputs/hydra_logs", version_base="1.2"):
        cfg_train = compose(config_name="config.yaml")

    cfg = OmegaConf.create({**cfg_eval, **cfg_train})
    if cfg.multi_agent.use:
        raise NotImplementedError("Open-loop evaluation supports single-agent SAC only.")
    cfg.env.dummy_update = bool(cfg.eval.dummy_update)
    saved_probes = os.path.abspath(os.path.join(training_name, "sac", "probes.csv"))
    if not os.path.isfile(saved_probes):
        raise FileNotFoundError(f"Training probe snapshot missing: {saved_probes}; migrate legacy runs explicitly")
    cfg.env.probes_file = saved_probes
    probe_params(cfg.env)
    return cfg


def solver_steps(cfg):
    """Solver time steps: SAC burn-in, then rollout, replay and constant episodes plus their resets per round."""
    reset_frames = int(cfg.env.reset_frames)
    burn_in = (1 + int(cfg.eval.initial_reset_frames) // reset_frames) * reset_frames
    rounds = 3 * int(cfg.eval.num_rounds) * (int(cfg.eval.episode_length) + reset_frames)
    guard = 1
    return int(cfg.env.steps_per_frame) * (burn_in + rounds + guard)


def launch_open_loop_eval(experiment, cfg):
    run_command = RUN_COMMAND if RUN_COMMAND and not cfg.eval.dummy_update else None
    kwargs = {"run_command": run_command} if run_command is not None else {}
    run_settings = experiment.create_run_settings(
        exe=PYTHON_BIN,
        exe_args=f"{MODEL_NAME}.py hydra.run.dir=outputs hydra.output_subdir=hydra_logs",
        **kwargs,
    )
    run_settings.set_tasks(1)
    run_settings.set_cpus_per_task(128)
    run_settings.set_nodes(1)
    run_settings.set_tasks_per_node(1)
    producer = experiment.create_model(MODEL_NAME, run_settings)

    files = [f"./eval/{MODEL_NAME}.py", "./sac/utils_sac.py"]
    files += ["./Solver/WF_enviroment.py", "./Solver/ADM_setup.py", "./Solver/farm.py"]
    producer.attach_generator_files(to_copy=files)
    experiment.generate(producer, overwrite=True)

    stage_probe_layout(producer.path, cfg.env)
    checkpoint_dir = os.path.join(cfg.eval.training_name, "sac", "checkpoints")
    models_dir = os.path.join(producer.path, "models")
    os.makedirs(models_dir, exist_ok=True)
    for name in ("actor", "critic"):
        shutil.copy2(os.path.join(checkpoint_dir, f"{name}_{cfg.eval.model_id}.pkl"), models_dir)
    shutil.copy2(os.path.join(checkpoint_dir, f"observation_{cfg.eval.model_id}.json"), models_dir)
    OmegaConf.save(cfg, os.path.join(producer.path, f"config_{MODEL_NAME}.yaml"))
    return producer


def wait_for_eval(experiment, rl_app, others):
    """Block until the evaluation finishes. Returns True only if it completed successfully."""
    while True:
        eval_status = experiment.get_status(rl_app)[0]
        if eval_status in TERMINAL_STATUSES:
            print(f"Open-loop evaluation finished with status {eval_status}.")
            return eval_status == SmartSimStatus.STATUS_COMPLETED
        ended = [entity.name for entity, status in zip(others, experiment.get_status(*others))
                 if status in TERMINAL_STATUSES]
        if ended:
            print(f"Stopping because these ended before the evaluation finished: {ended}.")
            return False
        time.sleep(POLL_INTERVAL_S)


def main(config_modifiers):
    cfg = load_config(config_modifiers)
    dummy_update = bool(cfg.eval.dummy_update)
    if not dummy_update:
        verified_solver(cfg.env)

    run_name = cfg.eval.run_name or (
        f"{cfg.eval.training_name}_open_loop_eval_{datetime.now().strftime('%Y-%m-%d_%H-%M')}"
    )
    print(f"Launching open-loop eval run at {run_name}.")
    exp = Experiment(run_name, launcher=LAUNCHER)

    db = launch_database(exp, DB_PORT)
    started_entities = [db]
    succeeded = False
    try:
        rl_app = launch_open_loop_eval(exp, cfg)
        simulations = []
        if not dummy_update:  # DummyTurbEnv does not talk to a solver
            steps = solver_steps(cfg)
            print(f"Each solver runs {steps} time steps.")
            for instance in range(1, int(cfg.eval.n_parallel) + 1):
                simulations.append(launch_solver(exp, instance=instance, cfg=cfg, simulation_steps=steps))

        exp.start(rl_app, block=False, summary=False)
        started_entities.append(rl_app)
        if simulations:
            time.sleep(SOLVER_START_DELAY_S)
            exp.start(*simulations, block=False, summary=False)
            started_entities.extend(simulations)
        succeeded = wait_for_eval(exp, rl_app, simulations + [db])
    finally:
        exp.stop(*started_entities)
        print(exp.summary())
    return succeeded


if __name__ == "__main__":
    sys.exit(0 if main(list(sys.argv[1:])) else 1)
