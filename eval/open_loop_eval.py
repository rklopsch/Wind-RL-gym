"""Open-loop evaluation of a trained single-agent SAC policy.

Each round runs three consecutive episodes in every parallel environment:

1. ``rollout``: the deterministic policy acts in closed loop and its actions are recorded.
2. ``replay``: the recorded actions are replayed open loop.
3. ``constant``: zero actions, with yaw held at the rollout's mean yaw angle.

Episodes are separated by the same automatic ``reset_frames`` reset as in SAC training. Rollout and replay
always start at zero yaw; only the reset before the constant episode targets the rollout's mean yaw.
"""

import logging
import os
import pickle
import sys
import time

import hydra
import numpy as np
import torch
from probe_layout import probe_params

from utils_sac import build_state, load_model, make_parallel_env

PHASES = ("rollout", "replay", "constant")
START_YAW_TOLERANCE = 1e-4


def make_env_params(cfg):
    episode_length = int(cfg.eval.episode_length)
    return {
        "n_turbines": cfg.env.turbines,
        "n_procs": cfg.env.n_processors_per_env,
        "n_envs": cfg.eval.n_parallel,
        **probe_params(cfg.env),
        "turbine_diameter": cfg.env.turbine_diameter,
        "turbine_spacing": cfg.env.turbine_spacing,
        "max_yaw_speed": cfg.env.max_yaw_speed,
        "max_yaw_angle": cfg.env.max_yaw_angle,
        "dt": cfg.env.steps_per_frame * 0.2,
        "reset_frames": cfg.env.reset_frames,
        "run_steps": episode_length * cfg.env.steps_per_frame,
        "penalty_scale": cfg.env.penalty_scale,
        "penalty_exp": cfg.env.penalty_exp,
        # Rollout and replay always start from zero yaw, as in training.
        "random_reset": False,
        "initial_angles": [0.0] * int(cfg.env.turbines),
        "reward_average_steps": cfg.env.reward_average_steps,
        "velocity_penalty_scale": cfg.env.velocity_penalty_scale,
        "difference_penalty_scale": cfg.env.difference_penalty_scale,
        "episode_length": episode_length,
        "communication_timeout_s": float(getattr(cfg.env, "communication_timeout_s", 1200.0)),
        "communication_poll_interval_s": float(getattr(cfg.env, "communication_poll_interval_s", 0.1)),
        "communication_slow_log_s": float(getattr(cfg.env, "communication_slow_log_s", 30.0)),
    }


def burn_in_frames(initial_reset_frames, reset_frames):
    """Control frames spent by the initial burn-in, identical to SAC training."""
    return (1 + int(initial_reset_frames) // int(reset_frames)) * int(reset_frames)


def required_control_frames(initial_reset_frames, reset_frames, episode_length, num_rounds):
    """Control frames each solver must run: burn-in, then three episodes plus automatic reset per round."""
    episode_frames = int(episode_length) + int(reset_frames)
    return burn_in_frames(initial_reset_frames, reset_frames) + 3 * int(num_rounds) * episode_frames


def phase_start_frame(burn_in, episode_length, reset_frames, round_idx, phase_idx):
    episode_frames = int(episode_length) + int(reset_frames)
    return int(burn_in) + (3 * int(round_idx) + int(phase_idx)) * episode_frames


def initial_reset(env, initial_reset_frames, reset_frames):
    obs, _ = env.reset()
    num_resets = int(initial_reset_frames) // int(reset_frames)
    for i in range(num_resets):
        obs, _ = env.reset()
        logging.info("%.0f%% done with initial reset.", 100 * (i + 1) / num_resets)
    return obs


def _validated_step_info(obs, infos, last_step, n_turbines, max_angle, step):
    alphas, mean_powers = [], []
    for env_idx, info in enumerate(infos):
        if last_step:
            observation = info.get("final_observation")
            if observation is None:
                raise RuntimeError(f"Environment {env_idx + 1} returned no final observation at step {step}.")
        else:
            observation = {key: obs[key][env_idx] for key in ("alpha", "power")}

        values = {}
        for key in ("alpha", "power"):
            reported = np.asarray(info[key], dtype=np.float32)
            observed = np.asarray(observation[key], dtype=np.float32)
            if reported.shape != (n_turbines,) or observed.shape != (n_turbines,):
                raise RuntimeError(f"Environment {env_idx + 1} returned an invalid {key} shape at step {step}.")
            if not np.isfinite(reported).all() or not np.isfinite(observed).all():
                raise RuntimeError(f"Environment {env_idx + 1} returned non-finite {key} at step {step}.")
            if not np.array_equal(reported, observed):
                raise RuntimeError(
                    f"Environment {env_idx + 1} {key} info differs from its observation at step {step}; "
                    "the transferred data may be corrupted."
                )
            values[key] = reported

        if np.any(np.abs(values["alpha"]) > max_angle + START_YAW_TOLERANCE):
            raise RuntimeError(f"Environment {env_idx + 1} returned out-of-range yaw at step {step}.")
        if not np.all(values["power"] == values["power"][0]):
            raise RuntimeError(f"Environment {env_idx + 1} returned inconsistent mean turbine power at step {step}.")
        alphas.append(values["alpha"])
        # Mean turbine power is broadcast to all turbines.
        mean_powers.append(values["power"][0])
    return np.stack(alphas), np.asarray(mean_powers, dtype=np.float32)


def run_episode(env, obs, episode_length, policy, max_angle, reset_options=None):
    """Run one episode of ``episode_length`` steps in every env.

    ``policy(step, obs)`` returns actions of shape ``[num_envs, n_turbines]``. ``reset_options`` is used for
    the automatic reset at the end of the episode. Returns the post-reset observation and the episode record.
    """
    start_alpha = np.asarray(obs["alpha"], dtype=np.float32).copy()
    n_turbines = start_alpha.shape[-1]
    actions, alphas, turbine_powers, rewards = [], [], [], []
    for step in range(episode_length):
        action = torch.as_tensor(policy(step, obs), dtype=torch.float32)
        last_step = step == episode_length - 1
        obs, reward, terminated, truncated, infos = env.step(
            action, reset_options=reset_options if last_step else None
        )
        episode_end = torch.logical_or(terminated, truncated).reshape(-1)
        if bool(episode_end.any()) and not (last_step and bool(episode_end.all())):
            raise RuntimeError(f"Unexpected episode end at step {step + 1}/{episode_length}.")
        if last_step and not bool(episode_end.all()):
            raise RuntimeError(f"Episodes did not end after {episode_length} steps; check env.episode_length.")

        if len(infos) != start_alpha.shape[0]:
            raise RuntimeError(f"Expected {start_alpha.shape[0]} environment results, got {len(infos)}.")
        step_alphas, step_powers = _validated_step_info(
            obs, infos, last_step, n_turbines, max_angle, step + 1
        )
        actions.append(action.detach().cpu().numpy().copy())
        alphas.append(step_alphas)
        turbine_powers.append(step_powers)
        rewards.append(reward.reshape(-1).detach().cpu().numpy().copy())

    mean_turbine_power = np.stack(turbine_powers, axis=1).astype(np.float32)
    farm_power = mean_turbine_power * n_turbines
    record = {
        "start_alpha": start_alpha,
        "actions": np.stack(actions, axis=1),
        "alpha": np.stack(alphas, axis=1),
        "mean_turbine_power": mean_turbine_power,
        "farm_power": farm_power,
        "mean_farm_power": farm_power.mean(axis=1),
        "reward": np.stack(rewards, axis=1),
    }
    return obs, record


def _check_start_yaw(phase, start_alpha, expected):
    error = float(np.max(np.abs(start_alpha - expected)))
    if not np.isfinite(error) or error > START_YAW_TOLERANCE:
        raise RuntimeError(f"{phase} episode started at yaw {start_alpha.tolist()}, expected {expected.tolist()}.")


def run_round(env, actor, obs, episode_length, max_angle, actions_path=None):
    """Run rollout, open-loop replay and constant-mean-yaw episodes. Returns the next obs and the round record."""

    def policy_action(_step, current_obs):
        action, _log_prob = actor.sample(build_state(current_obs), deterministic=True)
        return action

    def start_yaw(current_obs):
        return np.asarray(current_obs["alpha"], dtype=np.float32)

    zero_yaw = np.zeros_like(start_yaw(obs))
    _check_start_yaw("rollout", start_yaw(obs), zero_yaw)
    with torch.no_grad():
        obs, rollout = run_episode(env, obs, episode_length, policy_action, max_angle)
    if actions_path is not None:
        np.save(actions_path, rollout["actions"])

    mean_yaw = np.clip(rollout["alpha"].mean(axis=1), -max_angle, max_angle).astype(np.float32)
    constant_reset = [{"initial_angles": yaw.tolist()} for yaw in mean_yaw]
    recorded_actions = torch.as_tensor(rollout["actions"])
    _check_start_yaw("replay", start_yaw(obs), zero_yaw)
    obs, replay = run_episode(
        env, obs, episode_length, lambda step, _obs: recorded_actions[:, step], max_angle,
        reset_options=constant_reset,
    )

    zero_actions = torch.zeros_like(recorded_actions[:, 0])
    _check_start_yaw("constant", start_yaw(obs), mean_yaw)
    obs, constant = run_episode(env, obs, episode_length, lambda *_: zero_actions, max_angle)

    record = {
        "mean_yaw": mean_yaw,
        "rollout": rollout,
        "replay": replay,
        "constant": constant,
        "replay_yaw_max_error": float(np.max(np.abs(replay["alpha"] - rollout["alpha"]))),
        "constant_yaw_max_error": float(np.max(np.abs(constant["alpha"] - mean_yaw[:, None, :]))),
    }
    return obs, record


def log_round(round_idx, num_rounds, record):
    powers = {phase: record[phase]["mean_farm_power"] for phase in PHASES}
    logging.info(
        "Round %d/%d mean farm power [MW]: rollout=%.4f replay=%.4f constant=%.4f",
        round_idx + 1,
        num_rounds,
        *(float(powers[phase].mean()) for phase in PHASES),
    )
    for env_idx in range(len(powers["rollout"])):
        logging.info(
            "  env %d farm power [MW]: rollout=%.4f replay=%.4f constant=%.4f mean_yaw=[%s] deg",
            env_idx + 1,
            *(float(powers[phase][env_idx]) for phase in PHASES),
            ", ".join(f"{yaw:.2f}" for yaw in record["mean_yaw"][env_idx]),
        )
    for name in ("replay_yaw_max_error", "constant_yaw_max_error"):
        if record[name] > START_YAW_TOLERANCE:
            logging.warning("Round %d %s=%.3g degrees.", round_idx + 1, name, record[name])


def persist_eval_logs(logs, output_dir):
    path = os.path.join(output_dir, "eval_logs.pkl")
    tmp_path = path + ".tmp"
    with open(tmp_path, "wb") as f:
        pickle.dump(logs, f)
    os.replace(tmp_path, path)


@hydra.main(version_base="1.2", config_path="./", config_name="config_open_loop_eval")
def main(cfg: "DictConfig"):
    logging_stream = sys.stdout if cfg.logger.logging_stream == "stdout" else None
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s", stream=logging_stream)
    if cfg.multi_agent.use:
        raise NotImplementedError("Open-loop evaluation supports single-agent SAC only.")
    torch.set_num_threads(max(1, int(os.environ.get("SAC_TORCH_NUM_THREADS", 8))))

    output_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    results_dir = os.path.join(output_dir, "RESULTS")
    os.makedirs(results_dir, exist_ok=True)

    params = make_env_params(cfg)
    episode_length = params["episode_length"]
    reset_frames = int(cfg.env.reset_frames)
    num_rounds = int(cfg.eval.num_rounds)
    burn_in = burn_in_frames(cfg.eval.initial_reset_frames, reset_frames)

    actor, _ = load_model(cfg=cfg, env_params=params, path_to_model="models", id=cfg.eval.model_id)
    actor.eval()
    logging.info("Loaded actor_%s.pkl from %s.", cfg.eval.model_id, cfg.eval.training_name)

    logs = {
        "metadata": {
            "training_name": cfg.eval.training_name,
            "model_id": cfg.eval.model_id,
            "n_parallel": int(cfg.eval.n_parallel),
            "n_turbines": int(cfg.env.turbines),
            "episode_length": episode_length,
            "reset_frames": reset_frames,
            "steps_per_frame": int(cfg.env.steps_per_frame),
            "burn_in_frames": burn_in,
            "num_rounds": num_rounds,
            "power_units": "MW",
        },
        "rounds": [],
    }

    env = make_parallel_env(cfg, params, int(cfg.eval.n_parallel))
    try:
        logging.info("Initial reset: collecting %d frames.", burn_in)
        obs = initial_reset(env, cfg.eval.initial_reset_frames, reset_frames)
        start_time = time.time()
        for round_idx in range(num_rounds):
            actions_path = os.path.join(results_dir, f"round_{round_idx + 1:04d}_actions.npy")
            obs, record = run_round(env, actor, obs, episode_length, float(cfg.env.max_yaw_angle), actions_path)
            record["round"] = round_idx + 1
            for phase_idx, phase in enumerate(PHASES):
                record[phase]["control_frame_start"] = phase_start_frame(
                    burn_in, episode_length, reset_frames, round_idx, phase_idx
                )
            logs["rounds"].append(record)
            persist_eval_logs(logs, output_dir)
            log_round(round_idx, num_rounds, record)
        logging.info("Open-loop evaluation took %.2f seconds.", time.time() - start_time)
    finally:
        env.close()


if __name__ == "__main__":
    main()
