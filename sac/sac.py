import logging
import math
import os
import shutil
import sys
import time
from copy import deepcopy

import hydra
import numpy as np
import torch
from torch.nn import functional as F

from utils_sac import (
    ReplayBuffer,
    build_state,
    load_sac_checkpoint,
    make_parallel_env,
    make_sa_sac_agent,
    log_metrics,
    save_sac_checkpoint,
    should_log_now,
    soft_update,
)


@hydra.main(version_base="1.2", config_path="./", config_name="config_sac")
def main(cfg: "DictConfig"):
    device = "cpu"
    logging.info(f"Running on device: {device}.")

    logging_stream = sys.stdout if cfg.logger.logging_stream == "stdout" else None
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        stream=logging_stream,
    )

    if cfg.multi_agent.use:
        raise NotImplementedError("The plain-PyTorch SAC refactor currently supports single-agent mode only.")

    total_frames = cfg.collector.total_frames
    frames_per_batch = cfg.collector.frames_per_batch
    num_updates = frames_per_batch * cfg.optim.step_mult

    hydra_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    results_dir = os.path.join(hydra_dir, "RESULTS")
    os.makedirs(results_dir, exist_ok=True)

    params = {
        "n_turbines": cfg.env.turbines,
        "n_procs": cfg.env.n_processors_per_env,
        "n_envs": cfg.env.n_parallel,
        "probes_per_turbine": cfg.env.probes_per_turbine,
        "flow_field_directions": cfg.env.flow_field_directions,
        "turbine_diameter": cfg.env.turbine_diameter,
        "turbine_spacing": cfg.env.turbine_spacing,
        "max_yaw_speed": cfg.env.max_yaw_speed,
        "max_yaw_angle": cfg.env.max_yaw_angle,
        "dt": cfg.env.steps_per_frame * 0.2,
        "reset_frames": cfg.env.reset_frames,
        "run_steps": cfg.collector.max_episode_length * cfg.env.steps_per_frame,
        "penalty_scale": cfg.env.penalty_scale,
        "penalty_exp": cfg.env.penalty_exp,
        "random_reset": cfg.env.random_reset,
        "initial_angles": cfg.env.initial_angles,
        "reward_average_steps": cfg.env.reward_average_steps,
        "velocity_penalty_scale": cfg.env.velocity_penalty_scale,
        "difference_penalty_scale": cfg.env.difference_penalty_scale,
        "episode_length": cfg.collector.max_episode_length,
    }

    logging.info("Creating models")
    if cfg.checkpoint.load_from_checkpoint:
        checkpoint_dir = cfg.checkpoint.model_checkpoint_path
        actor, critic1, critic2, log_alpha = load_sac_checkpoint(
            cfg=cfg,
            env_params=params,
            path_to_model=checkpoint_dir,
            id=cfg.checkpoint.model_checkpoint_id,
        )
        collected_frames = cfg.checkpoint.model_checkpoint_id
        logging.info(f"Loaded checkpoint from frame {collected_frames}.")
    else:
        actor, critic1, critic2 = make_sa_sac_agent(cfg, params)
        log_alpha = torch.nn.Parameter(torch.tensor(math.log(cfg.optim.alpha_init), dtype=torch.float32))
        collected_frames = 0
        logging.info("Created fresh actor and critics.")

    actor, critic1, critic2 = actor.to(device), critic1.to(device), critic2.to(device)
    log_alpha = log_alpha.to(device)
    log_alpha.requires_grad_(True)

    target_critic1 = deepcopy(critic1).to(device)
    target_critic2 = deepcopy(critic2).to(device)
    target_critic1.eval()
    target_critic2.eval()
    for param in target_critic1.parameters():
        param.requires_grad_(False)
    for param in target_critic2.parameters():
        param.requires_grad_(False)

    optimizer_actor = torch.optim.Adam(
        actor.parameters(),
        lr=float(cfg.optim.lr) / cfg.optim.step_mult,
        weight_decay=cfg.optim.weight_decay,
        eps=cfg.optim.adam_eps,
    )
    optimizer_critic = torch.optim.Adam(
        list(critic1.parameters()) + list(critic2.parameters()),
        lr=float(cfg.optim.lr) / cfg.optim.step_mult,
        weight_decay=cfg.optim.weight_decay,
        eps=cfg.optim.adam_eps,
    )
    optimizer_alpha = torch.optim.Adam([log_alpha], lr=float(cfg.optim.entropy_lr) / cfg.optim.step_mult)

    train_env = make_parallel_env(cfg, params, cfg.env.n_parallel, device=device)
    replay_buffer = ReplayBuffer(
        capacity=cfg.replay_buffer.size,
        state_dim=actor.state_dim,
        action_dim=actor.action_dim,
    )
    if cfg.checkpoint.save_replay_buffer:
        saving_buffer = ReplayBuffer(
            capacity=math.ceil(cfg.collector.total_frames / cfg.env.n_parallel),
            state_dim=replay_buffer.state.shape[-1],
            action_dim=replay_buffer.action.shape[-1],
        )
    else:
        saving_buffer = None

    init_random_frames = cfg.collector.init_random_frames
    target_entropy = -float(replay_buffer.action.shape[-1])

    logging.info(
        f"Initial reset: collecting {(1 + cfg.env.initial_reset_frames // cfg.env.reset_frames) * cfg.env.reset_frames} frames."
    )
    current_obs, _ = train_env.reset()
    for i in range(cfg.env.initial_reset_frames // cfg.env.reset_frames):
        current_obs, _ = train_env.reset()
        logging.info(f"{100 * i / max(1, (cfg.env.initial_reset_frames // cfg.env.reset_frames))}% done with initial reset.")
    logging.info("100% done with initial reset.")

    logging.info("Starting training...")
    start_time = time.time()
    sampling_start = time.time()
    train_start_time = sampling_start
    logs = {}
    frames_since_update = 0
    steps_since_reset = 0

    while collected_frames < total_frames:
        log_info = {}
        sampling_time = time.time() - sampling_start

        state = build_state(current_obs).to(device)
        with torch.no_grad():
            action, _ = actor.sample(state, deterministic=False)

        next_obs, reward, terminated, truncated, infos = train_env.step(action)
        done = torch.logical_or(terminated, truncated)
        next_state = build_state(next_obs)

        replay_buffer.add(state, action, reward, next_state, done)
        if saving_buffer is not None:
            saving_buffer.add(state, action, reward, next_state, done)

        collected_frames += action.shape[0]
        frames_since_update += action.shape[0]
        steps_since_reset += 1
        current_obs = next_obs
        if bool(done.any()):
            steps_since_reset = 0

        if should_log_now(cfg, collected_frames, 1000):
            console_output = f"Frame {collected_frames}/{cfg.collector.total_frames}"
            time_passed = time.time() - train_start_time
            console_output += f" | {time_passed / 60:.0f} min" if time_passed / 60 > 1 else " | <1 min"
            logging.info(console_output)

        training_start = time.time()
        if collected_frames >= init_random_frames and replay_buffer.size >= cfg.optim.batch_size:
            for _ in range(num_updates):
                batch = replay_buffer.sample(cfg.optim.batch_size, device=device)
                batch_state = batch["state"]
                batch_action = batch["action"]
                batch_reward = batch["reward"]
                batch_next_state = batch["next_state"]
                batch_done = batch["done"].float()

                alpha = log_alpha.exp()

                with torch.no_grad():
                    next_action, next_log_prob = actor.sample(batch_next_state, deterministic=False)
                    target_q1 = target_critic1(batch_next_state, next_action)
                    target_q2 = target_critic2(batch_next_state, next_action)
                    target_q = torch.min(target_q1, target_q2) - alpha * next_log_prob
                    target_q = batch_reward + (1.0 - batch_done) * cfg.optim.gamma * target_q

                q1 = critic1(batch_state, batch_action)
                q2 = critic2(batch_state, batch_action)
                critic_loss = F.mse_loss(q1, target_q) + F.mse_loss(q2, target_q)

                for param in critic1.parameters():
                    param.requires_grad_(False)
                for param in critic2.parameters():
                    param.requires_grad_(False)

                new_action, log_prob = actor.sample(batch_state, deterministic=False)
                q1_new = critic1(batch_state, new_action)
                q2_new = critic2(batch_state, new_action)
                actor_loss = (alpha * log_prob - torch.min(q1_new, q2_new)).mean()
                alpha_loss = -(log_alpha * (log_prob + target_entropy).detach()).mean()
                for param in critic1.parameters():
                    param.requires_grad_(True)
                for param in critic2.parameters():
                    param.requires_grad_(True)

                optimizer_actor.zero_grad()
                actor_loss.backward()
                optimizer_actor.step()

                optimizer_critic.zero_grad()
                critic_loss.backward()
                optimizer_critic.step()

                optimizer_alpha.zero_grad()
                alpha_loss.backward()
                optimizer_alpha.step()

                soft_update(target_critic1, critic1, cfg.optim.target_update_polyak)
                soft_update(target_critic2, critic2, cfg.optim.target_update_polyak)

            log_info.update(
                {
                    "train/q_loss": critic_loss.item(),
                    "train/actor_loss": actor_loss.item(),
                    "train/alpha_loss": alpha_loss.item(),
                    "train/alpha": alpha.item(),
                    "train/entropy": (-log_prob).mean().item(),
                    "train/sampling_time": sampling_time,
                    "train/training_time": time.time() - training_start,
                }
            )

        if frames_since_update >= frames_per_batch:
            frames_since_update = 0
            if cfg.checkpoint.checkpoint_interval > 0 and (
                collected_frames % (frames_per_batch * cfg.checkpoint.checkpoint_interval) == 0
                or collected_frames >= total_frames
            ):
                output_dir = os.path.join(os.getcwd(), "checkpoints")
                save_sac_checkpoint(
                    cfg=cfg,
                    actor=actor,
                    critic1=critic1,
                    critic2=critic2,
                    log_alpha=log_alpha,
                    filepath=output_dir,
                    id=collected_frames,
                    replay_buffer=saving_buffer if cfg.checkpoint.save_replay_buffer else None,
                )
                logging.info(f"Checkpointed model at frame {collected_frames}.")
                if saving_buffer is not None:
                    logging.info(f"Checkpointed replay buffer at {output_dir}.")

        log_info.update(
            {
                "train/episode_reward": reward.mean().item(),
                "train/episode_length": float(steps_since_reset),
            }
        )

        log_metrics(logs, log_info)
        sampling_start = time.time()

    output_dir = os.path.join(os.getcwd(), "checkpoints")
    save_sac_checkpoint(
        cfg=cfg,
        actor=actor,
        critic1=critic1,
        critic2=critic2,
        log_alpha=log_alpha,
        filepath=output_dir,
        id=collected_frames,
        replay_buffer=saving_buffer if cfg.checkpoint.save_replay_buffer else None,
    )
    train_env.close()

    end_time = time.time()
    logging.info(f"Training took {end_time - start_time:.2f} seconds to finish")


if __name__ == "__main__":
    main()
