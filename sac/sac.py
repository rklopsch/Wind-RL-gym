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
from probe_layout import probe_params
from torch.nn import functional as F

from utils_sac import (
    ReplayBuffer,
    build_next_state,
    build_state,
    consume_collection_batches,
    load_sac_checkpoint,
    make_parallel_env,
    make_sa_sac_agent,
    log_metrics,
    persist_logs,
    save_sac_checkpoint,
    should_log_now,
    soft_update,
)


def configure_torch_threads(cfg):
    torch_threads = int(os.environ.get("SAC_TORCH_NUM_THREADS", getattr(cfg.optim, "torch_num_threads", 8)))
    torch_interop_threads = int(
        os.environ.get("SAC_TORCH_NUM_INTEROP_THREADS", getattr(cfg.optim, "torch_num_interop_threads", 1))
    )
    torch.set_num_threads(max(1, torch_threads))
    try:
        torch.set_num_interop_threads(max(1, torch_interop_threads))
    except RuntimeError:
        pass


def log_runtime_diagnostics(cfg):
    affinity_count = "n/a"
    if hasattr(os, "sched_getaffinity"):
        affinity_count = len(os.sched_getaffinity(0))
    env_worker_start_method = str(getattr(cfg.env, "worker_start_method", "spawn"))
    logging.info(
        "SAC runtime diagnostics: SLURM_CPUS_PER_TASK=%s SLURM_CPUS_ON_NODE=%s SLURM_JOB_CPUS_PER_NODE=%s affinity_cpus=%s torch_threads=%s torch_interop_threads=%s worker_start_method=%s worker_count=%s OMP_NUM_THREADS=%s MKL_NUM_THREADS=%s OPENBLAS_NUM_THREADS=%s NUMEXPR_NUM_THREADS=%s",
        os.environ.get("SLURM_CPUS_PER_TASK"),
        os.environ.get("SLURM_CPUS_ON_NODE"),
        os.environ.get("SLURM_JOB_CPUS_PER_NODE"),
        affinity_count,
        torch.get_num_threads(),
        torch.get_num_interop_threads(),
        env_worker_start_method,
        int(cfg.env.n_parallel),
        os.environ.get("OMP_NUM_THREADS"),
        os.environ.get("MKL_NUM_THREADS"),
        os.environ.get("OPENBLAS_NUM_THREADS"),
        os.environ.get("NUMEXPR_NUM_THREADS"),
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

    configure_torch_threads(cfg)
    total_frames = cfg.collector.total_frames
    frames_per_batch = cfg.collector.frames_per_batch
    updates_per_collection_batch = frames_per_batch * cfg.optim.step_mult

    hydra_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    results_dir = os.path.join(hydra_dir, "RESULTS")
    os.makedirs(results_dir, exist_ok=True)

    params = {
        "n_turbines": cfg.env.turbines,
        "n_procs": cfg.env.n_processors_per_env,
        "n_envs": cfg.env.n_parallel,
        **probe_params(cfg.env),
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
        "communication_timeout_s": float(getattr(cfg.env, "communication_timeout_s", 1200.0)),
        "communication_poll_interval_s": float(getattr(cfg.env, "communication_poll_interval_s", 0.1)),
        "communication_slow_log_s": float(getattr(cfg.env, "communication_slow_log_s", 30.0)),
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
    log_runtime_diagnostics(cfg)
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
    pending_collection_frames = collected_frames % frames_per_batch
    collection_batches_completed = collected_frames // frames_per_batch
    pending_batch_sampling_time = 0.0
    pending_rewards = []
    pending_episode_lengths = []
    steps_since_reset = 0

    while collected_frames < total_frames:
        step_sampling_time = time.time() - sampling_start

        state = build_state(current_obs).to(device)
        if collected_frames < init_random_frames:
            # Uniform random exploration before training starts (torchrl collector init_random_frames).
            action = actor.sample_uniform(state.shape[0])
        else:
            with torch.no_grad():
                action, _ = actor.sample(state, deterministic=False)

        env_step_start = time.time()
        next_obs, reward, terminated, truncated, infos = train_env.step(action)
        env_sampling_time = time.time() - env_step_start
        # Only true terminations stop bootstrapping; time-limit truncations are bootstrapped from the
        # final observation (the workers auto-reset, so next_obs holds the reset obs for finished envs).
        episode_end = torch.logical_or(terminated, truncated)
        next_state = build_next_state(next_obs, infos, episode_end)

        replay_buffer.add(state, action, reward, next_state, done=episode_end, terminated=terminated)
        if saving_buffer is not None:
            saving_buffer.add(state, action, reward, next_state, done=episode_end, terminated=terminated)

        previous_collected_frames = collected_frames
        collected_frames += action.shape[0]
        pending_batch_sampling_time += env_sampling_time
        steps_since_reset += 1
        current_obs = next_obs
        if bool(episode_end.any()):
            steps_since_reset = 0
        pending_rewards.append(reward.mean().item())
        pending_episode_lengths.append(float(steps_since_reset))

        if should_log_now(cfg, collected_frames, int(getattr(cfg.logger, "console_updates", 200)), previous_collected_frames):
            console_output = f"Frame {collected_frames}/{cfg.collector.total_frames}"
            time_passed = time.time() - train_start_time
            console_output += f" | {time_passed / 60:.0f} min" if time_passed / 60 > 1 else " | <1 min"
            logging.info(console_output)

        completed_batches, pending_collection_frames = consume_collection_batches(
            pending_collection_frames,
            action.shape[0],
            frames_per_batch,
        )
        if completed_batches <= 0:
            sampling_start = time.time()
            continue

        sampling_time_for_completed_batches = pending_batch_sampling_time
        avg_episode_reward = float(np.mean(pending_rewards)) if pending_rewards else np.nan
        avg_episode_length = float(np.mean(pending_episode_lengths)) if pending_episode_lengths else np.nan
        pending_batch_sampling_time = 0.0
        pending_rewards.clear()
        pending_episode_lengths.clear()

        previous_collection_batches = collection_batches_completed
        collection_batches_completed += completed_batches
        consumed_batch_frames = frames_per_batch * completed_batches
        log_info = {
            "train/episode_reward": avg_episode_reward,
            "train/episode_length": avg_episode_length,
            "train/collection_batches_completed": float(completed_batches),
            "train/collection_batch_frames": float(consumed_batch_frames),
            "train/sampling_time": sampling_time_for_completed_batches,
            "train/loop_overhead_time": step_sampling_time,
        }
        optimizer_time = 0.0
        updates_done = 0
        if collected_frames >= init_random_frames and replay_buffer.size >= cfg.optim.batch_size:
            training_start = time.time()
            updates_to_run = updates_per_collection_batch * completed_batches
            for _ in range(updates_to_run):
                batch = replay_buffer.sample(cfg.optim.batch_size, device=device)
                batch_state = batch["state"]
                batch_action = batch["action"]
                batch_reward = batch["reward"]
                batch_next_state = batch["next_state"]
                batch_terminated = batch["terminated"].float()

                alpha = log_alpha.exp()

                with torch.no_grad():
                    next_action, next_log_prob = actor.sample(batch_next_state, deterministic=False)
                    target_q1 = target_critic1(batch_next_state, next_action)
                    target_q2 = target_critic2(batch_next_state, next_action)
                    target_q = torch.min(target_q1, target_q2) - alpha * next_log_prob
                    target_q = batch_reward + (1.0 - batch_terminated) * cfg.optim.gamma * target_q

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

            updates_done = updates_to_run
            optimizer_time = time.time() - training_start
            log_info.update(
                {
                    "train/q_loss": critic_loss.item(),
                    "train/actor_loss": actor_loss.item(),
                    "train/alpha_loss": alpha_loss.item(),
                    "train/alpha": alpha.item(),
                    "train/entropy": (-log_prob).mean().item(),
                }
            )
        log_info["train/training_time"] = optimizer_time
        log_info["train/optimizer_updates"] = float(updates_done)
        total_batch_time = sampling_time_for_completed_batches + optimizer_time
        log_info["train/env_optimizer_fps"] = float(consumed_batch_frames) / max(total_batch_time, 1e-6)
        log_metrics(logs, log_info)

        logging.info(
            "Collection batches +%d (total %d) at frame %d: env_sampling=%.2fs optimizer=%.2fs updates=%d fps=%.2f",
            completed_batches,
            collection_batches_completed,
            collected_frames,
            sampling_time_for_completed_batches,
            optimizer_time,
            updates_done,
            log_info["train/env_optimizer_fps"],
        )

        if cfg.checkpoint.checkpoint_interval > 0 and (
            (collection_batches_completed // cfg.checkpoint.checkpoint_interval)
            > (previous_collection_batches // cfg.checkpoint.checkpoint_interval)
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
        persist_logs(logs)
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
    persist_logs(logs)
    train_env.close()

    end_time = time.time()
    logging.info(f"Training took {end_time - start_time:.2f} seconds to finish")


if __name__ == "__main__":
    main()
