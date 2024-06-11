# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
This script reproduces the Proximal Policy Optimization (PPO) Algorithm
results from Schulman et al. 2017 for the on MuJoCo Environments.
"""
import os
import sys
import time
import copy
import logging
import torch.optim
import tqdm
from tensordict import TensorDict
from torchrl.collectors import SyncDataCollector
from torchrl.data import LazyMemmapStorage, TensorDictReplayBuffer
from torchrl.data.replay_buffers.samplers import SamplerWithoutReplacement
from torchrl.envs import ExplorationType, set_exploration_type
from torchrl.objectives import ClipPPOLoss, ValueEstimators
from torchrl.objectives.value.advantages import GAE
from torchrl.record.loggers import generate_exp_name, get_logger
from utils_ppo import eval_model, make_env, make_parallel_env, make_ma_ppo_models, load_model
from utils.save_model import save_model
from omegaconf import OmegaConf
import wandb
import shutil
import hydra


@hydra.main(config_path="./", config_name="config_ppo", version_base="1.2")
def main(cfg: "DictConfig"):
    sys.path.append(os.getcwd())
    device = "cpu" if not torch.cuda.device_count() else "cuda"
    print(f'Running on {device}')
    print(f'cuda version:{torch.version.cuda}')

    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    # Correct for frame_skip
    frame_skip = cfg.collector.frame_skip
    total_frames = cfg.collector.total_frames // frame_skip
    frames_per_batch = cfg.collector.frames_per_batch // frame_skip
    max_episode_length = cfg.collector.max_episode_length // frame_skip
    mini_batch_size = cfg.loss.mini_batch_size // frame_skip
    test_interval = cfg.logger.test_interval // frame_skip
    n_environments = cfg.env.n_parallel

    dummy_update = cfg.env.dummy_update

    if not dummy_update:
        hydra_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
        results_dir = os.path.join(hydra_dir, 'RESULTS')
        os.makedirs(results_dir, exist_ok=True)

    params = {
        "n_turbines": cfg.env.turbines,
        "n_procs": cfg.env.n_processors_per_env,
        "n_envs": cfg.env.n_parallel,
        "probes_per_turbine": cfg.env.probes_per_turbine,
        "turbine_diameter": cfg.env.turbine_diameter,
        "turbine_spacing": cfg.env.turbine_spacing,
        "max_yaw_speed": cfg.env.max_yaw_speed,
        "max_yaw_angle": cfg.env.max_yaw_angle,
        "dt": cfg.env.steps_per_frame * 0.2,
        "run_steps": cfg.collector.max_episode_length * cfg.env.steps_per_frame,
    }

    test_params = copy.deepcopy(params)
    test_params["n_procs"]=cfg.env.n_processors_per_env*cfg.env.n_parallel
    test_params["n_envs"]=1

    # Create models
    if not cfg.optim.load_from_checkpoint:
        # Create a new model
        actor, critic = make_ma_ppo_models(params)
    else:
        # Load from specified checkpoint
        _, actor, critic = load_model(
            env_params=None,
            filepath=cfg.optim.model_checkpoint_path,
            id=cfg.optim.model_checkpoint_id,
            dummy_update=True)
    actor, critic = actor.to(device), critic.to(device)

    # Create environments
    train_env = make_parallel_env(params, n_environments, device=device, dummy_update=dummy_update)
    test_env = make_env(test_params, instance='TestEnv', save=True, device=device, dummy_update=dummy_update)
    test_env.eval()

    # Create collector
    collector = SyncDataCollector(
        train_env,
        policy=actor,
        frames_per_batch=frames_per_batch,
        total_frames=total_frames,
        device=device,
        storing_device=device,
        max_frames_per_traj=max_episode_length
    )

    # Create data buffer
    sampler = SamplerWithoutReplacement()
    data_buffer = TensorDictReplayBuffer(
        storage=LazyMemmapStorage(frames_per_batch),
        sampler=sampler,
        batch_size=mini_batch_size,
    )

    # Create replay buffer to remember entire history
    full_buffer = TensorDictReplayBuffer(
        storage=LazyMemmapStorage(total_frames),
    )

    # Create loss and adv modules
    loss_module = ClipPPOLoss(
        actor=actor,
        critic=critic,
        clip_epsilon=cfg.loss.clip_epsilon,
        loss_critic_type=cfg.loss.loss_critic_type,
        entropy_coef=cfg.loss.entropy_coef,
        critic_coef=cfg.loss.critic_coef,
        normalize_advantage=False,
    )
    loss_module.set_keys(  # We have to tell the loss where to find the keys
        reward=test_env.reward_key,
        action=test_env.action_key,
        sample_log_prob=("agents", "sample_log_prob"),
        value=("agents", "state_value"),
        # These last 2 keys will be expanded to match the reward shape
        done=("agents", "done"),
        terminated=("agents", "terminated"),
    )

    adv_module = GAE(
        gamma=cfg.loss.gamma,
        lmbda=cfg.loss.gae_lambda,
        value_network=critic,
        average_gae=False,
    )
    adv_module.set_keys(
        value=("agents", "state_value"),
        reward=test_env.reward_key,
    )

    # Create optimizers
    actor_optim = torch.optim.Adam(actor.parameters(), lr=cfg.optim.lr, eps=1e-5)
    critic_optim = torch.optim.Adam(critic.parameters(), lr=cfg.optim.lr, eps=1e-5)

    # Create logger
    exp_name = generate_exp_name("PPO_", cfg.env.env_name)
    if cfg.logger.project_name is None:
        raise ValueError("WandB project name must be specified in config.")
    wandb.init(
        mode=str(cfg.logger.mode),
        project=str(cfg.logger.project_name),
        entity=str(cfg.logger.team_name),
        name=exp_name,
        config=OmegaConf.to_container(cfg, resolve=True),
    )

    # Main loop
    collected_frames = 0
    num_network_updates = 0
    test_number = 0
    start_time = time.time()
    #pbar = tqdm.tqdm(total=total_frames)
    num_mini_batches = frames_per_batch // mini_batch_size
    total_network_updates = (
        (total_frames // frames_per_batch)
        * cfg.loss.ppo_epochs
        * num_mini_batches
        )

    sampling_start = time.time()

    # extract cfg variables
    cfg_loss_ppo_epochs = cfg.loss.ppo_epochs
    cfg_optim_anneal_lr = cfg.optim.anneal_lr
    cfg_optim_lr = cfg.optim.lr
    cfg_loss_anneal_clip_eps = cfg.loss.anneal_clip_epsilon
    cfg_loss_clip_epsilon = cfg.loss.clip_epsilon
    cfg_logger_num_test_episodes = cfg.logger.num_test_episodes
    cfg_logger_test_episode_length = cfg.logger.test_episode_length
    losses = TensorDict({}, batch_size=[cfg_loss_ppo_epochs, num_mini_batches])

    for i, data in enumerate(collector):

        log_info = {}
        sampling_time = time.time() - sampling_start
        frames_in_batch = data.numel()
        collected_frames += frames_in_batch * frame_skip
        #pbar.update(data.numel())
        logging.info(f"Training step {collected_frames}/{total_frames}.")

        data.set(
            ("next", "agents", "done"),
            data.get(("next", "done"))
            .unsqueeze(-1)
            .expand(data.get_item_shape(("next", test_env.reward_key))),
        )
        data.set(
            ("next", "agents", "terminated"),
            data.get(("next", "terminated"))
            .unsqueeze(-1)
            .expand(data.get_item_shape(("next", test_env.reward_key))),
        )
        data.set(
            ("next", "done"),
            data.get(("next", "done"))
            .unsqueeze(-1)
            .expand(data.get_item_shape(("next", test_env.reward_key))),
        )
        data.set(
            ("next", "terminated"),
            data.get(("next", "terminated"))
            .unsqueeze(-1)
            .expand(data.get_item_shape(("next", test_env.reward_key))),
        )
        # We need to expand the done and terminated to match the reward shape (this is expected by the value estimator)

        """
        # Get training rewards and episode lengths
        episode_rewards = data["next", "agents", "episode_reward"].sum(-2)
        episode_rewards = episode_rewards[data["next", "done"]]
        if len(episode_rewards) > 0:
            episode_length = data["next", "step_count"][data["next", "done"]]
            log_info.update(
                {
                    "train/reward": episode_rewards.mean().item(),
                    "train/episode_length": episode_length.sum().item()
                    / len(episode_length),
                }
            )
        """

        training_start = time.time()
        for j in range(cfg_loss_ppo_epochs):

            # Compute GAE
            with torch.no_grad():
                data = adv_module(data)
            data_reshape = data.reshape(-1)

            # Update the data buffers
            data_buffer.extend(data_reshape)
            full_buffer.extend(data_reshape)

            for k, batch in enumerate(data_buffer):

                # Get a data batch
                batch = batch.to(device)

                # Linearly decrease the learning rate and clip epsilon
                alpha = 1.0
                if cfg_optim_anneal_lr:
                    alpha = 1 - (num_network_updates / total_network_updates)
                    for group in actor_optim.param_groups:
                        group["lr"] = cfg_optim_lr * alpha
                    for group in critic_optim.param_groups:
                        group["lr"] = cfg_optim_lr * alpha
                if cfg_loss_anneal_clip_eps:
                    loss_module.clip_epsilon.copy_(cfg_loss_clip_epsilon * alpha)
                num_network_updates += 1

                # Forward pass PPO loss
                loss = loss_module(batch)
                losses[j, k] = loss.select(
                    "loss_critic", "loss_entropy", "loss_objective"
                ).detach()
                critic_loss = loss["loss_critic"]
                actor_loss = loss["loss_objective"] + loss["loss_entropy"]

                # Backward pass
                actor_loss.backward()
                critic_loss.backward()

                # Update the networks
                actor_optim.step()
                critic_optim.step()
                actor_optim.zero_grad()
                critic_optim.zero_grad()

        # Get training losses and times
        training_time = time.time() - training_start
        losses_mean = losses.apply(lambda x: x.float().mean(), batch_size=[])
        for key, value in losses_mean.items():
            log_info.update({f"train/{key}": value.item()})
        log_info.update(
            {
                "train/lr": alpha * cfg_optim_lr,
                "train/sampling_time": sampling_time,
                "train/training_time": training_time,
                "train/clip_epsilon": alpha * cfg_loss_clip_epsilon
                if cfg_loss_anneal_clip_eps
                else cfg_loss_clip_epsilon,
            }
        )

        # Get test rewards
        with torch.no_grad(), set_exploration_type(ExplorationType.MODE):
            if ((i - 1) * frames_in_batch * frame_skip) // test_interval < (
                i * frames_in_batch * frame_skip
            ) // test_interval:
                test_number += 1
                actor.eval()
                eval_start = time.time()

                test_rewards_mean, test_rewards_stdv, test_alpha_means, test_alpha_stdvs = eval_model(
                    actor, test_env, cfg.env.turbines,
                    num_episodes=cfg_logger_num_test_episodes,
                    episode_length=cfg_logger_test_episode_length
                )

                eval_time = time.time() - eval_start

                # Prepare to update log_info with dynamic alpha means and stdvs
                log_info.update({
                    "eval/reward_mean": test_rewards_mean,
                    "eval/reward_stdv": test_rewards_stdv,
                    "eval/time": eval_time,
                })

                # Dynamically update log_info for each turbine
                turbine_log = {}
                for idx, (mean, stdv) in enumerate(zip(test_alpha_means, test_alpha_stdvs), start=1):
                    turbine_log[f"eval/alpha_{idx}_mean"] = mean
                    turbine_log[f"eval/alpha_{idx}_stdv"] = stdv

                log_info.update(turbine_log)
                actor.train()

                # Copy LES data from evaluation into results directory in output
                if not dummy_update:
                    shutil.move('./LES_RUNS/TestEnv/Running/data', os.path.join(results_dir, f'TEST_{test_number}/data'))
                    for turb in range(test_env.n_turbs):
                        shutil.copy(f'./LES_RUNS/TestEnv/Running/disc{turb+1}.adm', os.path.join(results_dir, f'TEST_{test_number}'))

        if i % cfg.logger.checkpoint_interval == 0 or i == total_frames // frames_per_batch:
            output_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir + '/'
            full_buffer.dumps(output_dir + 'replay_buffer_checkpoint')
            logging.info(f"Checkpointed replay buffer. (Saved at {output_dir + 'replay_buffer_checkpoint'}).")
            save_model(test_env, actor, critic, output_dir, i)
            logging.info(f"Checkpointed model. (Saved at {output_dir}actor_{i}.pkl and {output_dir}critic_{i}.pkl")

        wandb.log(data=log_info, step=collected_frames)
        collector.update_policy_weights_()
        sampling_start = time.time()

    wandb.finish()
    end_time = time.time()
    execution_time = end_time - start_time
    print(f"Training took {execution_time:.2f} seconds to finish")


if __name__ == "__main__":
    main()
