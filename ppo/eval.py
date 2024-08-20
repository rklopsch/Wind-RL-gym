import os
import sys
sys.path.append(os.getcwd())
import time
import copy
import logging
import torch.optim
import tqdm
from tensordict import TensorDict
from torchrl.collectors import SyncDataCollector
from torchrl.data import LazyMemmapStorage, TensorDictReplayBuffer
from torchrl.data.replay_buffers.samplers import SamplerWithoutReplacement
from torchrl.objectives import ClipPPOLoss
from torchrl.objectives.value.advantages import GAE
from torchrl.record.loggers import generate_exp_name
from utils_ppo import make_parallel_env, make_ma_ppo_models, load_model, save_model, log_metrics
import shutil
import hydra
import numpy as np
import shutil


@hydra.main(config_path="./", config_name="config_ppo", version_base="1.2")
def main(cfg: "DictConfig"):
    device = "cpu"  # Run on CPU only
    logging.info(f'Running on device: {device}.')

    logging_stream = sys.stdout if cfg.logger.logging_stream == 'stdout' else None
    logging.basicConfig(
        level=logging.INFO, 
        format='%(asctime)s - %(levelname)s - %(message)s', 
        stream=logging_stream,
    )

    max_episode_length = cfg.eval.episode_length
    n_environments = cfg.eval.n_parallel
    dummy_update = cfg.env.dummy_update

    if not dummy_update:
        hydra_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
        results_dir = os.path.join(hydra_dir, 'RESULTS')
        os.makedirs(results_dir, exist_ok=True)

    # TO-DO: why does params have n_envs as a parameter? This is passed explicitly wherever necessary

    params = {
        "n_turbines": cfg.env.turbines,
        "n_procs": cfg.env.n_processors_per_env,
        "n_envs": cfg.eval.n_parallel,
        "probes_per_turbine": cfg.env.probes_per_turbine,
        "turbine_diameter": cfg.env.turbine_diameter,
        "turbine_spacing": cfg.env.turbine_spacing,
        "max_yaw_speed": cfg.env.max_yaw_speed,
        "max_yaw_angle": cfg.env.max_yaw_angle,
        "dt": cfg.env.steps_per_frame * 0.2,
        "reset_frames": cfg.env.reset_frames,
        "run_steps": cfg.collector.max_episode_length * cfg.env.steps_per_frame,
    }

    # Load the models to be evaluated
    # Copy the loaded models into the ppo/eval directory
    if not os.path.exists('eval'):
        os.mkdir('eval')
    for arch in ['actor', 'critic']:
        filename = f"{arch}_{cfg.eval.model_id}.pkl"
        if os.path.exists(filename):
            shutil.move(filename, "eval")
    # Load actor and critic
    actor, critic = load_model(
        env_params=params,
        id=cfg.eval.model_id,
        dummy_update=dummy_update)
    logging.info(f"Loaded actor_{cfg.eval.model_id}.pkl and critic_{cfg.eval.model_id}.pkl.")
    actor, critic = actor.to(device), critic.to(device)

    # Create environments
    eval_env = make_parallel_env(
        params,
        n_environments,
        device=device,
        dummy_update=dummy_update,
        eval_only=True,
    )

    # Create replay buffer to remember entire history
    full_buffer = TensorDictReplayBuffer(
        storage=LazyMemmapStorage(total_frames),
    )

    # Main loop
    collected_frames = 0 if not cfg.checkpoint.load_from_checkpoint else cfg.checkpoint.model_checkpoint_id
    start_time = time.time()
    sampling_start = time.time()

    # Set up empty dict for logging
    logs = {}

    for i, data in enumerate(collector):
        log_info = {}
        sampling_time = time.time() - sampling_start
        frames_in_batch = data.numel()
        collected_frames += frames_in_batch
        #pbar.update(data.numel())
        logging.info(f"Training step {collected_frames}/{total_frames}.")

        data.set(
            ("next", "agents", "done"),
            data.get(("next", "done"))
            .unsqueeze(-1)
            .expand(data.get_item_shape(("next", train_env.reward_key))),
        )
        data.set(
            ("next", "agents", "terminated"),
            data.get(("next", "terminated"))
            .unsqueeze(-1)
            .expand(data.get_item_shape(("next", train_env.reward_key))),
        )
        data.set(
            ("next", "done"),
            data.get(("next", "done"))
            .unsqueeze(-1)
            .expand(data.get_item_shape(("next", train_env.reward_key))),
        )
        data.set(
            ("next", "terminated"),
            data.get(("next", "terminated"))
            .unsqueeze(-1)
            .expand(data.get_item_shape(("next", train_env.reward_key))),
        )
        # We need to expand the done and terminated to match the reward shape (this is expected by the value estimator)

        # Get training rewards and episode lengths
        episode_rewards = data["next", "agents", "episode_reward"][data["next", "done"]]
        episode_length = data["next", "step_count"][data["next", "done"].all(-2)]
        if len(episode_length) > 0:
            log_info.update(
                {
                    "train/episode_reward": episode_rewards.mean().item(),
                    "train/episode_length": episode_length.sum().item() / len(episode_length),
                }
            )
        else:  # if no end of an episode is contained in the batch, fill the logs with NaN
            log_info.update(
                {
                    "train/episode_reward": np.nan,
                    "train/episode_length": np.nan,
                }
            )

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
                "step": collected_frames,
            }
        )

        if i % cfg.checkpoint.checkpoint_interval == 0 or i >= total_frames // frames_per_batch:
            # output_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir + '/'
            if not os.path.exists('checkpoints'):
                os.mkdir('checkpoints')
            output_dir = os.getcwd() + '/checkpoints/'
            # full_buffer.dumps(output_dir + 'replay_buffer_checkpoint')
            # logging.info(f"Checkpointed replay buffer. (Saved at {output_dir + 'replay_buffer_checkpoint'}).")
            save_model(actor, critic, output_dir, collected_frames)
            logging.info(f"Checkpointed model. (Saved at {output_dir}actor_{collected_frames}.pkl and {output_dir}critic_{collected_frames}.pkl")
            
        log_metrics(logs, log_info)
        collector.update_policy_weights_()
        sampling_start = time.time()

    end_time = time.time()
    execution_time = end_time - start_time
    logging.info(f"Training took {execution_time:.2f} seconds to finish")


if __name__ == "__main__":
    main()
