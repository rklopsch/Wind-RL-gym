import os
import sys
sys.path.append(os.getcwd())
import time
import logging
import torch.optim
from tensordict import TensorDict
from torchrl.data import LazyMemmapStorage, TensorDictReplayBuffer
from torchrl.envs.utils import set_exploration_type, ExplorationType
from utils_ppo import make_parallel_env, load_model, save_model, log_metrics
import shutil
import hydra
import numpy as np
import shutil
import pickle


def adjust_tensor_shapes(data, env):
    """
    Adjust the shapes of tensors for multiagent tensordict.
    """
    data.set(
        ("next", "agents", "done"),
        data.get(("next", "done"))
        .unsqueeze(-1)
        .expand(data.get_item_shape(("next", env.reward_key))),
    )
    data.set(
        ("next", "agents", "terminated"),
        data.get(("next", "terminated"))
        .unsqueeze(-1)
        .expand(data.get_item_shape(("next", env.reward_key))),
    )
    data.set(
        ("next", "done"),
        data.get(("next", "done"))
        .unsqueeze(-1)
        .expand(data.get_item_shape(("next", env.reward_key))),
    )
    data.set(
        ("next", "terminated"),
        data.get(("next", "terminated"))
        .unsqueeze(-1)
        .expand(data.get_item_shape(("next", env.reward_key))),
    )
    return data


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
        "run_steps": max_episode_length * cfg.env.steps_per_frame,
    }

    # Load the models to be evaluated
    # Copy the loaded models into the ppo/eval directory
    if not os.path.exists('models'):
        os.mkdir('models')
    for arch in ['actor', 'critic']:
        filename = f"{arch}_{cfg.eval.model_id}.pkl"
        if os.path.exists(filename):
            shutil.move(filename, "models")
    # Load actor and critic
    actor, critic = load_model(
        cfg=cfg,
        env_params=params,
        id=cfg.eval.model_id,
        path_to_model='models',
        dummy_update=dummy_update)
    logging.info(f"Loaded actor_{cfg.eval.model_id}.pkl and critic_{cfg.eval.model_id}.pkl.")
    actor, critic = actor.to(device), critic.to(device)

    # Create environments
    eval_env = make_parallel_env(
        cfg,
        params,
        n_environments,
        device=device,
        dummy_update=dummy_update,
        eval_only=True,
    )

    """
    # Create replay buffer to remember entire history
    full_buffer = TensorDictReplayBuffer(
        storage=LazyMemmapStorage(total_frames),
    )
    """

    # Timing
    start_time = time.time()
    logs = {}

    with set_exploration_type(ExplorationType.MEAN), torch.no_grad():
        data = eval_env.rollout(
            max_steps=max_episode_length,
            policy=actor,
            auto_reset=True,
        )
        data = adjust_tensor_shapes(data, eval_env)

        # Get rewards and episode lengths
        episode_rewards = data["next", "agents", "episode_reward"][data["next", "done"]]
        reward_shape = data.get_item_shape(("next", eval_env.reward_key))
        episode_rewards = episode_rewards.view(reward_shape[-2], reward_shape[0]).mean(dim=0)
        episode_length = data["next", "step_count"][data["next", "done"].all(-2)]
        rewards = data["next", "agents", "reward"].squeeze().mean(dim=-1)
        alpha = data["agents", "alpha"].squeeze()
        actions = data["agents", "action"].squeeze()
        if not len(episode_length) > 0:
            raise RuntimeWarning("The eval tensordict does not contain a finished episode.")
        
        for i in range(n_environments):
            logs[f"episode_reward_{i+1}"] = episode_rewards[i].item()
            logs[f"episode_length_{i+1}"] = episode_length[i].item()
            logs[f"rewards_{i+1}"] = rewards[i].detach().cpu().numpy()
            logs[f"alphas_{i+1}"] = alpha[i].detach().cpu().numpy()
            logs[f"actions_{i+1}"] = actions[i].detach().cpu().numpy()

    # End timing
    end_time = time.time()
    execution_time = end_time - start_time
    logging.info(f"Evaluation took {execution_time:.2f} seconds to finish")

    # Save logs to disk
    output_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir + '/'
    with open(output_dir + "eval_logs.pkl", "wb") as f:
        pickle.dump(logs, f)


if __name__ == "__main__":
    main()
