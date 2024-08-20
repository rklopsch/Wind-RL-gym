# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
print(os.getcwd())
import hydra
import torch.nn
import torch.optim
import pickle
import logging
from tensordict.nn import AddStateIndependentNormalScale, TensorDictModule
from tensordict.nn.distributions import NormalParamExtractor
from torchrl.data import CompositeSpec
from torchrl.envs import (
    ClipTransform,
    DoubleToFloat,
    ExplorationType,
    RewardSum,
    StepCounter,
    InitTracker,
    FiniteTensorDictCheck,
    TransformedEnv,
    # VecNorm,  # currently broken
    ParallelEnv,
    Compose,
    ObservationNorm,
    EnvCreator,
)
from torchrl.modules import MLP, ProbabilisticActor, TanhNormal, ValueOperator
from torchrl.modules.models.multiagent import MultiAgentMLP
import numpy as np


# ====================================================================
# Environment utils
# --------------------------------------------------------------------


def transforms():
    transform_list = [
        InitTracker(),
        RewardSum(),
        FiniteTensorDictCheck(),
    ]
    transforms = Compose(*transform_list)
    return transforms


def make_env(params, instance=None, save=False, device="cpu", dummy_update=False, add_transforms=True, eval_only=False):
    from WF_enviroment import TurbEnv
    env = TurbEnv(params, save=save, instance=instance, device=device, dummy_update=dummy_update)
    if add_transforms:
        env = TransformedEnv(env, transforms())
    if eval_only:
        env.eval()
    return env


# TO-DO: delete this
def make_smartsim_env(client, params, instance=None, save=False, device="cpu", dummy_update=False, add_transforms=True):
    from WF_enviroment import TurbEnv
    env = TurbEnv(client, params, save=save, instance=0, device=device, dummy_update=dummy_update)
    if add_transforms:
        env = TransformedEnv(env, transforms())
    return env


def make_parallel_env(params, num_envs, device="cpu", dummy_update=False, eval_only=False):
    function_list = [lambda i=i: make_env(params, instance=i, device=device, dummy_update=dummy_update, eval_only=eval_only) for i in
                     range(num_envs)]
    env = ParallelEnv(num_envs, function_list, )
    # serial_for_single=True)
    return env


# ====================================================================
# Model utils
# --------------------------------------------------------------------


def make_ppo_models_state(proof_environment):
    # Define input shape
    input_shape = proof_environment.observation_spec["observation"].shape

    # Define policy output distribution class
    num_outputs = proof_environment.action_spec.shape[-1]
    distribution_class = TanhNormal
    distribution_kwargs = {
        "min": proof_environment.action_spec.space.low,
        "max": proof_environment.action_spec.space.high,
        "tanh_loc": False,
    }

    # Define policy architecture
    policy_mlp = MLP(
        in_features=input_shape[-1],
        activation_class=torch.nn.Tanh,
        out_features=num_outputs,  # predict only loc
        num_cells=[64, 64],
    )

    # Initialize policy weights
    for layer in policy_mlp.modules():
        if isinstance(layer, torch.nn.Linear):
            torch.nn.init.orthogonal_(layer.weight, 1.0)
            layer.bias.data.zero_()

    # Add state-independent normal scale
    policy_mlp = torch.nn.Sequential(
        policy_mlp,
        AddStateIndependentNormalScale(
            proof_environment.action_spec.shape[-1], scale_lb=1e-8
        ),
    )

    # Add probabilistic sampling of the actions
    policy_module = ProbabilisticActor(
        TensorDictModule(
            module=policy_mlp,
            in_keys=["observation"],
            out_keys=["loc", "scale"],
        ),
        in_keys=["loc", "scale"],
        spec=CompositeSpec(action=proof_environment.action_spec),
        distribution_class=distribution_class,
        distribution_kwargs=distribution_kwargs,
        return_log_prob=True,
        default_interaction_type=ExplorationType.RANDOM,
    )

    # Define value architecture
    value_mlp = MLP(
        in_features=input_shape[-1],
        activation_class=torch.nn.Tanh,
        out_features=1,
        num_cells=[64, 64],
    )

    # Initialize value weights
    for layer in value_mlp.modules():
        if isinstance(layer, torch.nn.Linear):
            torch.nn.init.orthogonal_(layer.weight, 0.01)
            layer.bias.data.zero_()

    # Define value module
    value_module = ValueOperator(
        value_mlp,
        in_keys=["observation"],
    )

    return policy_module, value_module


def make_ppo_models(params):
    proof_environment = make_env(params, device="cpu", dummy_update=True)
    actor, critic = make_ppo_models_state(proof_environment)
    return actor, critic


def make_ma_ppo_models_state(proof_environment):
    # Policy
    actor_net = torch.nn.Sequential(
        MultiAgentMLP(
            n_agent_inputs=proof_environment.observation_spec["agents", "observation"].shape[-1],
            n_agent_outputs=2 * proof_environment.action_spec.shape[-1],
            n_agents=proof_environment.n_turbs,
            centralised=False,
            share_params=True,
            # device=cfg.train.device,
            depth=2,
            num_cells=256,
            activation_class=torch.nn.Tanh,
        ),
        NormalParamExtractor(),
    )
    policy_module = TensorDictModule(
        actor_net,
        in_keys=[("agents", "observation")],
        out_keys=[("agents", "loc"), ("agents", "scale")],
    )
    policy = ProbabilisticActor(
        module=policy_module,
        # spec=proof_environment.unbatched_action_spec,
        spec=proof_environment.action_spec,
        in_keys=[("agents", "loc"), ("agents", "scale")],
        out_keys=[proof_environment.action_key],
        distribution_class=TanhNormal,
        distribution_kwargs={
            # "min": proof_environment.unbatched_action_spec[("agents", "action")].space.low,
            "min": proof_environment.action_spec.space.low,
            # "max": proof_environment.unbatched_action_spec[("agents", "action")].space.high,
            "max": proof_environment.action_spec.space.high,
        },
        return_log_prob=True,
        log_prob_key=("agents", "sample_log_prob"),
    )

    # Critic
    module = MultiAgentMLP(
        n_agent_inputs=proof_environment.observation_spec["agents", "observation"].shape[-1],
        n_agent_outputs=1,
        n_agents=proof_environment.n_turbs,
        centralised=True,
        share_params=True,
        # device=cfg.train.device,
        depth=2,
        num_cells=256,
        activation_class=torch.nn.Tanh,
    )
    value_module = ValueOperator(
        module=module,
        in_keys=[("agents", "observation")],
        out_keys=[("agents", "state_value")]
    )

    return policy, value_module


def make_ma_ppo_models(params):
    proof_environment = make_env(params, device="cpu", dummy_update=True)
    actor, critic = make_ma_ppo_models_state(proof_environment)
    return actor, critic


def load_model(env_params, path_to_model, id, dummy_update=False):
    try:
        """
        # Load env transforms
        with open('checkpoints/env_transforms' + f"_{id}" + '.pkl', 'rb') as file:
            transforms_params = pickle.load(file)
        """
        # Filenames
        actor_path = path_to_model + '/actor' + f"_{id}" + '.pkl'
        critic_path = path_to_model + '/critic' + f"_{id}" + '.pkl'
        # Load model parameters
        with open(actor_path, 'rb') as file:
            actor_params = torch.load(file)
        with open(critic_path, 'rb') as file:
            critic_params = torch.load(file)
    except FileNotFoundError:
        print(f"File {actor_path} or {critic_path} has not been found.")
        return False

    device = "cpu" if not torch.cuda.device_count() else "cuda"
    # Build the env without transforms
    # Since the purpose of loading a trained model is to test, we only build a single env
    """
    env = make_env(
        env_params,
        instance='TestEnv',
        save=True,
        device=device,
        dummy_update=dummy_update,
        add_transforms=False,
    )

    # Rebuild the Transforms, but replacing the VecNorm with an ObservationNorm
    env = TransformedEnv(env, transforms())
    """

    # Instantiating the model with random params
    actor, critic = make_ma_ppo_models(env_params)
    actor, critic = actor.to(device), critic.to(device)
    # Inserting the loaded parameters
    actor.load_state_dict(actor_params)
    critic.load_state_dict(critic_params)

    return actor, critic


def save_model(actor, critic, filepath, id):
    # Read out loc and scale used in ObservationNorm, if used
    transform_list = transforms()
    obs_norm_transform = next((t for t in transform_list if isinstance(t, ObservationNorm)), None)
    if obs_norm_transform:
        norm_dict = {
            'loc': obs_norm_transform.loc,
            'scale': obs_norm_transform.scale,
        }
        # Save env transforms
        with open(filepath + 'env_transforms' + f"_{id}" + '.pkl', 'wb') as file:
            pickle.dump(norm_dict, file)
    # Save model
    with open(filepath + 'actor' + f"_{id}" + '.pkl', 'wb') as file:
        torch.save(actor.state_dict(), file)
    with open(filepath + 'critic' + f"_{id}" + '.pkl', 'wb') as file:
        torch.save(critic.state_dict(), file)

    return True


# ====================================================================
# Evaluation utils
# --------------------------------------------------------------------


def eval_model(actor, test_env, num_turbines, num_episodes=3, episode_length=1000):
    logging.info('Model evaluation')

    rewards = torch.zeros(num_episodes, 2)  # Mean and Std Dev for rewards
    alpha_means_list = [torch.zeros(num_episodes) for _ in range(num_turbines)]
    alpha_stds_list = [torch.zeros(num_episodes) for _ in range(num_turbines)]

    for episode_idx in range(num_episodes):
        td_test = test_env.rollout(
            policy=actor,
            auto_reset=True,
            auto_cast_to_device=True,
            break_when_any_done=False,
            max_steps=episode_length,
        )

        rewards[episode_idx, 0] = td_test["next", "agents", "reward"].mean().item()
        rewards[episode_idx, 1] = td_test["next", "agents", "reward"].std().item()

        alpha_means = td_test["agents", 'alpha'][:, :num_turbines].mean(dim=0)
        alpha_stds = td_test["agents", 'alpha'][:, :num_turbines].std(dim=0)

        for turbine_idx in range(num_turbines):
            alpha_means_list[turbine_idx][episode_idx] = alpha_means[turbine_idx]
            alpha_stds_list[turbine_idx][episode_idx] = alpha_stds[turbine_idx]

    # Compute the overall mean and standard deviation of rewards
    rewards_mean = rewards[:, 0].mean().item()
    rewards_stdv = rewards[:, 1].mean().item()

    # Compute the mean of means and mean of stds for alpha values across all episodes
    alpha_means_final = [alpha_means.mean().item() for alpha_means in alpha_means_list]
    alpha_stds_final = [alpha_stds.mean().item() for alpha_stds in alpha_stds_list]

    # Cleanup
    del td_test

    # Return matching the structure: mean/std for rewards, lists for alpha means/stds
    return rewards_mean, rewards_stdv, alpha_means_final, alpha_stds_final


# ====================================================================
# Logging utils
# --------------------------------------------------------------------


def log_metrics(logs, metrics):
    for metric_name, metric_value in metrics.items():
        if metric_name in logs.keys():
            logs[metric_name].append(metric_value)
        else:
            logs[metric_name] = [metric_value]
    # Save logs to disk
    output_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir + '/'
    with open(output_dir + "logs.pkl", "wb") as f:
        pickle.dump(logs, f)
