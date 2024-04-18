# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch.nn
import torch.optim
import pickle
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
from Solver.WF_enviroment import TurbEnv
import numpy as np


# ====================================================================
# Environment utils
# --------------------------------------------------------------------
def obs_normalisation():
    return {'loc': np.array([5.75, 0., 0., 5.75, 0., 5.75, 0., 5.75, 0., 5.75, 0.,
                             5.75, 0., 5.75, 0., 5.75, 0., 5.75, 0., 5.75, 0., 5.75,
                             0., 5.75, 0., 5.75, 0., 5.75, 0., 5.75, 0., 5.75, 0.,
                             5.75, 0., 5.75, 0., 5.75, 0., 5.75, 0., 5.75, 0., 5.75,
                             0., 5.75, 0., 5.75, 0., 5.75, 0., 5.75, 0.]),
            'scale': np.array([0.75, 0.75, 0.75, 0.75, 0.75, 0.75, 0.75, 0.75, 0.75, 0.75, 0.75,
                               0.75, 0.75, 0.75, 0.75, 0.75, 0.75, 0.75, 0.75, 0.75, 0.75, 0.75,
                               0.75, 0.75, 0.75, 0.75, 0.75, 0.75, 0.75, 0.75, 0.75, 0.75, 0.75,
                               0.75, 0.75, 0.75, 0.75, 0.75, 0.75, 0.75, 0.75, 0.75, 0.75, 0.75,
                               0.75, 0.75, 0.75, 0.75, 0.75, 0.75, 0.75, 0.75, 0.75])}


def add_env_transforms(env, obs_norm_params=None):
    # Load observation normalisation parameters from file
    if not obs_norm_params:
        obs_norm_params = obs_normalisation()
    assert obs_norm_params is not None

    transform_list = [
        InitTracker(),
        RewardSum(),
        FiniteTensorDictCheck(),
        ObservationNorm(
            loc=obs_norm_params['loc'],
            scale=obs_norm_params['scale'],
            in_keys=[('agents', 'observation')]
        )
    ]
    transforms = Compose(*transform_list)
    return TransformedEnv(env, transforms)


def make_env(params, instance=None, save=False, device="cpu", dummy_update=False, add_transforms=True):
    env = TurbEnv(params, save=save, instance=instance, device=device, dummy_update=dummy_update)
    if add_transforms:
        env = add_env_transforms(env)
    return env


def make_parallel_env(params, num_envs, device="cpu", dummy_update=False):
    """
    # Different way of creating parallel envs, this way the VecNorm is synchronised
    # However idk how to figure out the instance parameter here...
    env_creator = EnvCreator(lambda: make_env(params, device=device, dummy_update=dummy_update))
    env = ParallelEnv(num_envs, env_creator)
    env_creator.state_dict()["transforms.3._extra_state"]["td"]["agents_observation_count"].fill_(0.0)
    env_creator.state_dict()["transforms.3._extra_state"]["td"]["agents_observation_ssq"].fill_(0.0)
    env_creator.state_dict()["transforms.3._extra_state"]["td"]["agents_observation_sum"].fill_(0.0)
    # return env
    """
    function_list = [lambda i=i: make_env(params, instance=i, device=device, dummy_update=dummy_update) for i in
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
    proof_environment = make_env(params, device="cpu")
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


# ====================================================================
# Evaluation utils
# --------------------------------------------------------------------


def eval_model(actor, test_env, num_turbines, num_episodes=3, episode_length=1000):
    print('\n\nMODEL EVALUATION\n')

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
