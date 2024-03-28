# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch.nn
import torch.optim

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
)
from utils.vecnorm_fixed import VecNorm
from torchrl.modules import MLP, ProbabilisticActor, TanhNormal, ValueOperator
from torchrl.modules.models.multiagent import MultiAgentMLP
from Solver.WF_enviroment import TurbEnv


# ====================================================================
# Environment utils
# --------------------------------------------------------------------
def add_env_transforms(env, obs_norm_params=None):
    transform_list = [
        InitTracker(),
        RewardSum(),
        StepCounter(),
        FiniteTensorDictCheck(),
    ]
    if obs_norm_params is None:
        transform_list.append(VecNorm(in_keys=[("agents", "observation")], decay=0.99))
    else:
        for in_key, loc_scale_dict in obs_norm_params.items():
            transform_list.append(
                ObservationNorm(
                    loc=loc_scale_dict['loc'],
                    scale=loc_scale_dict['scale'],
                    in_keys=[in_key]
                )
            )

    transforms = Compose(*transform_list)
    return TransformedEnv(env, transforms)


def make_env(params, instance=None, save=False, device="cpu", dummy_update=False):
    base_env = TurbEnv(params, save=save, instance=instance, device=device, dummy_update=dummy_update)
    env = add_env_transforms(base_env)
    return env


def make_parallel_env(params, num_envs, device="cpu", dummy_update=False):
    # need to pass a separate instance id to each one
    # think there should be a better way to do this than defining separate functions for each
    # Also need to modify it to match length n_environments
    # https://pytorch.org/rl/tutorials/torchrl_envs.html#kwargs-for-parallel-environments may be better

    env = ParallelEnv(num_envs, [
            lambda: make_env(params, instance=0, device=device, dummy_update=dummy_update),
            lambda: make_env(params, instance=1, device=device, dummy_update=dummy_update)],)
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


def make_ma_ppo_models(params, dummy_update):
    proof_environment = make_env(params, device="cpu", dummy_update=dummy_update)
    actor, critic = make_ma_ppo_models_state(proof_environment)
    return actor, critic


# ====================================================================
# Evaluation utils
# --------------------------------------------------------------------


def eval_model(actor, test_env, num_episodes=3, episode_length=1000):
    print('\n\nMODEL EVALUATION\n')
    test_rewards_mean = []
    test_rewards_stdv = []
    test_alpha_1_mean = []
    test_alpha_2_mean = []
    test_alpha_1_stdv = []
    test_alpha_2_stdv = []
    for _ in range(num_episodes):
        td_test = test_env.rollout(
            policy=actor,
            auto_reset=True,
            auto_cast_to_device=True,
            break_when_any_done=False,
            max_steps=episode_length,
        )
        reward_mean = td_test["next", "agents", "reward"].mean().reshape(1)
        reward_stdv = td_test["next", "agents", "reward"].std().reshape(1)
        alpha_1_mean = td_test["agents", 'alpha'][:, 0].mean().reshape(1)
        alpha_2_mean = td_test["agents", 'alpha'][:, 1].mean().reshape(1)
        alpha_1_stdv = td_test["agents", 'alpha'][:, 0].std().reshape(1)
        alpha_2_stdv = td_test["agents", 'alpha'][:, 1].std().reshape(1)

        test_rewards_mean.append(reward_mean.cpu())
        test_rewards_stdv.append(reward_stdv.cpu())
        test_alpha_1_mean.append(alpha_1_mean.cpu())
        test_alpha_2_mean.append(alpha_2_mean.cpu())
        test_alpha_1_stdv.append(alpha_1_stdv.cpu())
        test_alpha_2_stdv.append(alpha_2_stdv.cpu())

    del td_test
    return (torch.cat(test_rewards_mean, 0).mean(),
            torch.cat(test_rewards_stdv, 0).mean(),
            torch.cat(test_alpha_1_mean, 0).mean(),
            torch.cat(test_alpha_2_mean, 0).mean(),
            torch.cat(test_alpha_1_stdv, 0).mean(),
            torch.cat(test_alpha_2_stdv, 0).mean())

