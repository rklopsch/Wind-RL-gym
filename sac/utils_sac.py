# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import tempfile
from contextlib import nullcontext
import torch
from pathlib import Path
import numpy as np
from tensordict.nn import InteractionType, TensorDictModule, TensorDictSequential
from tensordict.nn.distributions import NormalParamExtractor
from torch import nn, optim
from torchrl.collectors import SyncDataCollector
from torchrl.data import TensorDictPrioritizedReplayBuffer, TensorDictReplayBuffer
from torchrl.data.replay_buffers.storages import LazyMemmapStorage
from torchrl.envs.utils import ExplorationType, set_exploration_type
from torchrl.modules import MLP, ProbabilisticActor, ValueOperator
from torchrl.modules.distributions import TanhNormal
from torchrl.objectives import SoftUpdate, SACLoss
from utils.rng import env_seed
import wandb


# ====================================================================
# Collector and replay buffer
# ---------------------------
def make_collector(cfg, train_env, actor_model_explore):
    """Make collector."""
    collector = SyncDataCollector(
        train_env,
        actor_model_explore,
        init_random_frames=cfg.collector.init_random_frames // cfg.env.frame_skip,
        frames_per_batch=cfg.collector.frames_per_batch // cfg.env.frame_skip,
        total_frames=cfg.collector.total_frames // cfg.env.frame_skip,
        device=cfg.collector.device,
        # max_frames_per_traj=cfg.collector.max_episode_length // cfg.env.frame_skip,
    )
    collector.set_seed(env_seed(cfg))
    return collector


def make_replay_buffer(cfg, prefetch=3):
    batch_size = cfg.optim.batch_size
    buffer_size = cfg.replay_buffer.size // cfg.env.frame_skip
    buffer_scratch_dir = cfg.replay_buffer.scratch_dir
    device = torch.device(cfg.collector.device)
    with (
            tempfile.TemporaryDirectory()
            if buffer_scratch_dir is None
            else nullcontext(buffer_scratch_dir)
    ) as scratch_dir:
        if cfg.replay_buffer.prb:
            replay_buffer = TensorDictPrioritizedReplayBuffer(
                alpha=0.7,
                beta=0.5,
                pin_memory=False,
                prefetch=prefetch,
                storage=LazyMemmapStorage(
                    buffer_size,
                    scratch_dir=scratch_dir,
                    device=device,
                ),
                batch_size=batch_size,
            )
        else:
            replay_buffer = TensorDictReplayBuffer(
                pin_memory=False,
                prefetch=prefetch,
                storage=LazyMemmapStorage(
                    buffer_size,
                    scratch_dir=scratch_dir,
                    device=device,
                ),
                batch_size=batch_size,
            )
        return replay_buffer


# ====================================================================
# Model
# -----
def make_sac_agent(cfg, train_env, eval_env):
    path_to_model = cfg.env.path_to_cae_model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Define Actor Network
    in_keys_actor = ["observation"]
    out_keys_actor = ["_actor_net_out"]
    action_spec = train_env.action_spec
    if train_env.batch_size:
        action_spec = action_spec[(0,) * len(train_env.batch_size)]

    activation = nn.ReLU
    actor_net = MLP(
        num_cells=cfg.network.actor_hidden_sizes,
        out_features=2 * action_spec.shape[-1],
        activation_class=activation,
    )
    actor_net = TensorDictModule(
        actor_net,
        in_keys=in_keys_actor,
        out_keys=out_keys_actor,
    )
    actor_extractor = TensorDictModule(
        NormalParamExtractor(
            scale_mapping=f"biased_softplus_{cfg.network.default_policy_scale}",
            scale_lb=cfg.network.scale_lb,
        ),
        in_keys=out_keys_actor,
        out_keys=["loc", "scale"],
    )

    actor_module = TensorDictSequential(actor_net, actor_extractor)

    actor = ProbabilisticActor(
        spec=action_spec,
        in_keys=["loc", "scale"],
        module=actor_module,
        distribution_class=TanhNormal,
        distribution_kwargs={
            "min": action_spec.space.low,
            "max": action_spec.space.high,
            "tanh_loc": False,  # can be omitted since this is default value
        },
        default_interaction_type=InteractionType.RANDOM,
        return_log_prob=False,
    )

    # Define Critic Network
    qvalue_net_kwargs = {
        "num_cells": cfg.network.critic_hidden_sizes,
        "out_features": 1,
        "activation_class": activation,
    }
    qvalue_net = MLP(
        **qvalue_net_kwargs,
    )

    critic = ValueOperator(
        in_keys=["action"] + in_keys_actor,
        module=qvalue_net,
    )

    model = nn.ModuleList([actor, critic]).to(device)

    # Initialise models
    with torch.no_grad(), set_exploration_type(ExplorationType.RANDOM):
        td = eval_env.reset()
        td = td.to(device)
        for net in model:
            net(td)
    del td
    eval_env.close()

    return model, model[0], device


def make_loss_module(cfg, model):
    """Make loss module and target network updater."""
    # Create SAC loss
    loss_module = SACLoss(
        actor_network=model[0],
        qvalue_network=model[1],
        num_qvalue_nets=2,
        loss_function="l2",
        delay_actor=False,
        delay_qvalue=True,
        alpha_init=cfg.optim.alpha_init,
    )
    loss_module.make_value_estimator(gamma=cfg.optim.gamma)

    # Define Target Network Updater
    target_net_updater = SoftUpdate(loss_module, eps=cfg.optim.target_update_polyak)
    return loss_module, target_net_updater


def make_sac_optimizer(cfg, loss_module):
    critic_params = list(loss_module.qvalue_network_params.flatten_keys().values())
    actor_params = list(loss_module.actor_network_params.flatten_keys().values())

    trainable_actor_params = [p for p in actor_params if p.requires_grad]
    trainable_critic_params = [p for p in critic_params if p.requires_grad]

    optimizer_actor = optim.Adam(
        trainable_actor_params,
        lr=cfg.optim.lr,
        weight_decay=cfg.optim.weight_decay,
        eps=cfg.optim.adam_eps,
    )
    optimizer_critic = optim.Adam(
        trainable_critic_params,
        lr=cfg.optim.lr,
        weight_decay=cfg.optim.weight_decay,
        eps=cfg.optim.adam_eps,
    )
    optimizer_alpha = optim.Adam(
        [loss_module.log_alpha],
        lr=3.0e-4,
    )
    return optimizer_actor, optimizer_critic, optimizer_alpha


# ====================================================================
# General utils
# ---------


def log_metrics_wandb(metrics, step):
    wandb.log(data=metrics, step=step)


def log_metrics_offline(logs, metrics):
    for metric_name, metric_value in metrics.items():
        if metric_name in logs.keys():
            logs[metric_name].append(metric_value)
        else:
            logs[metric_name] = [metric_value]


def get_activation(cfg):
    if cfg.network.activation == "relu":
        return nn.ReLU
    elif cfg.network.activation == "tanh":
        return nn.Tanh
    elif cfg.network.activation == "leaky_relu":
        return nn.LeakyReLU
    else:
        raise NotImplementedError
