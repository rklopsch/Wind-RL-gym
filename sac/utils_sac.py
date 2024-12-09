import tempfile
from contextlib import nullcontext
import hydra
import pickle
import torch
from tensordict.nn import InteractionType, TensorDictModule, TensorDictSequential
from tensordict.nn.distributions import NormalParamExtractor
from torch import nn, optim
from torchrl.collectors import SyncDataCollector
from torchrl.data import TensorDictPrioritizedReplayBuffer, TensorDictReplayBuffer
from torchrl.data.replay_buffers.storages import LazyMemmapStorage
from torchrl.modules import MLP, ProbabilisticActor, ValueOperator
from torchrl.modules.distributions import TanhNormal
from torchrl.objectives import SoftUpdate, SACLoss
from tensordict.nn import TensorDictModule, TensorDictSequential
from tensordict.nn.distributions import NormalParamExtractor
from torchrl.envs import (
    ClipTransform,
    DoubleToFloat,
    CatFrames,
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


# ====================================================================
# Environment utils
# --------------------------------------------------------------------


def transforms(cfg, eval_only=False):
    transform_list = [
        InitTracker(),
        RewardSum(),
        FiniteTensorDictCheck(),
        CatFrames(N=cfg.env.frame_stack, dim=-1, in_keys=[("agents", "observation")]),
        ObservationNorm(
            loc=0.,
            scale=(4/cfg.env.max_yaw_angle),
            in_keys=[("agents", "alpha")],
            out_keys=[("agents", "alpha_normalised")]
        )
    ]
    if eval_only:
        transform_list.append(StepCounter(cfg.eval.episode_length))
    transforms = Compose(*transform_list)
    return transforms


def make_env(cfg, params, instance=None, save=False, device="cpu", add_transforms=True, eval_only=False):
    from WF_enviroment import TurbEnv
    env = TurbEnv(params, save=save, instance=instance, device=device)
    if add_transforms:
        env = TransformedEnv(env, transforms(cfg, eval_only))
    if eval_only:
        env.eval()
    return env


def make_parallel_env(cfg, params, num_envs, device="cpu", eval_only=False):
    function_list = [lambda i=i: make_env(cfg, params, instance=i, device=device, eval_only=eval_only) for i in
                     range(num_envs)]
    env = ParallelEnv(num_envs, function_list)
    return env


# ====================================================================
# Collector and replay buffer
# ---------------------------
def make_collector(cfg, train_env, actor, device):
    """Make collector."""
    collector = SyncDataCollector(
        train_env,
        actor,
        init_random_frames=cfg.collector.init_random_frames,
        frames_per_batch=cfg.collector.frames_per_batch,
        total_frames=cfg.collector.total_frames,
        device=device,
        max_frames_per_traj=cfg.collector.max_episode_length,
    )
    return collector


def make_replay_buffer(cfg, prefetch=3):
    batch_size = cfg.optim.batch_size
    buffer_size = cfg.replay_buffer.size
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
def make_sac_agent(cfg, params):
    # This is the single agent version
    # This is currently not implemented correctly

    device = "cpu"
    proof_environment = make_env(cfg, params, device=device)

    # Define input shape
    input_shape = proof_environment.observation_spec["observation"].shape
    
    # Define Actor Network
    action_spec = proof_environment.action_spec
    num_outputs = action_spec.shape[-1]
    in_keys_actor = ["observation"]
    out_keys_actor = ["_actor_net_out"]
    #if proof_environment.batch_size:
    #    action_spec = action_spec[(0,) * len(proof_environment.batch_size)]

    activation = nn.ReLU
    actor_net = MLP(
        in_features=input_shape[-1],
        num_cells=cfg.network.actor_hidden_sizes,
        out_features=2*num_outputs,
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
        "in_features": input_shape[-1],
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

    """
    # Initialise models
    # this should be removed
    with torch.no_grad(), set_exploration_type(ExplorationType.RANDOM):
        td = eval_env.reset()
        td = td.to(device)
        for net in model:
            net(td)
    del td
    eval_env.close()
    """

    return model, model[0]


def make_ma_sac_agents(cfg, params):
    device = "cpu"
    proof_environment = make_env(cfg, params, device=device)

    # Make actor network
    activation = nn.ReLU
    actor_module = TensorDictModule(
        MultiAgentMLP(
            n_agent_inputs=proof_environment.observation_spec["agents", "observation"].shape[-1] + 2,
            n_agent_outputs=2 * proof_environment.action_spec.shape[-1],
            n_agents=proof_environment.n_turbs,
            centralised=False,
            share_params=True,
            depth=cfg.network.actor_hidden_depth,
            num_cells=cfg.network.actor_hidden_size,
            activation_class=activation,
        ),
        in_keys=[("agents", "observation"), ("agents", "alpha_normalised"), ("agents", "pos_enc")],
        out_keys=[("agents", "actor_net_output")],
    )
    actor_extractor = TensorDictModule(
        NormalParamExtractor(
            scale_mapping=f"biased_softplus_{cfg.network.default_policy_scale}",
            scale_lb=cfg.network.scale_lb,
        ),
        in_keys=[("agents", "actor_net_output")],
        out_keys=[("agents", "loc"), ("agents", "scale")],
    )
    policy_module = TensorDictSequential(actor_module, actor_extractor)

    actor = ProbabilisticActor(
        spec=proof_environment.action_spec,
        in_keys=[("agents", "loc"), ("agents", "scale")],
        module=policy_module,
        distribution_class=TanhNormal,
        distribution_kwargs={
            "min": proof_environment.action_spec.space.low,
            "max": proof_environment.action_spec.space.high,
            "tanh_loc": False,  # can be omitted since this is default value
        },
        default_interaction_type=InteractionType.RANDOM,
        return_log_prob=False,
    )

    # Make critic
    module = MultiAgentMLP(
        n_agent_inputs=proof_environment.observation_spec["agents", "observation"].shape[-1] + proof_environment.action_spec.shape[-1] + 2,
        n_agent_outputs=1,
        n_agents=proof_environment.n_turbs,
        centralised=True,
        share_params=True,
        depth=cfg.network.critic_hidden_depth,
        num_cells=cfg.network.critic_hidden_size,
        activation_class=activation,
    )
    value_module = ValueOperator(
        module=module,
        in_keys=[proof_environment.action_key, ("agents", "observation"), ("agents", "alpha_normalised"), ("agents", "pos_enc")],
        out_keys=[("agents", "state_action_value")]
    )

    return actor, value_module
    

def make_loss_module(cfg, params, actor, critic):
    """Make loss module and target network updater."""
    # Create SAC loss
    loss_module = SACLoss(
        actor_network=actor,
        qvalue_network=critic,
        num_qvalue_nets=2,
        loss_function="l2",
        delay_actor=False,
        delay_qvalue=True,
        alpha_init=cfg.optim.alpha_init,
    )
    loss_module.make_value_estimator(gamma=cfg.optim.gamma)
    
    proof_env = make_env(cfg, params, device="cpu")
    loss_module.set_keys(  # We have to tell the loss where to find the keys
        reward=proof_env.reward_key,
        action=proof_env.action_key,
        # sample_log_prob=("agents", "sample_log_prob"),
        value=("agents", "state_action_value"),
        # These last 2 keys will be expanded to match the reward shape
        done=("agents", "done"),
        terminated=("agents", "terminated"),
    )

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


def should_log_now(cfg, frames, num_console_updates):
    return True
    return frames % (cfg.collector.total_frames // (num_console_updates) + 1) == 0
