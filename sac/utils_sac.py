import hydra
import math
import os
import pickle
from copy import deepcopy
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch
from torch import nn


def stack_obs_dict(obs_list):
    keys = obs_list[0].keys()
    out = {}
    for key in keys:
        values = [item[key] for item in obs_list]
        first = values[0]
        if isinstance(first, dict):
            out[key] = stack_obs_dict(values)
        else:
            tensors = [torch.as_tensor(v) for v in values]
            out[key] = torch.stack(tensors, dim=0)
    return out


class ParallelGymEnv:
    def __init__(self, env_fns):
        self.envs = [fn() for fn in env_fns]
        self.num_envs = len(self.envs)
        self.action_space = self.envs[0].action_space
        self.observation_space = self.envs[0].observation_space
        self._executor = ThreadPoolExecutor(max_workers=self.num_envs)

    def reset(self):
        results = list(self._executor.map(lambda env: env.reset(), self.envs))
        obs_list = [obs for obs, _ in results]
        infos = [info for _, info in results]
        return stack_obs_dict(obs_list), infos

    def step(self, actions):
        actions = torch.as_tensor(actions)
        if actions.dim() == 1:
            actions = actions.unsqueeze(0)
        if actions.shape[0] != self.num_envs:
            raise ValueError(f"Expected actions for {self.num_envs} envs, got shape {tuple(actions.shape)}")
        def _step_one(args):
            env, action = args
            obs, reward, term, trunc, info = env.step(action)
            final_obs = obs
            if term or trunc:
                reset_obs, reset_info = env.reset()
                info = dict(info)
                info["final_observation"] = final_obs
                info["final_info"] = reset_info
                obs = reset_obs
            return obs, reward, term, trunc, info

        results = list(self._executor.map(_step_one, zip(self.envs, actions)))
        obs_list = [obs for obs, _, _, _, _ in results]
        rewards = [torch.as_tensor(reward).reshape(1) for _, reward, _, _, _ in results]
        terminated = [torch.as_tensor(term, dtype=torch.bool).reshape(1) for _, _, term, _, _ in results]
        truncated = [torch.as_tensor(trunc, dtype=torch.bool).reshape(1) for _, _, _, trunc, _ in results]
        infos = [info for _, _, _, _, info in results]

        return (
            stack_obs_dict(obs_list),
            torch.stack(rewards, dim=0),
            torch.stack(terminated, dim=0),
            torch.stack(truncated, dim=0),
            infos,
        )

    def close(self):
        for env in self.envs:
            close = getattr(env, "close", None)
            if callable(close):
                close()
        self._executor.shutdown(wait=True)


def make_env(cfg, params, instance=None, save=False, device="cpu", eval_only=False):
    from WF_enviroment import TurbEnv

    return TurbEnv(params, multi_agent=cfg.multi_agent.use, save=save, instance=instance, device=device)


def make_parallel_env(cfg, params, num_envs, device="cpu", eval_only=False):
    env_fns = [
        lambda i=i: make_env(cfg, params, instance=i, device=device, eval_only=eval_only)
        for i in range(num_envs)
    ]
    return ParallelGymEnv(env_fns)


def _flatten_batch(x: torch.Tensor) -> torch.Tensor:
    if x.dim() == 0:
        return x.view(1, 1)
    if x.dim() == 1:
        return x.unsqueeze(0)
    return x.reshape(x.shape[0], -1)


def build_state(obs):
    observation = torch.as_tensor(obs["observation"], dtype=torch.float32)
    alpha = torch.as_tensor(obs["alpha_normalised"], dtype=torch.float32)
    obs_flat = _flatten_batch(observation)
    alpha_flat = _flatten_batch(alpha)
    return torch.cat([obs_flat, alpha_flat], dim=-1)


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_depth, hidden_size, output_dim, activation=nn.ReLU):
        super().__init__()
        layers = []
        in_dim = input_dim
        for _ in range(hidden_depth):
            layers.append(nn.Linear(in_dim, hidden_size))
            layers.append(activation())
            in_dim = hidden_size
        layers.append(nn.Linear(in_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class SACActor(nn.Module):
    def __init__(
        self,
        state_dim,
        action_dim,
        hidden_depth,
        hidden_size,
        action_low,
        action_high,
        log_std_min=-5.0,
        log_std_max=2.0,
    ):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.backbone = MLP(state_dim, hidden_depth, hidden_size, 2 * action_dim, activation=nn.ReLU)
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        action_low = torch.as_tensor(action_low, dtype=torch.float32)
        action_high = torch.as_tensor(action_high, dtype=torch.float32)
        self.register_buffer("action_scale", (action_high - action_low) / 2.0)
        self.register_buffer("action_bias", (action_high + action_low) / 2.0)

    def forward(self, state):
        mean_log_std = self.backbone(state)
        mean, log_std = mean_log_std.chunk(2, dim=-1)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        return mean, log_std

    def sample(self, state, deterministic=False):
        mean, log_std = self.forward(state)
        if deterministic:
            pre_tanh = mean
        else:
            std = log_std.exp()
            pre_tanh = mean + std * torch.randn_like(mean)

        squashed = torch.tanh(pre_tanh)
        action = squashed * self.action_scale + self.action_bias

        log_prob = None
        if not deterministic:
            std = log_std.exp()
            normal = torch.distributions.Normal(mean, std)
            log_prob = normal.log_prob(pre_tanh)
            log_prob = log_prob - torch.log(1 - squashed.pow(2) + 1e-6)
            log_prob = log_prob.sum(dim=-1, keepdim=True)
        return action, log_prob


class SACCritic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_depth, hidden_size):
        super().__init__()
        self.q = MLP(state_dim + action_dim, hidden_depth, hidden_size, 1, activation=nn.ReLU)

    def forward(self, state, action):
        x = torch.cat([state, action], dim=-1)
        return self.q(x)


class ReplayBuffer:
    def __init__(self, capacity, state_dim, action_dim):
        self.capacity = int(capacity)
        self.state = torch.zeros((self.capacity, state_dim), dtype=torch.float32)
        self.action = torch.zeros((self.capacity, action_dim), dtype=torch.float32)
        self.reward = torch.zeros((self.capacity, 1), dtype=torch.float32)
        self.next_state = torch.zeros((self.capacity, state_dim), dtype=torch.float32)
        self.done = torch.zeros((self.capacity, 1), dtype=torch.bool)
        self.index = 0
        self.size = 0

    def _store_one(self, state, action, reward, next_state, done):
        self.state[self.index] = state
        self.action[self.index] = action
        self.reward[self.index] = reward
        self.next_state[self.index] = next_state
        self.done[self.index] = done
        self.index = (self.index + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def add(self, state, action, reward, next_state, done):
        state = torch.as_tensor(state, dtype=torch.float32).detach().cpu()
        action = torch.as_tensor(action, dtype=torch.float32).detach().cpu()
        reward = torch.as_tensor(reward, dtype=torch.float32).detach().cpu()
        next_state = torch.as_tensor(next_state, dtype=torch.float32).detach().cpu()
        done = torch.as_tensor(done, dtype=torch.bool).detach().cpu()

        state = _flatten_batch(state)
        action = _flatten_batch(action)
        reward = reward.reshape(reward.shape[0], -1) if reward.dim() > 1 else reward.unsqueeze(-1)
        next_state = _flatten_batch(next_state)
        done = done.reshape(done.shape[0], -1) if done.dim() > 1 else done.unsqueeze(-1)

        for i in range(state.shape[0]):
            self._store_one(state[i], action[i], reward[i], next_state[i], done[i])

    def sample(self, batch_size, device="cpu"):
        if self.size == 0:
            raise RuntimeError("Replay buffer is empty.")
        idx = torch.randint(0, self.size, (batch_size,))
        return {
            "state": self.state[idx].to(device),
            "action": self.action[idx].to(device),
            "reward": self.reward[idx].to(device),
            "next_state": self.next_state[idx].to(device),
            "done": self.done[idx].to(device),
        }

    def state_dict(self):
        return {
            "capacity": self.capacity,
            "state": self.state[: self.size].clone(),
            "action": self.action[: self.size].clone(),
            "reward": self.reward[: self.size].clone(),
            "next_state": self.next_state[: self.size].clone(),
            "done": self.done[: self.size].clone(),
            "index": self.index,
            "size": self.size,
        }

    def load_state_dict(self, state_dict):
        n = min(state_dict["size"], self.capacity)
        self.state[:n] = state_dict["state"][:n]
        self.action[:n] = state_dict["action"][:n]
        self.reward[:n] = state_dict["reward"][:n]
        self.next_state[:n] = state_dict["next_state"][:n]
        self.done[:n] = state_dict["done"][:n]
        self.index = int(state_dict.get("index", n) % self.capacity)
        self.size = int(n)


def soft_update(target, source, tau):
    with torch.no_grad():
        for target_param, source_param in zip(target.parameters(), source.parameters()):
            target_param.data.mul_(tau).add_(source_param.data, alpha=1.0 - tau)


def make_sa_sac_agent(cfg, params):
    if cfg.multi_agent.use:
        raise NotImplementedError("Plain PyTorch SAC currently supports single-agent mode only.")

    state_dim = params["n_turbines"] * params["probes_per_turbine"] * len(params["flow_field_directions"]) + params["n_turbines"]
    action_dim = params["n_turbines"]

    actor = SACActor(
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_depth=cfg.network.actor_hidden_depth,
        hidden_size=cfg.network.actor_hidden_size,
        action_low=-1.0,
        action_high=1.0,
    )
    critic1 = SACCritic(
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_depth=cfg.network.critic_hidden_depth,
        hidden_size=cfg.network.critic_hidden_size,
    )
    critic2 = SACCritic(
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_depth=cfg.network.critic_hidden_depth,
        hidden_size=cfg.network.critic_hidden_size,
    )
    return actor, critic1, critic2


def save_model(cfg, actor, critic, filepath, id):
    os.makedirs(filepath, exist_ok=True)
    torch.save(actor.state_dict(), os.path.join(filepath, f"actor_{id}.pkl"))
    torch.save(critic.state_dict(), os.path.join(filepath, f"critic_{id}.pkl"))
    return True


def load_model(cfg, env_params, path_to_model, id):
    actor, critic1, _ = make_sa_sac_agent(cfg, env_params)
    actor_path = os.path.join(path_to_model, f"actor_{id}.pkl")
    critic_path = os.path.join(path_to_model, f"critic_{id}.pkl")
    if not os.path.exists(actor_path) or not os.path.exists(critic_path):
        raise FileNotFoundError(f"Missing checkpoint files: {actor_path} or {critic_path}")
    actor.load_state_dict(torch.load(actor_path, map_location="cpu"))
    critic1.load_state_dict(torch.load(critic_path, map_location="cpu"))
    return actor, critic1


def save_sac_checkpoint(cfg, actor, critic1, critic2, log_alpha, filepath, id, replay_buffer=None):
    os.makedirs(filepath, exist_ok=True)
    checkpoint = {
        "actor": actor.state_dict(),
        "critic1": critic1.state_dict(),
        "critic2": critic2.state_dict(),
        "log_alpha": log_alpha.detach().cpu(),
    }
    torch.save(checkpoint, os.path.join(filepath, f"sac_checkpoint_{id}.pt"))
    torch.save(actor.state_dict(), os.path.join(filepath, f"actor_{id}.pkl"))
    torch.save(critic1.state_dict(), os.path.join(filepath, f"critic_{id}.pkl"))
    torch.save(critic2.state_dict(), os.path.join(filepath, f"critic2_{id}.pkl"))
    if replay_buffer is not None:
        torch.save(replay_buffer.state_dict(), os.path.join(filepath, f"replay_buffer_{id}.pt"))


def load_sac_checkpoint(cfg, env_params, path_to_model, id):
    actor, critic1, critic2 = make_sa_sac_agent(cfg, env_params)
    checkpoint_path = os.path.join(path_to_model, f"sac_checkpoint_{id}.pt")
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        actor.load_state_dict(checkpoint["actor"])
        critic1.load_state_dict(checkpoint["critic1"])
        critic2.load_state_dict(checkpoint["critic2"])
        log_alpha = torch.nn.Parameter(checkpoint["log_alpha"].clone().detach().float())
        return actor, critic1, critic2, log_alpha

    actor_path = os.path.join(path_to_model, f"actor_{id}.pkl")
    critic_path = os.path.join(path_to_model, f"critic_{id}.pkl")
    critic2_path = os.path.join(path_to_model, f"critic2_{id}.pkl")
    if not os.path.exists(actor_path) or not os.path.exists(critic_path):
        raise FileNotFoundError(f"Missing checkpoint files: {actor_path} or {critic_path}")
    actor.load_state_dict(torch.load(actor_path, map_location="cpu"))
    critic1.load_state_dict(torch.load(critic_path, map_location="cpu"))
    if os.path.exists(critic2_path):
        critic2.load_state_dict(torch.load(critic2_path, map_location="cpu"))
    else:
        critic2.load_state_dict(deepcopy(critic1.state_dict()))
    log_alpha = torch.nn.Parameter(torch.tensor(math.log(float(cfg.optim.alpha_init)), dtype=torch.float32))
    return actor, critic1, critic2, log_alpha


def log_metrics(logs, metrics):
    for metric_name, metric_value in metrics.items():
        logs.setdefault(metric_name, []).append(metric_value)
    output_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir + "/"
    with open(output_dir + "logs.pkl", "wb") as f:
        pickle.dump(logs, f)


def should_log_now(cfg, frames, num_console_updates):
    return True


def update_data_shapes(train_env, data):
    return data
