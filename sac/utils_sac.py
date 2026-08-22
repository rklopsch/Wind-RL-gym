import hydra
import math
import os
import pickle
import multiprocessing as mp
import time
import traceback
from multiprocessing.connection import wait
from copy import deepcopy

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
    """Process-based vector environment.

    Each SmartRedis-connected environment must live in its own process to avoid
    shared-client/thread synchronization stalls.
    """

    def __init__(
        self,
        env_ctor,
        env_kwargs_list,
        worker_timeout_s=1200.0,
        start_method="spawn",
        parent_poll_interval_s=0.1,
        close_timeout_s=5.0,
    ):
        self.num_envs = len(env_kwargs_list)
        self.worker_timeout_s = float(worker_timeout_s)
        self.parent_poll_interval_s = float(parent_poll_interval_s)
        self.close_timeout_s = float(close_timeout_s)
        if self.num_envs <= 0:
            raise ValueError("ParallelGymEnv requires at least one environment")
        self._ctx = mp.get_context(start_method)
        self._workers = []
        self._closed = False
        for env_kwargs in env_kwargs_list:
            parent_conn, child_conn = self._ctx.Pipe(duplex=True)
            proc = self._ctx.Process(
                target=_env_worker_main,
                args=(child_conn, env_ctor, env_kwargs),
            )
            proc.daemon = True
            proc.start()
            child_conn.close()
            instance = env_kwargs.get("instance_label", env_kwargs.get("instance", len(self._workers)))
            self._workers.append(
                {
                    "process": proc,
                    "conn": parent_conn,
                    "instance": instance if instance is not None else len(self._workers),
                    "buffer": {},
                }
            )
        self.action_space = self._call_single_worker("get_action_space")
        self.observation_space = self._call_single_worker("get_observation_space")

    def _call_single_worker(self, command):
        worker = self._workers[0]
        req_id = f"{command}-{time.monotonic_ns()}"
        worker["conn"].send({"command": command, "request_id": req_id})
        msg = self._recv_for_worker(worker, req_id, self.worker_timeout_s, command)
        return msg["value"]

    def _recv_for_worker(self, worker, request_id, timeout_s, operation_name):
        conn = worker["conn"]
        buffer = worker["buffer"]
        if request_id in buffer:
            message = buffer.pop(request_id)
            if message.get("status") == "error":
                tb = message.get("traceback", "")
                raise RuntimeError(
                    f"Worker instance {worker['instance']} failed during {operation_name}: "
                    f"{message.get('error')}\n{tb}"
                )
            return message
        deadline = time.monotonic() + timeout_s
        errors = []
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                detail = f"worker instance {worker['instance']} did not respond to {operation_name} within {timeout_s:.2f}s"
                if errors:
                    detail += f"; errors: {' | '.join(errors)}"
                raise TimeoutError(detail)
            ready = wait([conn], timeout=min(self.parent_poll_interval_s, remaining))
            if not ready:
                continue
            try:
                message = conn.recv()
            except EOFError as exc:
                raise RuntimeError(f"Worker instance {worker['instance']} terminated unexpectedly during {operation_name}") from exc
            if message.get("request_id") != request_id:
                msg_req = message.get("request_id")
                if msg_req is not None:
                    buffer[msg_req] = message
                continue
            if message.get("status") == "error":
                tb = message.get("traceback", "")
                raise RuntimeError(
                    f"Worker instance {worker['instance']} failed during {operation_name}: "
                    f"{message.get('error')}\n{tb}"
                )
            return message

    def _collect_batch_results(self, request_id, operation_name, timeout_s):
        pending = {idx for idx in range(self.num_envs)}
        results = {}
        worker_errors = []
        deadline = time.monotonic() + timeout_s
        while pending:
            buffered_progress = False
            for worker_idx in sorted(list(pending)):
                worker = self._workers[worker_idx]
                buffer = worker["buffer"]
                if request_id not in buffer:
                    continue
                message = buffer.pop(request_id)
                pending.remove(worker_idx)
                buffered_progress = True
                if message.get("status") == "error":
                    worker_errors.append(
                        f"instance {worker['instance']}: {message.get('error')}\n{message.get('traceback', '')}"
                    )
                else:
                    results[worker_idx] = message["value"]
            if buffered_progress:
                continue

            remaining = deadline - time.monotonic()
            if remaining <= 0:
                unfinished = [self._workers[idx]["instance"] for idx in sorted(pending)]
                details = (
                    f"Timed out waiting for {operation_name} after {timeout_s:.2f}s. "
                    f"Unfinished environment instances: {unfinished}."
                )
                if worker_errors:
                    details += f" Worker errors already received: {' | '.join(worker_errors)}"
                raise TimeoutError(details)
            ready_conns = wait(
                [self._workers[idx]["conn"] for idx in sorted(pending)],
                timeout=min(self.parent_poll_interval_s, remaining),
            )
            if not ready_conns:
                continue
            for conn in ready_conns:
                worker_idx = next(
                    idx for idx in pending if self._workers[idx]["conn"] is conn
                )
                worker = self._workers[worker_idx]
                try:
                    message = conn.recv()
                except EOFError:
                    worker_errors.append(
                        f"instance {worker['instance']} connection closed unexpectedly during {operation_name}"
                    )
                    pending.remove(worker_idx)
                    continue
                if message.get("request_id") != request_id:
                    msg_req = message.get("request_id")
                    if msg_req is not None:
                        worker["buffer"][msg_req] = message
                    continue
                pending.remove(worker_idx)
                if message.get("status") == "error":
                    worker_errors.append(
                        f"instance {worker['instance']}: {message.get('error')}\n{message.get('traceback', '')}"
                    )
                    continue
                results[worker_idx] = message["value"]
        if worker_errors:
            raise RuntimeError(
                f"Worker errors during {operation_name}: {' | '.join(worker_errors)}"
            )
        return [results[idx] for idx in range(self.num_envs)]

    def reset(self):
        request_id = f"reset-{time.monotonic_ns()}"
        for worker in self._workers:
            worker["conn"].send({"command": "reset", "request_id": request_id})
        results = self._collect_batch_results(request_id, "reset", self.worker_timeout_s)
        obs_list = [obs for obs, _ in results]
        infos = [info for _, info in results]
        return stack_obs_dict(obs_list), infos

    def step(self, actions):
        actions = torch.as_tensor(actions)
        if actions.dim() == 1:
            actions = actions.unsqueeze(0)
        if actions.shape[0] != self.num_envs:
            raise ValueError(f"Expected actions for {self.num_envs} envs, got shape {tuple(actions.shape)}")
        request_id = f"step-{time.monotonic_ns()}"
        for idx, worker in enumerate(self._workers):
            worker["conn"].send(
                {
                    "command": "step",
                    "request_id": request_id,
                    "action": actions[idx].detach().cpu(),
                }
            )
        results = self._collect_batch_results(request_id, "step", self.worker_timeout_s)
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
        if self._closed:
            return None
        self._closed = True
        request_id = f"close-{time.monotonic_ns()}"
        for worker in self._workers:
            if worker["process"].is_alive():
                try:
                    worker["conn"].send({"command": "close", "request_id": request_id})
                except (BrokenPipeError, EOFError):
                    pass

        deadline = time.monotonic() + self.close_timeout_s
        pending = {idx for idx, worker in enumerate(self._workers) if worker["process"].is_alive()}
        while pending and time.monotonic() < deadline:
            ready_conns = wait(
                [self._workers[idx]["conn"] for idx in sorted(pending)],
                timeout=self.parent_poll_interval_s,
            )
            for conn in ready_conns:
                idx = next(i for i in pending if self._workers[i]["conn"] is conn)
                try:
                    message = conn.recv()
                except EOFError:
                    pending.remove(idx)
                    continue
                if message.get("request_id") == request_id:
                    pending.remove(idx)

        for worker in self._workers:
            proc = worker["process"]
            proc.join(timeout=0.2)
            if proc.is_alive():
                proc.terminate()
                proc.join(timeout=1.0)
            worker["conn"].close()
        return None


def make_env(cfg, params, instance=None, save=False, device="cpu", eval_only=False):
    from WF_enviroment import TurbEnv

    return TurbEnv(params, multi_agent=cfg.multi_agent.use, save=save, instance=instance, device=device)


class DummyTurbEnv:
    def __init__(self, params, multi_agent=False, save=False, instance=None, device="cpu"):
        self.instance = 0 if instance is None else int(instance)
        self.n_turbines = int(params["n_turbines"])
        probes_per_turbine = int(params["probes_per_turbine"])
        self.obs_size = int(self.n_turbines * probes_per_turbine * len(params["flow_field_directions"]))
        self.episode_length = int(params.get("episode_length", 200))
        self._step = 0
        self.action_space = {"shape": (self.n_turbines,), "low": -1.0, "high": 1.0}
        self.observation_space = {
            "alpha": {"shape": (self.n_turbines,)},
            "alpha_normalised": {"shape": (self.n_turbines,)},
            "observation": {"shape": (self.obs_size,)},
            "reward_buffer": {"shape": (1,)},
            "power": {"shape": (1,)},
        }

    def _obs(self):
        alpha = np.zeros((self.n_turbines,), dtype=np.float32)
        return {
            "alpha": alpha,
            "alpha_normalised": alpha,
            "observation": np.full((self.obs_size,), float(self.instance + self._step), dtype=np.float32),
            "reward_buffer": np.array([float(self._step)], dtype=np.float32),
            "power": np.array([float(self.instance)], dtype=np.float32),
        }

    def reset(self, seed=None, options=None):
        self._step = 0
        return self._obs(), {"dummy": True, "instance": self.instance}

    def step(self, action):
        self._step += 1
        reward = float(self.n_turbines) - float(np.mean(np.abs(np.asarray(action, dtype=np.float32))))
        terminated = False
        truncated = self._step >= self.episode_length
        return self._obs(), reward, terminated, truncated, {"dummy": True, "instance": self.instance}

    def close(self):
        return None


def make_parallel_env(cfg, params, num_envs, device="cpu", eval_only=False):
    use_dummy_env = bool(getattr(cfg.env, "dummy_update", False))
    if use_dummy_env:
        env_ctor = DummyTurbEnv
    else:
        from WF_enviroment import TurbEnv

        env_ctor = TurbEnv

    env_kwargs_list = [
        {
            "params": params,
            "multi_agent": cfg.multi_agent.use,
            "save": False,
            "instance": i,
            "instance_label": i + 1,
            "device": device,
        }
        for i in range(num_envs)
    ]
    return ParallelGymEnv(
        env_ctor=env_ctor,
        env_kwargs_list=env_kwargs_list,
        worker_timeout_s=float(getattr(cfg.env, "worker_timeout_s", 1200.0)),
        start_method=str(getattr(cfg.env, "worker_start_method", "spawn")),
        parent_poll_interval_s=float(getattr(cfg.env, "worker_poll_interval_s", 0.1)),
        close_timeout_s=float(getattr(cfg.env, "worker_close_timeout_s", 5.0)),
    )


def _env_worker_main(conn, env_ctor, env_kwargs):
    env = None
    current_request_id = None
    try:
        ctor_kwargs = dict(env_kwargs)
        ctor_kwargs.pop("instance_label", None)
        env = env_ctor(**ctor_kwargs)
        while True:
            message = conn.recv()
            command = message.get("command")
            current_request_id = message.get("request_id")
            if command == "get_action_space":
                conn.send({"status": "ok", "request_id": current_request_id, "value": env.action_space})
            elif command == "get_observation_space":
                conn.send({"status": "ok", "request_id": current_request_id, "value": env.observation_space})
            elif command == "reset":
                value = env.reset()
                conn.send({"status": "ok", "request_id": current_request_id, "value": value})
            elif command == "step":
                obs, reward, term, trunc, info = env.step(message.get("action"))
                final_obs = obs
                if term or trunc:
                    reset_obs, reset_info = env.reset()
                    info = dict(info)
                    info["final_observation"] = final_obs
                    info["final_info"] = reset_info
                    obs = reset_obs
                conn.send(
                    {
                        "status": "ok",
                        "request_id": current_request_id,
                        "value": (obs, reward, term, trunc, info),
                    }
                )
            elif command == "close":
                close_fn = getattr(env, "close", None)
                if callable(close_fn):
                    close_fn()
                conn.send({"status": "ok", "request_id": current_request_id, "value": None})
                break
            else:
                raise ValueError(f"Unknown worker command: {command}")
    except Exception as exc:
        tb = traceback.format_exc()
        try:
            conn.send(
                {
                    "status": "error",
                    "request_id": current_request_id,
                    "error": repr(exc),
                    "traceback": tb,
                }
            )
        except Exception:
            pass
    finally:
        if env is not None:
            close_fn = getattr(env, "close", None)
            if callable(close_fn):
                try:
                    close_fn()
                except Exception:
                    pass
        try:
            conn.close()
        except Exception:
            pass


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
