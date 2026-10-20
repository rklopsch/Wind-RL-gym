import hydra
import logging
import math
import os
import pickle
import multiprocessing as mp
import time
import traceback
from collections import deque
from multiprocessing.connection import wait
from copy import deepcopy

import numpy as np
import torch
from torch import nn


def consume_collection_batches(pending_frames, new_frames, frames_per_batch):
    pending = int(pending_frames) + int(new_frames)
    batch = int(frames_per_batch)
    if batch <= 0:
        raise ValueError("frames_per_batch must be positive")
    completed = pending // batch
    remaining = pending - completed * batch
    return completed, remaining


def compute_console_log_interval(total_frames, num_console_updates, frames_per_batch):
    total = max(1, int(total_frames))
    requested = max(1, int(num_console_updates))
    batch_floor = max(1, int(frames_per_batch))
    target_interval = max(batch_floor, math.ceil(total / requested))
    return batch_floor * max(1, math.ceil(target_interval / batch_floor))


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


def _stack_observation_space(space, key, num_frames):
    spaces_dict = getattr(space, "spaces", space)
    if not isinstance(spaces_dict, dict) or key not in spaces_dict:
        return space
    sub_space = spaces_dict[key]
    if isinstance(sub_space, dict):
        shape = tuple(sub_space["shape"])
        stacked = dict(sub_space)
        stacked["shape"] = (*shape[:-1], num_frames * shape[-1])
    else:
        shape = tuple(sub_space.shape)
        stacked = type(sub_space)(
            low=np.concatenate([np.asarray(sub_space.low)] * num_frames, axis=-1),
            high=np.concatenate([np.asarray(sub_space.high)] * num_frames, axis=-1),
            shape=(*shape[:-1], num_frames * shape[-1]),
            dtype=sub_space.dtype,
        )
    new_spaces = dict(spaces_dict)
    new_spaces[key] = stacked
    if space is spaces_dict:
        return new_spaces
    return type(space)(new_spaces)


class FrameStackEnv:
    """Concatenates the last ``num_frames`` entries of ``obs[key]`` along the last dim (oldest first).

    Mirrors torchrl's ``CatFrames(N=num_frames, dim=-1, in_keys=["observation"])`` with its default
    ``padding="same"``: after a reset, all frames are filled with the reset observation.
    """

    def __init__(self, env, num_frames, key="observation"):
        self.env = env
        self.num_frames = int(num_frames)
        if self.num_frames < 1:
            raise ValueError("num_frames must be >= 1")
        self.key = key
        self._frames = deque(maxlen=self.num_frames)
        self.action_space = env.action_space
        self.observation_space = _stack_observation_space(env.observation_space, key, self.num_frames)

    def _stacked_obs(self, obs):
        obs = dict(obs)
        obs[self.key] = torch.cat(list(self._frames), dim=-1)
        return obs

    def reset(self, *args, **kwargs):
        obs, info = self.env.reset(*args, **kwargs)
        frame = torch.as_tensor(obs[self.key]).clone()
        self._frames.clear()
        for _ in range(self.num_frames):
            self._frames.append(frame)
        return self._stacked_obs(obs), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self._frames.append(torch.as_tensor(obs[self.key]).clone())
        return self._stacked_obs(obs), reward, terminated, truncated, info

    def close(self):
        close_fn = getattr(self.env, "close", None)
        if callable(close_fn):
            return close_fn()
        return None


def make_env(cfg, params, instance=None, save=False, device="cpu", eval_only=False):
    from WF_enviroment import TurbEnv

    env = TurbEnv(params, multi_agent=cfg.multi_agent.use, save=save, instance=instance, device=device)
    frame_stack = get_frame_stack(cfg)
    if frame_stack > 1:
        env = FrameStackEnv(env, frame_stack)
    return env


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
            "frame_stack": get_frame_stack(cfg),
            "worker_omp_num_threads": int(getattr(cfg.env, "worker_omp_num_threads", 1)),
            "worker_mkl_num_threads": int(getattr(cfg.env, "worker_mkl_num_threads", 1)),
            "worker_torch_num_threads": int(getattr(cfg.env, "worker_torch_num_threads", 1)),
            "worker_torch_num_interop_threads": int(getattr(cfg.env, "worker_torch_num_interop_threads", 1)),
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
        frame_stack = int(ctor_kwargs.pop("frame_stack", 1) or 1)
        worker_omp_threads = ctor_kwargs.pop("worker_omp_num_threads", None)
        worker_mkl_threads = ctor_kwargs.pop("worker_mkl_num_threads", None)
        worker_torch_threads = ctor_kwargs.pop("worker_torch_num_threads", None)
        worker_torch_interop_threads = ctor_kwargs.pop("worker_torch_num_interop_threads", None)

        if worker_omp_threads is not None:
            os.environ["OMP_NUM_THREADS"] = str(int(worker_omp_threads))
        if worker_mkl_threads is not None:
            os.environ["MKL_NUM_THREADS"] = str(int(worker_mkl_threads))
        if worker_torch_threads is not None:
            torch.set_num_threads(max(1, int(worker_torch_threads)))
        if worker_torch_interop_threads is not None:
            try:
                torch.set_num_interop_threads(max(1, int(worker_torch_interop_threads)))
            except RuntimeError:
                # torch can only set inter-op threads before parallel work starts
                pass

        affinity_count = None
        if hasattr(os, "sched_getaffinity"):
            affinity_count = len(os.sched_getaffinity(0))
        logging.info(
            "Env worker startup instance=%s pid=%s affinity_cpus=%s torch_threads=%s torch_interop_threads=%s OMP_NUM_THREADS=%s MKL_NUM_THREADS=%s",
            ctor_kwargs.get("instance", "unknown"),
            os.getpid(),
            affinity_count if affinity_count is not None else "n/a",
            torch.get_num_threads(),
            torch.get_num_interop_threads(),
            os.environ.get("OMP_NUM_THREADS"),
            os.environ.get("MKL_NUM_THREADS"),
        )
        env = env_ctor(**ctor_kwargs)
        if frame_stack > 1:
            env = FrameStackEnv(env, frame_stack)
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


def build_next_state(next_obs, infos, episode_end):
    """State reached by each env's action.

    The workers auto-reset finished episodes, so ``next_obs`` already holds the reset observation for those
    envs. The transition must instead use the true last observation, ``info["final_observation"]``.
    """
    next_state = build_state(next_obs)
    episode_end = torch.as_tensor(episode_end, dtype=torch.bool).reshape(-1)
    for idx in torch.nonzero(episode_end, as_tuple=False).flatten().tolist():
        final_obs = infos[idx].get("final_observation") if infos[idx] is not None else None
        if final_obs is None:
            raise KeyError(f"Environment {idx} ended its episode but returned no 'final_observation' in info.")
        next_state[idx] = build_state(final_obs)[0]
    return next_state


def _inv_softplus(value):
    # Same as tensordict.nn.utils.inv_softplus: softplus(_inv_softplus(v)) == v.
    return math.log(max(math.expm1(float(value)), 1e-6))


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
    """Tanh-Normal policy equivalent to the former torchrl actor.

    The scale is parametrised like tensordict's
    ``NormalParamExtractor(scale_mapping="biased_softplus_{default_scale}", scale_lb=scale_lb)``:
    ``scale = max(softplus(x + inv_softplus(default_scale - min_val)) + min_val, scale_lb)``,
    so a zero network output maps to ``default_scale`` and the scale never drops below ``scale_lb``.
    """

    def __init__(
        self,
        state_dim,
        action_dim,
        hidden_depth,
        hidden_size,
        action_low,
        action_high,
        default_scale=1.0,
        scale_lb=0.1,
        scale_min_val=0.01,
    ):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.backbone = MLP(state_dim, hidden_depth, hidden_size, 2 * action_dim, activation=nn.ReLU)
        if float(default_scale) <= float(scale_min_val):
            raise ValueError(f"default_scale ({default_scale}) must be greater than scale_min_val ({scale_min_val}).")
        self.scale_min_val = float(scale_min_val)
        self.scale_lb = float(scale_lb)
        self.scale_bias = _inv_softplus(float(default_scale) - self.scale_min_val)
        action_low = torch.as_tensor(action_low, dtype=torch.float32)
        action_high = torch.as_tensor(action_high, dtype=torch.float32)
        self.register_buffer("action_scale", (action_high - action_low) / 2.0)
        self.register_buffer("action_bias", (action_high + action_low) / 2.0)

    def forward(self, state):
        net_out = self.backbone(state)
        mean, scale_param = net_out.chunk(2, dim=-1)
        std = nn.functional.softplus(scale_param + self.scale_bias) + self.scale_min_val
        std = std.clamp_min(self.scale_lb)
        return mean, std

    def sample(self, state, deterministic=False):
        mean, std = self.forward(state)
        if deterministic:
            pre_tanh = mean
        else:
            pre_tanh = mean + std * torch.randn_like(mean)

        action = torch.tanh(pre_tanh) * self.action_scale + self.action_bias

        log_prob = None
        if not deterministic:
            log_prob = torch.distributions.Normal(mean, std).log_prob(pre_tanh)
            # Numerically stable log|d tanh(x)/dx| = 2 * (log 2 - x - softplus(-2x)),
            # identical to torch.distributions.TanhTransform used by torchrl's TanhNormal.
            log_prob = log_prob - 2.0 * (math.log(2.0) - pre_tanh - nn.functional.softplus(-2.0 * pre_tanh))
            log_prob = log_prob - torch.log(self.action_scale)
            log_prob = log_prob.sum(dim=-1, keepdim=True)
        return action, log_prob

    def sample_uniform(self, batch_size):
        """Uniform random actions within the action bounds (torchrl collector ``rand_action``)."""
        noise = 2.0 * torch.rand((int(batch_size), self.action_dim), device=self.action_scale.device) - 1.0
        return noise * self.action_scale + self.action_bias


class SACCritic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_depth, hidden_size):
        super().__init__()
        self.q = MLP(state_dim + action_dim, hidden_depth, hidden_size, 1, activation=nn.ReLU)

    def forward(self, state, action):
        x = torch.cat([state, action], dim=-1)
        return self.q(x)


class ReplayBuffer:
    """Uniform replay buffer.

    ``done`` marks the end of an episode (terminated or truncated) and is kept for analysis of saved buffers.
    ``terminated`` marks true terminal states and is the only flag that stops bootstrapping in the TD target
    (same as torchrl's TD0 estimator), so time-limit truncations are bootstrapped.
    """

    def __init__(self, capacity, state_dim, action_dim):
        self.capacity = int(capacity)
        self.state = torch.zeros((self.capacity, state_dim), dtype=torch.float32)
        self.action = torch.zeros((self.capacity, action_dim), dtype=torch.float32)
        self.reward = torch.zeros((self.capacity, 1), dtype=torch.float32)
        self.next_state = torch.zeros((self.capacity, state_dim), dtype=torch.float32)
        self.done = torch.zeros((self.capacity, 1), dtype=torch.bool)
        self.terminated = torch.zeros((self.capacity, 1), dtype=torch.bool)
        self.index = 0
        self.size = 0

    def _store_one(self, state, action, reward, next_state, done, terminated):
        self.state[self.index] = state
        self.action[self.index] = action
        self.reward[self.index] = reward
        self.next_state[self.index] = next_state
        self.done[self.index] = done
        self.terminated[self.index] = terminated
        self.index = (self.index + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    @staticmethod
    def _as_column(x, dtype):
        x = torch.as_tensor(x, dtype=dtype).detach().cpu()
        return x.reshape(x.shape[0], -1) if x.dim() > 1 else x.unsqueeze(-1)

    def add(self, state, action, reward, next_state, done, terminated):
        state = _flatten_batch(torch.as_tensor(state, dtype=torch.float32).detach().cpu())
        action = _flatten_batch(torch.as_tensor(action, dtype=torch.float32).detach().cpu())
        reward = self._as_column(reward, torch.float32)
        next_state = _flatten_batch(torch.as_tensor(next_state, dtype=torch.float32).detach().cpu())
        done = self._as_column(done, torch.bool)
        terminated = self._as_column(terminated, torch.bool)

        for i in range(state.shape[0]):
            self._store_one(state[i], action[i], reward[i], next_state[i], done[i], terminated[i])

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
            "terminated": self.terminated[idx].to(device),
        }

    def state_dict(self):
        return {
            "capacity": self.capacity,
            "state": self.state[: self.size].clone(),
            "action": self.action[: self.size].clone(),
            "reward": self.reward[: self.size].clone(),
            "next_state": self.next_state[: self.size].clone(),
            "done": self.done[: self.size].clone(),
            "terminated": self.terminated[: self.size].clone(),
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
        # Buffers saved before "terminated" was stored: TurbEnv never terminates, only truncates.
        if "terminated" in state_dict:
            self.terminated[:n] = state_dict["terminated"][:n]
        else:
            self.terminated[:n] = False
        self.index = int(state_dict.get("index", n) % self.capacity)
        self.size = int(n)


def soft_update(target, source, tau):
    with torch.no_grad():
        for target_param, source_param in zip(target.parameters(), source.parameters()):
            target_param.data.mul_(tau).add_(source_param.data, alpha=1.0 - tau)


def get_frame_stack(cfg):
    return max(1, int(getattr(cfg.env, "frame_stack", 1)))


def make_sa_sac_agent(cfg, params):
    if cfg.multi_agent.use:
        raise NotImplementedError("Plain PyTorch SAC currently supports single-agent mode only.")

    obs_dim = params["n_turbines"] * params["probes_per_turbine"] * len(params["flow_field_directions"])
    state_dim = get_frame_stack(cfg) * obs_dim + params["n_turbines"]
    action_dim = params["n_turbines"]

    actor = SACActor(
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_depth=cfg.network.actor_hidden_depth,
        hidden_size=cfg.network.actor_hidden_size,
        action_low=-1.0,
        action_high=1.0,
        default_scale=float(cfg.network.default_policy_scale),
        scale_lb=float(cfg.network.scale_lb),
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


def persist_logs(logs, output_dir=None):
    if output_dir is None:
        try:
            output_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
        except Exception:
            output_dir = os.getcwd()
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "logs.pkl"), "wb") as f:
        pickle.dump(logs, f)


def should_log_now(cfg, frames, num_console_updates, previous_frames=None):
    interval = compute_console_log_interval(
        total_frames=cfg.collector.total_frames,
        num_console_updates=num_console_updates,
        frames_per_batch=cfg.collector.frames_per_batch,
    )
    prev = 0 if previous_frames is None else int(previous_frames)
    curr = int(frames)
    if curr >= int(cfg.collector.total_frames):
        return True
    return (prev // interval) < (curr // interval)


def update_data_shapes(train_env, data):
    return data
