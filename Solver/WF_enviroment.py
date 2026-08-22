import logging
import math
from typing import Any, Dict as TypingDict, Optional, Tuple

import numpy as np
import torch
from smartredis import Client

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError:
    class _Space:
        pass

    class Box(_Space):
        def __init__(self, low, high, shape, dtype=np.float32):
            self.low = np.full(shape, low, dtype=dtype)
            self.high = np.full(shape, high, dtype=dtype)
            self.shape = tuple(shape)
            self.dtype = dtype

    class DictSpace(_Space):
        def __init__(self, spaces_dict):
            self.spaces = spaces_dict

    class Env:
        metadata = {}

        def reset(self, seed=None, options=None):
            return None

    gym = type("gym", (), {"Env": Env})
    spaces = type("spaces", (), {"Box": Box, "Dict": DictSpace})


def ring_insert_shift(buffer: torch.Tensor, new_value: float) -> torch.Tensor:
    buffer[:-1] = buffer[1:].clone()
    buffer[-1] = new_value
    return buffer


class TurbEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(
        self,
        params: Optional[TypingDict[str, Any]] = None,
        multi_agent: bool = False,
        seed: Optional[int] = None,
        save: bool = False,
        instance: Optional[int] = None,
        device: str = "cpu",
    ):
        if params is None:
            raise ValueError("params must be provided")
        if multi_agent:
            raise NotImplementedError("The plain Gym environment currently supports single-agent mode only.")

        self.save = save
        self.device = torch.device(device)
        self.probes_per_turbine = params["probes_per_turbine"]
        lookup = {"ux": 0, "uy": 1, "uz": 2}
        flow_dirs = params["flow_field_directions"]
        if not all(p in lookup for p in flow_dirs):
            raise ValueError(
                "flow_field_directions must only contain 'ux', 'uy', 'uz'. "
                f"Got {flow_dirs}"
            )
        if len(set(flow_dirs)) != len(flow_dirs):
            raise ValueError(f"flow_field_directions must not contain duplicates. Got {flow_dirs}.")

        self.obs_idxs = list(map(lookup.get, flow_dirs))
        self.obs_per_probe = len(self.obs_idxs)
        self.obs_per_turbine = self.probes_per_turbine * self.obs_per_probe
        self.n_turbs = params["n_turbines"]
        self.total_probes = self.probes_per_turbine * self.n_turbs
        self.total_obs = self.obs_per_turbine * self.n_turbs
        self.max_speed = params["max_yaw_speed"]
        self.max_angle = params["max_yaw_angle"]
        self.dt = params["dt"]
        self.reset_frames = params["reset_frames"]
        self.max_episode_steps = int(params.get("episode_length", params.get("run_steps", 1)))
        self.instance = 0 if instance is None else instance + 1
        self.penalty_scale = params["penalty_scale"]
        self.penalty_exponent = params["penalty_exp"]
        self.random_reset = bool(params["random_reset"])
        self.initial_angles = np.asarray(params["initial_angles"], dtype=np.float32)
        if len(self.initial_angles) != self.n_turbs:
            raise ValueError(
                f"Number of initial angles {len(self.initial_angles)} does not match turbine count {self.n_turbs}."
            )
        self.reward_average_steps = int(params["reward_average_steps"])
        self.velocity_penalty_scale = params["velocity_penalty_scale"]
        self.difference_penalty_scale = params["difference_penalty_scale"]

        self.client = Client(address=None, cluster=False)
        self.client.put_tensor(f"{self.instance}_yaws_done", np.array([0]))
        self.client.put_tensor(f"{self.instance}_sim_done", np.array([0]))

        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(self.n_turbs,), dtype=np.float32)
        self.observation_space = spaces.Dict(
            {
                "alpha": spaces.Box(low=-self.max_angle, high=self.max_angle, shape=(self.n_turbs,), dtype=np.float32),
                "alpha_normalised": spaces.Box(low=-1.0, high=1.0, shape=(self.n_turbs,), dtype=np.float32),
                "observation": spaces.Box(
                    low=-np.inf, high=np.inf, shape=(self.total_obs,), dtype=np.float32
                ),
                "reward_buffer": spaces.Box(
                    low=-np.inf, high=np.inf, shape=(self.reward_average_steps,), dtype=np.float32
                ),
                "power": spaces.Box(low=-np.inf, high=np.inf, shape=(self.n_turbs,), dtype=np.float32),
            }
        )

        if seed is None:
            seed = torch.empty((), dtype=torch.int64).random_().item()
        self._set_seed(seed)

        self._alpha = torch.tensor(self.initial_angles, dtype=torch.float32, device=self.device)
        self._reward_buffer = -1.0 * torch.ones((self.reward_average_steps,), dtype=torch.float32, device=self.device)
        self._step_count = 0

    @staticmethod
    def _normalise_probe_data(arr):
        loc = np.array([6.0, 0.0, 0.0])
        scale = np.array([6.0, 4.0, 4.0])
        return (arr - loc) / scale

    @staticmethod
    def _position_encoding(n_turbs):
        idxs = torch.arange(n_turbs).view([n_turbs, 1])
        idxs = (torch.pi / 2) * idxs / (n_turbs - 1)
        return torch.sin(idxs)

    def _communicate(self, new_alpha: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        self.client.put_tensor(f"{self.instance}_yaws", new_alpha.detach().cpu().numpy().squeeze().astype(np.float64))
        self.client.put_tensor(f"{self.instance}_yaws_done", np.ones(1))

        while not self.client.poll_key(f"{self.instance}_sim_done", 100, 10) or not bool(
            self.client.get_tensor(f"{self.instance}_sim_done")[0]
        ):
            continue

        self.client.put_tensor(f"{self.instance}_sim_done", np.array([0]))

        turbine_powers = self.client.get_tensor(f"{self.instance}_turbine_powers")
        turbine_obs = np.zeros((self.total_probes, self.obs_per_probe))
        for i in range(self.total_probes):
            probe = self.client.get_tensor(f"{self.instance}_probe_{i + 1}")
            turbine_obs[i] = self._normalise_probe_data(probe)[self.obs_idxs]

        turbine_obs = turbine_obs.reshape(self.n_turbs, self.probes_per_turbine * self.obs_per_probe)
        turbine_powers = torch.tensor(turbine_powers / 1e06, dtype=torch.float32, device=self.device)
        observation = torch.tensor(turbine_obs, dtype=torch.float32, device=self.device)
        return turbine_powers, observation

    def _make_obs(self, alpha: torch.Tensor, observation: torch.Tensor, power: torch.Tensor) -> TypingDict[str, torch.Tensor]:
        return {
            "alpha": alpha.detach().cpu(),
            "alpha_normalised": (alpha / self.max_angle).detach().cpu(),
            "observation": observation.reshape(-1).detach().cpu(),
            "reward_buffer": self._reward_buffer.detach().cpu(),
            "power": power.detach().cpu(),
        }

    def _warmup_reset(self, alpha_start: torch.Tensor, alpha_target: torch.Tensor) -> None:
        steps_to_change_angle = max(1, math.ceil(self.max_angle / (self.max_speed * self.dt)))
        if steps_to_change_angle > self.reset_frames:
            raise ValueError(
                f"Must have at least {steps_to_change_angle} reset frames. Only have {self.reset_frames}."
            )

        for i in range(steps_to_change_angle):
            denom = max(1, steps_to_change_angle - 1)
            interpolated_angle = (steps_to_change_angle - 1 - i) / denom * alpha_start + i / denom * alpha_target
            self._communicate(interpolated_angle)

        for _ in range(self.reset_frames - steps_to_change_angle):
            self._communicate(alpha_target)

    def reset(self, seed: Optional[int] = None, options: Optional[TypingDict[str, Any]] = None):
        super().reset(seed=seed)
        logging.info("Resetting now")

        if self.random_reset:
            reset_angles = 0.75 * self.max_angle * (2 * torch.rand([self.n_turbs], device=self.device) - 1)
        else:
            reset_angles = torch.tensor(self.initial_angles, dtype=torch.float32, device=self.device)

        self._warmup_reset(self._alpha, reset_angles)

        self._alpha = torch.tensor(self.initial_angles, dtype=torch.float32, device=self.device)
        self._reward_buffer = -1.0 * torch.ones((self.reward_average_steps,), dtype=torch.float32, device=self.device)
        self._step_count = 0

        observation = torch.zeros((self.total_obs,), dtype=torch.float32, device=self.device)
        power = torch.zeros((self.n_turbs,), dtype=torch.float32, device=self.device)
        return self._make_obs(self._alpha, observation, power), {"reset_angles": reset_angles.detach().cpu().numpy()}

    def step(self, action):
        action = torch.as_tensor(action, dtype=torch.float32, device=self.device).view(self.n_turbs)
        action = action.clamp(-1.0, 1.0)
        u = action * self.max_speed

        new_alpha = (self._alpha + u * self.dt).clamp(-self.max_angle, self.max_angle)
        power, observation = self._communicate(new_alpha)

        angle_penalty = torch.mean(self.penalty_scale * (new_alpha / self.max_angle) ** self.penalty_exponent)
        velocity_penalty = self.velocity_penalty_scale * torch.mean((u / self.max_speed) ** 2)
        difference_penalty = self.difference_penalty_scale * torch.mean(
            (power.flatten().unsqueeze(0) - power.flatten().unsqueeze(1)) ** 2
        )
        power_mean = torch.mean(power)
        reward = power_mean - angle_penalty - velocity_penalty - difference_penalty

        self._reward_buffer = ring_insert_shift(self._reward_buffer, float(reward.item()))
        valid = self._reward_buffer != -1.0
        if torch.any(valid):
            reward = self._reward_buffer[valid].mean()
        power_out = torch.broadcast_to(power_mean, (self.n_turbs,))

        self._alpha = new_alpha.detach()
        self._step_count += 1
        terminated = False
        truncated = self._step_count >= self.max_episode_steps

        info = {
            "instant_reward": float(reward.item()),
            "power": power_out.detach().cpu().numpy(),
            "alpha": self._alpha.detach().cpu().numpy(),
        }
        return self._make_obs(self._alpha, observation, power_out), reward.detach().cpu(), terminated, truncated, info

    def close(self):
        return None

    def _set_seed(self, seed: Optional[int]):
        rng = torch.Generator(device=self.device)
        rng.manual_seed(int(seed))
        self.rng = rng
