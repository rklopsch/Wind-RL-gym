from collections import defaultdict
from typing import Optional

import numpy as np
import torch
import tqdm
from tensordict.nn import TensorDictModule
from tensordict.tensordict import TensorDict, TensorDictBase
from torch import nn

from torchrl.data import BoundedTensorSpec, CompositeSpec, UnboundedContinuousTensorSpec
from torchrl.envs import (
    CatTensors,
    EnvBase,
    Transform,
    TransformedEnv,
    UnsqueezeTransform,
)
from torchrl.envs.transforms.transforms import _apply_to_composite
from torchrl.envs.utils import check_env_specs, step_mdp


class TurbEnv(EnvBase):
    metadata = {}
    batch_locked = False

    def __init__(self,
                 client,
                 params=None,
                 seed=None,
                 save=False,
                 instance=None,
                 device="cpu",
                 dummy_update=False):

        if not params:
            params = self.gen_params()

        super().__init__(device=device, batch_size=[])

        self.save = save
        self.probes_per_turbine = params["probes_per_turbine"]
        self.obs_per_probe = 3
        self.obs_per_turbine = self.probes_per_turbine * self.obs_per_probe
        self.n_turbs = params["n_turbines"]
        self.total_obs = self.obs_per_turbine * self.n_turbs
        self.max_speed = params["max_yaw_speed"]  # maximum angular velocity of wind turbine
        self.max_angle = params["max_yaw_angle"]
        self.dt = params["dt"]

        self.client = client

        self._make_spec()
        if seed is None:
            seed = torch.empty((), dtype=torch.int64).random_().item()
        self.set_seed(seed)

        self.instance = 0  # Needs to be replaced with a list of instances

        self.dummy_update = dummy_update  # If True, perform a dummy update for testing

    def _communicate(self, new_alpha):
        ######### Communication with SmartRedis server ###########
        # Send yaws to X3D
        self.client.put_tensor(f"{self.instance}_yaws", new_alpha.detach().cpu().numpy().squeeze().astype(np.float64))
        # Set i_yaws_done flag to True (one)
        self.client.put_tensor(f"{self.instance}_yaws_done", np.ones(1)) # setting one as True

        # print(f"Set yaws to {new_alpha} for key {self.instance}_yaws")
        # print(f"Set yaws done to True for key {self.instance}_yaws_done")

        # Poll whether X3D simulation is done
        # This is done now for only one but we need to loop over ALL instances here
        while not self.client.get_tensor(f"{self.instance}_sim_done")[0]:
            continue

        # Here need to loop over ALL instances, and stack the results into one tensor
        turbine_powers = self.client.get_tensor(f"{self.instance}_turbine_powers")  # [n_turbs]
        turbine_obs = np.zeros((self.n_turbs*self.probes_per_turbine, self.obs_per_probe))
        for i in range(self.total_obs):
            turbine_obs[i] = self.client.get_tensor(f"{self.instance}_probe_{i}")  # [n_turbs*probes_per_turbine, obs_per_probe]
        turbine_obs = turbine_obs.reshape(self.n_turbs, self.probes_per_turbine, self.obs_per_probe)
        turbine_obs = turbine_obs.reshape(self.n_turbs, self.probes_per_turbine * self.obs_per_probe)  # [n_turbs, probes_per_turbine*obs_per_probe]

        # Here: Stack the outputs from all the parallel envs into one

        # Process the outputs from the solver
        turbine_powers /= 1e06
        farm_power = turbine_powers.mean(axis=-1)  # Compute mean over iterations many time steps
        farm_power = np.broadcast_to(farm_power, (self.n_turbs,))  # repeat farm power for all turbines
        # farm_power = turbine_powers.mean(axis=-1)  # Compute mean over iterations many time steps
        # turbine_obs = turbine_obs.mean(axis=-1)  # Compute mean over iterations many time steps
        # Missing turbine velocities here!! Stack the velocities, individual turbine power and yaw before the turbine_obs

        # Convert to Torch tensors
        power = torch.tensor(farm_power, dtype=torch.float32).to(self.device)
        observation = torch.tensor(turbine_obs, dtype=torch.float32).to(self.device)

        return power, observation

    def _step(self, tensordict):
        # All tensors are expected to be of shape [*batch_size, num_turbs, X]
        # where X = num_obs_per_turbine for observation
        #       X = num_actions_per_turbine for action
        #       X = 1 for reward

        # Retrieve action and previous alpha from tensordict
        action = tensordict.get(("agents", "action"))
        alpha = tensordict.get(("agents", "alpha"))
        u = action
        u = u.clamp(-self.max_speed, self.max_speed)
        
        # Compute new alpha
        new_alpha = alpha + u * self.dt
        new_alpha = new_alpha.clamp(-self.max_angle, self.max_angle)

        # print(f"In step (Turbenv): angles = {new_alpha}")

        power, observation = self._communicate(new_alpha)

        # Do a dummy update if desired - this probs needs to be changed
        self.dummy_update=False
        if self.dummy_update:
            power = torch.ones((*tensordict.shape, self.n_turbs, 1), device=self.device)
            observation = torch.zeros((*tensordict.shape, self.n_turbs, self.obs_per_turbine), device=self.device)
        self.dummy_update=False

        if len(power.shape) < len(tensordict.shape) + 2:
            reward = power.unsqueeze(dim=-1)
        else:
            reward = power
        done = torch.zeros((*tensordict.shape, 1), dtype=torch.bool)

        # Set i_sim_done flag to false (zero)
        # This needs to be done in a for loop for ALL instances here
        self.client.put_tensor(f"{self.instance}_sim_done", np.zeros(1)) 
        # print(f"Set sim done to True for key {self.instance}_sim_done")

        # print(f"observation shape {observation.shape}")

        source = {
            "done": done,
        }
        agent_tds = []
        for i in range(self.n_turbs):
            agent_out = TensorDict(
                {
                    "alpha": new_alpha[..., i, :],
                    "observation": observation[..., i, :],
                    "reward": reward[..., i, :],
                },
                ()  # tensordict.shape,
                )
            agent_tds.append(agent_out)

        # agent_tds = torch.stack(agent_tds, dim=1)
        agent_tds = torch.stack(agent_tds)
        agent_tds = agent_tds.to_tensordict()
        source.update({"agents": agent_tds})

        out = TensorDict(
            source=source,
            batch_size=self.batch_size,
            device=self.device,
        )

        return out

    def _reset(self, tensordict):
        print(f"Resetting now")

        for _ in range(150):
            _, _ = self._communicate(new_alpha=torch.zeros([self.n_turbs]))
        # Make sure the flags for yaws_done and sim_done are both set to False at the end of reset
        self.client.put_tensor(f"{self.instance}_yaws_done", np.array([0]))
        self.client.put_tensor(f"{self.instance}_sim_done", np.array([0]))

        # for non batch-locked envs, the input tensordict shape dictates the number
        # of simulators run simultaneously. In other contexts, the initial
        # random state's shape will depend upon the environment batch-size instead.
        alpha = torch.zeros((*self.batch_size, self.n_turbs, 1), device=self.device)
        observation = torch.zeros((*self.batch_size, self.n_turbs, self.obs_per_turbine), device=self.device)

        agent_tds = []
        for i in range(self.n_turbs):
            agent_out = TensorDict(
                {
                    "alpha": alpha[..., i, :],
                    "observation": observation[..., i, :],
                },
                ()  #self.batch_size,
            )
            agent_tds.append(agent_out)

        # agent_tds = torch.stack(agent_tds, dim=1)
        agent_tds = torch.stack(agent_tds)
        agent_tds = agent_tds.to_tensordict()

        out = TensorDict(
            {
                "agents": agent_tds,
            },
            batch_size=self.batch_size,
        )

        return out

    def _make_agent_spec(self):
        # Under the hood, this will populate self.output_spec["observation"]
        observation_spec = CompositeSpec(
                alpha=BoundedTensorSpec(
                    low=-self.max_angle,
                    high=self.max_angle,
                    shape=(*self.batch_size, 1),
                    dtype=torch.float32,
                    device=self.device
                ),
                observation=UnboundedContinuousTensorSpec(
                    shape=(*self.batch_size, self.obs_per_turbine),
                    dtype=torch.float32,
                    device=self.device
                ),
                # we need to add the "params" to the observation specs, as we want
                # to pass it at each step during a rollout
                # params=self.make_composite_from_td(td_params["params"]),
                shape=(),
                device=self.device
                )
        # action-spec will be automatically wrapped in input_spec when
        # `self.action_spec = spec` will be called supported
        action_spec = BoundedTensorSpec(
            low=-self.max_speed,
            high=self.max_speed,
            shape=(*self.batch_size, 1),
            dtype=torch.float32,
            device=self.device
        )
        reward_spec = UnboundedContinuousTensorSpec(
            shape=(*self.batch_size, 1),
            dtype=torch.float32,
            device=self.device
        )
        return action_spec, reward_spec, observation_spec

    def _make_spec(self):
        action_specs = []
        observation_specs = []
        reward_specs = []
        for i in range(self.n_turbs):
            agent_i_action_spec, agent_i_reward_spec, agent_i_observation_spec = self._make_agent_spec()
            action_specs.append(agent_i_action_spec)
            reward_specs.append(agent_i_reward_spec)
            observation_specs.append(agent_i_observation_spec)
        self.action_spec = CompositeSpec(
            {
                "agents": CompositeSpec(
                    {"action": torch.stack(action_specs, dim=0)}, shape=(self.n_turbs,)
                )
            }
        )
        self.reward_spec = CompositeSpec(
            {
                "agents": CompositeSpec(
                    {"reward": torch.stack(reward_specs, dim=0)}, shape=(self.n_turbs,)
                )
            }
        )
        self.observation_spec = CompositeSpec(
            {
                "agents": torch.stack(observation_specs, dim=0),
            },
            #shape=(self.n_turbs,)
        )
        # since the environment is stateless, we expect the previous output as input.
        # For this, EnvBase expects some state_spec to be available
        self.state_spec = self.observation_spec.clone()

    def make_composite_from_td(self, td):
        # custom funtion to convert a tensordict in a similar spec structure
        # of unbounded values.
        composite = CompositeSpec(
            {
                key: self.make_composite_from_td(tensor)
                if isinstance(tensor, TensorDictBase)
                else UnboundedContinuousTensorSpec(
                    dtype=tensor.dtype, device=self.device, shape=tensor.shape
                )
                for key, tensor in td.items()
            },
            shape=td.shape,
            device=self.device
        )
        return composite

    def _set_seed(self, seed: Optional[int]):
        rng = torch.Generator(device=self.device)
        rng.manual_seed(seed)
        self.rng = rng

    @staticmethod
    def gen_params() -> TensorDictBase:
        """Returns a dict containing the physical parameters such as speed and angle limits."""
        params = {
            "n_turbines": 3,
            "probes_per_turbine": 25,
            "turbine_diameter": 126,
            "turbine_spacing": 7,
            "max_yaw_speed": 0.25,
            "max_yaw_angle": 40,
            "dt": 10,
            "run_steps": 10,
        }
        return params


if __name__ == '__main__':

    test_env = TurbEnv(save=True, dummy_update=True)

    check_env_specs(test_env)

    print("action_keys:", test_env.action_keys)
    print("reward_keys:", test_env.reward_keys)
    print("observation_keys:", test_env.observation_spec)
    print("done_keys:", test_env.done_keys)
    rollout = test_env.rollout(max_steps=3)
    print(f"alpha = {rollout[('agents', 'alpha')][:, 1].mean()}")
    # print(f"Reward = {rollout['next', 'episode_reward'][rollout['next', 'done']][:, 1].mean()}")

    print("\nTesting environment rollout...")
    print(rollout)

