from collections import defaultdict
from typing import Optional

import numpy as np
import torch
import tqdm
from tensordict.nn import TensorDictModule
from tensordict.tensordict import TensorDict, TensorDictBase
from torch import nn

from Solver.ADM_runner import ADM
from Solver.farm import Turbine, Farm

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
                 params=None,
                 seed=None,
                 save=False,
                 device="cpu",
                 dummy_update=False):

        if params is None:
            params = self.gen_params().to(device)

        super().__init__(device=device, batch_size=[])

        self.save = save
        self.obs_per_turbine = params["params"]["probes_per_turbine"].item() * 2 + 3
        self.n_turbs = params["params"]["n_turbines"].item()
        self.n_agents = params["params"]["n_turbines"].item()
        self.total_obs = self.obs_per_turbine * self.n_turbs

        self._make_spec(params)
        if seed is None:
            seed = torch.empty((), dtype=torch.int64).random_().item()
        self.set_seed(seed)

        # set up a farm environment (probably better to pass this???)
        diameter = params["params"]["turbine_diameter"].item()
        spacing = params["params"]["turbine_spacing"].item()
        self.farm1 = Farm(diameter * spacing * self.n_turbs, diameter * 4,
                          self.n_turbs, Turbine(diameter, 90, yaw=0),
                          offset=[2 * diameter, 2 * diameter])
        self.farm1.grid(staggered=False)
        self.adm = ADM(self.farm1, (self.obs_per_turbine-3)//2)
        self.adm.total_timesteps = params["params"]["run_steps"] + self.adm.init_timesteps

        self.dummy_update = dummy_update  # If True, perform a dummy update for testing
        if not self.dummy_update:
            self.adm.run_precursor()
            self.adm.initialise_flow(self.adm.init_timesteps)
            self.adm.restart()

    def _step(self, tensordict):

        action = tensordict.get(("agents", "action"))
        print(action)
        # action = action.unbind(dim=1)
        alpha = tensordict["alpha"]
        # u = tensordict["action"].squeeze(-1)
        u = action
        u = u.clamp(-tensordict["params", "max_yaw_speed"], tensordict["params", "max_yaw_speed"])
        dt = tensordict["params", "dt"]
        
        new_alpha = alpha + u * dt
        new_alpha = new_alpha.clamp(-tensordict["params", "max_yaw_angle"], tensordict["params", "max_yaw_angle"])
        for i in range(len(new_alpha)):
            if new_alpha[i] == tensordict["params", "max_yaw_angle"]:
                u[i] = u[i].clamp(-tensordict["params", "max_yaw_speed"], 0)
            elif new_alpha[i] == -tensordict["params", "max_yaw_angle"]:
                u[i] = u[i].clamp(0, tensordict["params", "max_yaw_speed"])

        # update by running ADM
        if self.dummy_update:
            power = torch.ones((*tensordict.shape, 1), device=self.device)
            observation = torch.zeros((*tensordict.shape, self.total_obs), device=self.device)
        else:
            power, observation = (self.adm.advance(new_alpha.cpu(), save=self.save))
            power = power.to(self.device)
            observation = observation.to(self.device)

        reward = power.view(*tensordict.shape, 1)  # normalise?
        done = torch.zeros_like(reward, dtype=torch.bool)

        source = {"done": done}
        agent_tds = []
        for i in range(self.n_agents):
            agent_out = TensorDict(
                {
                    "alpha": new_alpha[i],
                    "observation": observation[i],
                    "params": tensordict["params"],
                    "reward": reward,
                    # "done": done,
                },
                tensordict.shape,
                )
            agent_tds.append(agent_out)

        # agent_tds = torch.stack(agent_tds, dim=1)
        agent_tds = torch.stack(agent_tds)
        agent_tds = agent_tds.to_tensordict()
        source.update({"agents": agent_tds})

        # print(f'\n\n SOURCE \n\n {source}')
        f = open("./source.txt", "w")
        f.write(f'{source}')
        f.close()
        out = TensorDict(
            source=source,
            batch_size=self.batch_size,
            # device=self.device,
        )

        return out

    def _reset(self, tensordict):
        print('\n\nRESETTING ENVIROMENT\n')

        if not self.dummy_update:
            self.adm.restart()

        if tensordict is None or tensordict.is_empty():
            # if no tensordict is passed, we generate a single set of hyperparameters
            # Otherwise, we assume that the input tensordict contains all the relevant
            # parameters to get started.
            tensordict = self.gen_params(batch_size=self.batch_size).to(self.device)

        high_alpha = torch.tensor(np.pi, device=self.device)
        low_alpha = -high_alpha

        # for non batch-locked envs, the input tensordict shape dictates the number
        # of simulators run simultaneously. In other contexts, the initial
        # random state's shape will depend upon the environment batch-size instead.
        alpha = (
            torch.rand((*tensordict.shape, self.n_turbs), generator=self.rng, device=self.device)
            * (high_alpha - low_alpha)
            + low_alpha
        )
        observation = torch.zeros((*tensordict.shape, self.total_obs), device=self.device)

        out = TensorDict(
            {
                "alpha": alpha,
                "observation": observation,
                "params": tensordict["params"],
            },
            batch_size=tensordict.shape,
        )
        return out

    def _make_agent_spec(self, td_params):
        # Under the hood, this will populate self.output_spec["observation"]
        observation_spec = CompositeSpec(
                alpha=BoundedTensorSpec(
                    low=-td_params["params", "max_yaw_angle"],
                    high=td_params["params", "max_yaw_angle"],
                    shape=(*self.batch_size,),
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
                params=self.make_composite_from_td(td_params["params"]),
                shape=(),
                device=self.device
                )
        # since the environment is stateless, we expect the previous output as input.
        # For this, EnvBase expects some state_spec to be available
        state_spec = observation_spec.clone()
        # action-spec will be automatically wrapped in input_spec when
        # `self.action_spec = spec` will be called supported
        action_spec = BoundedTensorSpec(
                low=-td_params["params", "max_yaw_speed"],
                high=td_params["params", "max_yaw_speed"],
            shape=(*self.batch_size,),
            dtype=torch.float32,
            device=self.device
        )
        reward_spec = UnboundedContinuousTensorSpec(
            shape=(*td_params.shape, 1),
            dtype=torch.float32,
            device=self.device
        )
        return action_spec, reward_spec, observation_spec, state_spec

    def _make_spec(self, td_params):
        action_specs = []
        observation_specs = []
        reward_specs = []
        state_specs = []
        for i in range(self.n_agents):
            agent_i_action_spec, agent_i_reward_spec, agent_i_observation_spec, agent_i_state_spec = self._make_agent_spec(td_params)
            action_specs.append(agent_i_action_spec)
            reward_specs.append(agent_i_reward_spec)
            observation_specs.append(agent_i_observation_spec)
            state_specs.append(agent_i_state_spec)
        self.action_spec = CompositeSpec(
            {
                "agents": CompositeSpec(
                    {"action": torch.stack(action_specs, dim=0)}, shape=(self.n_agents,)
                )
            }
        )
        self.reward_spec = CompositeSpec(
            {
                "agents": CompositeSpec(
                    {"reward": torch.stack(reward_specs, dim=0)}, shape=(self.n_agents,)
                )
            }
        )
        self.observation_spec = CompositeSpec(
            {
                "agents": CompositeSpec(
                    {"observation": torch.stack(observation_specs, dim=0)}, shape=(self.n_agents,)
                )
            }
        )
        self.state_spec = CompositeSpec(
            {
                "agents": CompositeSpec(
                    {"state": torch.stack(state_specs, dim=0)}, shape=(self.n_agents,)
                )
            }
        )

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
    def gen_params(batch_size=None) -> TensorDictBase:
        """Returns a tensordict containing the physical parameters such as speed and angle limits."""
        if batch_size is None:
            batch_size = []
        td = TensorDict(
            {
                "params": TensorDict(
                    {
                        "n_turbines": 3,
                        "probes_per_turbine": 25,
                        "turbine_diameter": 126,
                        "turbine_spacing": 7,
                        "max_yaw_speed": 0.25,
                        "max_yaw_angle": 40,
                        "dt": 10,
                        "run_steps":10,
                    },
                    [],
                )
            },
            [],
        )
        if batch_size:
            td = td.expand(batch_size).contiguous()
        return td


if __name__ == '__main__':

    test_env = TurbEnv(save=True, dummy_update=True)
    print("action_keys:", test_env.action_keys)
    print("reward_keys:", test_env.reward_keys)
    print("observation_keys:", test_env.observation_spec)
    print("done_keys:", test_env.done_keys)
    rollout = test_env.rollout(max_steps=3)
    print(f"alpha = {rollout['alpha'][:, 1].mean()}")
    # print(f"Reward = {rollout['next', 'episode_reward'][rollout['next', 'done']][:, 1].mean()}")

    print("\nTesting environment rollout...")
    print(rollout)

