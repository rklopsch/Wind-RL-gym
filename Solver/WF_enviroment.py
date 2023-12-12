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
                 device="cpu"):
        if params is None:
            params = self.gen_params()

        super().__init__(device=device, batch_size=[])
        self.n_obs = 2
        self.n_turbs = 2
        self.total_obs = 6
        self._make_spec(params)
        if seed is None:
            seed = torch.empty((), dtype=torch.int64).random_().item()
        self.set_seed(seed)

        # self.device = device

        # set up a farm environment (probably better to pass this???)
        self.farm1 = Farm(126 * 14, 126 * 4, 3, Turbine(126, 90, yaw=0), offset=[2 * 126, 2 * 126])
        self.farm1.grid(staggered=False)
        self.adm = ADM(self.farm1)

        self.dummy_update = True  # If True, perform a dummy update for testing

    def _step(self, tensordict):
        alpha = tensordict["alpha"]
        u = tensordict["action"].squeeze(-1)
        u = u.clamp(-tensordict["params", "max_speed"], tensordict["params", "max_speed"])
        dt = tensordict["params", "dt"]
        
        new_alpha = alpha + u * dt
        new_alpha = u.clamp(-tensordict["params", "max_angle"], tensordict["params", "max_angle"])

        # update by running ADM
        if self.dummy_update:
            power = torch.ones((*tensordict.shape, 1), device=self.device)
            observation = torch.zeros((*tensordict.shape, self.total_obs), device=self.device)
        else:
            power, observation = self.adm.advance(new_alpha)

        reward = power.view(*tensordict.shape, 1)  # normalise?
        done = torch.zeros_like(reward, dtype=torch.bool)
        
        out = TensorDict(
            {
                "alpha": new_alpha,
                "observation": observation,
                "params": tensordict["params"],
                "reward": reward,
                "done": done,
            },
            tensordict.shape,
        )        
        return out

    def _reset(self, tensordict):
        if tensordict is None or tensordict.is_empty():
            # if no tensordict is passed, we generate a single set of hyperparameters
            # Otherwise, we assume that the input tensordict contains all the relevant
            # parameters to get started.
            tensordict = self.gen_params(batch_size=self.batch_size)

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

    def _make_spec(self, td_params):
        # Under the hood, this will populate self.output_spec["observation"]
        self.observation_spec = CompositeSpec(
                alpha=BoundedTensorSpec(
                    low=-td_params["params", "max_angle"],
                    high=td_params["params", "max_angle"],
                    shape=(*self.batch_size, self.n_turbs),
                    dtype=torch.float32,
                      ),
                observation=UnboundedContinuousTensorSpec(
                    shape=(*self.batch_size, self.total_obs),
                    dtype=torch.float32,
                    device=self.device
                ),
                # we need to add the "params" to the observation specs, as we want
                # to pass it at each step during a rollout
                params=self.make_composite_from_td(td_params["params"]),
                shape=(),
                )
        # since the environment is stateless, we expect the previous output as input.
        # For this, EnvBase expects some state_spec to be available
        self.state_spec = self.observation_spec.clone()
        # action-spec will be automatically wrapped in input_spec when
        # `self.action_spec = spec` will be called supported
        self.action_spec = BoundedTensorSpec(
                low=-td_params["params", "max_speed"],
                high=td_params["params", "max_speed"],
            shape=(*self.batch_size, self.n_turbs),
            dtype=torch.float32,
                )
        self.reward_spec = UnboundedContinuousTensorSpec(shape=(*td_params.shape, 1))

    def make_composite_from_td(self, td):
        # custom funtion to convert a tensordict in a similar spec structure
        # of unbounded values.
        composite = CompositeSpec(
            {
                key: self.make_composite_from_td(tensor)
                if isinstance(tensor, TensorDictBase)
                else UnboundedContinuousTensorSpec(
                    dtype=tensor.dtype, device=tensor.device, shape=tensor.shape
                )
                for key, tensor in td.items()
            },
            shape=td.shape,
        )
        return composite

    def _set_seed(self, seed: Optional[int]):
        rng = torch.manual_seed(seed)
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
                        "max_speed": 2,
                        "max_angle": 60,
                        "dt": 0.05,
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

    test_env = TurbEnv()
    rollout = test_env.rollout(max_steps=3)
    print("Testing environment rollout...")
    print(rollout)

