# from __future__ import annotations

from torchrl.envs.transforms import TransformedEnv, Transform, Compose, ObservationNorm

# from __future__ import annotations

import collections
import importlib.util
import multiprocessing as mp
import warnings
from copy import copy
from functools import wraps
from textwrap import indent
from typing import Any, Dict, List, Optional, OrderedDict, Sequence, Tuple, Union

import numpy as np

import torch

from tensordict import (
    is_tensor_collection,
    NonTensorData,
    set_lazy_legacy,
    TensorDict,
    TensorDictBase,
    unravel_key,
    unravel_key_list,
)
from tensordict._tensordict import _unravel_key_to_tuple
from tensordict.nn import dispatch, TensorDictModuleBase
from tensordict.utils import expand_as_right, expand_right, NestedKey
from torch import nn, Tensor
from torch.utils._pytree import tree_map
from torchrl._utils import _replace_last

from torchrl.data.tensor_specs import (
    BinaryDiscreteTensorSpec,
    BoundedTensorSpec,
    CompositeSpec,
    ContinuousBox,
    DiscreteTensorSpec,
    MultiDiscreteTensorSpec,
    MultiOneHotDiscreteTensorSpec,
    OneHotDiscreteTensorSpec,
    TensorSpec,
    UnboundedContinuousTensorSpec,
)
from torchrl.envs.common import _EnvPostInit, EnvBase, make_tensordict
from torchrl.envs.transforms import functional as F
from torchrl.envs.transforms.utils import (
    _get_reset,
    _set_missing_tolerance,
    check_finite,
)
from torchrl.envs.utils import _sort_keys, _update_during_reset, step_mdp
from torchrl.objectives.value.functional import reward2go


class PinMemoryTransform(Transform):
    """Calls pin_memory on the tensordict to facilitate writing on CUDA devices."""

    def __init__(self):
        super().__init__()

    def _call(self, tensordict: TensorDictBase) -> TensorDictBase:
        return tensordict.pin_memory()

    forward = _call

    def _reset(
        self, tensordict: TensorDictBase, tensordict_reset: TensorDictBase
    ) -> TensorDictBase:
        with _set_missing_tolerance(self, True):
            tensordict_reset = self._call(tensordict_reset)
        return tensordict_reset


def _sum_left(val, dest):
    while val.ndimension() > dest.ndimension():
        val = val.sum(0)
    return val


class VecNorm(Transform):
    """Moving average normalization layer for torchrl environments.

    VecNorm keeps track of the summary statistics of a dataset to standardize
    it on-the-fly. If the transform is in 'eval' mode, the running
    statistics are not updated.

    If multiple processes are running a similar environment, one can pass a
    TensorDictBase instance that is placed in shared memory: if so, every time
    the normalization layer is queried it will update the values for all
    processes that share the same reference.

    To use VecNorm at inference time and avoid updating the values with the new
    observations, one should substitute this layer by `vecnorm.to_observation_norm()`.

    Args:
        in_keys (sequence of NestedKey, optional): keys to be updated.
            default: ["observation", "reward"]
        out_keys (sequence of NestedKey, optional): destination keys.
            Defaults to ``in_keys``.
        shared_td (TensorDictBase, optional): A shared tensordict containing the
            keys of the transform.
        decay (number, optional): decay rate of the moving average.
            default: 0.99
        eps (number, optional): lower bound of the running standard
            deviation (for numerical underflow). Default is 1e-4.
        shapes (List[torch.Size], optional): if provided, represents the shape
            of each in_keys. Its length must match the one of ``in_keys``.
            Each shape must match the trailing dimension of the corresponding
            entry.
            If not, the feature dimensions of the entry (ie all dims that do
            not belong to the tensordict batch-size) will be considered as
            feature dimension.

    Examples:
        >>> from torchrl.envs.libs.gym import GymEnv
        >>> t = VecNorm(decay=0.9)
        >>> env = GymEnv("Pendulum-v0")
        >>> env = TransformedEnv(env, t)
        >>> tds = []
        >>> for _ in range(1000):
        ...     td = env.rand_step()
        ...     if td.get("done"):
        ...         _ = env.reset()
        ...     tds += [td]
        >>> tds = torch.stack(tds, 0)
        >>> print((abs(tds.get(("next", "observation")).mean(0))<0.2).all())
        tensor(True)
        >>> print((abs(tds.get(("next", "observation")).std(0)-1)<0.2).all())
        tensor(True)

    """

    def __init__(
        self,
        in_keys: Sequence[NestedKey] | None = None,
        out_keys: Sequence[NestedKey] | None = None,
        shared_td: Optional[TensorDictBase] = None,
        lock: mp.Lock = None,
        decay: float = 0.9999,
        eps: float = 1e-4,
        shapes: List[torch.Size] = None,
    ) -> None:
        if lock is None:
            lock = mp.Lock()
        if in_keys is None:
            in_keys = ["observation", "reward"]
        if out_keys is None:
            out_keys = copy(in_keys)
        super().__init__(in_keys=in_keys, out_keys=out_keys)
        self._td = shared_td
        if shared_td is not None and not (
            shared_td.is_shared() or shared_td.is_memmap()
        ):
            raise RuntimeError(
                "shared_td must be either in shared memory or a memmap " "tensordict."
            )
        if shared_td is not None:
            for key in in_keys:
                if (
                    (key + "_sum" not in shared_td.keys())
                    or (key + "_ssq" not in shared_td.keys())
                    or (key + "_count" not in shared_td.keys())
                ):
                    raise KeyError(
                        f"key {key} not present in the shared tensordict "
                        f"with keys {shared_td.keys()}"
                    )
        self.lock = lock
        self.decay = decay
        self.shapes = shapes
        self.eps = eps

    def _key_str(self, key):
        if not isinstance(key, str):
            key = "_".join(key)
        return key

    def _reset(
        self, tensordict: TensorDictBase, tensordict_reset: TensorDictBase
    ) -> TensorDictBase:
        with _set_missing_tolerance(self, True):
            tensordict_reset = self._call(tensordict_reset)
        return tensordict_reset

    def _call(self, tensordict: TensorDictBase) -> TensorDictBase:
        if self.lock is not None:
            self.lock.acquire()

        for key in self.in_keys:
            if key not in tensordict.keys(include_nested=True):
                continue
            self._init(tensordict, key)
            # update and standardize
            new_val = self._update(
                key, tensordict.get(key), N=max(1, tensordict.numel())
            )

            tensordict.set(key, new_val)

        if self.lock is not None:
            self.lock.release()

        return tensordict

    forward = _call

    def _init(self, tensordict: TensorDictBase, key: str) -> None:
        key_str = self._key_str(key)
        if self._td is None or key_str + "_sum" not in self._td.keys():
            if key is not key_str and key_str in tensordict.keys():
                raise RuntimeError(
                    f"Conflicting key names: {key_str} from VecNorm and input tensordict keys."
                )
            if self.shapes is None:
                td_view = tensordict.view(-1)
                td_select = td_view[0]
                item = td_select.get(key)
                d = {key_str + "_sum": torch.zeros_like(item)}
                d.update({key_str + "_ssq": torch.zeros_like(item)})
            else:
                idx = 0
                for in_key in self.in_keys:
                    if in_key != key:
                        idx += 1
                    else:
                        break
                shape = self.shapes[idx]
                item = tensordict.get(key)
                d = {
                    key_str
                    + "_sum": torch.zeros(shape, device=item.device, dtype=item.dtype)
                }
                d.update(
                    {
                        key_str
                        + "_ssq": torch.zeros(
                            shape, device=item.device, dtype=item.dtype
                        )
                    }
                )

            d.update(
                {
                    key_str
                    + "_count": torch.zeros(1, device=item.device, dtype=torch.float)
                }
            )
            if self._td is None:
                self._td = TensorDict(d, batch_size=[])
            else:
                self._td.update(d)
        else:
            pass

    def _update(self, key, value, N) -> torch.Tensor:
        key = self._key_str(key)
        _sum = self._td.get(key + "_sum")
        _ssq = self._td.get(key + "_ssq")
        _count = self._td.get(key + "_count")

        _sum = self._td.get(key + "_sum")
        value_sum = _sum_left(value, _sum)
        _sum *= self.decay
        _sum += value_sum
        self._td.set_(
            key + "_sum",
            _sum,
        )

        _ssq = self._td.get(key + "_ssq")
        value_ssq = _sum_left(value.pow(2), _ssq)
        _ssq *= self.decay
        _ssq += value_ssq
        self._td.set_(
            key + "_ssq",
            _ssq,
        )

        _count = self._td.get(key + "_count")
        _count *= self.decay
        _count += N
        self._td.set_(
            key + "_count",
            _count,
        )

        mean = _sum / _count
        std = (_ssq / _count - mean.pow(2)).clamp_min(self.eps).sqrt()
        return (value - mean) / std.clamp_min(self.eps)

    def to_observation_norm(self) -> Union[Compose, ObservationNorm]:
        """Converts VecNorm into an ObservationNorm class that can be used at inference time."""
        out = []
        for key in self.in_keys:
            key = self._key_str(key)
            _sum = self._td.get(key + "_sum")
            _ssq = self._td.get(key + "_ssq")
            _count = self._td.get(key + "_count")
            mean = _sum / _count
            std = (_ssq / _count - mean.pow(2)).clamp_min(self.eps).sqrt()

            _out = ObservationNorm(
                loc=mean,
                scale=std,
                standard_normal=True,
                in_keys=self.in_keys,
            )
            if len(self.in_keys) == 1:
                return _out
            else:
                out += ObservationNorm
        return Compose(*out)

    @staticmethod
    def build_td_for_shared_vecnorm(
        env: EnvBase,
        keys: Optional[Sequence[str]] = None,
        memmap: bool = False,
    ) -> TensorDictBase:
        """Creates a shared tensordict for normalization across processes.

        Args:
            env (EnvBase): example environment to be used to create the
                tensordict
            keys (sequence of NestedKey, optional): keys that
                have to be normalized. Default is `["next", "reward"]`
            memmap (bool): if ``True``, the resulting tensordict will be cast into
                memmory map (using `memmap_()`). Otherwise, the tensordict
                will be placed in shared memory.

        Returns:
            A memory in shared memory to be sent to each process.

        Examples:
            >>> from torch import multiprocessing as mp
            >>> queue = mp.Queue()
            >>> env = make_env()
            >>> td_shared = VecNorm.build_td_for_shared_vecnorm(env,
            ...     ["next", "reward"])
            >>> assert td_shared.is_shared()
            >>> queue.put(td_shared)
            >>> # on workers
            >>> v = VecNorm(shared_td=queue.get())
            >>> env = TransformedEnv(make_env(), v)

        """
        raise NotImplementedError("this feature is currently put on hold.")
        sep = ".-|-."
        if keys is None:
            keys = ["next", "reward"]
        td = make_tensordict(env)
        keys = {key for key in td.keys() if key in keys}
        td_select = td.select(*keys)
        td_select = td_select.flatten_keys(sep)
        if td.batch_dims:
            raise RuntimeError(
                f"VecNorm should be used with non-batched environments. "
                f"Got batch_size={td.batch_size}"
            )
        keys = list(td_select.keys())
        for key in keys:
            td_select.set(key + "_ssq", td_select.get(key).clone())
            td_select.set(
                key + "_count",
                torch.zeros(
                    *td.batch_size,
                    1,
                    device=td_select.device,
                    dtype=torch.float,
                ),
            )
            td_select.rename_key_(key, key + "_sum")
        td_select.exclude(*keys).zero_()
        td_select = td_select.unflatten_keys(sep)
        if memmap:
            return td_select.memmap_()
        return td_select.share_memory_()

    def get_extra_state(self) -> OrderedDict:
        return collections.OrderedDict({"lock": self.lock, "td": self._td})

    def set_extra_state(self, state: OrderedDict) -> None:
        lock = state["lock"]
        if lock is not None:
            """
            since locks can't be serialized, we have use cases for stripping them
            for example in ParallelEnv, in which case keep the lock we already have
            to avoid an updated tensor dict being sent between processes to erase locks
            """
            self.lock = lock
        td = state["td"]
        if td is not None and not td.is_shared():
            raise RuntimeError(
                "Only shared tensordicts can be set in VecNorm transforms"
            )
        self._td = td

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(decay={self.decay:4.4f},"
            f"eps={self.eps:4.4f}, keys={self.in_keys})"
        )

    def __getstate__(self) -> Dict[str, Any]:
        state = self.__dict__.copy()
        _lock = state.pop("lock", None)
        if _lock is not None:
            state["lock_placeholder"] = None
        return state

    def __setstate__(self, state: Dict[str, Any]):
        if "lock_placeholder" in state:
            state.pop("lock_placeholder")
            _lock = mp.Lock()
            state["lock"] = _lock
        self.__dict__.update(state)