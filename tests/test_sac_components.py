import importlib.util
import math
import os
import sys
import types
import unittest
from pathlib import Path
from unittest import mock

REPO_ROOT = str(Path(__file__).resolve().parents[1])
SAC_DIR = os.path.join(REPO_ROOT, "sac")
if SAC_DIR not in sys.path:
    sys.path.insert(0, SAC_DIR)

HAS_RUNTIME_DEPS = all(
    importlib.util.find_spec(mod_name) is not None for mod_name in ("numpy", "torch", "hydra")
)

if HAS_RUNTIME_DEPS:
    import numpy as np  # noqa: E402
    import torch  # noqa: E402
    from utils_sac import (  # noqa: E402
        FrameStackEnv,
        ReplayBuffer,
        SACActor,
        build_next_state,
        build_state,
        make_parallel_env,
        make_sa_sac_agent,
    )


def _load_wf_environment(client_cls):
    """Import Solver/WF_enviroment.py with a stub smartredis module so it can be tested without SmartRedis."""
    fake_smartredis = types.ModuleType("smartredis")
    fake_smartredis.Client = client_cls
    path = os.path.join(REPO_ROOT, "Solver", "WF_enviroment.py")
    spec = importlib.util.spec_from_file_location("_wf_enviroment_under_test", path)
    module = importlib.util.module_from_spec(spec)
    with mock.patch.dict(sys.modules, {"smartredis": fake_smartredis}):
        spec.loader.exec_module(module)
    return module


def _make_cfg(frame_stack=1, dummy_update=True):
    class Cfg:
        class env:
            pass

        class multi_agent:
            use = False

        class network:
            actor_hidden_depth = 1
            actor_hidden_size = 8
            critic_hidden_depth = 1
            critic_hidden_size = 8
            default_policy_scale = 1.0
            scale_lb = 0.1

    Cfg.env.frame_stack = frame_stack
    Cfg.env.dummy_update = dummy_update
    Cfg.env.worker_timeout_s = 30.0
    Cfg.env.worker_start_method = "spawn"
    Cfg.env.worker_poll_interval_s = 0.01
    Cfg.env.worker_close_timeout_s = 1.0
    return Cfg


class _CountingEnv:
    """Observation "observation" equals [step_count, step_count]; truncates after ``truncate_after`` steps."""

    def __init__(self, truncate_after=3):
        self.truncate_after = truncate_after
        self.step_count = 0
        self.action_space = {"shape": (1,)}
        self.observation_space = {
            "observation": {"shape": (2,)},
            "alpha_normalised": {"shape": (1,)},
        }

    def _obs(self):
        return {
            "observation": torch.full((2,), float(self.step_count)),
            "alpha_normalised": torch.zeros(1),
        }

    def reset(self, seed=None, options=None):
        self.step_count = 0
        return self._obs(), {}

    def step(self, action):
        self.step_count += 1
        return self._obs(), 0.0, False, self.step_count >= self.truncate_after, {}


@unittest.skipUnless(HAS_RUNTIME_DEPS, "Requires numpy, torch, and hydra dependencies")
class SACActorTests(unittest.TestCase):
    def _actor(self, default_scale=1.0, scale_lb=0.1, action_low=-1.0, action_high=1.0):
        return SACActor(
            state_dim=4,
            action_dim=3,
            hidden_depth=1,
            hidden_size=8,
            action_low=action_low,
            action_high=action_high,
            default_scale=default_scale,
            scale_lb=scale_lb,
        )

    def _set_last_layer(self, actor, mean_value, scale_param_value):
        last = actor.backbone.net[-1]
        with torch.no_grad():
            last.weight.zero_()
            last.bias[: actor.action_dim] = mean_value
            last.bias[actor.action_dim:] = scale_param_value

    def test_zero_scale_output_maps_to_default_policy_scale(self):
        for default_scale in (1.0, 0.5):
            actor = self._actor(default_scale=default_scale)
            self._set_last_layer(actor, 0.0, 0.0)
            _, std = actor(torch.randn(5, 4))
            torch.testing.assert_close(std, torch.full((5, 3), default_scale))

    def test_scale_is_lower_bounded_and_unbounded_above(self):
        actor = self._actor(scale_lb=0.1)
        self._set_last_layer(actor, 0.0, -50.0)
        _, std = actor(torch.randn(2, 4))
        torch.testing.assert_close(std, torch.full((2, 3), 0.1))

        self._set_last_layer(actor, 0.0, 20.0)
        _, std = actor(torch.randn(2, 4))
        self.assertTrue(bool((std > 10.0).all()))

    def test_scale_matches_tensordict_biased_softplus(self):
        actor = self._actor(default_scale=1.0, scale_lb=0.1)
        x = torch.linspace(-6.0, 6.0, 25)
        bias = math.log(math.expm1(1.0 - 0.01))  # tensordict inv_softplus(bias - min_val)
        expected = (torch.nn.functional.softplus(x + bias) + 0.01).clamp_min(0.1)
        actual = (torch.nn.functional.softplus(x + actor.scale_bias) + actor.scale_min_val).clamp_min(actor.scale_lb)
        torch.testing.assert_close(actual, expected)

    def test_log_prob_matches_tanh_transformed_normal(self):
        torch.manual_seed(0)
        actor = self._actor()
        state = torch.randn(64, 4)
        mean, std = actor(state)
        torch.manual_seed(1)
        action, log_prob = actor.sample(state)
        torch.manual_seed(1)
        pre_tanh = mean + std * torch.randn_like(mean)
        reference = torch.distributions.TransformedDistribution(
            torch.distributions.Independent(torch.distributions.Normal(mean, std), 1),
            [torch.distributions.TanhTransform(cache_size=1)],
        )
        reference_action = reference.transforms[0](pre_tanh)
        torch.testing.assert_close(action, reference_action)
        torch.testing.assert_close(log_prob.squeeze(-1), reference.log_prob(reference_action), rtol=1e-4, atol=1e-4)

    def test_log_prob_stays_finite_when_saturated(self):
        actor = self._actor()
        self._set_last_layer(actor, 30.0, 0.0)
        action, log_prob = actor.sample(torch.randn(16, 4))
        self.assertTrue(bool(torch.isfinite(log_prob).all()))
        self.assertTrue(bool((action <= 1.0).all()))

    def test_log_prob_accounts_for_action_rescaling(self):
        torch.manual_seed(0)
        unit_actor = self._actor()
        wide_actor = self._actor(action_low=-2.0, action_high=2.0)
        wide_actor.backbone.load_state_dict(unit_actor.backbone.state_dict())
        state = torch.randn(8, 4)
        torch.manual_seed(3)
        _, unit_log_prob = unit_actor.sample(state)
        torch.manual_seed(3)
        _, wide_log_prob = wide_actor.sample(state)
        torch.testing.assert_close(wide_log_prob, unit_log_prob - 3 * math.log(2.0))

    def test_sample_uniform_covers_action_bounds(self):
        torch.manual_seed(0)
        actor = self._actor()
        actions = actor.sample_uniform(10_000)
        self.assertEqual(actions.shape, (10_000, 3))
        self.assertTrue(bool((actions >= -1.0).all() and (actions <= 1.0).all()))
        self.assertLess(actions.min().item(), -0.95)
        self.assertGreater(actions.max().item(), 0.95)
        self.assertLess(abs(actions.mean().item()), 0.05)


@unittest.skipUnless(HAS_RUNTIME_DEPS, "Requires numpy, torch, and hydra dependencies")
class TransitionTests(unittest.TestCase):
    def test_build_next_state_uses_final_observation_for_finished_episodes(self):
        next_obs = {
            "observation": torch.tensor([[0.0, 0.0], [5.0, 5.0]]),
            "alpha_normalised": torch.tensor([[0.0], [0.5]]),
        }
        infos = [
            {"final_observation": {"observation": torch.tensor([9.0, 9.0]), "alpha_normalised": torch.tensor([3.0])}},
            {},
        ]
        episode_end = torch.tensor([[True], [False]])
        next_state = build_next_state(next_obs, infos, episode_end)
        torch.testing.assert_close(next_state, torch.tensor([[9.0, 9.0, 3.0], [5.0, 5.0, 0.5]]))

    def test_build_next_state_requires_final_observation(self):
        next_obs = {"observation": torch.zeros(1, 2), "alpha_normalised": torch.zeros(1, 1)}
        with self.assertRaises(KeyError):
            build_next_state(next_obs, [{}], torch.tensor([[True]]))

    def test_replay_buffer_keeps_terminated_separate_from_done(self):
        buffer = ReplayBuffer(capacity=4, state_dim=2, action_dim=1)
        buffer.add(
            torch.zeros(2, 2),
            torch.zeros(2, 1),
            torch.ones(2, 1),
            torch.ones(2, 2),
            done=torch.tensor([[True], [False]]),
            terminated=torch.tensor([[False], [False]]),
        )
        self.assertEqual(buffer.done[:2].flatten().tolist(), [True, False])
        self.assertEqual(buffer.terminated[:2].flatten().tolist(), [False, False])
        batch = buffer.sample(8)
        self.assertIn("terminated", batch)
        self.assertFalse(bool(batch["terminated"].any()))

        legacy = buffer.state_dict()
        legacy.pop("terminated")
        restored = ReplayBuffer(capacity=4, state_dim=2, action_dim=1)
        restored.load_state_dict(legacy)
        self.assertEqual(restored.size, 2)
        self.assertFalse(bool(restored.terminated.any()))


@unittest.skipUnless(HAS_RUNTIME_DEPS, "Requires numpy, torch, and hydra dependencies")
class FrameStackTests(unittest.TestCase):
    def test_same_padding_on_reset_and_oldest_first_order(self):
        env = FrameStackEnv(_CountingEnv(), num_frames=3)
        obs, _ = env.reset()
        torch.testing.assert_close(obs["observation"], torch.zeros(6))
        obs, *_ = env.step(None)
        torch.testing.assert_close(obs["observation"], torch.tensor([0.0, 0.0, 0.0, 0.0, 1.0, 1.0]))
        obs, *_ = env.step(None)
        torch.testing.assert_close(obs["observation"], torch.tensor([0.0, 0.0, 1.0, 1.0, 2.0, 2.0]))
        obs, *_ = env.step(None)
        torch.testing.assert_close(obs["observation"], torch.tensor([1.0, 1.0, 2.0, 2.0, 3.0, 3.0]))
        obs, _ = env.reset()
        torch.testing.assert_close(obs["observation"], torch.zeros(6))
        self.assertEqual(env.observation_space["observation"]["shape"], (6,))
        self.assertEqual(env.observation_space["alpha_normalised"]["shape"], (1,))

    def test_parallel_env_stacks_frames_and_matches_agent_input_size(self):
        cfg = _make_cfg(frame_stack=3, dummy_update=True)
        params = {
            "n_turbines": 2,
            "probes_per_turbine": 3,
            "flow_field_directions": ["ux"],
            "episode_length": 2,
        }
        actor, critic1, _ = make_sa_sac_agent(cfg, params)
        self.assertEqual(actor.state_dim, 3 * 6 + 2)
        env = make_parallel_env(cfg, params, 2, device="cpu")
        try:
            obs, _ = env.reset()
            self.assertEqual(obs["observation"].shape, (2, 18))
            self.assertEqual(build_state(obs).shape, (2, actor.state_dim))
            actions = actor.sample_uniform(2)
            env.step(actions)
            next_obs, _, terminated, truncated, infos = env.step(actions)
            self.assertTrue(bool(truncated.all()))
            self.assertFalse(bool(terminated.any()))
            episode_end = torch.logical_or(terminated, truncated)
            next_state = build_next_state(next_obs, infos, episode_end)
            self.assertEqual(next_state.shape, (2, actor.state_dim))
            # DummyTurbEnv observation is instance + step; final obs stacks steps [0, 1, 2], reset obs is all 0.
            for i in range(2):
                expected_final = torch.cat([torch.full((6,), float(i + s)) for s in (0, 1, 2)])
                torch.testing.assert_close(next_state[i, :18], expected_final)
                torch.testing.assert_close(next_obs["observation"][i], torch.full((18,), float(i)))
            with torch.no_grad():
                q = critic1(next_state, actions)
            self.assertEqual(q.shape, (2, 1))
        finally:
            env.close()


@unittest.skipUnless(HAS_RUNTIME_DEPS, "Requires numpy, torch, and hydra dependencies")
class TurbEnvObservationTests(unittest.TestCase):
    class FakeClient:
        n_probes = 6

        def __init__(self, *args, **kwargs):
            self.store = {}

        def put_tensor(self, key, value):
            self.store[key] = np.array(value)
            if key.endswith("_yaws_done") and bool(np.asarray(value).reshape(-1)[0]):
                instance = key.split("_")[0]
                self.store[f"{instance}_sim_done"] = np.array([1])
                self.store[f"{instance}_turbine_powers"] = np.array([1.0e6, 2.0e6, 3.0e6])
                for i in range(self.n_probes):
                    self.store[f"{instance}_probe_{i + 1}"] = np.array([12.0, 0.0, 4.0])

        def poll_key(self, key, *_):
            return key in self.store

        def get_tensor(self, key):
            return self.store[key]

    def _params(self):
        return {
            "n_turbines": 3,
            "probes_per_turbine": 2,
            "flow_field_directions": ["ux", "uz"],
            "max_yaw_speed": 1.0,
            "max_yaw_angle": 40,
            "dt": 10.0,
            "reset_frames": 5,
            "episode_length": 2,
            "penalty_scale": 0.0,
            "penalty_exp": 2,
            "random_reset": False,
            "initial_angles": [0, 0, 0],
            "reward_average_steps": 1,
            "velocity_penalty_scale": 0.0,
            "difference_penalty_scale": 0.0,
        }

    def test_alpha_normalised_uses_former_observation_norm_scale(self):
        wf = _load_wf_environment(self.FakeClient)
        env = wf.TurbEnv(self._params(), instance=0)
        env.reset()
        obs, _, terminated, truncated, _ = env.step(np.array([1.0, -1.0, 0.0], dtype=np.float32))
        torch.testing.assert_close(obs["alpha"], torch.tensor([10.0, -10.0, 0.0]))
        torch.testing.assert_close(obs["alpha_normalised"], torch.tensor([1.0, -1.0, 0.0]))
        self.assertEqual(float(env.observation_space.spaces["alpha_normalised"].high[0]), 4.0)
        self.assertEqual(obs["observation"].shape, (12,))
        self.assertFalse(terminated)
        self.assertFalse(truncated)
        for _ in range(5):
            obs, _, _, _, _ = env.step(np.array([1.0, 1.0, 1.0], dtype=np.float32))
        # Starting from [10, -10, 0] degrees, five +10 degree steps saturate all turbines at max_yaw_angle.
        torch.testing.assert_close(obs["alpha_normalised"], torch.tensor([4.0, 4.0, 4.0]))


if __name__ == "__main__":
    unittest.main()
