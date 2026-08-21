import os
import sys
import time
import unittest
import importlib.util

REPO_ROOT = "/home/runner/work/Wind-RL-gym/Wind-RL-gym"
SAC_DIR = os.path.join(REPO_ROOT, "sac")
SOLVER_DIR = os.path.join(REPO_ROOT, "Solver")
if SAC_DIR not in sys.path:
    sys.path.insert(0, SAC_DIR)
if SOLVER_DIR not in sys.path:
    sys.path.insert(0, SOLVER_DIR)

HAS_RUNTIME_DEPS = all(
    importlib.util.find_spec(mod_name) is not None for mod_name in ("numpy", "torch", "hydra")
)

if HAS_RUNTIME_DEPS:
    import numpy as np  # noqa: E402
    import torch  # noqa: E402
    from utils_sac import ParallelGymEnv  # noqa: E402
    try:
        from WF_enviroment import TurbEnv  # noqa: E402

        HAS_WF_ENV = True
    except Exception:
        HAS_WF_ENV = False
else:
    np = None
    torch = None
    ParallelGymEnv = None
    HAS_WF_ENV = False


class DummyEnv:
    def __init__(
        self,
        instance=0,
        truncate_after=2,
        step_delay_s=0.0,
        fail_on_step=False,
        close_delay_s=0.0,
    ):
        self.instance = instance
        self.truncate_after = truncate_after
        self.step_delay_s = step_delay_s
        self.fail_on_step = fail_on_step
        self.close_delay_s = close_delay_s
        self.step_count = 0
        self.action_space = ("dummy_action_space", instance)
        self.observation_space = ("dummy_observation_space", instance)

    def _obs(self):
        return {
            "alpha": np.array([self.instance], dtype=np.float32),
            "alpha_normalised": np.array([self.instance / 10.0], dtype=np.float32),
            "observation": np.array([self.instance, self.step_count, 1.0], dtype=np.float32),
            "reward_buffer": np.array([float(self.step_count)], dtype=np.float32),
            "power": np.array([self.instance + 0.5], dtype=np.float32),
        }

    def reset(self, seed=None, options=None):
        self.step_count = 0
        return self._obs(), {"instance": self.instance, "reset": True}

    def step(self, action):
        if self.fail_on_step:
            raise RuntimeError(f"step failed for instance {self.instance}")
        if self.step_delay_s > 0:
            time.sleep(self.step_delay_s)
        self.step_count += 1
        reward = float(self.instance + self.step_count)
        terminated = False
        truncated = self.step_count >= self.truncate_after
        return self._obs(), reward, terminated, truncated, {"instance": self.instance, "step_count": self.step_count}

    def close(self):
        if self.close_delay_s > 0:
            time.sleep(self.close_delay_s)
        return None


@unittest.skipUnless(HAS_RUNTIME_DEPS, "Requires numpy, torch, and hydra dependencies")
class ParallelGymEnvTests(unittest.TestCase):
    def test_reset_and_step_stacking_and_autoreset(self):
        env = ParallelGymEnv(
            env_ctor=DummyEnv,
            env_kwargs_list=[{"instance": 1}, {"instance": 2}],
            worker_timeout_s=2.0,
            start_method="spawn",
            parent_poll_interval_s=0.01,
        )
        try:
            obs, infos = env.reset()
            self.assertEqual(obs["observation"].shape, (2, 3))
            self.assertEqual([info["instance"] for info in infos], [1, 2])

            actions = torch.zeros((2, 1), dtype=torch.float32)
            _, _, _, truncated, infos = env.step(actions)
            self.assertFalse(bool(truncated[0]))
            self.assertFalse(bool(truncated[1]))

            next_obs, rewards, _, truncated, infos = env.step(actions)
            self.assertEqual(next_obs["observation"].shape, (2, 3))
            self.assertEqual(rewards.shape, (2, 1))
            self.assertTrue(bool(truncated[0]))
            self.assertTrue(bool(truncated[1]))
            self.assertIn("final_observation", infos[0])
            self.assertIn("final_info", infos[0])
        finally:
            env.close()

    def test_worker_exception_propagates(self):
        env = ParallelGymEnv(
            env_ctor=DummyEnv,
            env_kwargs_list=[{"instance": 5, "fail_on_step": True}],
            worker_timeout_s=1.0,
            start_method="spawn",
            parent_poll_interval_s=0.01,
        )
        try:
            env.reset()
            with self.assertRaises(RuntimeError) as ctx:
                env.step(torch.zeros((1, 1), dtype=torch.float32))
            msg = str(ctx.exception)
            self.assertIn("instance 5", msg)
            self.assertIn("step failed", msg)
        finally:
            env.close()

    def test_parent_timeout_reports_unfinished_and_worker_errors(self):
        env = ParallelGymEnv(
            env_ctor=DummyEnv,
            env_kwargs_list=[
                {"instance": 10, "step_delay_s": 2.0},
                {"instance": 11, "fail_on_step": True},
            ],
            worker_timeout_s=0.3,
            start_method="spawn",
            parent_poll_interval_s=0.01,
        )
        try:
            env.reset()
            with self.assertRaises(TimeoutError) as ctx:
                env.step(torch.zeros((2, 1), dtype=torch.float32))
            msg = str(ctx.exception)
            self.assertIn("Unfinished environment instances", msg)
            self.assertIn("10", msg)
            self.assertIn("instance 11", msg)
        finally:
            env.close()

    def test_close_forced_termination_fallback(self):
        env = ParallelGymEnv(
            env_ctor=DummyEnv,
            env_kwargs_list=[{"instance": 99, "close_delay_s": 3.0}],
            worker_timeout_s=1.0,
            start_method="spawn",
            parent_poll_interval_s=0.01,
            close_timeout_s=0.1,
        )
        start = time.monotonic()
        env.close()
        elapsed = time.monotonic() - start
        self.assertLess(elapsed, 2.0)
        for worker in env._workers:
            self.assertFalse(worker["process"].is_alive())


@unittest.skipUnless(HAS_RUNTIME_DEPS and HAS_WF_ENV, "WF_enviroment/runtime dependencies unavailable")
class TurbEnvTimeoutDiagnosticsTests(unittest.TestCase):
    def test_communicate_timeout_message_has_instance_phase_and_flags(self):
        class FakeClient:
            def __init__(self):
                self.store = {
                    "7_yaws_done": np.array([1]),
                    "7_sim_done": np.array([0]),
                }

            def put_tensor(self, key, value):
                self.store[key] = np.array(value)

            def poll_key(self, key, *_):
                return key in self.store

            def get_tensor(self, key):
                return self.store[key]

        env = TurbEnv.__new__(TurbEnv)
        env.instance = 7
        env.client = FakeClient()
        env.communication_timeout_s = 0.05
        env.communication_poll_interval_s = 0.01
        env.communication_slow_log_s = 0.0

        with self.assertRaises(TimeoutError) as ctx:
            env._communicate(torch.zeros(3, dtype=torch.float32))
        msg = str(ctx.exception)
        self.assertIn("instance=7", msg)
        self.assertIn("phase=7_sim_done", msg)
        self.assertIn("yaws_done=1", msg)
        self.assertIn("sim_done=0", msg)


if __name__ == "__main__":
    unittest.main()
