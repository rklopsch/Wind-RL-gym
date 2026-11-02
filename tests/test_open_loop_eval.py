import importlib.util
import os
import sys
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = str(Path(__file__).resolve().parents[1])
for subdir in ("sac", "Solver", "eval"):
    path = os.path.join(REPO_ROOT, subdir)
    if path not in sys.path:
        sys.path.insert(0, path)
from probe_layout import ProbeLayout

HAS_RUNTIME_DEPS = all(
    importlib.util.find_spec(mod_name) is not None for mod_name in ("numpy", "torch", "hydra")
)

if HAS_RUNTIME_DEPS:
    import numpy as np  # noqa: E402
    import torch  # noqa: E402
    from open_loop_eval import (  # noqa: E402
        _check_start_yaw,
        _validated_step_info,
        burn_in_frames,
        phase_start_frame,
        required_control_frames,
        run_round,
    )
    from utils_sac import make_parallel_env  # noqa: E402

N_ENVS = 2
N_TURBINES = 3
EPISODE_LENGTH = 4
MAX_ANGLE = 40.0


class _Cfg:
    class env:
        dummy_update = True
        worker_timeout_s = 30.0
        worker_start_method = "spawn"
        worker_poll_interval_s = 0.01
        worker_close_timeout_s = 1.0

    class multi_agent:
        use = False


class _ScriptedActor:
    """Deterministic actor whose action depends only on the env index and the current yaw."""

    def __init__(self):
        self.deterministic_calls = []

    def sample(self, state, deterministic=False):
        self.deterministic_calls.append(deterministic)
        alpha_normalised = state[:, -N_TURBINES:]
        direction = torch.tensor([[1.0, -0.5, 0.25], [-1.0, 0.5, 0.0]])
        return (direction * (1.0 - alpha_normalised.abs())).clamp(-1.0, 1.0), None


@unittest.skipUnless(HAS_RUNTIME_DEPS, "Requires numpy, torch, and hydra dependencies")
class OpenLoopEvalTests(unittest.TestCase):
    def _env(self):
        params = {
            "n_turbines": N_TURBINES,
            "probe_layout": ProbeLayout(((1.0, 2.0, 3.0),) * 2, ((0,),) * 2),
            "episode_length": EPISODE_LENGTH,
            "max_yaw_angle": MAX_ANGLE,
            "max_yaw_speed": 1.0,
            "dt": 5.0,
        }
        return make_parallel_env(_Cfg, params, N_ENVS)

    def test_round_replays_actions_and_holds_mean_yaw(self):
        env = self._env()
        actor = _ScriptedActor()
        try:
            obs, _ = env.reset()
            with tempfile.TemporaryDirectory() as tmp:
                actions_path = os.path.join(tmp, "round_0001_actions.npy")
                for _ in range(2):  # the second round must start from zero yaw again
                    obs, record = run_round(env, actor, obs, EPISODE_LENGTH, MAX_ANGLE, actions_path)
                    np.testing.assert_array_equal(np.load(actions_path), record["rollout"]["actions"])
        finally:
            env.close()

        self.assertTrue(all(actor.deterministic_calls))
        rollout, replay, constant = record["rollout"], record["replay"], record["constant"]
        self.assertEqual(rollout["actions"].shape, (N_ENVS, EPISODE_LENGTH, N_TURBINES))
        np.testing.assert_array_equal(rollout["start_alpha"], np.zeros((N_ENVS, N_TURBINES)))
        np.testing.assert_array_equal(replay["start_alpha"], np.zeros((N_ENVS, N_TURBINES)))
        np.testing.assert_array_equal(replay["actions"], rollout["actions"])
        np.testing.assert_array_equal(replay["alpha"], rollout["alpha"])
        self.assertEqual(record["replay_yaw_max_error"], 0.0)

        np.testing.assert_allclose(record["mean_yaw"], rollout["alpha"].mean(axis=1), rtol=1e-6)
        np.testing.assert_allclose(constant["start_alpha"], record["mean_yaw"], rtol=1e-6)
        np.testing.assert_array_equal(constant["actions"], np.zeros_like(rollout["actions"]))
        self.assertLess(record["constant_yaw_max_error"], 1e-5)

        for phase in (rollout, replay, constant):
            self.assertEqual(phase["farm_power"].shape, (N_ENVS, EPISODE_LENGTH))
            np.testing.assert_allclose(phase["farm_power"], N_TURBINES * phase["mean_turbine_power"])
            np.testing.assert_allclose(phase["mean_farm_power"], phase["farm_power"].mean(axis=1))
        np.testing.assert_allclose(replay["mean_farm_power"], rollout["mean_farm_power"])

    def test_frame_layout_matches_sac_burn_in(self):
        self.assertEqual(burn_in_frames(5000, 150), 34 * 150)
        self.assertEqual(required_control_frames(5000, 150, 2000, 2), 34 * 150 + 6 * 2150)
        self.assertEqual(phase_start_frame(100, 20, 5, round_idx=0, phase_idx=0), 100)
        self.assertEqual(phase_start_frame(100, 20, 5, round_idx=1, phase_idx=2), 100 + 5 * 25)

    def test_step_rejects_wrong_number_of_reset_options(self):
        env = self._env()
        try:
            env.reset()
            with self.assertRaises(ValueError):
                env.step(torch.zeros((N_ENVS, N_TURBINES)), reset_options=[None])
        finally:
            env.close()

    def test_round_rejects_mismatched_step_data(self):
        env = self._env()
        original_step = env.step
        try:
            for key in ("alpha", "power"):
                for terminal in (False, True):
                    with self.subTest(key=key, terminal=terminal):
                        obs, _ = env.reset()

                        def corrupt_step(actions, reset_options=None):
                            next_obs, reward, terminated, truncated, infos = original_step(
                                actions, reset_options=reset_options
                            )
                            if bool(truncated[0]) == terminal:
                                infos[0][key][0] += 1.0
                            return next_obs, reward, terminated, truncated, infos

                        env.step = corrupt_step
                        with self.assertRaisesRegex(RuntimeError, f"Environment 1 {key} info differs"):
                            run_round(env, _ScriptedActor(), obs, EPISODE_LENGTH, MAX_ANGLE)
                        env.step = original_step
        finally:
            env.close()

    def test_step_data_rejects_invalid_values(self):
        for key, value, message in (
            ("alpha", 41.0, "out-of-range yaw"),
            ("power", float("nan"), "non-finite power"),
            ("power", 2.0, "inconsistent mean turbine power"),
        ):
            with self.subTest(key=key, value=value):
                obs = {
                    "alpha": torch.zeros((1, N_TURBINES)),
                    "power": torch.ones((1, N_TURBINES)),
                }
                info = {
                    "alpha": np.zeros(N_TURBINES, dtype=np.float32),
                    "power": np.ones(N_TURBINES, dtype=np.float32),
                }
                obs[key][0, 0] = value
                info[key][0] = value
                with self.assertRaisesRegex(RuntimeError, message):
                    _validated_step_info(obs, [info], False, N_TURBINES, MAX_ANGLE, 1)

        with self.assertRaisesRegex(RuntimeError, "no final observation"):
            _validated_step_info(obs, [info], True, N_TURBINES, MAX_ANGLE, 2)

    def test_nonfinite_start_yaw_is_rejected(self):
        with self.assertRaisesRegex(RuntimeError, "rollout episode started at yaw"):
            _check_start_yaw("rollout", np.array([[float("nan")]]), np.zeros((1, 1)))


if __name__ == "__main__":
    unittest.main()
