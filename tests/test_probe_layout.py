import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "Solver"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from probe_layout import load_probe_layout
from solver_runtime import _verified_build
from probe_layout import checkpoint_contract, validate_checkpoint_contract


class ProbeLayoutTests(unittest.TestCase):
    def load(self, text):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "probes.csv"
            path.write_text(text, encoding="ascii")
            return load_probe_layout(path)

    def test_heterogeneous_components_and_roundtrip(self):
        layout = self.load("x,y,z,ux,uy,uz,p\n1,2,3,1,0,0,1\n4,5,6,0,1,0,0\n7,8,9,1,1,1,1\n")
        self.assertEqual(layout.total_probes, 3)
        self.assertEqual(layout.total_obs, 7)
        self.assertEqual(layout.selections, ((0, 3), (1,), (0, 1, 2, 3)))
        layout.validate_domain((10, 10, 10))
        with self.assertRaisesRegex(ValueError, "outside"):
            layout.validate_domain((5, 10, 10))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "roundtrip.csv"
            layout.write(path)
            self.assertEqual(load_probe_layout(path), layout)
            self.assertEqual(load_probe_layout(path).fingerprint, layout.fingerprint)

    def test_pressure_only_and_header_order(self):
        layout = self.load("p,z,y,x,uz,uy,ux\n1,3,2,1,0,0,0\n")
        self.assertEqual(layout.coordinates, ((1.0, 2.0, 3.0),))
        self.assertEqual(layout.selections, ((3,),))

    def test_invalid_layouts(self):
        header = "x,y,z,ux,uy,uz,p\n"
        for text in ("", header, header + "1,2,3,0,0,0,0\n",
                     header + "nan,2,3,1,0,0,0\n", header + "1,2,3,2,0,0,0\n",
                     header + "1,2,3,1,0\n", header + "1,2,3,1,0,0,0,extra\n",
                     "x,y,z,ux,uy,uz,ux\n1,2,3,1,0,0,0\n"):
            with self.subTest(text=text), self.assertRaises(ValueError):
                self.load(text)

    def test_reject_upstream_solver_before_launch(self):
        _verified_build.cache_clear()
        with mock.patch("solver_runtime.subprocess.check_output", return_value="git@github.com:admole/Incompact3d.git\n"):
            with self.assertRaisesRegex(ValueError, "origin must be"):
                _verified_build("/unused", "/unused/build")

    def test_reject_missing_fork_build(self):
        _verified_build.cache_clear()
        with tempfile.TemporaryDirectory() as directory:
            with mock.patch("solver_runtime.subprocess.check_output", return_value="https://github.com/rklopsch/Incompact3d-WindRL.git\n"):
                with self.assertRaisesRegex(FileNotFoundError, "build the fork"):
                    _verified_build(directory, str(Path(directory) / "build"))

    def test_checkpoint_detects_same_width_reordered_features(self):
        import json
        from types import SimpleNamespace
        cfg = SimpleNamespace(env={"frame_stack": 1})
        layout = self.load("x,y,z,ux,uy,uz,p\n1,2,3,1,0,0,0\n4,5,6,0,0,0,1\n")
        params = {"probe_layout": layout, "n_turbines": 2}
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "Missing"):
                validate_checkpoint_contract(cfg, params, directory, 1)
            path = Path(directory) / "observation_1.json"
            path.write_text(json.dumps(checkpoint_contract(cfg, params)), encoding="ascii")
            validate_checkpoint_contract(cfg, params, directory, 1)
            params["probe_layout"] = type(layout)(layout.coordinates[::-1], layout.selections[::-1])
            with self.assertRaisesRegex(ValueError, "differs"):
                validate_checkpoint_contract(cfg, params, directory, 1)

    def test_staged_csv_is_self_contained(self):
        import subprocess
        from omegaconf import OmegaConf
        from probe_layout import stage_probe_layout
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "source.csv"
            source.write_text("x,y,z,ux,uy,uz,p\n1,2,3,0,0,0,1\n", encoding="ascii")
            worker = Path(directory) / "worker"
            worker.mkdir()
            stage_probe_layout(worker, OmegaConf.create({"probes_file": str(source)}))
            source.unlink()
            code = "from probe_layout import load_probe_layout; print(load_probe_layout('probes.csv').total_obs)"
            result = subprocess.check_output([sys.executable, "-c", code], cwd=worker, text=True)
            self.assertEqual(result.strip(), "1")

    def test_namelist_replaces_old_probe_coordinates(self):
        import f90nml
        from Solver.ADM_setup import ADMSimulation
        from types import SimpleNamespace
        layout = self.load("x,y,z,ux,uy,uz,p\n1,2,3,1,0,0,0\n4,5,6,0,0,0,1\n")
        farm = SimpleNamespace(n_turbines=3, turbines=[SimpleNamespace(diam=126)])
        simulation = ADMSimulation(farm, timesteps=10, control_freq=5, probe_layout=layout)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "input.i3d"
            path.write_text("&ProbesParam\nxyzprobes(1:3,1)=10,20,30\nxyzprobes(1:3,2)=40,50,60\nxyzprobes(1:3,3)=70,80,90\n/\n", encoding="ascii")
            simulation.add_probes(directory)
            coordinates = f90nml.read(path)["probesparam"]["xyzprobes"]
            self.assertEqual(coordinates, [list(position) for position in layout.coordinates])

    def test_setup_case_starts_statistics_before_first_control_output(self):
        import f90nml
        from types import SimpleNamespace
        from Solver.ADM_setup import ADMSimulation

        layout = self.load("x,y,z,ux,uy,uz,p\n1,2,3,1,0,0,1\n")
        farm = mock.Mock(n_turbines=3, lx=1260, lz=126, offset=[252, 378],
                         turbines=[SimpleNamespace(diam=126)])
        template_path = Path(__file__).resolve().parents[1] / "Solver/ADM/Base/input.i3d"
        for control_freq in (5, 50):
            with self.subTest(control_freq=control_freq), tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "input.i3d"
                template = f90nml.read(template_path)
                template["statistics"].update(nstat=2, initstat=1000, istatfreq=7)
                template.write(path)
                simulation = ADMSimulation(farm, timesteps=2000, control_freq=control_freq,
                                           probe_layout=layout)
                with mock.patch("Solver.ADM_setup.find_grid_dimensions", return_value=(16, 16, 16)):
                    simulation.setup_case(directory)
                generated = f90nml.read(path)
                statistics = generated["statistics"]
                self.assertEqual(statistics["nstat"], 1)
                self.assertEqual(statistics["initstat"], generated["basicparam"]["ifirst"])
                self.assertEqual(statistics["istatfreq"], 1)
                self.assertLessEqual(statistics["initstat"], generated["admparam"]["icontrolfreq"])
                self.assertEqual(generated["admparam"]["icontrolfreq"], control_freq)
                self.assertEqual(generated["inoutparam"]["nprobes"], layout.total_probes)


if __name__ == "__main__":
    unittest.main()