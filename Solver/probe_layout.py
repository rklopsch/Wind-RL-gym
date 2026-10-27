import csv
import hashlib
import json
import math
import shutil
from dataclasses import dataclass
from pathlib import Path


COMPONENTS = ("ux", "uy", "uz", "p")
MODULE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = MODULE_ROOT.parent if MODULE_ROOT.name == "Solver" else MODULE_ROOT


@dataclass(frozen=True)
class ProbeLayout:
    coordinates: tuple
    selections: tuple

    @property
    def total_probes(self):
        return len(self.coordinates)

    @property
    def total_obs(self):
        return sum(len(selection) for selection in self.selections)

    @property
    def fingerprint(self):
        payload = json.dumps((self.coordinates, self.selections), separators=(",", ":"))
        return hashlib.sha256(payload.encode("ascii")).hexdigest()

    def validate_domain(self, dimensions):
        for probe_id, position in enumerate(self.coordinates, start=1):
            for axis, coordinate, extent in zip(("x", "y", "z"), position, dimensions):
                if not 0 <= coordinate <= extent:
                    raise ValueError(f"Probe {probe_id}: {axis}={coordinate} outside [0, {extent}]")

    def write(self, path):
        with Path(path).open("w", newline="", encoding="ascii") as stream:
            writer = csv.writer(stream, lineterminator="\n")
            writer.writerow(("x", "y", "z", *COMPONENTS))
            for position, selection in zip(self.coordinates, self.selections):
                writer.writerow((*position, *(int(index in selection) for index in range(4))))


def load_probe_layout(path):
    path = Path(path).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    coordinates = []
    selections = []
    with path.open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        required = ("x", "y", "z", *COMPONENTS)
        if reader.fieldnames is None or sorted(reader.fieldnames) != sorted(required):
            raise ValueError(f"{path}: expected exactly the columns {','.join(required)}")
        for row in reader:
            try:
                if None in row or any(value is None for value in row.values()):
                    raise ValueError("incorrect number of columns")
                position = tuple(float(row[axis]) for axis in ("x", "y", "z"))
                if not all(math.isfinite(value) for value in position):
                    raise ValueError("coordinates must be finite")
                flags = tuple(row[component].strip() for component in COMPONENTS)
                if any(flag not in ("0", "1") for flag in flags):
                    raise ValueError("component flags must be 0 or 1")
                selection = tuple(index for index, flag in enumerate(flags) if flag == "1")
                if not selection:
                    raise ValueError("each probe must select at least one component")
            except (TypeError, ValueError) as error:
                raise ValueError(f"{path}:{reader.line_num}: {error}") from error
            coordinates.append(position)
            selections.append(selection)
    if not coordinates:
        raise ValueError(f"{path}: at least one probe is required")
    return ProbeLayout(tuple(coordinates), tuple(selections))


def probe_params(env):
    for obsolete in ("probes_per_turbine", "flow_field_directions"):
        if obsolete in env:
            raise ValueError(f"Replace env.{obsolete} with env.probes_file and per-probe CSV flags")
    staged = Path(__file__).resolve().parent / "probes.csv"
    layout = load_probe_layout(staged if staged.is_file() else env.probes_file)
    offset = float(env.get("pressure_offset", 0.0))
    scale = float(env.get("pressure_scale", 1.0))
    if not math.isfinite(offset) or not math.isfinite(scale) or scale <= 0:
        raise ValueError("pressure_offset must be finite and pressure_scale must be positive and finite")
    return {"probe_layout": layout, "pressure_offset": offset, "pressure_scale": scale}


def observation_contract(params):
    return {
        "layout": params["probe_layout"].fingerprint,
        "pressure_offset": float(params.get("pressure_offset", 0.0)),
        "pressure_scale": float(params.get("pressure_scale", 1.0)),
        "velocity_offset": [6.0, 0.0, 0.0],
        "velocity_scale": [6.0, 4.0, 4.0],
    }


def stage_probe_layout(directory, env):
    directory = Path(directory)
    params = probe_params(env)
    params["probe_layout"].write(directory / "probes.csv")
    shutil.copy2(__file__, directory / "probe_layout.py")
    metadata = observation_contract(params)
    metadata["features"] = [
        {"probe": probe_id, "position": position, "component": COMPONENTS[index]}
        for probe_id, (position, selection) in enumerate(
            zip(params["probe_layout"].coordinates, params["probe_layout"].selections), start=1
        )
        for index in selection
    ]
    (directory / "probe_layout.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="ascii")
    return str((directory / "probes.csv").resolve())


def checkpoint_contract(cfg, params):
    return {**observation_contract(params), "frame_stack": int(cfg.env.get("frame_stack", 1)),
            "n_turbines": int(params["n_turbines"])}


def save_checkpoint_contract(cfg, directory, checkpoint_id):
    params = {**probe_params(cfg.env), "n_turbines": cfg.env.turbines}
    path = Path(directory) / f"observation_{checkpoint_id}.json"
    path.write_text(json.dumps(checkpoint_contract(cfg, params), indent=2) + "\n", encoding="ascii")


def validate_checkpoint_contract(cfg, params, directory, checkpoint_id):
    path = Path(directory) / f"observation_{checkpoint_id}.json"
    if not path.is_file():
        raise ValueError(f"Missing checkpoint observation contract: {path}; migrate legacy checkpoints explicitly")
    if json.loads(path.read_text()) != checkpoint_contract(cfg, params):
        raise ValueError("Checkpoint probe layout, normalisation, frame stack or turbine count differs from this run")