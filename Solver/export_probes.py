import argparse
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf

from Solver.farm import Farm, Turbine
from Solver.probe_layout import ProbeLayout


def legacy_layout(env, count):
    if count <= 0 or count % 7:
        raise ValueError("The legacy grid requires a positive multiple of seven probes per turbine")
    diameter = env.turbine_diameter
    farm = Farm(diameter * env.turbine_spacing * (env.turbines - 1), diameter, env.turbines,
                Turbine(diameter, env.turbine_height),
                offset=[(env.turbine_spacing - 1) / 2 * diameter,
                        (env.turbine_spacing_z - 1) / 2 * diameter])
    farm.grid()
    streamwise, lateral = np.meshgrid(np.linspace(-2, 3, count // 7), np.linspace(-1, 1, 7))
    positions = tuple(
        (float(dx * diameter + turbine.location[0]), float(turbine.hub_height),
         float(dz * diameter + turbine.location[1]))
        for turbine in farm.turbines
        for dx, dz in zip(streamwise.ravel(), lateral.ravel())
    )
    return ProbeLayout(positions, ((0,),) * len(positions))


def main():
    parser = argparse.ArgumentParser(description="Export the former turbine-relative ux probe grid to CSV")
    parser.add_argument("--config", required=True)
    parser.add_argument("--count", required=True, type=int)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    layout = legacy_layout(OmegaConf.load(args.config).env, args.count)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    layout.write(output)
    print(f"Exported {layout.total_probes} probes to {output}")


if __name__ == "__main__":
    main()