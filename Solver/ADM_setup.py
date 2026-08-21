import torch
import os
import shutil
import numpy as np
import f90nml
from Solver.farm import Turbine, Farm
from itertools import combinations_with_replacement
from math import prod


def next_prime_product(start):
    primes = [2, 3, 5]  # Use small primes

    current = start
    while True:
        # Iterate over increasing lengths of combinations
        for combination_length in range(1, 14):
            for comb in combinations_with_replacement(primes, combination_length):
                if prod(comb) == current:
                    return current
        current += 1


def find_grid_dimensions(lx, ly, lz, delta):
    # Estimate maximum nx, ny, nz values
    min_nx = int(lx // delta)
    min_ny = int(ly // delta)
    min_nz = int(lz // delta)

    # Generate all combinations of products of primes
    nx = next_prime_product(min_nx)
    ny = next_prime_product(min_ny)
    nz = next_prime_product(min_nz)

    return nx, ny, nz


def test_grid_dimensions():
    # Example usage for find_grid_dimensions()
    lx, ly, lz = 10 * 7 * 126 + 4 * 126., 500., 126 * 7 * 10.
    delta = 2.
    nx, ny, nz = find_grid_dimensions(lx, ly, lz, delta)
    print(f"Optimal nx, ny, nz: {lx // delta}, {ly // delta}, {lz // delta}")
    print(f"Found prime combinations nx, ny, nz: {nx}, {ny}, {nz}")
    print(f"With dx, dy, dz: {lx / nx}, {ly / ny}, {lz / nz}")


class ADMSimulation:

    def __init__(self, farm, timesteps, control_freq, probes_per_turbine=25, windspeed=10, instance=0):
        self.farm = farm
        self.n_turbines = self.farm.n_turbines
        self.windspeed = windspeed
        self.probes_per_turbine = probes_per_turbine
        self.diameter = farm.turbines[0].diam
        self.gridsize = self.diameter/10.
        self.dt = 0.2 * self.gridsize / self.windspeed
        # TODO: probably want to set dt in sims
        self.total_timesteps = timesteps
        self.control_freq = control_freq
        self.instance = instance

    def setup_precursor(self, directory):

        ly = 500
        lz = self.farm.lz + self.farm.offset[1] + self.farm.offset[1]
        lx = lz*32
        nx, ny, nz = find_grid_dimensions(lx, ly, lz, self.gridsize)
        ny += 1

        # Update start and end time
        shutil.move(os.path.join(directory, 'input.i3d'),
                    os.path.join(directory, 'old_input.i3d'))
        patch_nml = {'BasicParam': {
                        'ifirst': 1,
                        'ilast': self.total_timesteps,
                        'xlx': lx,
                        'yly': ly,
                        'zlz': lz,
                        'nx': nx,
                        'ny': ny,
                        'nz': nz,
                    },
                     'InOutParam': {
                        'ntimesteps': 5000,
                        'ioutput': self.total_timesteps//100
                        }
                     }
        f90nml.patch(os.path.join(directory, 'old_input.i3d'),
                     patch_nml, os.path.join(directory, 'input.i3d'))

    def setup_case(self, directory, save_flow=False, precursor_root=None):

        lx = self.farm.lx + self.farm.offset[0] + 7*self.diameter
        ly = 500
        lz = self.farm.lz + self.farm.offset[1] + self.farm.offset[1]
        nx, ny, nz = find_grid_dimensions(lx, ly, lz, self.gridsize)
        nx += 1
        ny += 1
        if save_flow:
            save_interval = self.control_freq
        else:
            save_interval = self.total_timesteps

        precursor_dir = os.path.join(os.path.dirname(directory), f'precursor_{self.instance}')
        source_precursor_dir = None
        if precursor_root:
            source_precursor_dir = os.path.join(os.path.abspath(os.path.expanduser(str(precursor_root))),
                                               f'precursor_{self.instance}')

        inflow_dir = None
        if source_precursor_dir is not None:
            inflow_dir = os.path.join(source_precursor_dir, 'out')
        else:
            inflow_dir = os.path.join(precursor_dir, 'out')

        has_precursor_inflow = os.path.isdir(inflow_dir) and any(
            'inflow' in name.lower() for name in os.listdir(inflow_dir)
        )

        self.add_probes(directory)
        # Set-up ADM turbine parameters
        yaw = np.zeros(self.farm.n_turbines)
        self.farm.set_yaw(yaw)
        self.farm.write_adm(os.path.join(directory, 'adm'))

        # Update input parameters
        shutil.move(os.path.join(directory, 'input.i3d'),
                    os.path.join(directory, 'old_input.i3d'))

        patch_nml = {'BasicParam': {
            'ifirst': 1,
            'ilast': self.total_timesteps,
            'xlx': lx,
            'yly': ly,
            'zlz': lz,
            'nx': nx,
            'ny': ny,
            'nz': nz,
            'iin': 3 if has_precursor_inflow else 1,
            },
            'InOutParam': {
                'irestart': 0,
                'icheckpoint': self.total_timesteps,
                'ioutput': save_interval,
                'ilist': self.total_timesteps // 1000,
                'inflowpath': f'{inflow_dir}/' if has_precursor_inflow else '',
                'ntimesteps': 5000,
                'ninflows': self.total_timesteps // 5000 if has_precursor_inflow else 1,
                'nprobes': self.probes_per_turbine * self.farm.n_turbines
            },
            'ADMParam': {
                'Ndiscs': self.farm.n_turbines,
                'instance': self.instance,
                'iturboutput': self.control_freq,
                'icontrolfreq': self.control_freq
            }
        }
        if not has_precursor_inflow:
            print(f"No precursor inflow found for instance {self.instance}; falling back to built-in inflow profile.")
        f90nml.patch(os.path.join(directory, 'old_input.i3d'),
                     patch_nml, os.path.join(directory, 'input.i3d'))

    def add_probes(self, directory):
        probes_per_turbine = self.probes_per_turbine
        probe_rows = 7
        nrows = probes_per_turbine // probe_rows
        x, z = np.meshgrid(
            np.linspace(-2, 3, nrows),
            np.linspace(-1, 1, probe_rows))
        x *= self.farm.turbines[0].diam
        z *= self.farm.turbines[0].diam
        y = np.ones(probes_per_turbine) * self.farm.turbines[0].hub_height

        probe_locations = np.zeros((3, self.farm.n_turbines*probes_per_turbine))
        for i, turb in enumerate(self.farm.turbines):
            probe_locations[0, i*probes_per_turbine:(i+1)*probes_per_turbine] = x.flatten() + turb.location[0]
            probe_locations[1, i*probes_per_turbine:(i+1)*probes_per_turbine] = y.flatten()
            probe_locations[2, i*probes_per_turbine:(i+1)*probes_per_turbine] = z.flatten() + turb.location[1]

        probe_dictionary = {}
        for i, loc in enumerate(probe_locations.T):
            for dim in range(3):
                probe_dictionary[f'xyzprobes({dim + 1},{i + 1})'] = loc[dim]

        # Update i3d file
        shutil.move(os.path.join(directory, 'input.i3d'),
                    os.path.join(directory, 'old_input.i3d'))
        patch_nml = {'ProbesParam': probe_dictionary}
        f90nml.patch(os.path.join(directory, 'old_input.i3d'),
                     patch_nml, os.path.join(directory, 'input.i3d'))
