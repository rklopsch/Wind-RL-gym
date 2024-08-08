import torch
import os
import subprocess
import shutil
import numpy as np
import f90nml
from Solver.farm import Turbine, Farm

is_verbose = False


def make_even(i):
    i += i % 2
    return int(i)


def make_odd(i):
    i -= i % 2 - 1
    return int(i)


class ADMSimulation:

    def __init__(self, farm, timesteps, probes_per_turbine=25, windspeed=10, instance=0):
        self.farm = farm
        self.n_turbines = self.farm.n_turbines
        self.windspeed = windspeed
        self.probes_per_turbine = probes_per_turbine
        self.diameter = farm.turbines[0].diam
        self.gridsize = self.diameter/10.
        self.lx = farm.lx + farm.offset[0] + 7*self.diameter
        self.ly = 500
        self.lz = farm.lz + farm.offset[1] + farm.offset[1]
        self.nx = make_odd(self.lx // self.gridsize)
        self.ny = make_odd(self.ly // self.gridsize)
        self.nz = make_even(self.lz // self.gridsize)
        self.dt = 0.2 * self.gridsize / self.windspeed
        self.total_timesteps = timesteps
        self.instance = instance

    def setup_precursor(self, directory):

        # Update start and end time
        shutil.move(os.path.join(directory, 'input.i3d'),
                    os.path.join(directory, 'old_input.i3d'))
        patch_nml = {'BasicParam': {
                        'ifirst': 1,
                        'ilast': self.total_timesteps,
                        'xlx': self.ly*4,
                        'yly': self.ly,
                        'zlz': self.lz,
                        'nx': self.ny*4,
                        'ny': self.ny,
                        'nz': self.nz,
                    },
                     'InOutParam': {
                        'ntimesteps': self.total_timesteps//100,
                        'ioutput': self.total_timesteps//100
                        }
                     }
        f90nml.patch(os.path.join(directory, 'old_input.i3d'),
                     patch_nml, os.path.join(directory, 'input.i3d'))

    def setup_case(self, directory):
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
            'xlx': self.lx,
            'yly': self.ly,
            'zlz': self.lz,
            'nx': self.nx,
            'ny': self.ny,
            'nz': self.nz,
        },
            'Statistics': {
                'initstat': self.stat_timesteps,
            },
            'InOutParam': {
                'irestart': 0,
                'icheckpoint': self.total_timesteps,
                'ioutput': self.total_timesteps,
                'ilist': self.total_timesteps // 1000,
                # 'inflowpath': relative_precursor,
                'ntimesteps': self.total_timesteps // 100,
                'ninflows': 100,
                'nprobes': self.probes_per_turbine * self.farm.n_turbines
            },
            'ADMParam': {
                'Ndiscs': self.farm.n_turbines,
                'instance': self.instance
            }
        }
        f90nml.patch(os.path.join(directory, 'old_input.i3d'),
                     patch_nml, os.path.join(directory, 'input.i3d'))

    def add_probes(self, directory):
        probes_per_turbine = self.probes_per_turbine
        probe_spacing = self.farm.turbines[0].diam/2
        nrows = probes_per_turbine // 5
        x, z = np.meshgrid(np.arange(probe_spacing, (nrows+1)*probe_spacing, probe_spacing),
                           np.arange(-2*probe_spacing, 2*probe_spacing+1, probe_spacing))
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


if __name__ == '__main__':

    farm1 = Farm(126*14, 126*4, 3, Turbine(126, 90, yaw=0), offset=[2 * 126, 2*126])
    farm1.grid()
    case = ADM(farm1, 25)
