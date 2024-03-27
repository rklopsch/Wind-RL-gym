import torch
import os
from datetime import datetime
import sys
import subprocess
import shutil
import numpy as np
import f90nml
from Solver.farm import Turbine, Farm
from utils import is_verbose


def make_even(i):
    i += i % 2
    return int(i)


def make_odd(i):
    i -= i % 2 - 1
    return int(i)


class ADM:

    def __init__(self, farm, probes_per_turbine, windspeed=10, base_dir=None, device='cpu'):
        self.device = torch.device(device)
        self.farm = farm
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
        total_flowthroughs = 6
        stat_flowthroughs = 1
        init_flowthroughs = 4
        self.total_timesteps = int(self.lx / self.windspeed / self.dt * total_flowthroughs)
        self.init_timesteps = int(self.lx / self.windspeed / self.dt * init_flowthroughs)
        self.stat_timesteps = int(self.lx / self.windspeed / self.dt * (total_flowthroughs - stat_flowthroughs))

        if base_dir is None:
            self.dir = './LES_RUNS'
        else:
            self.dir = os.path.join('./LES_RUNS', base_dir)

        self.run_dir = os.path.join(self.dir, 'Running')
        self.precursor_dir = os.path.join(self.dir, 'PrecursorABL')
        self.initialise_dir = os.path.join(self.dir, 'Initialisation')

    def run_precursor(self):
        if os.path.isdir(self.precursor_dir):
            if is_verbose():
                print(f'Using precursor simulation that already exists in {self.precursor_dir}')
        else:
            base_dir = './Solver/ADM/precursor_Base'
            shutil.copytree(base_dir, self.precursor_dir, dirs_exist_ok=True)

            # Update start and end time
            shutil.move(os.path.join(self.precursor_dir, 'input.i3d'),
                        os.path.join(self.precursor_dir, 'old_input.i3d'))
            patch_nml = {'BasicParam': {
                            'ifirst': 1,
                            'ilast': self.total_timesteps,
                            'xlx': self.ly*2,
                            'yly': self.ly,
                            'zlz': self.lz,
                            'nx': self.ny*2,
                            'ny': self.ny,
                            'nz': self.nz,
                        },
                         'InOutParam': {
                            'ntimesteps': self.total_timesteps//10,
                            'ioutput': self.total_timesteps//10
                            }
                         }
            f90nml.patch(os.path.join(self.precursor_dir, 'old_input.i3d'),
                         patch_nml, os.path.join(self.precursor_dir, 'input.i3d'))

            # Run for iterations
            if is_verbose():
                print(f'Running XCompact3D precursor simulation for ABL from 0 to {self.total_timesteps}')
            subprocess.run(os.path.join(self.precursor_dir, 'run.sh'))

    def initialise_flow(self, iterations=100):
        if os.path.isdir(self.initialise_dir):
            if is_verbose():
                print(f'Using initialisation simulation that already exists in {self.initialise_dir}')
        else:
            base_dir = './Solver/ADM/Base'
            shutil.copytree(base_dir, self.initialise_dir, dirs_exist_ok=True)
            self.add_probes(self.initialise_dir)
            # set up case for ADM with
            yaw = np.zeros(self.farm.n_turbines)
            self.farm.set_yaw(yaw)
            relative_precursor = os.path.join('../', os.path.basename(self.precursor_dir), 'out/')
            # Update *.ad file
            self.farm.write_adm(os.path.join(self.initialise_dir, 'adm'))

            # Update start and end time
            shutil.move(os.path.join(self.initialise_dir, 'input.i3d'),
                        os.path.join(self.initialise_dir, 'old_input.i3d'))

            patch_nml = {'BasicParam': {
                            'ifirst': 1,
                            'ilast': iterations,
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
                            'icheckpoint': iterations,
                            'ioutput': iterations,
                            'ilist': iterations//10,
                            'inflowpath': relative_precursor,
                            'ntimesteps': self.total_timesteps//10,
                            'nprobes': self.probes_per_turbine*self.farm.n_turbines
                            },
                         'ADMParam': {
                            'Ndiscs': self.farm.n_turbines
                            }
                         }
            f90nml.patch(os.path.join(self.initialise_dir, 'old_input.i3d'),
                         patch_nml, os.path.join(self.initialise_dir, 'input.i3d'))

            # Run for iterations
            if is_verbose():
                print(f'Initialisation Xcompact3D case from {1} to {iterations}')
            subprocess.run(os.path.join(self.initialise_dir, 'run.sh'))

    def advance(self, yaws, iterations=50, save=False):
        yaw = yaws.numpy()
        if is_verbose():
            print(f'Turbine yaw angles = {yaw}')
        # set up case for ADM
        self.farm.set_yaw(yaw)

        # Update *.ad file
        self.farm.write_adm(os.path.join(self.run_dir, 'adm'))

        # Update case parameters input.i3d file
        ioutput = iterations if save else iterations*1000
        shutil.move(os.path.join(self.run_dir, 'input.i3d'), os.path.join(self.run_dir, 'old_input.i3d'))
        old_input = f90nml.read(os.path.join(self.run_dir, 'old_input.i3d'))
        old_ilast = old_input['BasicParam']['ilast']
        patch_nml = {'BasicParam': {
                        'ifirst': old_ilast + 1,
                        'ilast': old_ilast + iterations
                        },
                     'InOutParam': {
                        'irestart': 1,
                        'icheckpoint': iterations,
                        'ioutput': ioutput,
                        'ilist': iterations
                        }
                     }
        f90nml.patch(os.path.join(self.run_dir, 'old_input.i3d'), patch_nml, os.path.join(self.run_dir, 'input.i3d'))

        # Run for iterations
        if is_verbose():
            print(f'Running xcompact from {old_ilast+1} to {old_ilast+iterations}')
        subprocess.run(os.path.join(self.run_dir, 'run.sh'))

        # Retrieve Power
        turbine_obs = np.zeros((self.farm.n_turbines, 3+2*self.probes_per_turbine))
        farm_power = 0

        for iturb in range(self.farm.n_turbines):
            fname = os.path.join(self.run_dir, f'disc{iturb + 1}.adm')
            turbine_velocity, turbine_power = np.loadtxt(fname, usecols=(2, 3), skiprows=1, unpack=True)
            turbine_power /= 1e06
            turbine_power = turbine_power[-iterations:-1].mean()
            turbine_velocity = turbine_velocity[-iterations:-1].mean()
            farm_power += turbine_power
            turbine_obs[iturb][0] = turbine_velocity
            turbine_obs[iturb][1] = turbine_power
            turbine_obs[iturb][2] = yaw[iturb]
            # Read probes
            for iobs in range(self.probes_per_turbine):
                fname = os.path.join(self.run_dir, f'probes/probe{iobs*(iturb+1)+1:04}')
                probe_u, probe_w = np.loadtxt(fname, usecols=(1, 3), unpack=True)
                probe_u = probe_u[-iterations:-1].mean()
                probe_w = probe_w[-iterations:-1].mean()
                turbine_obs[iturb][iobs*2+3:(iobs+1)*2+3] = [probe_u, probe_w]

        if is_verbose():
            print(f'Farm Power = {farm_power}')

        return torch.tensor(farm_power, dtype=torch.float32), torch.tensor(turbine_obs, dtype=torch.float32)

    def restart(self, case_name=None):
        if case_name is not None:
            self.run_dir = os.path.join(self.dir, case_name)
        if is_verbose():
            print(f'copying {self.initialise_dir} to {self.run_dir}')
        shutil.copytree(self.initialise_dir, self.run_dir, dirs_exist_ok=True)

    def add_probes(self, directory):
        probes_per_turbine = self.probes_per_turbine
        probe_spacing = self.farm.turbines[0].diam/2
        nrows = probes_per_turbine // 5
        x, z = np.meshgrid(np.arange(probe_spacing, (nrows+1)*probe_spacing, probe_spacing),
                           np.arange(-2*probe_spacing, 2*probe_spacing+1, probe_spacing))
        y = np.ones(probes_per_turbine) * self.farm.turbines[0].hub_height

        probe_locations = np.zeros((3, self.farm.n_turbines*probes_per_turbine))
        for i, turb in enumerate(self.farm.turbines):
            probe_locations[0, i*probes_per_turbine:(i+1)*probes_per_turbine] = x.flatten() - turb.location[0]
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
    case = ADM(farm1, 25, base_dir='test')
    # case.total_timesteps = 3000
    case.run_precursor()
    case.initialise_flow(case.init_timesteps)
    case.restart()

    for i in range(20):
        print(f'iteration {i}')
        case.advance(torch.ones(farm1.n_turbines) * 2*i, save=True)
