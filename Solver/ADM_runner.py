import torch
import os
import sys
import subprocess
import shutil
import numpy as np
import f90nml
from Solver.farm import Turbine, Farm


class ADM:

    def __init__(self, farm, dt=0.2, device='cpu'):
        self.device = torch.device(device)
        self.farm = farm
        self.dt = dt
        self.total_timesteps = 2000
        self.dx = farm.turbines[0].diam/10.
        # ect...
        # TODO: setup case parameters (from farm)

        self.run_dir = './Solver/ADM/TESTING'
        self.precursor_dir = './Solver/ADM/TESTINGprecursor'
        self.initialise_dir = './Solver/ADM/TESTINGinitialisation'

    def run_precursor(self):
        base_dir = './Solver/ADM/precursor_Base'
        shutil.copytree(base_dir, self.precursor_dir, dirs_exist_ok=True)

        # Update start and end time
        shutil.move(os.path.join(self.precursor_dir, 'input.i3d'),
                    os.path.join(self.precursor_dir, 'old_input.i3d'))
        patch_nml = {'BasicParam': {
                        'ifirst': 1,
                        'ilast': self.total_timesteps
                        },
                     'InOutParam': {
                        'ntimesteps': self.total_timesteps//10,
                        'ioutput': self.total_timesteps//10
                        }
                     }
        f90nml.patch(os.path.join(self.precursor_dir, 'old_input.i3d'),
                     patch_nml, os.path.join(self.precursor_dir, 'input.i3d'))

        # Run for iterations
        print(f'Running XCompact3D precursor simulation for ABL from 0 to {self.total_timesteps}')
        subprocess.run(os.path.join(self.precursor_dir, 'run.sh'))

    def initialise_flow(self, iterations=100):
        base_dir = './Solver/ADM/Base'
        shutil.copytree(base_dir, self.initialise_dir, dirs_exist_ok=True)
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
                        'ilast': iterations
                        },
                     'InOutParam': {
                        'irestart': 0,
                        'icheckpoint': iterations,
                        'ioutput': iterations,
                        'ilist': iterations//10,
                        'inflowpath': relative_precursor,
                        'ntimesteps': self.total_timesteps//10
                        }
                     }
        f90nml.patch(os.path.join(self.initialise_dir, 'old_input.i3d'),
                     patch_nml, os.path.join(self.initialise_dir, 'input.i3d'))

        # Run for iterations
        print(f'Initialisation Xcompact3D case from {1} to {iterations}')
        subprocess.run(os.path.join(self.initialise_dir, 'run.sh'))

    def advance(self, yaws, iterations=50, save=False):
        nturbs = np.shape(yaws.numpy())[0]+1
        yaw = np.zeros(nturbs)
        yaw[:nturbs-1] = yaws.numpy()
        yaw[-1] = 0
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
        print(f'Running xcompact from {old_ilast+1} to {old_ilast+iterations}')
        subprocess.run(os.path.join(self.run_dir, 'run.sh'))

        # Retrieve Power
        turbine_obs = np.empty((nturbs, 4))
        farm_power = 0

        for i in range(nturbs):
            fname = os.path.join(self.run_dir, f'disc{i + 1}.adm')
            turbine_velocity, turbine_power = np.loadtxt(fname, usecols=(2, 3), skiprows=1, unpack=True)
            turbine_power /= 1e06
            turbine_power = turbine_power[-iterations:-1].mean()
            turbine_velocity = turbine_velocity[-iterations:-1].mean()
            farm_power += turbine_power
            # Read probes
            fname = os.path.join(self.run_dir, f'probes/probe000{i+1}')
            probe_u, probe_w = np.loadtxt(fname, usecols=(1, 3), unpack=True)
            probe_u = probe_u[-iterations:-1].mean()
            probe_w = probe_w[-iterations:-1].mean()
            turbine_obs[i] = [turbine_velocity, turbine_power, probe_u, probe_w]

        print(f'Farm Power = {farm_power}')
        print(f'Farm Observations = {turbine_obs}')
        return torch.tensor(farm_power, dtype=torch.float32), torch.tensor(turbine_obs, dtype=torch.float32).flatten()

    def restart(self, case_name=None):
        if case_name==None:
            case_name = self.run_dir
        shutil.copytree(self.initialise_dir, case_name, dirs_exist_ok=True)


if __name__ == '__main__':

    farm1 = Farm(126*14, 126*4, 3, Turbine(126, 90, yaw=0), offset=[2 * 126, 2*126])
    farm1.grid()
    case = ADM(farm1)
    case.total_timesteps = 3000
    case.run_precursor()
    case.initialise_flow(2000)
    case.restart()

    for i in range(20):
        print(f'iteration {i}')
        case.advance(torch.ones(farm1.n_turbines-1) * 2*i, save=True)
