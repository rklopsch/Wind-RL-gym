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
        # TODO: setup and run precursor simulation
        # TODO: setup and run initialisation case (currently just copying this)

        # setup running directory
        base_dir = './Solver/ADM/Base'  # TODO: change to basedir
        self.run_dir = './Solver/ADM/TESTING'
        self.precursor_dir = './Solver/ADM/TESTINGprecursor'
        shutil.copytree(base_dir, self.run_dir, dirs_exist_ok=True)
        # Run xCompact3d for initialisation
        # print('\nINITIALISING XCOMPACT3D CASE')
        # currently don't need to run as copying pre run case
        # self.initialise_flow(5000)

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
                        'ntimesteps': self.total_timesteps//10
                        }
                    }
        f90nml.patch(os.path.join(self.precursor_dir, 'old_input.i3d'),
                     patch_nml, os.path.join(self.precursor_dir, 'input.i3d'))

        # Run for iterations
        print(f'Running XCompact3D precursor simulation for ABL')
        subprocess.run(os.path.join(self.precursor_dir, 'run.sh'))

    def initialise_flow(self, iterations=100):
        # set up case for ADM with
        yaw = np.zeros(self.farm.n_turbines)
        self.farm.set_yaw(yaw)
        relative_precursor = os.path.join('../', os.path.basename(self.precursor_dir), 'out')
        # Update *.ad file
        self.farm.write_adm(os.path.join(self.run_dir, 'adm'))

        # Update start and end time
        shutil.move(os.path.join(self.run_dir, 'input.i3d'),
                    os.path.join(self.run_dir, 'old_input.i3d'))

        patch_nml = {'BasicParam': {
                        'ifirst': 1,
                        'ilast': iterations
                        },
                     'InOutParam': {
                        'irestart': 0,
                        'icheckpoint': iterations,
                        'ioutput': iterations,
                        'ilist': iterations,
                        'inflowpath': relative_precursor,
                        'ntimesteps': self.total_timesteps//10
                        }
                     }
        f90nml.patch(os.path.join(self.run_dir, 'old_input.i3d'),
                     patch_nml, os.path.join(self.run_dir, 'input.i3d'))

        # Run for iterations
        print(f'Initialisation Xcompact3D case from {1} to {iterations}')
        subprocess.run(os.path.join(self.run_dir, 'run.sh'))

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
        turbine_obs = np.empty((nturbs, 2))
        farm_power = 0

        for i in range(nturbs):
            fname = os.path.join(self.run_dir, f'disc{i + 1}.adm')
            turbine_data = np.loadtxt(fname, usecols=(2, 3), skiprows=1)  # , unpack=True)
            turbine_obs[i] = turbine_data[-1]
            farm_power += turbine_data[-1][1] / 1e06
            # turbine_data[:][0] is instantaneous wind speed
            # turbine_data[:][1] is instantaneous power
        print(f'Farm Power = {farm_power}')
        print(f'Farm Observations = {turbine_obs}')
        return torch.tensor(farm_power, dtype=torch.float32), torch.tensor(turbine_obs, dtype=torch.float32).flatten()


if __name__ == '__main__':

    farm1 = Farm(126*14, 126*4, 3, Turbine(126, 90, yaw=0), offset=[2 * 126, 2*126])
    farm1.grid()
    case = ADM(farm1)
    case.run_precursor()
    case.initialise_flow(1000)

    for i in range(20):
        print(f'iteration {i}')
        case.advance(torch.ones(farm1.n_turbines-1) * 2*i)
