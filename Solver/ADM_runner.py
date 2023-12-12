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
        self.dx = farm.turbines[0].diam/10.
        # ect...
        # TODO: setup case parameters (from farm)
        # TODO: setup and run precursor simulation
        # TODO: setup and run initialisation case (currently just copying this)

        # setup running directory
        base_dir = './Solver/ADM/initialise'
        self.run_dir = './Solver/ADM/running'
        shutil.copytree(base_dir, self.run_dir, dirs_exist_ok=True)
        # Run xCompact3d for initialisation
        print('\nINITIALISING XCOMPACT3D CASE')
        # currently don't need to run as copying pre run case
        # subprocess.run(os.path.join(self.run_dir, 'run.sh'))

    def advance(self, yaws, iterations=50):

        nturbs = np.shape(yaws.numpy())[0]+1
        yaw = np.zeros(nturbs)
        yaw[:nturbs-1] = yaws.numpy()
        yaw[-1] = 0
        print(yaw)
        # set up case for ADM
        self.farm.set_yaw(yaw)
        path = f'./Solver/ADM/running/'

        # Update *.ad file
        self.farm.write_adm(os.path.join(path, 'adm'))

        # Update start and end time
        shutil.move(os.path.join(path, 'input.i3d'), os.path.join(path, 'old_input.i3d'))
        old_input = f90nml.read(os.path.join(path, 'old_input.i3d'))
        old_ilast = old_input['BasicParam']['ilast']
        patch_nml = {'BasicParam': {
                        'ifirst': old_ilast + 1,
                        'ilast': old_ilast + iterations
                        },
                     'InOutParam': {
                        'irestart': 1,
                        'icheckpoint': iterations,
                        'ioutput': iterations,
                        'ilist': iterations
                        }
                     }
        f90nml.patch(os.path.join(path, 'old_input.i3d'), patch_nml, os.path.join(path, 'input.i3d'))

        # Run for iterations
        print(f'Running xcompact from {old_ilast+1} to {old_ilast+iterations}')
        subprocess.run(os.path.join(path, 'run.sh'))

        # Retrieve Power
        turbine_obs = np.empty((nturbs, 2))
        farm_power = 0

        for i in range(nturbs):
            fname = os.path.join(path, f'disc{i + 1}.adm')
            turbine_data = np.loadtxt(fname, usecols=(2, 3), skiprows=1)  # , unpack=True)
            turbine_obs[i] = turbine_data[-1]
            farm_power += turbine_data[-1][0] / 1e06
            # turbine_data[:][0] is instantaneous wind speed
            # turbine_data[:][1] is instantaneous power
        print(f'Farm Power = {farm_power}')
        print(f'Farm Observations = {turbine_obs}')
        return torch.tensor(farm_power), torch.tensor(turbine_obs).flatten()


if __name__ == '__main__':

    farm1 = Farm(126 * 14, 126*4, 3, Turbine(126, 90, yaw=0), offset=[2 * 126, 2*126])
    farm1.grid(staggered=False)
    case = ADM(farm1)

    for i in range(20):
        print(f'iteration {i}')
        case.advance(torch.ones(farm1.n_turbines-1) * 2*i)
