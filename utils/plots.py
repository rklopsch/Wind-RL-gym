from functools import partial

import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt
import matplotlib.animation as ani
from Solver.WF_enviroment import TurbEnv
from Solver.farm import Turbine, Farm
import hydra
import tqdm
# import fonts
import os
import re

def plot_slice(t, casename, env, ax, height=90):
    for a in ax:
        a.clear()
    path = casename
    fileu = os.path.join(path, f'data/ux-{t}.bin')
    filep = os.path.join(path, f'data/pp-{t}.bin')
    gridx = np.linspace(0, env.adm.lx, env.adm.nx)
    gridy = np.linspace(0, env.adm.ly, env.adm.ny)
    gridz = np.linspace(0, env.adm.lz, env.adm.nz)

    u = np.fromfile(fileu, dtype=np.float64)
    ur = u.reshape((env.adm.nx, env.adm.ny, env.adm.nz), order='F')
    p = np.fromfile(filep, dtype=np.float64)
    pr = p.reshape((env.adm.nx, env.adm.ny, env.adm.nz), order='F')
    slice_loc = int(height / env.adm.ly * env.adm.ny)
    contour_u = ax[0].contourf(gridx/1000, gridz/1000, ur[:, slice_loc, :].T,
                               cmap='inferno_r', levels=100, vmin=0, vmax=10)
    contour_p = ax[1].contourf(gridx/1000, gridz/1000, pr[:, slice_loc, :].T,
                               cmap='Blues_r', levels=100, vmin=-10, vmax=10)
    for a in ax:
        a.set_aspect('equal')
        a.set_xlabel(r'$x \; (km)$')
        a.set_ylabel(r'$y \; (km)$')


# def plot_power(t, casename, env, ax):

class TimeSeries:
    def __init__(self, casename, env):
        self.casename = casename
        self.nturbines = env.n_turbs
        self.data = []

    def collect_data(self):
        for turbine in range(self.nturbines):
            file = os.path.join(self.casename, f'disc{turbine+1}.adm')
            dat = pd.read_csv(file, sep="\s+|, ", engine='python')
            self.data.append(dat)

    def draw(self, it, ax):
        total_power = 0
        for turbine in range(self.nturbines):
            ax.plot(self.data[turbine]['Time'][:it], self.data[turbine]['Power'][:it] / 1e6,
                    color='k',
                    alpha=0.3,
                    label=f'Turbine {turbine + 1}')
            total_power += self.data[turbine]['Power'][:it]
        ax.plot(self.data[turbine]['Time'][:it], total_power / 1e6,
                color='k',
                alpha=1,
                label=f'Total')
        # ax.legend(loc='upper right', ncol=len(cases), columnspacing=0.5, frameon=False)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Power (MW)')
        ax.set_xlim(self.data[0]['Time'][0], self.data[0]['Time'].iloc[-1])
        ax.set_ylim(0, max(total_power/1e6))
        ax.legend(frameon=False)


def extract_integers_from_filenames(directory):
    pattern = r'^snapshot-(\d+)\.xdmf'
    integers = []
    for filename in os.listdir(directory):
        match = re.match(pattern, filename)
        if match:
            integers.append(int(match.group(1)))
    integers.sort()
    return integers


@hydra.main(config_path="../ppo/", config_name="config_ppo", version_base="1.2")
def main(cfg: "DictConfig"):

    casename = './outputs/2024-04-18/14-15-25/RESULTS/TEST_2'

    params = {
        "n_turbines": cfg.env.turbines,
        "probes_per_turbine": cfg.env.probes_per_turbine,
        "turbine_diameter": cfg.env.turbine_diameter,
        "turbine_spacing": cfg.env.turbine_spacing,
        "max_yaw_speed": cfg.env.max_yaw_speed,
        "max_yaw_angle": cfg.env.max_yaw_angle,
        "dt": cfg.env.steps_per_frame * 0.2,
        "run_steps": cfg.collector.total_frames * cfg.env.steps_per_frame,
    }
    env = TurbEnv(params, dummy_update=True)

    # Plotting velocity and pressure
    fig, axs = plt.subplots(2, 1,
                            figsize=(10, 6),
                            constrained_layout=True,
                            sharex=True,
                            sharey=True)

    snaps = extract_integers_from_filenames(os.path.join(casename, 'data'))
    iters = tqdm.tqdm(snaps, desc="Animation Iteration", position=0)
    anim = ani.FuncAnimation(fig, partial(plot_slice, casename=casename, env=env, ax=axs), frames=iters)
    anim.save(os.path.join(casename, 'slice.mp4'), fps=20, dpi=400)#codec='h263p')

    # Plotting Time series
    fig, axs = plt.subplots(1, 1,
                            figsize=(6, 4),
                            constrained_layout=True)
    series1 = TimeSeries(casename, env)
    series1.collect_data()
    series1.draw(-1, axs)
    fig.savefig(os.path.join(casename, 'power.pdf'))


if __name__ == "__main__":
    main()
