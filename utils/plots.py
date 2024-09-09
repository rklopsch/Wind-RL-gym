# from functools import partial

import numpy as np
import pandas as pd
# import torch
from matplotlib import pyplot as plt
# import matplotlib.animation as ani
# from Solver.WF_enviroment import TurbEnv
# from Solver.farm import Turbine, Farm
import hydra
# import tqdm
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
    def __init__(self, casename, n_turbs):
        self.casename = casename
        self.nturbines = n_turbs
        self.data = []

    def collect_data(self):
        for turbine in range(self.nturbines):
            file = os.path.join(self.casename, f'disc{turbine+1}.adm')
            dat = pd.read_csv(file, sep="\s+|, ", engine='python')
            self.data.append(dat)

    def draw_power(self, it, ax):
        total_power = 0
        for turbine in range(self.nturbines):
            # ax.plot(self.data[turbine]['Time'][:it], self.data[turbine]['Power'][:it] / 1e6,
            #         color='k',
            #         alpha=0.3,
            #         label=f'Turbine {turbine + 1}')
            total_power += self.data[turbine]['Power'][:it]
        ax.plot(self.data[turbine]['Time'][:it], total_power[:it] / 1e6,
                color='k',
                alpha=1,
                label=f'Reinforcement Learning')
        ax.plot([self.data[turbine]['Time'][0], self.data[turbine]['Time'][it]], [16.46, 16.46],
                color='r',
                label=f'Bayesian Optimisation')
        ax.plot([self.data[turbine]['Time'][0], self.data[turbine]['Time'][it]], [16.46/1.338, 16.46/1.338],
                color='b',
                label=f'Baseline')
        # ax.legend(loc='upper right', ncol=len(cases), columnspacing=0.5, frameon=False)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Power (MW)')
        ax.set_xlim(self.data[0]['Time'][0], self.data[0]['Time'].iloc[-1])
        ax.set_ylim(0, max(total_power/1e6))
        # only use first labels in legend
        handles, labels = ax.get_legend_handles_labels()
        handles = handles[:3]
        labels = labels[:3]
        ax.legend(handles, labels, frameon=False, ncols=1)

    def draw_yaw(self, it, ax):
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22',
                  '#17becf']
        for turbine in range(self.nturbines):
        # for turbine in [0]:
                ax.plot(self.data[turbine]['Time'][:it], self.data[turbine]['YawAng'][:it],
                    color=colors[turbine],
                    alpha=0.1,
                    label=f'Turbine {turbine + 1}')
        # ax.legend(loc='upper right', ncol=len(cases), columnspacing=0.5, frameon=False)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel(r'$\text{Yaw}_i (^{\circ})$')
        ax.set_xlim(self.data[0]['Time'].iloc[0], self.data[0]['Time'].iloc[-1])
        ax.set_ylim(-45, 45)

        # Step 2: only use first labels in legend
        handles, labels = ax.get_legend_handles_labels()
        handles = handles[:self.nturbines]
        labels = labels[:self.nturbines]

        # Adjust the alpha of handles in the legend
        # for handle in handles:
        #     handle.set_alpha(1.0)

        # Step 4: Display the Legend
        ax.legend(handles, labels, frameon=False, ncols=3)


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


    # params = {
    #     "n_turbines": cfg.env.turbines,
    #     "probes_per_turbine": cfg.env.probes_per_turbine,
    #     "turbine_diameter": cfg.env.turbine_diameter,
    #     "turbine_spacing": cfg.env.turbine_spacing,
    #     "max_yaw_speed": cfg.env.max_yaw_speed,
    #     "max_yaw_angle": cfg.env.max_yaw_angle,
    #     "dt": cfg.env.steps_per_frame * 0.2,
    #     "run_steps": cfg.collector.total_frames * cfg.env.steps_per_frame,
    # }
    # env = TurbEnv(params, dummy_update=True)

    # Plotting velocity and pressure
    # fig, axs = plt.subplots(2, 1,
    #                         figsize=(10, 6),
    #                         constrained_layout=True,
    #                         sharex=True,
    #                         sharey=True)
    #
    # snaps = extract_integers_from_filenames(os.path.join(casename, 'data'))
    # iters = tqdm.tqdm(snaps, desc="Animation Iteration", position=0)
    # anim = ani.FuncAnimation(fig, partial(plot_slice, casename=casename, env=env, ax=axs), frames=iters)
    # anim.save(os.path.join(casename, 'slice.mp4'), fps=20, dpi=400)#codec='h263p')

    fig_power, axs_power = plt.subplots(1, 1,
                            figsize=(6, 4),
                            constrained_layout=True)
    fig_yaw, axs_yaw = plt.subplots(1, 1,
                            figsize=(6, 4),
                            constrained_layout=True)

    for env in range(32):
        casename = f'./training_ppo/WindFarm_{env+1}'
        print(casename)

        series1 = TimeSeries(casename, cfg.env.turbines)
        series1.collect_data()

        series1.draw_power(-1, axs_power)
        fig_power.savefig(os.path.join(casename, 'power.pdf'))

        # Plotting Yaw Time series

        series1.draw_yaw(-1, axs_yaw)
        fig_yaw.savefig(os.path.join(casename, 'yaw.pdf'))


if __name__ == "__main__":
    main()
