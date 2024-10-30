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
import yaml
from scipy.signal import welch


class TimeSeries:
    def __init__(self, casename, config, eval=True):
        self.casename = casename
        self.data = []
        self.U = 8
        self.n_turbs = config['env']['turbines']
        self.D = config['env']['turbine_diameter']
        self.dt = 0.2 * config['env']['steps_per_frame']
        self.spacing = config['env']['turbine_spacing'] * self.D
        self.domain_length = self.D * (config['env']['turbine_spacing'] * (self.n_turbs - 0.5) - 0.5)
        self.start_time = config['env']['initial_reset_frames'] * self.dt
        if eval:
            self.end_time = self.start_time + config['eval']['episode_length'] * self.dt
        else:
            self.end_time = self.start_time + config['collector']['max_episode_length'] * self.dt

    def collect_data(self):
        for turbine in range(self.n_turbs):
            file = os.path.join(self.casename, f'disc{turbine+1}.adm')
            dat = pd.read_csv(file, sep="\s+|, ", engine='python')
            dat['NonDimTime'] = dat['Time'] * self.U / self.D
            self.data.append(dat)

    def draw_power(self, it, ax):
        total_power = 0
        for turbine in range(self.n_turbs):
            ax.plot(self.data[turbine]['Time'][:it], self.data[turbine]['Power'][:it] / 1e6,
                    label=f'Turbine {turbine + 1}')
            total_power += self.data[turbine]['Power'][:it]
        ax.plot(self.data[turbine]['Time'][:it], total_power[:it] / 1e6,
                color='k',
                alpha=1,
                label=f'Total')

        # ax.plot([self.data[turbine]['Time'][0], self.data[turbine]['Time'][it]], [16.46, 16.46],
        #         color='r',
        #         label=f'Bayesian Optimisation')
        # ax.plot([self.data[turbine]['Time'][0], self.data[turbine]['Time'][it]], [16.46/1.338, 16.46/1.338],
        #         color='b',
        #         label=f'Baseline')
        # ax.legend(loc='upper right', ncol=len(cases), columnspacing=0.5, frameon=False)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Power (MW)')
        ax.set_xlim(self.data[0]['Time'][0], self.data[0]['Time'].iloc[-1])
        # ax.set_ylim(0, max(total_power/1e6))
        ax.set_xlim(self.start_time, self.end_time)

        # only use first labels in legend
        handles, labels = ax.get_legend_handles_labels()
        handles = handles[:4]
        labels = labels[:4]
        ax.legend(handles, labels, frameon=False, ncols=5)

    def draw_power_psd(self, it, ax):
        fs = 1/self.dt
        for turbine in range(self.n_turbs):
            freqs, psd = welch(self.data[turbine]['Power'][3000:it], fs, nperseg=1024)
            ax.semilogx(freqs, psd*freqs, label=f'Turbine {turbine + 1}')

        ax.set_xlabel(r'$f \; (Hz)$')
        ax.set_ylabel(r'$f \, PSD \; (W^2)$')
        ax.legend(frameon=False, ncols=1)

        self.add_frequecies(ax, max(psd*freqs))

    def draw_yaw_psd(self, it, ax):
        fs = 1/self.dt
        for turbine in range(self.n_turbs):
            freqs, psd = welch(self.data[turbine]['YawAng'][3000:it], fs, nperseg=1024)
            ax.semilogx(freqs, psd * freqs, label=f'Turbine {turbine + 1}')

        ax.set_xlabel(r'$f \; (Hz)$')
        ax.set_ylabel(r'$f \, PSD$')
        ax.legend(frameon=False, ncols=1)

        self.add_frequecies(ax, max(psd*freqs))

    def add_frequecies(self, ax, label_max):
        ax.axvspan(0.15 * self.U / self.D, 0.25 * self.U / self.D, color='k', alpha=0.1)
        ax.text(0.015 * 1.1, label_max, 'Wake Meandering', rotation=90, va='top')
        ax.axvline(1 / self.dt, color='k', linestyle='--')
        ax.text(1 / self.dt * 1.1, label_max, 'RL Sampling', rotation=90, va='top')
        ax.axvline(1 / self.dt / 2, color='k', linestyle='--')
        ax.text(1 / self.dt / 2 * 1.1, label_max, 'RL Nyquist', rotation=90, va='top')
        ax.axvline(self.U / self.domain_length, color='k', linestyle='--', alpha=0.3)
        ax.text(self.U / self.domain_length * 1.1, label_max, 'Domain Flow Through', rotation=90, va='top',
                alpha=0.3)
        ax.axvline(self.U / self.spacing, color='k', linestyle='--', alpha=0.3)
        ax.text(self.U / self.spacing * 1.1, label_max, 'Turbine Spacing', rotation=90, va='top', alpha=0.3)
        ax.axvline(1 / (self.end_time - self.start_time), color='k', linestyle='--')
        ax.text(1 / (self.end_time - self.start_time) * 1.1, label_max, 'Episode Length', rotation=90, va='top', alpha=0.3)

    def draw_yaw(self, it, ax):
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22',
                  '#17becf']
        for turbine in range(self.n_turbs):
                ax.plot(self.data[turbine]['Time'][:it], self.data[turbine]['YawAng'][:it],
                    color=colors[turbine],
                    alpha=1,
                    label=f'Turbine {turbine + 1}')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel(r'$\text{Yaw}_i (^{\circ})$')
        ax.set_xlim(self.data[0]['Time'].iloc[0], self.data[0]['Time'].iloc[-1])
        ax.set_ylim(-45, 45)
        ax.set_xlim(self.start_time, self.end_time)

        # only use first labels in legend
        handles, labels = ax.get_legend_handles_labels()
        handles = handles[:self.n_turbs]
        labels = labels[:self.n_turbs]
        ax.legend(handles, labels, frameon=False, ncols=5)
        # ax.legend(loc='upper right', ncol=len(cases), columnspacing=0.5, frameon=False)


# @hydra.main(config_path="../ppo/", config_name="config_ppo", version_base="1.2")
# def main(cfg: "DictConfig"):
def main():

    fig, axs = plt.subplots(2, 2,
                            figsize=(12, 8),
                            constrained_layout=True,
                            gridspec_kw={'width_ratios': [2, 1]})
                            # sharey=True)

    casename = f'./training_ppo_28-10-24'
    with open(os.path.join(casename, "ppo/config_ppo.yaml"), "r") as file:
        config = yaml.safe_load(file)
    print(casename)

    for env in range(2):
    # for env in [0]:
        environment_name = f'./training_ppo_28-10-24/WindFarm_{env+1}'
        print(environment_name)
        series1 = TimeSeries(environment_name, config)
        series1.collect_data()
        series1.draw_yaw_psd(-1, axs[0, 1])
        series1.draw_yaw(-1, axs[0, 0])
        series1.draw_power_psd(-1, axs[1, 1])
        series1.draw_power(4998, axs[1, 0])
        fig.savefig(os.path.join(environment_name, 'time_series.pdf'))
    axs[1, 0].set_ylim(0, 7)
    plt.show()

    fig.savefig(os.path.join(casename, 'time_series_all.pdf'))


if __name__ == "__main__":
    main()
