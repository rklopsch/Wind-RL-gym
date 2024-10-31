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
        self.freq_data = []
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

    def calculate_psds(self):
        fs = 1/self.dt
        total_steps = int((self.end_time - self.start_time)//self.dt)-1
        for turbine in range(self.n_turbs):
            freqs, psd_power = welch(self.data[turbine]['Power'][int(self.start_time//self.dt):-1], fs, nperseg=total_steps)
            freqs, psd_yaw = welch(self.data[turbine]['YawAng'][int(self.start_time//self.dt):-1], fs, nperseg=total_steps)
            df_psd = pd.DataFrame({'Frequency': freqs, 'Power_PSD': psd_power, 'YawAng_PSD': psd_yaw})
            self.freq_data.append(df_psd)

    def draw_power(self, it, ax):
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22',
                  '#17becf']
        total_power = 0
        for turbine in range(self.n_turbs):
            ax.plot(self.data[turbine]['Time'][:it], self.data[turbine]['Power'][:it] / 1e6,
                    color=colors[turbine],
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
        ax.set_xlim(self.start_time, self.end_time)
        ax.set_ylim(0, max(total_power[int(self.start_time//self.dt):]/1e6))

        # only use first labels in legend
        handles, labels = ax.get_legend_handles_labels()
        handles = handles[:self.n_turbs]
        labels = labels[:self.n_turbs]
        ax.legend(handles, labels, frameon=False, ncols=5)

    def draw_power_psd(self, it, ax):
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22',
                  '#17becf']
        for turbine in range(self.n_turbs):
            ax.semilogx(self.freq_data[turbine]['Frequency'],
                        self.freq_data[turbine]['Power_PSD'] * self.freq_data[turbine]['Frequency'],
                        color=colors[turbine], label=f'Turbine {turbine + 1}')
        ax.set_xlabel(r'$f \; (Hz)$')
        ax.set_ylabel(r'$f \, PSD \; (W^2)$')
        # only use first labels in legend
        handles, labels = ax.get_legend_handles_labels()
        handles = handles[:self.n_turbs]
        labels = labels[:self.n_turbs]
        ax.legend(handles, labels, frameon=False, ncols=1)

        self.add_frequecies(ax, max(self.freq_data[turbine]['Power_PSD'] * self.freq_data[turbine]['Frequency']))

    def draw_yaw_psd(self, it, ax):
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22',
                  '#17becf']
        for turbine in range(self.n_turbs):
            ax.semilogx(self.freq_data[turbine]['Frequency'],
                        self.freq_data[turbine]['YawAng_PSD'] * self.freq_data[turbine]['Frequency'],
                        color=colors[turbine], label=f'Turbine {turbine + 1}')

        ax.set_xlabel(r'$f \; (Hz)$')
        ax.set_ylabel(r'$f \, PSD$')
        # only use first labels in legend
        handles, labels = ax.get_legend_handles_labels()
        handles = handles[:self.n_turbs]
        labels = labels[:self.n_turbs]
        ax.legend(handles, labels, frameon=False, ncols=1)

        self.add_frequecies(ax, max(self.freq_data[turbine]['YawAng_PSD'] * self.freq_data[turbine]['Frequency']))

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
                            figsize=(14, 8),
                            constrained_layout=True,
                            gridspec_kw={'width_ratios': [2, 1]})
                            # sharey=True)

    casename = f'./eval_ppo_31-10-24'
    with open(os.path.join(casename, "eval/config_ppo.yaml"), "r") as file:
        config = yaml.safe_load(file)
    print(casename)

    for env in range(config['eval']['n_parallel']):
        for ax in axs.flatten():
            ax.cla()
        environment_name = os.path.join(casename, f'WindFarm_{env+1}')
        print(environment_name)
        series1 = TimeSeries(environment_name, config)
        series1.collect_data()
        series1.calculate_psds()
        series1.draw_yaw_psd(-1, axs[0, 1])
        series1.draw_yaw(-1, axs[0, 0])
        series1.draw_power_psd(-1, axs[1, 1])
        series1.draw_power(4998, axs[1, 0])
        fig.savefig(os.path.join(environment_name, 'time_series.pdf'))

    fig.savefig(os.path.join(casename, 'time_series_all.pdf'))
    fig.savefig(os.path.join(casename, 'time_series_all.png'), dpi=600)

    plt.show()



if __name__ == "__main__":
    main()
