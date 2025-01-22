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
import sys
import re
import yaml
from scipy.signal import welch


class TimeSeries:
    def __init__(self, casename, config, mode='eval'):
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
        self.episode_length = config['collector']['max_episode_length']
        self.reset_length = config['env']['reset_frames']
        self.init_length = (config['env']['initial_reset_frames'] // self.reset_length) * self.reset_length
        if mode == 'eval':
            print(config['env']['reset_frames'])
            self.start_time = (config['env']['initial_reset_frames'] + config['env']['reset_frames']) * self.dt
            self.end_time = self.start_time + config['eval']['episode_length'] * self.dt
        else:
            self.start_time = config['env']['initial_reset_frames'] * self.dt
            self.end_time = (self.start_time +
                             (config['collector']['max_episode_length'] + config['env']['reset_frames'])
                             * config['collector']['total_frames'] / config['env']['n_parallel'] / config['collector']['max_episode_length']
                             * self.dt)

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
            freqs, psd_power = welch(self.data[turbine]['Power_ave'][int(self.start_time//self.dt):-1], fs, nperseg=500)  # total_steps//2)
            freqs, psd_yaw = welch(self.data[turbine]['YawAng'][int(self.start_time//self.dt):-1], fs, nperseg=500)  # total_steps//2)
            df_psd = pd.DataFrame({'Frequency': freqs, 'Power_PSD': psd_power, 'YawAng_PSD': psd_yaw})
            self.freq_data.append(df_psd)

    def time_average(self, turbine):
        # Remove start
        # print(f'origional data {self.data[turbine].shape}')
        df_filtered = self.data[turbine].iloc[self.init_length+1*self.reset_length:].reset_index(drop=True)
        # print(f'filtered data {df_filtered.shape}')
        cycle_length = self.episode_length + self.reset_length
        mask = np.arange(len(df_filtered)) % cycle_length >= self.reset_length
        episode_data = df_filtered[mask].copy()
        # ... = np.arange(len(episode_data)) // self.episode_length
        # l = []
        # for i in range(len(episode_data)//self.episode_length):
        #   l += self.episode_length * [i]
        episode_data.loc[:, 'chunk_id'] = (np.arange(len(df_filtered))[mask] // (self.episode_length+self.reset_length))
        episode_averaged_power = episode_data.groupby('chunk_id')['Power_ave'].mean()
        x_positions = [(cycle_length * i + self.episode_length // 2) * self.dt + self.start_time for i in episode_averaged_power.index]
        return x_positions, episode_averaged_power, mask

    def draw_power(self, it, ax):
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22',
                  '#17becf']
        total_power = 0
        for turbine in range(self.n_turbs):
            ax.plot(self.data[turbine]['Time'][:it], self.data[turbine]['Power_ave'][:it] / 1e6,
                    color=colors[turbine],
                    alpha=0.05,
                    label=f'_Turbine {turbine + 1}')
            x, y, mask = self.time_average(turbine)
            # ax.scatter(self.start_time+mask*10, np.ones_like(mask))
            ax.step(x, y/1e6, where='pre', label='Episode Mean', color=colors[turbine], linewidth=2)
            # ax.step(self.data[turbine]['Time'][self.init_length:it], mask[:-1], where='mid', label='Episode Mean', color=colors[turbine], linewidth=2)
            total_power += self.data[turbine]['Power_ave'][:it]
        ax.plot(self.data[turbine]['Time'][:it-1], total_power[:it] / 1e6,
                color='k',
                alpha=0.05,
                label=f'_Total')
        average_power = np.mean(total_power[int(self.start_time//self.dt):])/1e6
        # ax.plot([self.start_time, self.end_time], [average_power, average_power], 'k', label='Total Mean')

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
        # ax.set_ylim(0, max(total_power[int(self.start_time//self.dt):]/1e6))

        # only use first labels in legend
        handles, labels = ax.get_legend_handles_labels()
        handles = handles[:self.n_turbs+2]
        labels = labels[:self.n_turbs+2]
        ax.legend(handles, labels, frameon=False, ncols=5)
        plt.show()

    def draw_yaw(self, it, ax):
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22',
                  '#17becf']
        for turbine in range(self.n_turbs):
                ax.plot(self.data[turbine]['Time'][:it], self.data[turbine]['YawAng'][:it],
                    color=colors[turbine],
                    alpha=0.05,
                    label=f'_Turbine {turbine + 1}')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel(r'$\text{Yaw}_i \; (^{\circ})$')
        ax.set_xlim(self.data[0]['Time'].iloc[0], self.data[0]['Time'].iloc[-1])
        ax.set_ylim(-45, 45)
        ax.set_xlim(self.start_time, self.end_time)

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
        ax.set_ylabel(r'$f \, \text{PSD}(P_i) \; (MW^2)$')
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
        ax.set_ylabel(r'$f \, \text{PSD}(\theta) \; (^{\circ 2})$')
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


class EnsembleAverage:
    def __init__(self, time_series_instances):
        self.instances = time_series_instances

        self.ensemble_mean = []
        self.ensemble_psd_mean = []
        self.ensemble_err = []
        self.ensemble_psd_err = []
        for turbine_idx in range(self.instances[0].n_turbs):
            all_dfs = [instance.data[turbine_idx] for instance in self.instances]
            self.ensemble_mean.append(pd.concat(all_dfs).groupby(level=0).mean())
            self.ensemble_err.append(pd.concat(all_dfs).groupby(level=0).sem())
            all_dfs = [instance.freq_data[turbine_idx] for instance in self.instances]
            self.ensemble_psd_mean.append(pd.concat(all_dfs).groupby(level=0).mean())
            self.ensemble_psd_err.append(pd.concat(all_dfs).groupby(level=0).sem())

    def plot_time(self, variable: str, ax: plt.Axes, sum: bool = False, scale=1.0):
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22',
                  '#17becf']
        total = 0
        for turbine_idx in range(len(self.ensemble_mean)):
            mean = self.ensemble_mean[turbine_idx]
            err = self.ensemble_err[turbine_idx]
            color = colors[turbine_idx % len(colors)]
            ax.plot(mean['Time'], mean[variable]/scale, color=color, label=f'Turbine {turbine_idx + 1}')
            ax.fill_between(mean['Time'],
                            (mean[variable] - err[variable])/scale,
                            (mean[variable] + err[variable])/scale,
                            color=color, alpha=0.3)
            total += mean[variable]
        if sum:
            ax.plot(mean['Time'], total/scale, color='k', label=f'Total')
            ax.set_ylim(0, max(total[int(self.instances[0].start_time//self.instances[0].dt):]/scale))

    def draw_power(self, it, ax):
        self.plot_time('Power_ave', ax, sum=True, scale=1e6)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Power (MW)')
        ax.set_xlim(self.instances[0].start_time, self.instances[0].end_time)
        # ax.set_ylim(0, max(total_power[int(self.instances[0].start_time//self.instances[0].dt):]/1e6))
        ax.legend(frameon=False, ncols=5)

    def draw_yaw(self, it, ax):
        self.plot_time('YawAng', ax)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel(r'$\text{Yaw}_i \; (^{\circ})$')
        ax.set_ylim(-45, 45)
        ax.set_xlim(self.instances[0].start_time, self.instances[0].end_time)
        ax.legend(frameon=False, ncols=5)

    def plot_freq(self, variable: str, ax: plt.Axes):
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22',
                  '#17becf']
        for turbine_idx in range(len(self.ensemble_psd_mean)):
            mean = self.ensemble_psd_mean[turbine_idx]
            err = self.ensemble_psd_err[turbine_idx]
            color = colors[turbine_idx % len(colors)]
            ax.semilogx(mean['Frequency'], mean[variable] * mean['Frequency'], color=color, label=f'Turbine {turbine_idx+1}')
            ax.fill_between(mean['Frequency'],
                            (mean[variable] - err[variable]) * mean['Frequency'],
                            (mean[variable] + err[variable]) * mean['Frequency'],
                            color=color, alpha=0.3)
        self.instances[0].add_frequecies(ax, max(mean[variable] * mean['Frequency']))

    def draw_power_psd(self, it, ax):
        self.plot_freq('Power_PSD', ax)
        ax.set_xlabel(r'$f \; (Hz)$')
        ax.set_ylabel(r'$f \, \text{PSD}(P_i) \; (MW^2)$')
        ax.legend(frameon=False, ncols=1, loc='upper left')

    def draw_yaw_psd(self, it, ax):
        self.plot_freq('YawAng_PSD', ax)
        ax.set_xlabel(r'$f \; (Hz)$')
        ax.set_ylabel(r'$f \, \text{PSD}(\theta) \; (^{\circ 2})$')
        ax.legend(frameon=False, ncols=1, loc='upper left')


# @hydra.main(config_path="../ppo/", config_name="config_ppo", version_base="1.2")
# def main(cfg: "DictConfig"):
def main():

    fig, axs = plt.subplots(2, 2,
                            figsize=(12, 6),
                            constrained_layout=True,
                            gridspec_kw={'width_ratios': [2, 1]},
                            sharex='col')

    casename = sys.argv[1]

    if 'eval' in casename:
        mode = 'eval'
    elif 'ppo' in casename:
        mode = 'ppo'
    elif 'sac' in casename:
        mode = 'sac'
    else:
        print('Unknown Method')

    with open(os.path.join(casename, f"{mode}/outputs/hydra_logs/config.yaml"), "r") as file:
        config = yaml.safe_load(file)
    print(casename)

    # plot time series and psd for each environment and
    # Collect time series data from multiple environments
    time_series_instances = []
    if mode == 'eval':
        n_environments = config['eval']['n_parallel']
    else:
        n_environments = config['env']['n_parallel']

    for env in range(n_environments):
        # for ax in axs.flatten():
        #     ax.cla()
        environment_name = os.path.join(casename, f'WindFarm_{env+1}')
        print(environment_name)
        series = TimeSeries(environment_name, config, mode=mode)
        series.collect_data()
        series.calculate_psds()
        # series.draw_yaw_psd(-1, axs[0, 1])
        series.draw_yaw(-1, axs[0, 0])
        # series.draw_power_psd(-1, axs[1, 1])
        # series.draw_power(-1, axs[1, 0])
        # fig.savefig(os.path.join(environment_name, 'time_series.pdf'))
        time_series_instances.append(series)

    # Ensemble averaging
    ensemble = EnsembleAverage(time_series_instances)

    # Plot ensemble results
    # fig, axs = plt.subplots(2, 2,
    #                         figsize=(14, 8),
    #                         constrained_layout=True,
    #                         gridspec_kw={'width_ratios': [2, 1]},
    #                         sharex = 'col')
    # ensemble.plot_ensemble_results(axs[1], axs[0])
    ensemble.draw_yaw_psd(-1, axs[0, 1])
    ensemble.draw_yaw(-1, axs[0, 0])
    ensemble.draw_power_psd(-1, axs[1, 1])
    fig.savefig(os.path.join(casename, 'ensemble_time_series.pdf'))
    ensemble.draw_power(-1, axs[1, 0])

    plt.show()


if __name__ == "__main__":
    main()
