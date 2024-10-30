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
from scipy.signal import welch


class TimeSeries:
    def __init__(self, casename, n_turbs):
        self.casename = casename
        self.nturbines = n_turbs
        self.data = []
        self.U = 8
        self.D = 126

    def collect_data(self):
        for turbine in range(self.nturbines):
            file = os.path.join(self.casename, f'disc{turbine+1}.adm')
            dat = pd.read_csv(file, sep="\s+|, ", engine='python')
            dat['NonDimTime'] = dat['Time'] * self.U / self.D
            self.data.append(dat)

    def draw_power(self, it, ax):
        total_power = 0
        for turbine in range(self.nturbines):
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
        ax.set_ylim(0, max(total_power/1e6))
        # only use first labels in legend
        handles, labels = ax.get_legend_handles_labels()
        handles = handles[:4]
        labels = labels[:4]
        ax.legend(handles, labels, frameon=False, ncols=5)

    def draw_power_psd(self, it, ax):
        total_power = 0
        for turbine in range(self.nturbines):
            total_power += self.data[turbine]['Power'][3000:it]
        fs = 0.1
        # Compute the PSD using Welch's method
        freqs, psd = welch(total_power, fs, nperseg=1024)

        for turbine in range(self.nturbines):
            freqs, psd = welch(self.data[turbine]['Power'][3000:it], fs, nperseg=1024)
            ax.semilogx(freqs, psd*freqs, label=f'Turbine {turbine + 1}')

        ax.set_xlabel(r'$f \; (Hz)$')
        ax.set_ylabel(r'$f \, PSD \; (W^2)$')
        ax.legend(frameon=False, ncols=1)

        ax.axvspan(0.15*self.U/self.D, 0.25*self.U/self.D, color='k', alpha=0.1)
        ax.text(0.015 * 1.1, 6e10, 'Wake Meandering', rotation=90, va='top')
        ax.axvline(1/10, color='k', linestyle='--')
        ax.text(1/10 * 1.1, 6e10, 'RL Sampling', rotation=90, va='top')
        ax.axvline(1/10/2, color='k', linestyle='--')
        ax.text(1/10/2 * 1.1, 6e10, 'RL Nyquist', rotation=90, va='top')
        ax.axvline(8 / 2394, color='k', linestyle='--', alpha=0.3)
        ax.text(8 / 2394 * 1.1, 6e10, 'Domain Flow Through', rotation=90, va='top', alpha=0.3)
        ax.axvline(8 / (5*126), color='k', linestyle='--', alpha=0.3)
        ax.text(8 / (5*126) * 1.1, 6e10, 'Turbine Spacing', rotation=90, va='top', alpha=0.3)

    def draw_yaw_psd(self, it, ax):
        fs = 0.1

        for turbine in range(self.nturbines):
            freqs, psd = welch(self.data[turbine]['YawAng'][3000:it], fs, nperseg=1024)
            ax.semilogx(freqs, psd * freqs, label=f'Turbine {turbine + 1}')


        ax.set_xlabel(r'$f \; (Hz)$')
        ax.set_ylabel(r'$f \, PSD$')
        ax.legend(frameon=False, ncols=1)

        ax.axvspan(0.15*self.U/self.D, 0.25*self.U/self.D, color='k', alpha=0.1)
        ax.text(0.015 * 1.1, 50, 'Wake Meandering', rotation=90, va='top')
        ax.axvline(1/10, color='k', linestyle='--')
        ax.text(1/10 * 1.1, 50, 'RL Sampling', rotation=90, va='top')
        ax.axvline(1/10/2, color='k', linestyle='--')
        ax.text(1/10/2 * 1.1, 50, 'RL Nyquist', rotation=90, va='top')
        ax.axvline(8 / 2394, color='k', linestyle='--', alpha=0.3)
        ax.text(8 / 2394 * 1.1, 50, 'Domain Flow Through', rotation=90, va='top', alpha=0.3)
        ax.axvline(8 / (5*126), color='k', linestyle='--', alpha=0.3)
        ax.text(8 / (5*126) * 1.1, 50, 'Turbine Spacing', rotation=90, va='top', alpha=0.3)

    def draw_yaw(self, it, ax):
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22',
                  '#17becf']
        for turbine in range(self.nturbines):
                ax.plot(self.data[turbine]['Time'][:it], self.data[turbine]['YawAng'][:it],
                    color=colors[turbine],
                    alpha=1,
                    label=f'Turbine {turbine + 1}')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel(r'$\text{Yaw}_i (^{\circ})$')
        ax.set_xlim(self.data[0]['Time'].iloc[0], self.data[0]['Time'].iloc[-1])
        ax.set_ylim(-45, 45)

        # only use first labels in legend
        handles, labels = ax.get_legend_handles_labels()
        handles = handles[:self.nturbines]
        labels = labels[:self.nturbines]
        ax.legend(handles, labels, frameon=False, ncols=5)
        # ax.legend(loc='upper right', ncol=len(cases), columnspacing=0.5, frameon=False)


@hydra.main(config_path="../ppo/", config_name="config_ppo", version_base="1.2")
def main(cfg: "DictConfig"):

    fig, axs = plt.subplots(2, 2,
                            figsize=(12, 8),
                            constrained_layout=True,)
                            # sharey=True)

    # for env in range(1):
    for env in [0]:
        casename = f'./eval_ppo_28-10-24/WindFarm_{env+1}'
        print(casename)
        series1 = TimeSeries(casename, 3)
        series1.collect_data()
        series1.draw_yaw_psd(-1, axs[0, 1])
        series1.draw_yaw(-1, axs[0, 0])
        series1.draw_power_psd(-1, axs[1, 1])
        series1.draw_power(4998, axs[1, 0])
        fig.savefig(os.path.join(casename, 'all.pdf'))
    axs[0, 0].set_xlim(30000, 50000)
    axs[1, 0].set_xlim(30000, 50000)
    axs[1, 0].set_ylim(0, 7)
    plt.show()

    fig.savefig(os.path.join(casename, 'time_series.pdf'))


if __name__ == "__main__":
    main()
