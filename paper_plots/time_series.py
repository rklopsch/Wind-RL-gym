import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib import cm, ticker
import matplotlib.animation as ani
from functools import partial
import os
import tqdm
from utils import fonts


def draw_yaw(data, it, ax):
    t_on = 52500
    for turbine in range(3):
            ax.plot(data[turbine]['Time'][START_TIME:it]-t_on, data[turbine]['YawAng'][START_TIME:it],
                label=f'Turbine {turbine + 1}')
    # ax.set_xlabel('Time (s)')
    ax.set_ylabel(r'$\text{Yaw}_i \; (^{\circ})$')
    ax.axvspan(data[turbine]['Time'][START_TIME]-t_on, 0, color='k', alpha=0.1)
    ax.set_ylim(-40, 40)
    ax.set_xlim(data[turbine]['Time'][START_TIME]-t_on, data[turbine]['Time'][END_TIME]-t_on)
    ax.set_xticklabels([])
    ax.grid(alpha=0.3)
    ax.legend(frameon=False, ncols=3, columnspacing=0.8, handlelength=1)


def draw_power(data, it, ax):
    t_on = 52500
    total_power = 0
    for turbine in range(3):
        ax.plot(data[turbine]['Time'][START_TIME:it]-t_on,
                data[turbine]['Power_ave'][START_TIME:it] / 1e6,
                alpha=0.5,
                label=f'_Turbine {turbine + 1}')
        total_power += data[turbine]['Power_ave']
    ax.plot(data[turbine]['Time'][START_TIME:it]-t_on, total_power[START_TIME:it] / 1e6,
            color='k',
            label=f'_Total')
    ax.plot([data[turbine]['Time'][START_TIME]-t_on, min(0, data[turbine]['Time'][it]-t_on)], [3.21, 3.21], 'k--')
    if it*10 > t_on:
        ax.plot([0, data[turbine]['Time'][it]-t_on], [3.35, 3.35], 'k', linestyle='--')

    # average_power = np.mean(total_power[int(start_time//dt):])/1e6
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Power (MW)')
    ax.axvspan(data[turbine]['Time'][START_TIME]-t_on, 0, color='k', alpha=0.1)
    ax.set_xlim(data[turbine]['Time'][START_TIME]-t_on, data[turbine]['Time'][END_TIME]-t_on)
    ax.set_ylim(0, 5.5)
    ax.set_yticks([0, 2, 4,])
    # ax.set_ylim(0, max(total_power[int(start_time//dt):]/1e6))
    ax.grid(alpha=0.3)
    # ax.legend(frameon=False, ncols=3)


def plot_all(t, casename, data, ax, fig):
    for a in ax:
        a.clear()

    draw_yaw(data, t, ax[0])
    draw_power(data, t, ax[1])
    # fig.legend(frameon=False, ncols=3, loc='lower left', columnspacing=0.8,
    #         bbox_to_anchor=(0.09, 0.45), handlelength=1)


START_TIME = 5200
END_TIME = 5700
# START_TIME = 5240
# END_TIME = 5260

def main():
    fig, axs = plt.subplots(2, 1,
                            figsize=(6, 3),
                            constrained_layout=True,)

    # directory = './sa_sac_array_2/evaluation/saved/training_sasac_array_8_eval_2025-02-26_15-02_CONTROLLER/WindFarm_1'
    directory = './sa_sac_array_2/evaluation/long/RL/training_sasac_array_8_eval_2025-02-10_15-46/WindFarm_3'

    data = []
    for turbine in range(3):
        data_file = os.path.join(directory, f'disc{turbine + 1}.adm')
        dat = pd.read_csv(data_file, sep="\s+|, ", engine='python')
        data.append(dat)

    plot_all(END_TIME, casename=directory, data=data, ax=axs, fig=fig)
    fig.savefig('time_series.png', dpi=400)
    fig.savefig('time_series.pdf')


if __name__ == '__main__':
    main()
