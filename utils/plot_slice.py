import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib import cm, ticker
import matplotlib.animation as ani
from functools import partial
import os
import tqdm
# import fonts_video
import fonts


def plot_slice(t, casename, ax, height=90):
    path = casename
    lx, ly, lz = 2394.0, 500.0, 882.0
    nx, ny, nz = 193, 41, 72
    fileu = os.path.join(path, f'data/ux-{t}.bin')
    filep = os.path.join(path, f'data/pp-{t}.bin')
    gridx = np.linspace(0, lx, nx)
    gridy = np.linspace(0, ly, ny)
    gridz = np.linspace(0, lz, nz)

    u = np.fromfile(fileu, dtype=np.float64)
    ur = u.reshape((nx, ny, nz), order='F')
    p = np.fromfile(filep, dtype=np.float64)
    pr = p.reshape((nx, ny, nz), order='F')
    slice_loc = int(height / ly * ny)
    contour_u = ax.contourf(gridx/1000, gridz/1000, ur[:, slice_loc, :].T,
                            # locator=ticker.LogLocator(),
                            cmap='Blues_r', levels=100,
                            # extend='both',
                            vmin=2, vmax=12)
    ax.set_aspect('equal')
    ax.set_xlabel(r'$x \; (km)$', labelpad=0)
    ax.set_ylabel(r'$y \; (km)$')


# Add turbine locations from plotting.py
def plot_turbine(ax, xc, zc, diam=126.0, yaw=0):
    x = [xc-diam/2*np.sin(np.radians(yaw)),
         xc+diam/2*np.sin(np.radians(yaw))]
    z = [zc-diam/2*np.cos(np.radians(yaw)),
         zc+diam/2*np.cos(np.radians(yaw))]
    ax.plot(np.array(x)/1000., np.array(z)/1000., c='k', linewidth=2)
    ax.plot(np.array(x)/1000., np.array(z)/1000.)
    # ax.scatter(xc/1000., zc/1000., c='w', edgecolor='none')
    ax.scatter(xc/1000., zc/1000., edgecolor='k')
    # plot_probes(ax, xc, zc, 77)


def plot_probes(ax, xt, zt, nprobes, diam=126.0):
    turbine_spacing = 5
    probe_rows = 7
    nrows = nprobes // probe_rows
    x, z = np.meshgrid(
            np.linspace(-2, 3, nrows),
            np.linspace(-1, 1, probe_rows))
    x *= diam
    z *= diam
    x += xt
    z += zt
    print(f'Requested {nprobes} sensors per turbine')
    print(f'Placed {len(x.flatten())} sensors per turbine')

    ax.scatter(x/1000., z/1000., color='w', marker='x', s=10, linewidths=1)


def draw_yaw(data, it, ax):
    t_on = 52500
    for turbine in range(3):
            ax.plot(data[turbine]['Time'][START_TIME:it]-t_on, data[turbine]['YawAng'][START_TIME:it],
                label=f'Turbine {turbine + 1}')
    # ax.set_xlabel('Time (s)')
    ax.set_ylabel(r'$\text{Yaw}_i \; (^{\circ})$')
    ax.axvspan(data[turbine]['Time'][START_TIME]-t_on, 0, color='k', alpha=0.2)
    ax.set_ylim(-40, 40)
    ax.set_xlim(data[turbine]['Time'][START_TIME]-t_on, data[turbine]['Time'][END_TIME]-t_on)
    ax.set_xticklabels([])
    ax.grid(alpha=0.3)


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
    ax.axvspan(data[turbine]['Time'][START_TIME]-t_on, 0, color='k', alpha=0.2)
    ax.set_xlim(data[turbine]['Time'][START_TIME]-t_on, data[turbine]['Time'][END_TIME]-t_on)
    ax.set_ylim(0, 5.5)
    ax.set_yticks([0, 2, 4,])
    # ax.set_ylim(0, max(total_power[int(start_time//dt):]/1e6))
    ax.grid(alpha=0.3)
    # ax.legend(frameon=False, ncols=3)


def plot_all(t, casename, data, ax, fig):
    for a in ax:
        a.clear()
    plot_slice(t, casename=casename, ax=ax[0])
    for turbine, x in enumerate([252, 882, 1512]):
        plot_turbine(ax[0], x, 441, yaw=data[turbine]['YawAng'][t])

    ax[0].set_xlim(left=0.)

    ax[0].text(0.77, 0.85, 'Control', transform = ax[0].transAxes, fontsize=16, color='k')
    if t < 5250:
        ax[0].text(0.92, 0.85, 'Off', transform = ax[0].transAxes, fontsize=16, color='r')
    else:
        ax[0].text(0.92, 0.85, 'On', transform = ax[0].transAxes, fontsize=16, color=(0, 0.8, 0))

    draw_yaw(data, t, ax[1])
    draw_power(data, t, ax[2])
    fig.legend(frameon=False, ncols=3, loc='lower left', fontsize='x-small', columnspacing=0.8,
            bbox_to_anchor=(0.09, 0.45), handlelength=1)


START_TIME = 5200
END_TIME = 5600
# START_TIME = 5240
# END_TIME = 5260

def main():
    fig, axs = plt.subplots(3, 1,
                            figsize=(6, 4.5),
                            gridspec_kw={'height_ratios': [2.5, 1, 1]},
                            constrained_layout=True,)
    axs[0].set_aspect('equal')

    # directory = './sa_sac_array/visualisation/training_sasac_array_5_eval_2025-01-23_11-48/WindFarm_1'
    directory = './sa_sac_array_2/evaluation/saved/training_sasac_array_8_eval_2025-02-26_15-02_CONTROLLER/WindFarm_1'

    data = []
    for turbine in range(3):
        data_file = os.path.join(directory, f'disc{turbine + 1}.adm')
        dat = pd.read_csv(data_file, sep="\s+|, ", engine='python')
        # dat['NonDimTime'] = dat['Time'] * self.U / self.D
        data.append(dat)

    iters = tqdm.tqdm(range(START_TIME, END_TIME), desc="Iteration", position=0)
    anim = ani.FuncAnimation(fig, partial(plot_all, casename=directory, data=data, ax=axs, fig=fig), frames=iters)
    anim.save(f'animations/evaluation_colored.mp4', fps=10, dpi=400)  # codec='h263p')


if __name__ == '__main__':
    main()
