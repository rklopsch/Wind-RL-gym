from functools import partial
import numpy as np
import pandas as pd
# import torch
from matplotlib import pyplot as plt
import matplotlib.animation as ani
import matplotlib.colors as mcolors
import matplotlib.ticker as ticker
from scipy.fft import fft, fftfreq
import pyvista as pv
import os
import sys
import tqdm
from utils import fonts


def plot_slice(t, casename, ax, height=90):
    for a in ax:
        a.clear()
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
    contour_u = ax[0].contourf(gridx/1000, gridz/1000, ur[:, slice_loc, :].T,
                               cmap='plasma', levels=100, extend='both', vmin=4, vmax=12)
    contour_p = ax[1].contourf(gridx/1000, gridz/1000, pr[:, slice_loc, :].T,
                               cmap='Blues_r', levels=100, vmin=-10, vmax=10)
    for a in ax:
        a.set_aspect('equal')
        a.set_xlabel(r'$x \; (km)$')
        a.set_ylabel(r'$y \; (km)$')


def collect_data(casename, slice_height=90):
    path = casename

    u_all_timesteps = []
    v_all_timesteps = []
    w_all_timesteps = []

    iters = tqdm.tqdm(range(5401, 7400), desc="Time Step", position=0)
    for t in iters:
        fileu = os.path.join(path, f'data/ux-{t}.bin')
        filev = os.path.join(path, f'data/uy-{t}.bin')
        filew = os.path.join(path, f'data/uz-{t}.bin')
        # filep = os.path.join(path, f'data/pp-{t}.bin')

        lx, ly, lz = 2394.0, 500.0, 882.0
        nx, ny, nz = 193, 41, 72

        u = np.fromfile(fileu, dtype=np.float64)
        v = np.fromfile(filev, dtype=np.float64)
        w = np.fromfile(filew, dtype=np.float64)

        u = u.reshape(nx, ny, nz, order='F')
        v = v.reshape(nx, ny, nz, order='F')
        w = w.reshape(nx, ny, nz, order='F')

        slice_loc = int(slice_height / ly * ny)
        u = u[:, slice_loc, :]
        v = v[:, slice_loc, :]
        w = w[:, slice_loc, :]

        u_all_timesteps.append(u)
        v_all_timesteps.append(v)
        w_all_timesteps.append(w)

    # Stack to create a combined data matrix
    data_matrix = np.stack((u_all_timesteps, v_all_timesteps, w_all_timesteps))

    return data_matrix
    # return u_matrix


def calculate_stats(data):
    # mean = np.mean(data, axis=-1)[:, np.newaxis]
    data_mean = np.mean(data, axis=1)[:, np.newaxis, :, :]
    data_prime = data - data_mean

    # R = np.tensordot(data, data, axes=(0, 0))

    uu = np.mean(data_prime[0] * data_prime[0], axis=0)
    vv = np.mean(data_prime[1] * data_prime[1], axis=0)
    ww = np.mean(data_prime[2] * data_prime[2], axis=0)
    uv = np.mean(data_prime[0] * data_prime[1], axis=0)
    uw = np.mean(data_prime[0] * data_prime[2], axis=0)
    vw = np.mean(data_prime[1] * data_prime[2], axis=0)

    data_prime_2_mean = np.stack((uu, vv, ww, uv, uw, vw))

    return data_mean, data_prime_2_mean


# Add turbine locations from plotting.py
def plot_turbine(ax, xc, zc, diam=126.0, yaw=0):
    x = [xc-diam/2*np.sin(np.radians(yaw)),
         xc+diam/2*np.sin(np.radians(yaw))]
    z = [zc-diam/2*np.cos(np.radians(yaw)),
         zc+diam/2*np.cos(np.radians(yaw))]
    ax.plot(x, z, c='k')
    ax.scatter(xc, zc, c='k', edgecolor='none')


def create_figure(mean, p2m, u_base=7.5, figure_name='flow_stats', diff=False):

    lx, ly, lz = 2394.0, 500.0, 882.0
    nx, ny, nz = 193, 41, 72
    num_grid_points = nx * ny * nz

    fig, axs = plt.subplots(3, 3,
                            figsize=(6, 4),
                            constrained_layout=True,)
    axs = axs.ravel()

    gridx = np.linspace(0, lx, nx)
    gridy = np.linspace(0, ly, ny)
    gridz = np.linspace(0, lz, nz)

    labels = [r"$\overline{u}/U_0$", r"$\overline{v}/U_0$", r"$\overline{w}/U_0$",
              r"$\overline{u'u'}/U_0^2$", r"$\overline{v'v'}/U_0^2$", r"$\overline{w'w'}/U_0^2$",
              r"$\overline{u'v'}/U_0^2$", r"$\overline{u'w'}/U_0^2$", r"$\overline{v'w'}/U_0^2$"]
    if not diff:
        cmaps = ['Blues_r', 'RdBu', 'RdBu', 'plasma', 'plasma', 'plasma', 'plasma', 'plasma', 'plasma']
    else:
        cmaps = ['RdBu']*len(labels)
    for dir, dat in enumerate([mean[0]/u_base, mean[1]/u_base, mean[2]/u_base,
                               p2m[0]/u_base**2, p2m[1]/u_base**2, p2m[2]/u_base**2,
                               p2m[3]/u_base**2, p2m[4]/u_base**2, p2m[5]/u_base**2]):
        # axs[mean].contourf(u_mean[:, slice_loc, :], cmap='viridis')
        data_min = min(dat.min(), -dat.max()) if (dir==1 or dir==2 or diff) else dat.min() 
        data_max = max(-dat.min(), dat.max())
        contour = axs[dir].contourf(gridx / 1000, gridz / 1000, dat.squeeze().T,
                               cmap=cmaps[dir], levels=100,
                               vmin=data_min, vmax=data_max)
        # axs[dir].colorbar()
        cbar = fig.colorbar(contour, location='top')
        cbar.locator = ticker.MaxNLocator(nbins=4)
        cbar.update_ticks()
        cbar.ax.set_xlabel(labels[dir])
        axs[dir].set_xlabel(r'$x \; (km)$')
        axs[dir].set_ylabel(r'$z \; (km)$')
        axs[dir].set_aspect('equal')

        plot_turbine(axs[dir], 2*126/1000, 3.5*126/1000, diam=126.0/1000, yaw=0)
        plot_turbine(axs[dir], 7*126/1000, 3.5*126/1000, diam=126.0/1000, yaw=0)
        plot_turbine(axs[dir], 12*126/1000, 3.5*126/1000, diam=126.0/1000, yaw=0)

        axs[dir].label_outer()


    fig.subplots_adjust(hspace=0, wspace=0)
    fig.savefig(f'{figure_name}.png', dpi=400)


def main():

    # if len(sys.argv) != 3:
    #     print("Usage: python pod_plot.py <path/to/windfarm/directory>")
    #     sys.exit(1)

    casename_zero = sys.argv[1]
    casename_rl = sys.argv[2]
    # data = collect_data('./POD_zero_2')
    # data = collect_data('./sa_sac_array_2/evaluation/saved/training_sasac_array_8_eval_2025-02-26_15-02/WindFarm_1')

    slice_height = 90

    zero_data = collect_data(casename_zero, slice_height=slice_height)
    zero_mean, zero_p2m = calculate_stats(zero_data)

    rl_data = collect_data(casename_rl, slice_height=slice_height)
    rl_mean, rl_p2m = calculate_stats(rl_data)

    diff_mean = rl_mean - zero_mean
    diff_p2m = rl_p2m - zero_p2m

    u_base = np.mean(zero_mean[0, 0, 0, :])
    print(f'{u_base=}')

    create_figure(zero_mean, zero_p2m, u_base, 'zero_stats_slice')
    create_figure(rl_mean, rl_p2m, u_base, 'rl_stats_slice')
    create_figure(diff_mean, diff_p2m, u_base, 'diff_stats_slice', diff=True)


if __name__ == '__main__':
    main()

