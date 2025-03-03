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


def collect_data(casename):
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

        # Select only the first half of y 
        nx, ny, nz = 193, 41, 72
        ny_new = ny // 2
        indices = []
        for k in range(nz):  # Iterate over z
            for j in range(ny_new):  # Only the first half of y
                for i in range(nx):  # Iterate over x
                    idx = i + j * nx + k * (nx * ny)  # Fortran-style index calculation
                    indices.append(idx)
        indices = np.array(indices)

        u = np.fromfile(fileu, dtype=np.float64)[indices]
        v = np.fromfile(filev, dtype=np.float64)[indices]
        w = np.fromfile(filew, dtype=np.float64)[indices]

        # ur = u.reshape((nx, ny, nz), order='F')

        u_all_timesteps.append(u)
        v_all_timesteps.append(v)
        w_all_timesteps.append(w)

    # Stack to create a combined data matrix
    u_matrix = np.column_stack(u_all_timesteps)
    v_matrix = np.column_stack(v_all_timesteps)
    w_matrix = np.column_stack(w_all_timesteps)
    data_matrix = np.vstack((u_matrix, v_matrix, w_matrix))

    return data_matrix
    # return u_matrix


# Add turbine locations from plotting.py
def plot_turbine(ax, xc, zc, diam=126.0, yaw=0):
    x = [xc-diam/2*np.sin(np.radians(yaw)),
         xc+diam/2*np.sin(np.radians(yaw))]
    z = [zc-diam/2*np.cos(np.radians(yaw)),
         zc+diam/2*np.cos(np.radians(yaw))]
    ax.plot(x, z, c='k')
    ax.scatter(xc, zc, c='k', edgecolor='none')


def main():

    if len(sys.argv) != 2:
        print("Usage: python pod_plot.py <path/to/windfarm/directory>")
        sys.exit(1)

    casename = sys.argv[1]
    # data = collect_data('./POD_zero_2')
    # data = collect_data('./sa_sac_array_2/evaluation/saved/training_sasac_array_8_eval_2025-02-26_15-02/WindFarm_1')
    data = collect_data(casename)
    print(np.shape(data))


    slice_height = 90
    lx, ly, lz = 2394.0, 500.0, 882.0
    nx, ny, nz = 193, 41, 72
    ny = ny//2
    ly = ly/2
    num_grid_points = nx * ny * nz
    gridx = np.linspace(0, lx, nx)
    gridy = np.linspace(0, ly, ny)
    gridz = np.linspace(0, lz, nz)

    # Extract u, v, w components of the mean and reshape to original grid dimensions
    nt = np.shape(data)[-1]
    u = data[:num_grid_points, :].reshape(nx, ny, nz, nt, order='F')  # Reshape to (Nx, Ny, Nz)
    v = data[num_grid_points:2 * num_grid_points, :].reshape(nx, ny, nz, nt, order='F')
    w = data[2 * num_grid_points:, :].reshape(nx, ny, nz, nt, order='F')

    slice_loc = int(slice_height / ly * ny)

    u = u[:, slice_loc, :, :]
    v = v[:, slice_loc, :, :]
    w = w[:, slice_loc, :, :]

    # mean = np.mean(data, axis=-1)[:, np.newaxis]
    u_mean = np.mean(u, axis=-1)[:, :, np.newaxis]
    v_mean = np.mean(v, axis=-1)[:, :, np.newaxis]
    w_mean = np.mean(w, axis=-1)[:, :, np.newaxis]

    u_prime = u - u_mean
    v_prime = v - v_mean
    w_prime = w - w_mean

    uu = np.mean(u_prime * u_prime, axis=-1)
    vv = np.mean(v_prime * v_prime, axis=-1)
    ww = np.mean(w_prime * w_prime, axis=-1)
    uv = np.mean(u_prime * v_prime, axis=-1)
    uw = np.mean(u_prime * w_prime, axis=-1)
    vw = np.mean(v_prime * w_prime, axis=-1)



    fig, axs = plt.subplots(3, 3,
                            figsize=(6, 4),
                            constrained_layout=True,)
    axs = axs.ravel()

    cmaps = ['Blues_r', 'RdBu', 'RdBu', 'plasma', 'plasma', 'plasma', 'plasma', 'plasma', 'plasma']
    labels = [r'$\overline{u}$', r'$\overline{v}$', r'$\overline{w}$',
              r'$\overline{uu}$', r'$\overline{vv}$', r'$\overline{ww}$',
              r'$\overline{uv}$', r'$\overline{uw}$', r'$\overline{vw}$']
    for dir, dat in enumerate([u_mean, v_mean, w_mean, uu, vv, ww, uv, uw, vw]):
        # axs[mean].contourf(u_mean[:, slice_loc, :], cmap='viridis')
        data_min = min(dat.min(), -dat.max()) if (dir==1 or dir==2) else dat.min() 
        data_max = max(-dat.min(), dat.max())
        contour = axs[dir].contourf(gridx / 1000, gridz / 1000, dat.squeeze().T,
                               cmap=cmaps[dir], levels=100,
                               vmin=data_min, vmax=data_max)
        # axs[dir].colorbar()
        cbar = fig.colorbar(contour, location='top')
        cbar.locator = ticker.MaxNLocator(nbins=5)
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
    fig.savefig('flow_stats_zero.png', dpi=400)


if __name__ == '__main__':
    main()

