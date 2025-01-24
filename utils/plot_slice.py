import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib import cm, ticker
import matplotlib.animation as ani
from functools import partial
import os
import tqdm
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
    ax.set_xlabel(r'$x \; (km)$')
    ax.set_ylabel(r'$y \; (km)$')


# Add turbine locations from plotting.py
def plot_turbine(ax, xc, zc, diam=126.0, yaw=0):
    x = [xc-diam/2*np.sin(np.radians(yaw)),
         xc+diam/2*np.sin(np.radians(yaw))]
    z = [zc-diam/2*np.cos(np.radians(yaw)),
         zc+diam/2*np.cos(np.radians(yaw))]
    ax.plot(np.array(x)/1000., np.array(z)/1000., c='w')
    ax.scatter(xc/1000., zc/1000., c='w', edgecolor='none')
    # plot_probes(ax, xc, zc, 60)


def plot_probes(ax, xt, zt, nprobes, diam=126.0):
    probe_spacing = 1.0/2
    nrows = nprobes // 5
    x, z = np.meshgrid(
            np.linspace(-(nrows-1)/2*probe_spacing, (nrows-1)/2*probe_spacing, nrows),
            np.linspace(-2*probe_spacing, 2*probe_spacing, 5))
    x *= diam
    z *= diam
    x += xt
    z += zt

    ax.scatter(x/1000., z/1000., color='w', marker='x', s=10, linewidths=1)

def plot_all(t, casename, data, ax):
    # for a in ax:
    ax.clear()
    plot_slice(t, casename=casename, ax=ax)
    for turbine, x in enumerate([252, 882, 1512]):
        plot_turbine(ax, x, 441, yaw=data[turbine]['YawAng'][t])

    ax.set_xlim(left=0.)


def main():
    fig, axs = plt.subplots(1, 1,
                            figsize=(7, 3),
                            constrained_layout=True,)

    directory = './sa_sac_array/visualisation/training_sasac_array_5_eval_2025-01-23_11-48/WindFarm_1'

    data = []
    for turbine in range(3):
        data_file = os.path.join(directory, f'disc{turbine + 1}.adm')
        dat = pd.read_csv(data_file, sep="\s+|, ", engine='python')
        # dat['NonDimTime'] = dat['Time'] * self.U / self.D
        data.append(dat)

    iteration = 6000
    plot_all(iteration, casename=directory, data=data, ax=axs)
    fig.savefig('flow_probes.png', dpi=400)

    iters = tqdm.tqdm(range(5000, 6600), desc="Iteration", position=0)
    anim = ani.FuncAnimation(fig, partial(plot_all, casename=directory, data=data, ax=axs), frames=iters)
    anim.save(f'animations/control_gamma0.99_blue.mp4', fps=10, dpi=400)  # codec='h263p')


if __name__ == '__main__':
    main()
