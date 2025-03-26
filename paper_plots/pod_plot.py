from functools import partial
import numpy as np
import pandas as pd
# import torch
from matplotlib import pyplot as plt
import matplotlib.animation as ani
import matplotlib.colors as mcolors
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

    iters = tqdm.tqdm(range(5401, 5900), desc="Time Step", position=0)
    for t in iters:
        fileu = os.path.join(path, f'data/ux-{t}.bin')
        filev = os.path.join(path, f'data/uy-{t}.bin')
        filew = os.path.join(path, f'data/uz-{t}.bin')
        # filep = os.path.join(path, f'data/pp-{t}.bin')

        # Select only the first half of y 
        nx, ny, nz = 193, 41, 72
        # ny_new = ny // 2
        ny_new = ny
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
    data -= np.mean(data, axis=1)[:, np.newaxis]
    U, S, Vt = np.linalg.svd(data, full_matrices=False)

    # Check the shapes of the SVD components
    print("Shape of U (spatial modes):", U.shape)
    print("Shape of S (singular values):", S.shape)
    print("Shape of Vt (temporal modes):", Vt.shape)


    dt = 10
    Nt = Vt.shape[1]

    # Frequency range for each mode
    frequencies = fftfreq(Nt, dt)

    # Apply FFT to each mode's time series
    mode_frequencies = []
    for mode_time_series in Vt:
        # Compute the FFT of the temporal coefficient (time series) of each mode
        mode_fft = fft(mode_time_series)
        # Get the power spectral density (PSD) or magnitude squared of the FFT
        mode_power = np.abs(mode_fft)**2
        # Append frequency and power pairs for analysis
        mode_frequencies.append((frequencies, mode_power))


    # Find the dominant frequency for each mode
    dominant_frequencies = []
    for frequencies, power in mode_frequencies:
        # Consider only the positive half of the spectrum (since FFT gives symmetric result)
        pos_mask = frequencies > 0
        frequencies = frequencies[pos_mask]
        power = power[pos_mask]

        # Find the frequency with maximum power
        max_power_index = np.argmax(power)
        dominant_frequency = frequencies[max_power_index]
        dominant_frequencies.append(dominant_frequency)




    num_modes_to_plot = 10
    starting_mode = 0
    slice_height = 90
    lx, ly, lz = 2394.0, 500.0, 882.0
    nx, ny, nz = 193, 41, 72
    # ny = ny//2
    # ly = ly/2
    num_grid_points = nx * ny * nz
    gridx = np.linspace(0, lx, nx)
    gridy = np.linspace(0, ly, ny)
    gridz = np.linspace(0, lz, nz)

    # reshape
    u = U.reshape(nx, ny, nz, 3, -1, order='F')  # Reshape to (Nx, Ny, Nz)

    # PLOTTING SLICES AT HUB HEIGHT
    fig, axs = plt.subplots(num_modes_to_plot, 4,
                            figsize=(8, 0.85*num_modes_to_plot),
                            gridspec_kw={'width_ratios': [2, 2, 2, 1]},
                            constrained_layout=True,)

    slice_loc = int(slice_height / ly * ny)
    min_value = np.min(u[:, slice_loc, :, :, :])
    max_value = np.max(u[:, slice_loc, :, :, :])

    for mode in range(num_modes_to_plot):
        for dir in [0, 1, 2]:
            contour = axs[mode, dir].contourf(gridx / 1000, gridz / 1000, u[:, slice_loc, :, dir, mode].T,
                                   cmap='RdBu', levels=100,
                                   norm=mcolors.TwoSlopeNorm(vmin=min_value, vcenter=0, vmax=max_value))
            # axs[mode].colorbar()
            # cbar = fig.colorbar(contour)
            # cbar.ax.set_ylabel('pod value')
            axs[-1, dir].set_xlabel(r'$x \; (km)$')
            axs[mode, dir].set_aspect('equal')

            plot_turbine(axs[mode, dir], 2*126/1000, 3.5*126/1000, diam=126.0/1000, yaw=0)
            plot_turbine(axs[mode, dir], 7*126/1000, 3.5*126/1000, diam=126.0/1000, yaw=0)
            plot_turbine(axs[mode, dir], 12*126/1000, 3.5*126/1000, diam=126.0/1000, yaw=0)

            axs[mode, dir].label_outer()


        axs[mode, 0].set_ylabel(r'$z \; (km)$')

        frequency = mode_frequencies[mode+starting_mode][0]
        power = mode_frequencies[mode+starting_mode][1]
        axs[mode, 3].semilogx(frequency[frequency>0], power[frequency>0]*frequency[frequency>0], c='k')
        axs[mode, 3].set_ylim(0, 0.25)
        axs[mode, 3].set_ylabel(r'$f$ Power')
        if mode != num_modes_to_plot-1:
            axs[mode, 3].set_xticklabels([])
        axs[mode, 3].set_yticks([0.0, 0.1, 0.2])
        axs[-1, 3].set_xlabel(r'$f \; (Hz)$')

        axs[mode, 0].annotate(f'Mode {mode+starting_mode+1}', xy=(0, 0.5), xytext=(-axs[mode, 0].yaxis.labelpad - 5, 0),
                              xycoords=axs[mode, 0].yaxis.label, textcoords='offset points',
                              size='large', ha='right', va='center', rotation=90)


    fig.subplots_adjust(hspace=0, wspace=0)
    # plt.show()
    fig.savefig('pod_y.png', dpi=400)



    # PLOTTING SLICES AT CENTERLINE
    fig, axs = plt.subplots(num_modes_to_plot, 4,
                            figsize=(8, 0.85*num_modes_to_plot),
                            gridspec_kw={'width_ratios': [2, 2, 2, 1]},
                            constrained_layout=True,)

    slice_loc = int(nz//2)
    min_value = np.min(u[:, :, slice_loc, :, :])
    max_value = np.max(u[:, :, slice_loc, :, :])

    for mode in range(num_modes_to_plot):
        for dir in [0, 1, 2]:
            contour = axs[mode, dir].contourf(gridx / 1000, gridy / 1000, u[:, :, slice_loc, dir, mode].T,
                                   cmap='RdBu', levels=100,
                                   norm=mcolors.TwoSlopeNorm(vmin=min_value, vcenter=0, vmax=max_value))
            # axs[mode].colorbar()
            # cbar = fig.colorbar(contour)
            # cbar.ax.set_ylabel('pod value')
            axs[-1, dir].set_xlabel(r'$x \; (km)$')
            axs[mode, dir].set_aspect('equal')

            plot_turbine(axs[mode, dir], 2*126/1000, 90/1000, diam=126.0/1000, yaw=0)
            plot_turbine(axs[mode, dir], 7*126/1000, 90/1000, diam=126.0/1000, yaw=0)
            plot_turbine(axs[mode, dir], 12*126/1000, 90/1000, diam=126.0/1000, yaw=0)

            axs[mode, dir].label_outer()


        axs[mode, 0].set_ylabel(r'$y \; (km)$')

        frequency = mode_frequencies[mode+starting_mode][0]
        power = mode_frequencies[mode+starting_mode][1]
        axs[mode, 3].semilogx(frequency[frequency>0], power[frequency>0]*frequency[frequency>0], c='k')
        axs[mode, 3].set_ylim(0, 0.25)
        axs[mode, 3].set_ylabel(r'$f$ Power')
        if mode != num_modes_to_plot-1:
            axs[mode, 3].set_xticklabels([])
        axs[mode, 3].set_yticks([0.0, 0.1, 0.2])
        axs[-1, 3].set_xlabel(r'$f \; (Hz)$')

        axs[mode, 0].annotate(f'Mode {mode+starting_mode+1}', xy=(0, 0.5), xytext=(-axs[mode, 0].yaxis.labelpad - 5, 0),
                              xycoords=axs[mode, 0].yaxis.label, textcoords='offset points',
                              size='large', ha='right', va='center', rotation=90)


    fig.subplots_adjust(hspace=0, wspace=0)
    # plt.show()
    fig.savefig('pod_z.png', dpi=400)


    # PLOTTING MODE ENERGIES
    mode_energy = S**2
    total_energy = np.sum(mode_energy[:])
    energy_percent = (mode_energy / total_energy) * 100
    fig, ax = plt.subplots(figsize=(6, 2.75))
    ax.semilogy(energy_percent[:-1], 'ko-')
    ax.set_xlabel('Mode Number')
    ax.set_ylabel('Energy (\%)')
    # ax.set_ylim(np.min(energy_percent[:-1])/2, 20)

    # from mpl_toolkits.axes_grid1.inset_locator import inset_axes
    x1, x2, y1, y2 = 0, 20, energy_percent[22], 10  # subregion of the original image
    mini_ax = ax.inset_axes([0.5, 0.35, 0.47, 0.6],
        xlim=(x1, x2), ylim=(y1, y2), xticklabels=[0, 5, 10, 15, 20], yticklabels=[])
    mini_ax.semilogy(energy_percent[:], 'ko-')
    ax.indicate_inset_zoom(mini_ax, edgecolor="black", alpha=1)


    cumulative_energy = np.cumsum(energy_percent[:])
    plt.figure(figsize=(10, 5))
    plt.plot(cumulative_energy, 'ko-')
    # plt.axhline(y=90, color='r', linestyle='--', label='90\% Energy Threshold')
    plt.xlabel('Mode Number')
    plt.ylabel('Cumulative Energy (\%)')
    # plt.ylim(0, 100)
    plt.legend()

    fig.savefig('pod_energy.png', dpi=400)
    fig.savefig('pod_energy.pdf')

if __name__ == '__main__':
    main()
