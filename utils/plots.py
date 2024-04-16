from functools import partial

import numpy as np
import torch
from matplotlib import pyplot as plt
import matplotlib.animation as ani
from Solver.WF_enviroment import TurbEnv
from Solver.farm import Turbine, Farm
import hydra
import tqdm
# import fonts
import os


def plot_slice(t, casename, env, ax, mean=False):
    # ax.clear()
    path = casename
    if mean:
        file = os.path.join(path, f'statistics/umean.dat{env.adm.total_timesteps}')
    else:
        file = os.path.join(path, f'data/ux-{t}.bin')
    gridx = np.linspace(0, env.adm.lx, env.adm.nx)
    gridy = np.linspace(0, env.adm.ly, env.adm.ny)
    gridz = np.linspace(0, env.adm.lz, env.adm.nz)

    data = np.fromfile(file, dtype=np.float64)
    datar = data.reshape((env.adm.nx, env.adm.ny, env.adm.nz), order='F')
    slice_loc = int(90 / env.adm.ly * env.adm.ny)
    ax.contourf(gridx/1000, gridz/1000, datar[:, slice_loc, :].T, cmap='Blues_r', levels=100)
    ax.set_aspect('equal')
    # adm_file = os.path.join(path, 'adm.ad')
    # yaws = np.loadtxt(adm_file, usecols=3, skiprows=1, unpack=True)
    # env.adm.farm.set_yaw(yaws)
    # env.adm.farm.plot_turbines(ax)
    ax.set_xlabel(r'$x \; (km)$')
    ax.set_ylabel(r'$y \; (km)$')


@hydra.main(config_path="../ppo/", config_name="config_ppo", version_base="1.2")
def main(cfg: "DictConfig"):

    casename = './TEST_DATA/'

    fig, axs = plt.subplots(1, 1,
                            figsize=(10, 3),
                            constrained_layout=True,
                            sharex=True,
                            sharey=True)

    params = {
        "n_turbines": cfg.env.turbines,
        "probes_per_turbine": cfg.env.probes_per_turbine,
        "turbine_diameter": cfg.env.turbine_diameter,
        "turbine_spacing": cfg.env.turbine_spacing,
        "max_yaw_speed": cfg.env.max_yaw_speed,
        "max_yaw_angle": cfg.env.max_yaw_angle,
        "dt": cfg.env.steps_per_frame * 0.2,
        "run_steps": cfg.collector.total_frames * cfg.env.steps_per_frame,
    }
    env = TurbEnv(params, dummy_update=True)

    optimisation_iter = 30
    iters = tqdm.tqdm(range(1000, 1010), desc="Optimisation Iteration", position=0)

    anim = ani.FuncAnimation(fig, partial(plot_slice, casename=casename, env=env, ax=axs, mean=False), frames=iters)
    anim.save(f'animations/test.mp4', fps=10,)#codec='h263p')

    # for i in range(1000, 1010):
    #     print(i)
    #     plot_slice(i, casename, env, axs[0], mean=False)


if __name__ == "__main__":
    main()
