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


def plot_slice(t, casename, env, ax, height=90):
    for a in ax:
        a.clear()
    path = casename
    fileu = os.path.join(path, f'data/ux-{t}.bin')
    filep = os.path.join(path, f'data/pp-{t}.bin')
    gridx = np.linspace(0, env.adm.lx, env.adm.nx)
    gridy = np.linspace(0, env.adm.ly, env.adm.ny)
    gridz = np.linspace(0, env.adm.lz, env.adm.nz)

    u = np.fromfile(fileu, dtype=np.float64)
    ur = u.reshape((env.adm.nx, env.adm.ny, env.adm.nz), order='F')
    p = np.fromfile(filep, dtype=np.float64)
    pr = p.reshape((env.adm.nx, env.adm.ny, env.adm.nz), order='F')
    slice_loc = int(height / env.adm.ly * env.adm.ny)
    contour_u = ax[0].contourf(gridx/1000, gridz/1000, ur[:, slice_loc, :].T,
                               cmap='plasma', levels=100, extend='both', vmin=4, vmax=12)
    contour_p = ax[1].contourf(gridx/1000, gridz/1000, pr[:, slice_loc, :].T,
                               cmap='Blues_r', levels=100, vmin=-10, vmax=10)
    for a in ax:
        a.set_aspect('equal')
        a.set_xlabel(r'$x \; (km)$')
        a.set_ylabel(r'$y \; (km)$')


@hydra.main(config_path="../ppo/", config_name="config_ppo", version_base="1.2")
def main(cfg: "DictConfig"):

    casename = '/home/amole/Downloads/TEST_8'

    fig, axs = plt.subplots(2, 1,
                            figsize=(10, 6),
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
    iters = tqdm.tqdm(range(129, 1128), desc="Optimisation Iteration", position=0)

    anim = ani.FuncAnimation(fig, partial(plot_slice, casename=casename, env=env, ax=axs), frames=iters)
    anim.save(f'animations/test2.mp4', fps=20, dpi=400)#codec='h263p')

    # for i in range(1000, 1010):
    #     print(i)
    #     plot_slice(i, casename, env, axs[0], mean=False)


if __name__ == "__main__":
    main()
