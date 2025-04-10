import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np
from utils import fonts
import re
import pickle
from scipy.signal import welch
import seaborn as sns
import pandas as pd
import scipy


def time_shift(yaws, shift, turbine):
    if turbine == 1:
        if shift > 0:
            yaws_shifted = [yaws[shift:, 0], yaws[:-shift, 1], yaws[:-shift, 2]]
        elif shift == 0:
            yaws_shifted = [yaws[:, 0], yaws[:, 1], yaws[:, 2]]
        else:
            yaws_shifted = [yaws[:shift, 0], yaws[-shift:, 1], yaws[-shift:, 2]]
    elif turbine == 2:
        if shift > 0:
            yaws_shifted = [yaws[:-shift, 0], yaws[shift:, 1], yaws[:-shift, 2]]
        elif shift == 0:
            yaws_shifted = [yaws[:, 0], yaws[:, 1], yaws[:, 2]]
        else:
            yaws_shifted = [yaws[-shift:, 0], yaws[:shift, 1], yaws[-shift:, 2]]
    return yaws_shifted


def find_num_eps_envs(data):
    # detect the number of envs and episodes
    num_episodes = 0
    num_envs = 0
    for key in data.keys():
        match = re.search(r"EPISODE_(.+)", key)
        if match:
            episode_number = int(match.group(1))
            num_episodes = max(num_episodes, episode_number)
        match = re.search(r"_ENV_(.+?)_EPISODE_", key)
        if match:
            env_number = int(match.group(1))
            num_envs = max(num_envs, env_number)
    return num_episodes, num_envs


def extract_angles(data, shift=0, turbine=1):
    angles = []
    n_eps, n_envs = find_num_eps_envs(data)
    for ep in range(1, n_eps + 1):
        for ev in range(1, n_envs + 1):
            # for ep in range(1, 2):
            # for ev in range(1, 4):
            yaws = data[f"alphas_ENV_{ev}_EPISODE_{ep}"]
            yaws_shifted = time_shift(yaws, shift, turbine)
            # yaws_shifted = yaws
            angles.append(yaws_shifted)
    angles = np.asarray(angles).transpose(0, 2, 1)
    reshaped_arr = angles.reshape(-1, 3)
    df = pd.DataFrame(reshaped_arr, columns=['Angle 1', 'Angle 2', 'Angle 3'])
    return df


def extract_flow(data, shift=0, turbine=1, sensor=25):
    observations = []
    n_eps, n_envs = find_num_eps_envs(data)
    for ep in range(1, n_eps + 1):
        for ev in range(1, n_envs + 1):
            obs = data[f"observations_ENV_{ev}_EPISODE_{ep}"]
            observations.append(obs)
    observations = np.asarray(observations).transpose(0, 2, 1)
    observation_range = sensor + 77*(turbine)
    observations = observations[:, observation_range, :] * 6. + 6.
    observations = observations.reshape(-1)
    return observations


def extract_barycenter(data, shift=0, turbine=0):
    observations = []
    n_eps, n_envs = find_num_eps_envs(data)
    for ep in range(1, n_eps + 1):
        for ev in range(1, n_envs + 1):
            obs = data[f"observations_ENV_{ev}_EPISODE_{ep}"]
            observations.append(obs)
    observations = np.asarray(observations).transpose(0, 2, 1)
    print(f'{observations.shape=}')
    # observation_range = range(4+77*(turbine-1), 77+77*(turbine-1), 11)
    indices = np.concatenate([np.arange(1, 3),
                              np.arange(12, 14),
                              np.arange(23, 25),
                              np.arange(34, 36),
                              np.arange(45, 47),
                              np.arange(56, 58),
                              np.arange(67, 69),
                              ]) + 77*(turbine)
    # indices = np.concatenate([[0], [11], [22], [33], [44], [55], [66]]) + 77*(turbine)
    print(f'{indices=}')
    observations = observations[:, indices, :] * 6. + 6.
    print(f'{observations.shape=}')
    observations = np.asarray(observations).transpose(0, 2, 1)
    observations = observations.reshape(-1, 7*2)
    # observations = observations.reshape(-1, 7*1)
    # probe_locations = np.repeat(np.linspace(-1, 1, 7)[1:-1], 2)
    probe_locations = np.repeat(np.linspace(-1, 1, 7), 2)
    # probe_locations = np.repeat(np.linspace(-1, 1, 7), 1)

    print(f'{probe_locations=}')
    print(f'{probe_locations.shape=}')
    print(f'{observations.shape=}')

    weighted_sums = np.sum(observations * probe_locations, axis=1)
    total_weights = np.sum(observations, axis=1)

    barycenters = weighted_sums / total_weights
    return barycenters


if __name__ == '__main__':

    # Load data
    # with open(f'./long_evals_final/RL/eval_logs.pkl', 'rb') as f:
    with open(f'./final/evaluation/RL/eval_logs.pkl', 'rb') as f:
        rl_data = pickle.load(f)

    barycenters = extract_barycenter(rl_data, turbine=0)

    fig, axs = plt.subplots(1, 1, figsize=(8, 3),
                            constrained_layout=True, )

    axs.plot(barycenters)
    fig.savefig('barycenters.pdf')
    mm = 1/25.4  # millimeters in inches
    fig, axes = plt.subplots(1, 3, figsize=(183*mm, 55*mm),
                             constrained_layout=True, )
    palette = sns.color_palette()

    angles = extract_angles(rl_data)
    x = np.arange(-40, 41, 1)
    for turb in range(3):
        barycenters = extract_barycenter(rl_data, turbine=turb)
        print(f'Plotting KDE for turbine {turb+1} barycenter')
        sns.kdeplot(x=barycenters, y=angles[f"Angle {turb+1}"],
                    ax=axes[turb], fill=True, cmap='Greys', levels=100)
        # sns.histplot(x=angles[f"Angle {turb+1}"], y=barycenters,
        #              ax=axes[turb], cmap='Greys', bins=100, pthresh=0.05, cbar=False)
        axes[turb].set_ylim(-40, 40)
        # axes[turb].set_xlabel(fr'Barycenter {turb+1}, $\left<z\right>_u(x_{turb+1})/D$')
        axes[turb].set_xlabel(f'Barycenter {turb+1}')
        axes[turb].set_ylabel(fr'Angle {turb+1} $(^\circ)$')
        # axes[turb].set_ylabel(fr'Angle {turb+1}, $\alpha_{turb+1}$')
        # axes[turb].grid(alpha=0.3, linestyle=':')
        axes[turb].set_yticks([-40, -20, 0, 20, 40])
        # axes[turb].set_xlim(-0.2, 0.2)
    axes[0].set_xlim(-0.1, 0.1)
    axes[0].set_aspect(1./400)
    axes[1].set_xlim(-0.2, 0.2)
    axes[1].set_aspect(1./200)
    axes[2].set_xlim(-0.2, 0.2)
    axes[2].set_aspect(1./200)


    fig.savefig("barycenter_correlation.png")
    fig.savefig("barycenter_correlation.pdf")
    plt.close()


    fig, axes = plt.subplots(1, 3, figsize=(6, 2),
                             constrained_layout=True, )
    palette = sns.color_palette()

    angles = extract_angles(rl_data)
    x = np.arange(-40, 41, 1)
    for turb in range(3):
        barycenters = extract_barycenter(rl_data, turbine=turb)
        print(f'Plotting KDE for turbine {turb+1} barycenter')
        # sns.kdeplot(x=angles[f"Angle {turb+1}"], y=barycenters,
        #             ax=axes[turb], fill=True, cmap='Greys', levels=100)
        sns.histplot(x=barycenters,
                     ax=axes[turb], bins=100, pthresh=0.05, cbar=False)
        axes[turb].set_xlabel(f'Barycenter {turb+1}')
        # axes[turb].grid(alpha=0.3, linestyle=':')


    fig.savefig("barycenter_distribution.png", dpi=400)
    fig.savefig("barycenter_distribution.pdf")
    plt.close()


    All_Sensors = False

    if All_Sensors:

        for turb in range(3):
            fig, axes = plt.subplots(7, 11, figsize=(18, 14),
                                     constrained_layout=True, )
            palette = sns.color_palette()

            angles = extract_angles(rl_data)
            x = np.arange(-40, 41, 1)
            for i, sensor in enumerate(range(0,77,11)):
                for j in range(11):
                    flow = extract_flow(rl_data, turbine=turb, sensor=sensor+j)
                    print(f'Plotting KDE for turbine {turb+1} sensor {sensor+j}')
                    # sns.kdeplot(x=angles[f"Angle {turb+1}"], y=barycenters,
                    #             ax=axes[turb], fill=True, cmap='Greys', levels=100)
                    sns.histplot(x=angles[f"Angle {turb+1}"], y=flow,
                                 ax=axes[i, j], cmap='Greys', bins=100, pthresh=0.05, cbar=False)
                    axes[i, j].set_xlim(-40, 40)
                    axes[i, j].set_ylabel(f'Sensor {sensor+j}')
                    # axes[i, j].grid(alpha=0.3, linestyle=':')
                    axes[i, j].set_xticks([-40, -20, 0, 20, 40])


            fig.savefig(f"flow_correlation_turbine_{turb}.png", dpi=400)
            fig.savefig(f"flow_correlation_turbine_{turb}.pdf")
            plt.close()

