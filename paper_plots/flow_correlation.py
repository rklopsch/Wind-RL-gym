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
    observation_range = sensor + 77*(turbine-1)
    observations = observations[:, observation_range, :] * 6. + 6.
    observations = observations.reshape(-1)
    return observations


def extract_barycenter(data, shift=0, turbine=1):
    observations = []
    n_eps, n_envs = find_num_eps_envs(data)
    for ep in range(1, n_eps + 1):
        for ev in range(1, n_envs + 1):
            obs = data[f"observations_ENV_{ev}_EPISODE_{ep}"]
            observations.append(obs)
    observations = np.asarray(observations).transpose(0, 2, 1)
    # observation_range = range(4+77*(turbine-1), 77+77*(turbine-1), 11)
    indices = np.concatenate([np.arange(11, 16),
                              np.arange(22, 27),
                              np.arange(44, 49),
                              np.arange(55, 60),
                              ]) + 77*(turbine-1)
    observations = observations[:, indices, :] * 6. + 6.
    observations = observations.reshape(-1, 5*4)
    probe_locations = np.repeat(np.linspace(-1, 1, 7)[1:-1], 4)

    weighted_sums = np.sum(observations * probe_locations, axis=1)
    total_weights = np.sum(observations, axis=1)

    barycenters = weighted_sums / total_weights
    return barycenters


if __name__ == '__main__':

    # Load data
    # with open(f'./long_evals_final/RL/eval_logs.pkl', 'rb') as f:
    with open(f'./final/evaluation/RL/eval_logs.pkl', 'rb') as f:
        rl_data = pickle.load(f)

    barycenters = extract_barycenter(rl_data)

    fig, axs = plt.subplots(1, 1,
                            figsize=(8, 3),
                            constrained_layout=True, )

    axs.plot(barycenters)
    fig.savefig('barycenters.pdf')

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
        sns.histplot(x=angles[f"Angle {turb+1}"], y=barycenters,
                     ax=axes[turb], cmap='Greys', bins=100, pthresh=0.05, cbar=False)
        axes[turb].set_xlim(-40, 40)
        axes[turb].set_ylabel(f'Barycenter {turb}')
        axes[turb].grid(alpha=0.3, linestyle=':')
        axes[turb].set_xticks([-40, -20, 0, 20, 40])


    fig.savefig("barycenter_correlation.png", dpi=400)
    fig.savefig("barycenter_correlation.pdf")
    plt.close()

    fig, axes = plt.subplots(7, 3, figsize=(6, 14),
                             constrained_layout=True, )
    palette = sns.color_palette()

    angles = extract_angles(rl_data)
    x = np.arange(-40, 41, 1)
    for i, sensor in enumerate(range(3,77,11)):
        for turb in range(3):
            barycenters = extract_flow(rl_data, turbine=turb, sensor=sensor)
            print(f'Plotting KDE for turbine {turb+1} sensor {sensor}')
            # sns.kdeplot(x=angles[f"Angle {turb+1}"], y=barycenters,
            #             ax=axes[turb], fill=True, cmap='Greys', levels=100)
            sns.histplot(x=angles[f"Angle {turb+1}"], y=barycenters,
                         ax=axes[i, turb], cmap='Greys', bins=100, pthresh=0.05, cbar=False)
            axes[i, turb].set_xlim(-40, 40)
            axes[i, turb].set_ylabel(f'Sensor {sensor}')
            axes[i, turb].grid(alpha=0.3, linestyle=':')
            axes[i, turb].set_xticks([-40, -20, 0, 20, 40])


    fig.savefig("flow_correlation.png", dpi=400)
    fig.savefig("flow_correlation.pdf")
    plt.close()
    # plt.show()
