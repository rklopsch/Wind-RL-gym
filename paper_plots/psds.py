import matplotlib.pyplot as plt
import numpy as np
from utils import fonts
import re
import pickle
from scipy.signal import welch


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

def extract_mean_powers(data):
    mean_powers = []
    n_eps, n_envs = find_num_eps_envs(data)
    for ep in range(1, n_eps+1):
        for ev in range(1, n_envs+1):
            powers = data[f"power_ENV_{ev}_EPISODE_{ep}"]
            ep_power = powers.mean()
            ep_power *= 3
            mean_powers.append(ep_power)
    return np.asarray(mean_powers)


def extract_psd_actions(data):
    action_psds = []
    n_eps, n_envs = find_num_eps_envs(data)
    fs = 1/10
    for ep in range(1, n_eps+1):
        for ev in range(1, n_envs+1):
            yaws = data[f"alphas_ENV_{ev}_EPISODE_{ep}"]
            turbine_psds = []
            for turbine in range(3):
                freqs, psd_yaw = welch(yaws[:, turbine], fs, nperseg=500)
                turbine_psds.append(psd_yaw)
            action_psds.append(turbine_psds)
    return np.asarray(action_psds), np.asarray(freqs)


def extract_psd_power(data):
    power_psds = []
    n_eps, n_envs = find_num_eps_envs(data)
    fs = 1/10
    for ep in range(1, n_eps+1):
        for ev in range(1, n_envs+1):
            powers = data[f"power_ENV_{ev}_EPISODE_{ep}"]
            # turbine_psds = []
            # for turbine in range(3):
            freqs, psd_power = welch(powers, fs, nperseg=500)
                # turbine_psds.append(psd_yaw)
            power_psds.append(psd_power)
    return np.asarray(power_psds), np.asarray(freqs)


def extract_psd_powers(file):
    with open(file, 'rb') as f:
        data = pickle.load(f)
    power_psds = []
    n_eps, n_envs = find_num_eps_envs(data)
    fs = 1/10
    for ep in range(1, n_eps+1):
        for ev in range(1, n_envs+1):
            powers = data[f"powers_ENV_{ev}_EPISODE_{ep}"]
            turbine_psds = []
            for turbine in range(3):
                freqs, psd_power = welch(powers[:, turbine], fs, nperseg=500)
                turbine_psds.append(psd_power)
            power_psds.append(turbine_psds)
    return np.asarray(power_psds), np.asarray(freqs)

if __name__ == '__main__':

    # Load data
    with open(f'./final/evaluation/RL/eval_logs.pkl', 'rb') as f:
        rl_data = pickle.load(f)
    rl_yaw_psds, freqs = extract_psd_actions(rl_data)
    # rl_power_psds, freqs = extract_psd_power(rl_data)
    rl_powers_psds, freqs = extract_psd_powers(f'./final/evaluation/RL/turbine_powers.pkl')


    with open(f'./final/evaluation/Zero/eval_logs.pkl', 'rb') as f:
        zero_data = pickle.load(f)
    # zero_power_psds, freqs = extract_psd_power(zero_data)
    zero_powers_psds, freqs = extract_psd_powers(f'./final/evaluation/Zero/turbine_powers.pkl')
    freqs *= 126/7.5

    # with open(f'./final/evaluation/Zero/eval_logs.pkl', 'rb') as f:
    #     data = pickle.load(f)
    # powers = extract_mean_powers(data)
    # base_power = np.mean(powers)


    # Create the box plot
    fig, ax = plt.subplots(1, 2, figsize=(6, 2.5), constrained_layout=True)
    fig.set_figwidth(6)
    n_eps, n_envs = find_num_eps_envs(rl_data)

    ax[0].semilogx(freqs, (freqs*np.mean(rl_yaw_psds, 0)).T)
    for turbine in range(3):
        ax[0].fill_between(
            freqs,
            (freqs*(np.mean(rl_yaw_psds[:, turbine, :], 0)-1.96*np.std(rl_yaw_psds[:, turbine, :], 0)/np.sqrt(n_envs*n_eps))).T,
            (freqs*(np.mean(rl_yaw_psds[:, turbine, :], 0)+1.96*np.std(rl_yaw_psds[:, turbine, :], 0)/np.sqrt(n_envs*n_eps))).T,
            alpha=0.3
        )
    ax[0].set_ylabel("PSD of yaw angle")
    ax[0].set_xlabel("St")
    ax[0].set_xlim(np.min(freqs), np.max(freqs))
    ax[0].set_ylim(0)
    # ax[0].grid(True)
    ax[0].legend(["Turbine 1", "Turbine 2", "Turbine 3"], loc="upper right")

    # for data in [rl_powers_psds, zero_power_psds]:
    for data in [zero_powers_psds]:
        ax[1].semilogx(freqs, (freqs*np.mean(data, 0)).T, linestyle=':')
        for turbine in range(3):
            ax[1].fill_between(freqs,
                (freqs*(np.mean(data[:, turbine, :], 0)-1.96*np.std(data[:, turbine, :], 0)/np.sqrt(n_envs*n_eps))).T,
                (freqs*(np.mean(data[:, turbine, :], 0)+1.96*np.std(data[:, turbine, :], 0)/np.sqrt(n_envs*n_eps))).T,
                alpha=0.3)
    plt.gca().set_prop_cycle(None)
    for data in [rl_powers_psds]:
        ax[1].semilogx(freqs, (freqs*np.mean(data, 0)).T)
        for turbine in range(3):
            ax[1].fill_between(freqs,
                (freqs*(np.mean(data[:, turbine, :], 0)-1.96*np.std(data[:, turbine, :], 0)/np.sqrt(n_envs*n_eps))).T,
                (freqs*(np.mean(data[:, turbine, :], 0)+1.96*np.std(data[:, turbine, :], 0)/np.sqrt(n_envs*n_eps))).T,
                alpha=0.3)
    ax[1].set_ylabel("PSD of farm power")
    ax[1].set_xlabel("St")
    ax[1].set_xlim(np.min(freqs), np.max(freqs))
    ax[1].set_ylim(0)
    # ax[1].legend(["Turbine 1", "Turbine 2", "Turbine 3"], loc="upper right")
    ax[1].legend(["Greedy", "_", "_", "_", "_", "_", "RL"], loc="upper right")


    # Show the plot
    fig.savefig('psd.pdf')
    fig.savefig('psd.png', dpi=400)

