import matplotlib.pyplot as plt
import numpy as np
from utils import fonts
import re
import pickle
from scipy.signal import welch
import matplotlib.ticker as mtick


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

def extract_powers(data):
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


if __name__ == '__main__':

    with open(f'./final/evaluation/Zero/eval_logs.pkl', 'rb') as f:
        data = pickle.load(f)
    powers = extract_powers(data)
    base_power = np.mean(powers)

    frequencies = [5.0e-02,
                   2.5e-02,
                   1.25e-02,
                   6.25e-03,
                   3.125e-03,
                   1.5625e-03,
                   7.8125e-04,
                   3.90625e-04,]
    all_powers = []
    all_psds = []
    for i in range(len(frequencies)):
        # Load data
        with open(f'./sa_sac_array_2/evaluation/frequencies/pkls/{i}/fixed_actions_logs.pkl', 'rb') as f:
            data = pickle.load(f)
        powers = extract_powers(data)
        psds, freqs = extract_psd_actions(data)
        all_powers.append(powers)
        all_psds.append(psds)


    # Create the figure
    fig, ax = plt.subplots(1, 2, figsize=(6, 2.5), constrained_layout=True)
    fig.set_figwidth(6)
    num_envs = 16

    ax[0].set_xscale('log')
    means = (np.mean(all_powers, -1)-base_power)/base_power
    stderr = np.std(all_powers, -1)/np.sqrt(num_envs)/base_power
    ax[0].plot(frequencies, means, 'k-', marker='x', markersize=4)
    # ax[0].errorbar(frequencies, (np.mean(all_powers, -1)-base_power)/base_power*100, np.std(all_powers, -1)/np.sqrt(num_envs)/base_power*100, color='k')
    ax[0].fill_between(
        frequencies,
        means-1.96*stderr,
        means+1.96*stderr,
        color='k', alpha=0.2
    )
    ax[0].yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1, decimals=0))
    ax[0].set_ylabel("Mean farm power")
    ax[0].set_xlabel("Cut-off frequency (Hz)")
    # ax.set_xticks(frequencies)
    # ax.grid(True)

    # for i, frequency in enumerate(frequencies):
    #     mean = np.mean(all_powers[i])
    #     ax[0].text(frequency, mean/base_power, f'{frequency:.2e}', ha='center', va='bottom', fontsize=12)

    for rl_psds in all_psds:
        plt.gca().set_prop_cycle(None)
        n_eps, n_envs = find_num_eps_envs(data)
        ax[1].set_xscale('log')
        ax[1].plot(freqs, (freqs*np.mean(rl_psds, 0)).T)
        for turbine in range(3):
            ax[1].fill_between(
                freqs,
                (freqs*(np.mean(rl_psds[:, turbine, :], 0))).T,
                alpha=0.1
            )
        # for turbine in range(3):
        #     ax[1].fill_between(
        #         freqs,
        #         (freqs*(np.mean(rl_psds[:, turbine, :], 0)-1.96*np.std(rl_psds[:, turbine, :], 0)/np.sqrt(n_envs*n_eps))).T,
        #         (freqs*(np.mean(rl_psds[:, turbine, :], 0)+1.96*np.std(rl_psds[:, turbine, :], 0)/np.sqrt(n_envs*n_eps))).T,
        #         alpha=0.3
        #     )
    ax[1].set_ylabel("PSD of yaw angle")
    ax[1].set_xlabel("Frequency (Hz)")
    # ax.grid(True)

    for x in frequencies:
        for a in ax:
            a.axvline(x, color='k', linestyle=':', alpha=0.25)
    ax[1].legend(["Turbine 1", "Turbine 2", "Turbine 3"], loc="upper right", frameon=True)

    # Show the plot
    fig.savefig('smoothing_frequencies.pdf')
    fig.savefig('smoothing_frequencies.png', dpi=400)

