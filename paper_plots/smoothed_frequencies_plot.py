import matplotlib.pyplot as plt
import numpy as np
from utils import fonts
import re
import pickle


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


if __name__ == '__main__':
    frequencies = [5.0e-02,
                   2.5e-02,
                   1.25e-02,
                   6.25e-03,
                   3.125e-03,
                   1.5625e-03,
                   7.8125e-04,
                   3.90625e-04,]
    all_powers = []
    for i in range(len(frequencies)):
        # Load data
        with open(f'./sa_sac_array_2/evaluation/frequencies/pkls/{i}/fixed_actions_logs.pkl', 'rb') as f:
            data = pickle.load(f)
        powers = extract_powers(data)
        # print(data.keys())
        all_powers.append(powers)


    with open(f'./final/evaluation/Zero/eval_logs.pkl', 'rb') as f:
        data = pickle.load(f)
    powers = extract_powers(data)
    base_power = np.mean(powers)


    # Create the box plot
    fig, ax = plt.subplots(figsize=(6, 3), constrained_layout=True)
    fig.set_figwidth(6)
    num_envs = 16

    ax.set_xscale('log')
    ax.plot(frequencies, np.mean(all_powers, -1)/base_power, 'k-', marker='x')
    ax.fill_between(
        frequencies,
        (np.mean(all_powers, -1)-1.96*np.std(all_powers, -1)/np.sqrt(num_envs))/base_power,
        (np.mean(all_powers, -1)+1.96*np.std(all_powers, -1)/np.sqrt(num_envs))/base_power,
        color='k', alpha=0.3
    )
    ax.set_ylabel("Mean farm power")
    ax.set_xlabel("Cut-off frequency (Hz)")
    # ax.set_xticks(frequencies)
    # ax.grid(True)

    for i, frequency in enumerate(frequencies):
        mean = np.mean(all_powers[i])
        ax.text(frequency, mean/base_power, f'{frequency:.2e}', ha='center', va='bottom', fontsize=12)


    # Labels and title
    # plt.xlabel("Datasets")
    # plt.ylabel("Mean farm power (MW)")

    # Add legend
    # plt.legend(["Zero yaw", "Static BO", "RL"], loc="upper right")

    # Show the plot
    fig.savefig('smoothing_frequencies.pdf')
    fig.savefig('smoothing_frequencies.png', dpi=400)

