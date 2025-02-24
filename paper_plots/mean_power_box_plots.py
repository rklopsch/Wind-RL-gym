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
    # Load best training run data
    with open('../outputs/long_eval/long_eval_logs.pkl', 'rb') as f:
        rl_data = pickle.load(f)

    rl_powers = extract_powers(rl_data)

    # Create the box plot
    fig, ax = plt.subplots(figsize=(6, 3))
    fig.set_figwidth(6)
    plt.boxplot([rl_powers, rl_powers, rl_powers], tick_labels=["Greedy", "Static BO", "RL"])
    plt.grid(True)

    # Labels and title
    # plt.xlabel("Datasets")
    plt.ylabel("Mean farm power (MW)")

    # Add legend
    # plt.legend(["Zero yaw", "Static BO", "RL"], loc="upper right")

    # Show the plot
    plt.savefig('mean_power_box_plots.png')

