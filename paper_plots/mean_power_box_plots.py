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
    with open('final/evaluation/RL/eval_logs.pkl', 'rb') as f:
        rl_data = pickle.load(f)
    rl_powers = extract_powers(rl_data)

    with open('final/evaluation/BO/eval_logs.pkl', 'rb') as f:
        bo_data = pickle.load(f)
    bo_powers = extract_powers(bo_data)

    with open('final/evaluation/Zero/eval_logs.pkl', 'rb') as f:
        zero_data = pickle.load(f)
    zero_powers = extract_powers(zero_data)

    # Create the box plot
    fig, ax = plt.subplots(figsize=(6, 3), constrained_layout=True)
    fig.set_figwidth(6)
    # ax.boxplot([zero_powers, bo_powers, rl_powers], tick_labels=["Greedy", "Static BO", "RL"])
    vp = ax.violinplot([zero_powers, bo_powers, rl_powers], showmeans=True, showmedians=False, quantiles=[[0.25, 0.75], [0.25, 0.75], [0.25, 0.75]], showextrema=True)
    ax.yaxis.grid(True)
    ax.set_xticks([1, 2, 3], labels=["Greedy", "Static BO", "RL"])
    ax.set_ylabel("Mean farm power (MW)")
    vp['cquantiles'].set_alpha(0.2)
    vp['cmins'].set_alpha(0.2)
    vp['cmaxes'].set_alpha(0.2)
    # vp['cbars'].set_color('k')
    # vp['cmeans'].set_color('k')
    # vp['cquantiles'].set_color('k')
    # plt.grid(True)

    for i, data in enumerate([zero_powers, bo_powers, rl_powers]):
        mean = np.mean(data)
        ax.text(i+1.02, mean, f'{mean:.2f} MW', ha='left', va='bottom', fontsize=12)


    # Labels and title
    # plt.xlabel("Datasets")
    # plt.ylabel("Mean farm power (MW)")

    # Add legend
    # plt.legend(["Zero yaw", "Static BO", "RL"], loc="upper right")

    # Show the plot
    fig.savefig('mean_power_box_plots.pdf')
    fig.savefig('mean_power_box_plots.png', dpi=400)

