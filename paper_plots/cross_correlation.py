import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
from utils import fonts
import re
import pickle
from scipy.signal import welch
import seaborn as sns
import pandas as pd


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


def extract_angles(data):
    angles = []
    n_eps, n_envs = find_num_eps_envs(data)
    for ep in range(1, n_eps+1):
        for ev in range(1, n_envs+1):
            yaws = data[f"alphas_ENV_{ev}_EPISODE_{ep}"]
            angles.append(yaws)
    angles = np.asarray(angles).transpose(0, 2, 1)
    reshaped_arr = angles.reshape(-1, 3)
    df = pd.DataFrame(reshaped_arr, columns=['Angle 1', 'Angle 2', 'Angle 3'])
    return df


if __name__ == '__main__':

    # Load data
    with open(f'./long_evals_final/RL/eval_logs.pkl', 'rb') as f:
        rl_data = pickle.load(f)

    fig, axs = plt.subplots(1, 3,
                            figsize=(8, 4),
                            constrained_layout=True,)

    best_shift = []
    best_shift_value = []

    angles = extract_angles(rl_data)
    for i, pair in enumerate([(1,2), (1,3), (2,3)]):
        assert pair[0] < pair[1]
        angles_A = angles[f'Angle {pair[0]}']
        angles_B = angles[f'Angle {pair[1]}']
        correlation = signal.correlate(angles_A, angles_B, mode="full")
        lags = signal.correlation_lags(angles_A.size, angles_A.size, mode="full")
        lag = lags[np.argmax(correlation)]

        axs[i].plot(lags, correlation)
        turbs = pair[1]-pair[0]
        axs[i].axvline((-126*5*turbs/7.5)/10, color='k', linestyle=':')
        best_shift.append(lag)
        best_shift_value.append(np.max(correlation))
        axs[i].axvline(best_shift[-1], linestyle='--')
        axs[i].set_title(f'Turbines {pair[0]}, {pair[1]} | lag {lag} frames')
        axs[i].set_xlabel('Time lag (RL frames)')
        axs[i].set_ylabel('Cross Correlation')
        axs[i].set_xlim((-50, 50))
        # axs[i].grid(axis='x')
        axs[i].grid(alpha=0.3, linestyle=':')

        print(f"Pair {pair[0]}, {pair[1]}: lag {lag} | time in free stream {-126*5*turbs/7.5/10}")

    fig.savefig("cross_correlation.png", dpi=400)
    fig.savefig("cross_correlation.pdf")
    plt.close(fig)


    """
    # Plot correlations with shifted data
    fig, axes = plt.subplots(1, 3, figsize=(6, 2),
                            constrained_layout=True,)

    for i, pair in enumerate([(1,2), (1,3), (2,3)]):

        axes[i].set_aspect('equal')
        print(f'extracted shift of {best_shift[i]}')
        print(f'with correlation of {best_shift_value[i]:.3f}')
        shifted_angles = extract_angles(rl_data, best_shift[i], pair[0])
        print(f'plotting kde {i+1}')
        # sns.kdeplot(data=shifted_angles, x=f"Angle {pair[0]}", y=f"Angle {pair[1]}", ax=axes[i], color='k')
        sns.kdeplot(data=shifted_angles, x=f"Angle {pair[0]}", y=f"Angle {pair[1]}", ax=axes[i], fill=True, cmap='Greys', levels=100)
        axes[i].set_xlim(-40, 40)
        axes[i].set_ylim(-40, 40)
        axes[i].grid(alpha=0.3, linestyle=':')
        axes[i].set_xticks([-40, -20, 0, 20, 40])

    # THESE VALUES ARE TAKEN FROM THE ANGLE DISTRIBUTION PLOT
    # TODO: ADD ANGLE DISTRIBUTION CODE HERE?
    axes[0].scatter([-22.3, 23.3], [-8.3, 12.0], c='r', marker='x')
    axes[1].scatter([-22.3, 23.3], [6.7, -1.0], c='r', marker='x')
    axes[2].scatter([-8.3, 12.0], [6.7, -1.0], c='r', marker='x')

    axes[0].scatter([-20, 20], [-13, 13], c='b', marker='+')
    axes[1].scatter([-20, 20], [3, -3], c='b', marker='+')
    axes[2].scatter([-13, 13], [3, -3], c='b', marker='+')

    fig.savefig("shifted_correlation.png", dpi=400)
    fig.savefig("shifted_correlation.pdf")
    # plt.show()
    """

