import matplotlib.pyplot as plt
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


def extract_angles(data, shift=1, turbine=1):
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


if __name__ == '__main__':

    # Load data
    with open(f'./long_evals_final/RL/eval_logs.pkl', 'rb') as f:
        rl_data = pickle.load(f)

    fig, axs = plt.subplots(1, 3,
                            figsize=(8, 3),
                            constrained_layout=True, )

    compute_CI = False
    best_shift = []
    best_shift_value = []
    best_shift_pvalue = []
    correlation_type = 'kendall'
    assert correlation_type in ['spearman', 'pearson', 'kendall']

    for i, pair in enumerate([(1, 2), (1, 3), (2, 3)]):
        assert pair[0] < pair[1]
        correlations = []
        pvalues = []
        bootstrap_stderror = []
        x = range(-50, 50)
        for shift in x:
            angles = extract_angles(rl_data, shift, pair[0])
            angles_A = angles[f'Angle {pair[0]}']
            angles_B = angles[f'Angle {pair[1]}']
            if correlation_type == 'pearson':
                cor = scipy.stats.pearsonr(angles_A, angles_B)
            elif correlation_type == 'spearman':
                cor = scipy.stats.spearmanr(angles_A, angles_B)
            elif correlation_type == 'kendall':
                cor = scipy.stats.kendalltau(angles_A, angles_B)
            p_value = cor.pvalue
            cor = cor.statistic
            correlations.append(cor)
            pvalues.append(p_value)

        correlations = np.asarray(correlations)
        bootstrap_stderror = np.asarray(bootstrap_stderror)
        axs[i].plot(np.asarray(x) * 10, correlations)
        if compute_CI:
            axs[i].fill_between(x, correlations - 1.96 * bootstrap_stderror, correlations + 1.96 * bootstrap_stderror,
                                alpha=0.3)
        turbs = pair[1] - pair[0]
        axs[i].axvline(-126 * 5 * turbs / 7.5, color='k', linestyle=':')
        best_shift.append(x[np.argmax(np.abs(correlations))])
        best_shift_value.append(correlations[np.argmax(np.abs(correlations))])
        best_shift_pvalue.append(pvalues[np.argmax(np.abs(correlations))])
        axs[i].axvline(best_shift[-1] * 10, linestyle='--')
        axs[i].set_title(f'Turbine {pair[0]} - Turbine {pair[1]}')
        axs[i].set_xlabel('Time lag (s)')
        axs[i].set_ylabel('Angle Correlation')
        # axs[i].grid(axis='x')
        axs[i].grid(alpha=0.3, linestyle=':')

    fig.savefig(f"cross_correlation_{correlation_type}.png", dpi=400)
    fig.savefig(f"cross_correlation_{correlation_type}.pdf")
    plt.close(fig)

    for i in range(3):
        pair = [(1, 2), (1, 3), (2, 3)][i]
        output = f"PAIR {pair} | LAG {best_shift[i]} | CORRELATION {best_shift_value[i]} | PVALUE {best_shift_pvalue[i]}"
        print(output)

    # Fit linear models between the time-shifted signals
    slopes = []
    intercepts = []
    stds = []
    residuals = []
    for i, pair in enumerate([(1, 2), (1, 3), (2, 3)]):
        shift = best_shift[i]
        angles = extract_angles(rl_data, shift, pair[0])
        angles_A = angles[f'Angle {pair[0]}']
        angles_B = angles[f'Angle {pair[1]}']
        regression = scipy.stats.linregress(angles_A, angles_B)
        slopes.append(regression.slope)
        intercepts.append(regression.intercept)
        std = np.linalg.norm((intercepts[i] + slopes[i] * angles_A) - angles_B) / (np.sqrt(angles_A.size)-1)
        stds.append(std)
        residuals.append(np.mean(np.abs((intercepts[i] + slopes[i] * angles_A) - angles_B)))
        print(f"PAIR {pair} | slope {slopes[i]:.2f} | intercept {intercepts[i]:.2f} | std {stds[i]:.2f} | mean residual {residuals[i]:.5f}")

    # Plot correlations with shifted data
    fig, axes = plt.subplots(1, 3, figsize=(6, 2),
                             constrained_layout=True, )

    x = np.arange(-40, 41, 1)
    for i, pair in enumerate([(1, 2), (1, 3), (2, 3)]):
        axes[i].set_aspect('equal')
        shifted_angles = extract_angles(rl_data, best_shift[i], pair[0])
        print(f'Plotting KDE for {pair}')
        # sns.kdeplot(data=shifted_angles, x=f"Angle {pair[0]}", y=f"Angle {pair[1]}", ax=axes[i], color='k')
        sns.kdeplot(data=shifted_angles, x=f"Angle {pair[0]}", y=f"Angle {pair[1]}", ax=axes[i], fill=True,
                    cmap='Greys', levels=100)
        axes[i].plot(x, intercepts[i] + slopes[i] * x, c='k', alpha=0.5)
        # axes[i].plot(x, intercepts[i] + slopes[i] * x + 1.96 * stds[i], c='k', alpha=0.5, linestyle='--')
        # axes[i].plot(x, intercepts[i] + slopes[i] * x - 1.96 * stds[i], c='k', alpha=0.5, linestyle='--')
        axes[i].plot(x, intercepts[i] + slopes[i] * x + stds[i], c='k', alpha=0.5, linestyle='--')
        axes[i].plot(x, intercepts[i] + slopes[i] * x - stds[i], c='k', alpha=0.5, linestyle='--')
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
    plt.close()
    # plt.show()
