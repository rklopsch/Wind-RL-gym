import matplotlib.pyplot as plt
import numpy as np
from utils import fonts
import re
import pickle
import seaborn as sns
from scipy.stats import gaussian_kde


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
    print(f"Found {n_eps} episodes and {n_envs} envs")
    for ep in range(1, n_eps+1):
        for ev in range(1, n_envs+1):
            powers = data[f"power_ENV_{ev}_EPISODE_{ep}"]
            ep_power = powers.mean()
            ep_power *= 3
            mean_powers.append(ep_power)
    return np.asarray(mean_powers)


def extract_raw_powers(data):
    list_powers = []
    n_eps, n_envs = find_num_eps_envs(data)
    print(f"Found {n_eps} episodes and {n_envs} envs")
    for ep in range(1, n_eps+1):
        for ev in range(1, n_envs+1):
            powers = data[f"power_ENV_{ev}_EPISODE_{ep}"]
            powers *= 3.0  # problematic, this modifies in place...
            list_powers.append(powers)
    return np.asarray(list_powers)


if __name__ == '__main__':
    # Load best training run data
    with open('long_evals_final/RL/eval_logs.pkl', 'rb') as f:
        rl_data = pickle.load(f)
    rl_powers = extract_mean_powers(rl_data)

    with open('long_evals_final/BO/eval_logs.pkl', 'rb') as f:
        bo_data = pickle.load(f)
    bo_powers = extract_mean_powers(bo_data)

    with open('long_evals_final/zero/eval_logs.pkl', 'rb') as f:
        zero_data = pickle.load(f)
    zero_powers = extract_mean_powers(zero_data)

    # Create the figure with two subplots
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    palette = sns.color_palette()

    # First subplot: Violin plot
    ax1 = axes[0]
    vp = ax1.violinplot([zero_powers, bo_powers, rl_powers], showmeans=False, showmedians=True,
                        quantiles=[[0.25, 0.75], [0.25, 0.75], [0.25, 0.75]], showextrema=True)
    ax1.yaxis.grid(True)
    ax1.set_xticks([1, 2, 3], labels=["Greedy", "Static BO", "RL"])
    ax1.set_ylabel("Mean farm power (MW)")
    ax1.set_xlim(0.5, 3.5)  # Expand x-axis limits to prevent text overflow

    # Adjust transparency of quantile lines
    vp['cquantiles'].set_alpha(0.4)
    vp['cmins'].set_alpha(0.4)
    vp['cmaxes'].set_alpha(0.4)

    def colour_shuffle(i):
        if i==0: return 2
        if i==1: return 1
        if i==2: return 0

    # Set color properties
    for i, p in enumerate(vp['bodies']):
        p.set_facecolor(palette[colour_shuffle(i)])
        p.set_edgecolor('black')
        p.set_alpha(0.7)
    """
    for p in vp['bodies']:
        p.set_color('k')
    """
    vp['cbars'].set_color('k')
    vp['cmedians'].set_color('k')
    vp['cmaxes'].set_color('k')
    vp['cmins'].set_color('k')
    vp['cquantiles'].set_color('k')

    # Add median text labels
    for i, data in enumerate([zero_powers, bo_powers, rl_powers]):
        value = np.median(data)
        ax1.text(i + 1.02, value, f'{value:.2f} MW', ha='left', va='bottom', fontsize=12)

    # Second subplot: KDE plot
    ax2 = axes[1]
    rl_raw_powers = extract_raw_powers(rl_data).flatten()
    bo_raw_powers = extract_raw_powers(bo_data).flatten()
    zero_raw_powers = extract_raw_powers(zero_data).flatten()

    # Compute medians
    rl_median = np.median(rl_raw_powers)
    bo_median = np.median(bo_raw_powers)
    zero_median = np.median(zero_raw_powers)

    # Compute means
    rl_mean = np.mean(rl_raw_powers)
    bo_mean = np.mean(bo_raw_powers)
    zero_mean = np.mean(zero_raw_powers)

    # Compute KDE curves manually and normalize them to the mean of the data
    def scaled_kde(data, mean_value):
        kde = gaussian_kde(data)
        x_vals = np.linspace(min(data), max(data), 1000)
        kde_vals = kde(x_vals)
        scale_factor = mean_value / np.trapezoid(kde_vals, x_vals)  # Scale integral to match mean
        return x_vals, kde_vals * scale_factor, scale_factor

    # Compute KDE curves manually to extract y-values at median locations
    def kde_y_at_x(data, x, scale):
        kde = gaussian_kde(data)
        return scale*kde(x)  # Returns the estimated density at x

    rl_x, rl_kde_vals, rl_scale = scaled_kde(rl_raw_powers, rl_mean)
    bo_x, bo_kde_vals, bo_scale = scaled_kde(bo_raw_powers, bo_mean)
    zero_x, zero_kde_vals, zero_scale = scaled_kde(zero_raw_powers, zero_mean)

    rl_median_y = kde_y_at_x(rl_raw_powers, rl_median, rl_scale)
    bo_median_y = kde_y_at_x(bo_raw_powers, bo_median, bo_scale)
    zero_median_y = kde_y_at_x(zero_raw_powers, zero_median, zero_scale)

    # Plot KDEs with consistent colors
    # sns.kdeplot(rl_raw_powers, label='RL', ax=ax2, color=palette[0])
    # sns.kdeplot(bo_raw_powers, label='Static BO', ax=ax2, color=palette[1])
    # sns.kdeplot(zero_raw_powers, label='Greedy', ax=ax2, color=palette[2])

    # Plot scaled KDEs with consistent colors
    ax2.plot(rl_x, rl_kde_vals, label='RL', color=palette[0])
    ax2.plot(bo_x, bo_kde_vals, label='Static BO', color=palette[1])
    ax2.plot(zero_x, zero_kde_vals, label='Greedy', color=palette[2])

    # Add truncated vertical lines up to KDE peak at the median
    ax2.vlines(rl_median, ymin=0, ymax=rl_median_y, color=palette[0], linestyle='dashed', linewidth=1.3)
    ax2.vlines(bo_median, ymin=0, ymax=bo_median_y, color=palette[1], linestyle='dashed', linewidth=1.3)
    ax2.vlines(zero_median, ymin=0, ymax=zero_median_y, color=palette[2], linestyle='dashed', linewidth=1.3)

    ax2.set_xlim(1.8, 6)  # Set x-axis limits
    ax2.set_xlabel("Instantaneous Power Output (MW)", labelpad=-0.5)
    ax2.set_ylabel("Density")
    ax2.legend()

    # Save combined figure
    fig.savefig('combined_violin_kde.png', dpi=400)
    fig.savefig('combined_violin_kde.pdf')
    plt.close(fig)

