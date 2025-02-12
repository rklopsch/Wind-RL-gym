import numpy as np
import sys
import matplotlib.pyplot as plt
import torch
# from load_buffer import load_buffer


# We define the base power output here
BASE_POWER = 3.223854866218793
BASE_POWER_STDERROR = 0.009575164430978821
STD_POWER = 0.49171201813667925
STD_POWER_STDERROR = 0.00496220315074398


if __name__ == '__main__':

    fig1, ax1 = plt.subplots(1, 1,
                            figsize=(6, 6),
                            constrained_layout=True,)
    fig2, ax2 = plt.subplots(1, 1,
                            figsize=(6, 6),
                            constrained_layout=True,)
    fig3, ax3 = plt.subplots(1, 1,
                            figsize=(6, 6),
                            constrained_layout=True,)
    fig4, ax4 = plt.subplots(1, 1,
                            figsize=(6, 6),
                            constrained_layout=True,)

    # Check if an argument is provided
    if len(sys.argv) < 2:
        print("Usage: python shorten_replay_buffer.py <filename_of_buffers>")
        sys.exit(1)

    gamma = [0.95, 0.8, 0.9, 0.98, 0.99]
    alpha_init = [1, 10, 100]
    yaw_speed = [1.0, 0.5, 2.0]
    frame_stack = [5, 1]
    episode_length = [500, 125, 250, 1000, 2000]
    learning_rate = [3e-4, 1e-4, 3e-5, 3e-6]
    batch_frames = [1024, 512, 128]
    probes_per_turbine = [60, 40, 20, 10, 5]

    cmap = plt.get_cmap("tab10")

    for i in range(len(sys.argv)-1):

        # Get the command-line argument
        filename = sys.argv[i+1]

        # Load replay buffer
        # replay_buffer = load_buffer(filename)
        replay_buffer = torch.load(filename)

        # Extract data
        has_power = ("_data", "next", "power") in replay_buffer.keys(include_nested=True)
        # rewards = replay_buffer.get(("_data", "next", "reward")).squeeze()
        if has_power:
            # powers = replay_buffer.get(("_data", "next", "power")).squeeze()
            episode_power = replay_buffer.get(("_data", "next", "episode_power")).squeeze().numpy()
        episode_reward = replay_buffer.get(("_data", "next", "episode_reward")).squeeze().numpy()
        done = replay_buffer.get(("_data", "next", "done")).squeeze().numpy()
        num_envs = done.shape[-1]

        # Find out where the episodes have ended
        # Everything is synchronised so we can just look at one env
        episode_ends = np.where(done[:, 0])
        episode_reward = episode_reward[episode_ends]
        if has_power:
            episode_power = episode_power[episode_ends]

        # Constants for normalisation
        episode_length = episode_ends[0][0] + 1  # num of RL frames in one episode
        dt = 10  # 10 seconds per time step
        n_turbs = 3  # Number of turbines is fixed here.

        # We compute rewards as the MEAN across turbines, transform this back to a SUM across turbines
        # Episode reward is the sum of rewards in an episode, we want the mean across time
        episode_reward *= n_turbs
        episode_reward /= episode_length
        episode_reward /= BASE_POWER
        if has_power:
            episode_power *= n_turbs
            episode_power /= episode_length
            episode_power /= BASE_POWER

        # Take mean across environment dimension
        mean_episode_reward = np.mean(episode_reward, axis=-1)
        std_episode_reward = np.std(episode_reward, axis=-1)
        if has_power:
            mean_episode_power = np.mean(episode_power, axis=-1)
            std_episode_power = np.std(episode_power, axis=-1)

        # Plot mean with 95% confidence band
        # REWARD
        x = num_envs * np.arange(1, episode_reward.shape[0]+1)
        ax1.plot(x, mean_episode_reward, label=f'probes_per_turbine={probes_per_turbine[i]}', color=cmap(i))
        ax1.fill_between(
            x,
            mean_episode_reward-1.98*std_episode_reward/np.sqrt(num_envs),
            mean_episode_reward+1.98*std_episode_reward/np.sqrt(num_envs),
            alpha=0.3
        )
        # POWER
        if has_power:
            x = num_envs * np.arange(1, episode_power.shape[0] + 1)
            plt.plot(x, mean_episode_power)
            plt.plot(x, mean_episode_power, label=f'probes_per_turbine={probes_per_turbine[i]}', color=cmap(i))
            plt.fill_between(
                x,
                mean_episode_power - 1.98 * std_episode_power / np.sqrt(num_envs),
                mean_episode_power + 1.98 * std_episode_power / np.sqrt(num_envs),
                alpha=0.3
            )
    ax1.set_xlabel('Number of episodes (across all environments)')
    ax1.set_ylabel('Mean episodic reward (in MW) normalised by base mean power')
    ax1.legend(frameon=False)
    ax1.set_title(f'Mean and 95% CI for {num_envs} training environments')
    fig1.savefig('mean_episode_reward.png')

    ax2.set_xlabel('Number of episodes (across all environments)')
    ax2.set_ylabel('Mean episodic power (in MW) normalised by base power')
    ax2.set_title(f'Mean and 95% CI for {num_envs} training environments')
    fig2.savefig('mean_episode_power.png')

    # Plot all episode rewards (across environments) individually
    # ONLY DO THIS FOR LAST CASE GIVEN OTHERWISE TOO MESSY
    # REWARD
    for i in range(episode_reward.shape[-1]):
        y = episode_reward[:, i]
        ax3.plot(y, alpha=0.5)
    ax3.set_xlabel('Number of episodes per individual environment')
    ax3.set_ylabel('Episodic reward normalised by mean base power')
    ax3.set_title('Individual training envs')
    fig3.savefig('individual_episode_reward.png')

    if has_power:
        for i in range(episode_power.shape[-1]):
            y = episode_power[:, i]
            ax4.plot(y, alpha=0.5)
        ax4.set_xlabel('Number of episodes per individual environment')
        ax4.set_ylabel('Episodic power normalised by mean base power')
        ax4.set_title('Individual training envs')
        fig4.savefig('individual_episode_power.png')
