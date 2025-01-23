import numpy as np
import sys
import matplotlib.pyplot as plt
import torch
# from load_buffer import load_buffer


# We define the base power output here
BASE_POWER = 3.25


if __name__ == '__main__':

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
        rewards = replay_buffer.get(("_data", "next", "reward")).squeeze()
        episode_reward = replay_buffer.get(("_data", "next", "episode_reward")).squeeze().numpy()
        done = replay_buffer.get(("_data", "next", "done")).squeeze().numpy()
        num_envs = done.shape[-1]

        # Find out where the episodes have ended
        # Everything is synchronised so we can just look at one env
        episode_ends = np.where(done[:, 0])
        episode_reward = episode_reward[episode_ends]

        # Constants for normalisation
        episode_length = episode_ends[0][0] + 1  # num of RL frames in one episode
        dt = 10  # 10 seconds per time step

        # We compute rewards as the MEAN across turbines, transform this back to a SUM across turbines
        # Episode reward is the sum of rewards in an episode, we want the mean across time
        episode_reward *= num_envs
        episode_reward /= (episode_length * dt)

        # Normalise the rewards by the mean base power
        episode_reward /= BASE_POWER

        # Take mean across environment dimension
        mean_episode_reward = np.mean(episode_reward, axis=-1)
        std_episode_reward = np.std(episode_reward, axis=-1)
        max_episode_reward = np.max(episode_reward, axis=-1)
        min_episode_reward = np.min(episode_reward, axis=-1)

        # Plot mean with 95% confidence band
        x = num_envs * np.arange(1, episode_reward.shape[0]+1)
        plt.plot(x, mean_episode_reward, label=f'probes_per_turbine={probes_per_turbine[i]}', color=cmap(i))
        # plt.plot(x, max_episode_reward, linestyle='--', color=cmap(i), alpha=0.5)
        # plt.plot(x, min_episode_reward, linestyle='--', color=cmap(i), alpha=0.5)
        plt.fill_between(
            x,
            (mean_episode_reward-1.98*std_episode_reward/np.sqrt(num_envs)),
            (mean_episode_reward+1.98*std_episode_reward/np.sqrt(num_envs)),
            alpha=0.3,
            color=cmap(i)
        )
    plt.xlabel('Number of episodes (across all environments)')
    plt.ylabel('Mean episodic reward (in MW)')
    plt.legend(frameon=False)
    plt.title(f'Mean and 95% CI for {num_envs} training environments')
    plt.savefig('mean_episode_reward.png')


    # plt.close()

    # # Plot all episode rewards (across environments) individually
    # for i in range(episode_reward.shape[-1]):
    #     y = episode_reward[:, i]
    #     plt.plot(y, alpha=0.5)
    # plt.xlabel('Number of episodes per individual environment')
    # plt.ylabel('Episodic reward')
    # plt.title('Individual training envs')
    # plt.savefig('individual_episode_reward.png')
    # plt.close()

