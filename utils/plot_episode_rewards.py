import numpy as np
import matplotlib.pyplot as plt
from load_buffer import load_buffer


# We define the base power output here
BASE_POWER = 3.25


if __name__ == '__main__':
    # This script reads data from a replay buffer to plot the episodic reward
    print(f"Using mean base power of {BASE_POWER} to normalise plots.")

    # Load replay buffer
    replay_buffer = load_buffer('replay_buffer_test/replay_buffer_checkpoint_ppo.pt')

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

    # Plot mean with 95% confidence band
    x = num_envs * np.arange(1, episode_reward.shape[0]+1)
    plt.plot(x, mean_episode_reward)
    plt.fill_between(
        x,
        mean_episode_reward-1.98*std_episode_reward/np.sqrt(num_envs),
        mean_episode_reward+1.98*std_episode_reward/np.sqrt(num_envs),
        alpha=0.3
    )
    plt.xlabel('Number of episodes (across all environments)')
    plt.ylabel('Mean episodic reward (in MW)')
    plt.title(f'Mean and 95% CI for {num_envs} training environments')
    plt.savefig('mean_episode_reward.png')
    plt.close()

    # Plot all episode rewards (across environments) individually
    for i in range(episode_reward.shape[-1]):
        y = episode_reward[:, i]
        plt.plot(y, alpha=0.5)
    plt.xlabel('Number of episodes per individual environment')
    plt.ylabel('Episodic reward')
    plt.title('Individual training envs')
    plt.savefig('individual_episode_reward.png')
    plt.close()

