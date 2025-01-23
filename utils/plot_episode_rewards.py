import numpy as np
import matplotlib.pyplot as plt
from load_buffer import load_buffer


# We define the base power output here
BASE_POWER = 3.223854866218793
BASE_POWER_STDERROR = 0.009575164430978821
STD_POWER = 0.49171201813667925
STD_POWER_STDERROR = 0.00496220315074398


if __name__ == '__main__':
    # This script reads data from a replay buffer to plot the episodic reward
    print(f"Using mean base power of {BASE_POWER} to normalise plots.")

    # Load replay buffer
    replay_buffer = load_buffer('replay_buffer_test/replay_buffer_checkpoint_ppo.pt')

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

    # We compute rewards as the MEAN across turbines, transform this back to a SUM across turbines
    # Episode reward is the sum of rewards in an episode, we want the mean across time
    episode_reward *= num_envs
    episode_reward /= (episode_length * dt)
    episode_reward /= BASE_POWER
    if has_power:
        episode_power *= num_envs
        episode_power /= (episode_length * dt)
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
    plt.plot(x, mean_episode_reward)
    plt.fill_between(
        x,
        mean_episode_reward-1.98*std_episode_reward/np.sqrt(num_envs),
        mean_episode_reward+1.98*std_episode_reward/np.sqrt(num_envs),
        alpha=0.3
    )
    plt.xlabel('Number of episodes (across all environments)')
    plt.ylabel('Mean episodic reward (in MW) normalised by base mean power')
    plt.title(f'Mean and 95% CI for {num_envs} training environments')
    plt.savefig('mean_episode_reward.png')
    plt.close()

    # POWER
    if has_power:
        x = num_envs * np.arange(1, episode_power.shape[0] + 1)
        plt.plot(x, mean_episode_power)
        plt.fill_between(
            x,
            mean_episode_power - 1.98 * std_episode_power / np.sqrt(num_envs),
            mean_episode_power + 1.98 * std_episode_power / np.sqrt(num_envs),
            alpha=0.3
        )
        plt.xlabel('Number of episodes (across all environments)')
        plt.ylabel('Mean episodic power (in MW) normalised by base power')
        plt.title(f'Mean and 95% CI for {num_envs} training environments')
        plt.savefig('mean_episode_power.png')
        plt.close()

    # Plot all episode rewards (across environments) individually
    # REWARD
    for i in range(episode_reward.shape[-1]):
        y = episode_reward[:, i]
        plt.plot(y, alpha=0.5)
    plt.xlabel('Number of episodes per individual environment')
    plt.ylabel('Episodic reward normalised by mean base power')
    plt.title('Individual training envs')
    plt.savefig('individual_episode_reward.png')
    plt.close()

    if has_power:
        for i in range(episode_power.shape[-1]):
            y = episode_power[:, i]
            plt.plot(y, alpha=0.5)
        plt.xlabel('Number of episodes per individual environment')
        plt.ylabel('Episodic power normalised by mean base power')
        plt.title('Individual training envs')
        plt.savefig('individual_episode_power.png')
        plt.close()
