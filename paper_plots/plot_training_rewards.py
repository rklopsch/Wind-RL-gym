import numpy as np
import matplotlib.pyplot as plt
import torch
import matplotlib.ticker as mtick
from utils import fonts
import pickle


# We define the base power output here
BASE_POWER = 3.21075


if __name__ == '__main__':

    fig, axes = plt.subplots(1, 2,
                            figsize=(10, 4),
                            constrained_layout=True,)
    # Load replay buffer
    # replay_buffer = load_buffer(filename)
    replay_buffer = torch.load('replay_buffer_final/replay_buffer_checkpoint.pt')
    episode_power = replay_buffer.get(("_data", "next", "episode_power")).squeeze().numpy()
    done = replay_buffer.get(("_data", "next", "done")).squeeze().numpy()
    num_envs = done.shape[-1]

    # Load losses
    with open('training_losses_final/training_logs.pkl', 'rb') as f:
        loss_logs = pickle.load(f)
    q_loss = np.asarray(loss_logs['train/q_loss'])
    a_loss = np.asarray(loss_logs['train/actor_loss'])
    alpha_loss = np.asarray(loss_logs['train/alpha_loss'])
    entropy = np.asarray(loss_logs['train/entropy'])

    # Find out where the episodes have ended
    # Everything is synchronised so we can just look at one env
    episode_ends = np.where(done[:, 0])
    episode_power = episode_power[episode_ends]

    # Constants for normalisation
    episode_length = episode_ends[0][0] + 1  # num of RL frames in one episode
    dt = 10  # 10 seconds per time step
    n_turbs = 3  # Number of turbines is fixed here.

    episode_power *= n_turbs
    episode_power /= episode_length
    episode_power /= BASE_POWER
    episode_power -= 1.0
    episode_power *= 100

    # Take mean across environment dimension
    mean_episode_power = np.mean(episode_power, axis=-1)
    std_episode_power = np.std(episode_power, axis=-1)

    # Plot mean with 95% confidence band
    # POWER
    x = num_envs * np.arange(1, episode_power.shape[0] + 1)
    axes[0].plot(x, mean_episode_power)
    axes[0].fill_between(
        x,
        mean_episode_power - 1.98 * std_episode_power / np.sqrt(num_envs),
        mean_episode_power + 1.98 * std_episode_power / np.sqrt(num_envs),
        alpha=0.3
    )
    axes[0].axhline(y=0, color='k', linestyle='--', linewidth=1)
    axes[0].set_xlabel('Episode number')
    axes[0].set_ylabel('Normalised farm power')
    axes[0].yaxis.set_major_formatter(mtick.PercentFormatter(xmax=100, decimals=0))

    axes[1].plot(entropy+3)
    # axes[1].plot(alpha_loss, label="alpha l")
    # axes[1].plot(np.abs(q_loss), label="q l")
    # axes[1].plot(np.abs(a_loss), label="a l")
    axes[1].set_xlabel('Training steps')
    axes[1].set_ylabel('Normalised entropy')
    #axes[1].set_yscale('symlog')


    fig.savefig('training_reward.png')
    plt.close()

