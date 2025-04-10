import numpy as np
import matplotlib.pyplot as plt
import torch
import matplotlib.ticker as mtick
import tqdm
import matplotlib.animation as ani
from functools import partial
# from utils import fonts_video
from utils import fonts
import pickle


# We define the base power output here
BASE_POWER = 3.21075

def update_plot(it, ax, x, mean_episode_power, std_episode_power):

    ax.clear()
    ax.plot(x[:it], mean_episode_power[:it])
    ax.fill_between(
            x[:it],
            mean_episode_power[:it] - 1.98 * std_episode_power[:it] / np.sqrt(num_envs),
            mean_episode_power[:it] + 1.98 * std_episode_power[:it] / np.sqrt(num_envs),
        alpha=0.3
    )
    ax.axhline(y=0, color='k', linestyle='--', linewidth=1)
    ax.set_xlabel('Episode number')
    ax.set_ylabel('Normalised farm power')
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=100, decimals=0))
    ax.grid(linestyle=':', alpha=0.5)
    ax.set_xlim(min(x), max(x))
    ax.set_ylim(-25, 5)

if __name__ == '__main__':

    animation = False

    fig, axes = plt.subplots(1, 1,
                            figsize=(3, 2),
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


    print('Plotting Figure...')
    update_plot(it=episode_power.shape[0], ax=axes, x=x, mean_episode_power=mean_episode_power, std_episode_power=std_episode_power)
    fig.savefig('training_reward.png')
    fig.savefig('training_reward.pdf')

    if animation:

        print('Creating Animation...')
        iters = tqdm.tqdm(range(0, len(x)), desc="Iteration", position=0)
        anim = ani.FuncAnimation(fig, partial(update_plot, ax=axes, x=x, mean_episode_power=mean_episode_power, std_episode_power=std_episode_power), frames=iters)
        anim.save(f'animations/training.mp4', fps=3, dpi=400)  # codec='h263p')


