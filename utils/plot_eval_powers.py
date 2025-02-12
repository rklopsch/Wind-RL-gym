import numpy as np
import pickle
import sys
import os
import matplotlib.pyplot as plt
import re


dt = 10  # 10 seconds per time step
BASE_POWER = 3.223854866218793


if __name__ == '__main__':
    # This script is designed to import the eval logs obtained from running the eval script
    # and plotting the angles and actions of those episodes

    if len(sys.argv) != 2:
        print("Usage: python script.py <path_to_eval_logs>")
        sys.exit(1)

    filename = sys.argv[1]

    with open(filename, 'rb') as f:
        data = pickle.load(f)

    dir_name = os.path.dirname(filename)
    if len(dir_name) == 0:
        dir_name = './'
    pic_path = dir_name + '/power_plots/'
    if not os.path.exists(pic_path):
        os.makedirs(pic_path)

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

    # find episode length
    episode_length = data[f'episode_length_ENV_{num_envs}_EPISODE_{num_episodes}']

    # fit things into an array of shape [num envs x num episodes]
    pows = np.zeros([num_envs, num_episodes])
    for env_number in range(1, num_envs+1):
        for episode_number in range(1, num_episodes+1):    
            episode_power = data[f'episode_power_ENV_{env_number}_EPISODE_{episode_number}']
            episode_power *= 3
            episode_power /= episode_length
            episode_power /= BASE_POWER

            powers = data[f'power_ENV_{env_number}_EPISODE_{episode_number}']
            episode_powers_2 = powers.mean()
            episode_powers_2 *= 3
            episode_powers_2 /= BASE_POWER

            # Just to be sure, we compute the power in two different ways here and check they are reasonably similar
            assert np.abs((episode_powers_2 - episode_power) / episode_power) < 0.001

            pows[env_number-1, episode_number-1] = episode_power

    x = np.arange(1,num_episodes+1)
    plt.plot(x, pows.mean(axis=0))
    plt.fill_between(
        x, 
        pows.mean(axis=0)-1.96*pows.std(axis=0)/np.sqrt(num_envs),
        pows.mean(axis=0)+1.96*pows.std(axis=0)/np.sqrt(num_envs),
        alpha=0.3
    )
    plt.xlabel("Number of episodes")
    plt.ylabel("Normalised episode power")
    plt.xticks(x,x)
    plt.grid(True)
    plt.savefig(pic_path + f'episode_power_by_episode.png')
    plt.close()

    for env in range(num_envs):
        plt.plot(x, pows[env,:], label=f"Env {env+1}")
    plt.xlabel("Number of episodes")
    plt.ylabel("Normalised episode power")
    plt.xticks(x,x)
    plt.grid(True)
    plt.legend()
    plt.savefig(pic_path + f'episode_power_by_episode_individual_envs.png')
    plt.close()

    print(f"Saved plots to {os.path.abspath(pic_path)}.")

    # Compute overall mean
    print('---------------')
    print(f"Overall mean relative power using {num_envs} envs and {num_episodes} episodes per env:")
    print(f"{pows.mean():.5f} ({100*pows.std()/np.sqrt(num_envs*num_episodes):.5f}% stderror | {100*pows.std()/(np.sqrt(num_envs*num_episodes)*np.abs(pows.mean())):.5f}% relative error to mean power)")
    print('---------------')

    # Compute standard error when we compute the mean using k samples out of n_envs x n_episodes
    overall_std = np.std(pows)
    print(f"Using __ samples yields stderror __")
    for k in range(1, num_envs * num_episodes, (num_envs * num_episodes)//10):
        print(f"k={k} \t {overall_std/np.sqrt(k):.5f} ({100*overall_std/(np.sqrt(k)*np.abs(np.mean(pows))):.2f}% of mean relative power)")

    


