import numpy as np
import pickle
import sys
import os
import matplotlib.pyplot as plt
import re


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
    pic_path = dir_name + '/yaw_action_plots/'
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

    for episode_number in range(1, num_episodes+1):
        for env_number in range(1, num_envs+1):
            fig, axes = plt.subplots(1, 2, figsize=(10, 4))
            alpha_arr = data[f'alphas_ENV_{env_number}_EPISODE_{episode_number}']
            action_arr = data[f'actions_ENV_{env_number}_EPISODE_{episode_number}']
            for turb in range(alpha_arr.shape[-1]):
                axes[0].plot(alpha_arr[:, turb], label=f"Turbine {turb+1}")
                axes[1].plot(action_arr[:, turb], label=f"Turbine {turb+1}")
            axes[0].legend()
            axes[1].legend()
            axes[0].set_xlabel('RL frames')
            axes[1].set_xlabel('RL frames')
            axes[0].grid(True)
            axes[1].grid(True)
            axes[0].set_ylabel('Turbine yaw angles (degrees)')
            axes[1].set_ylabel('Turbine actions')
            fig.suptitle(f"Environment {env_number} | Episode {episode_number}")
            plt.tight_layout()
            plt.savefig(pic_path + f'action_angle_plot_ENV{env_number}_EP{episode_number}.png')
            plt.close()

    print(f"Saved plots to {os.path.abspath(pic_path)}.")

    


