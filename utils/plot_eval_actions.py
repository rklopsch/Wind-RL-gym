import numpy as np
import pickle
import sys
import os
import matplotlib.pyplot as plt


if __name__ == '__main__':
    # This script is designed to import the eval logs obtained from running the eval script
    # and plotting the angles and actions of those episodes

    if len(sys.argv) != 2:
        print("Usage: python script.py <path_to_eval_logs>")
        sys.exit(1)

    filename = sys.argv[1]

    with open(filename, 'rb') as f:
        data = pickle.load(f)

    for key, arr in data.items():
        if not key.startswith('alpha'):
            continue
        env_number = key[-11]
        episode_number = key[-1]
        for turb in range(arr.shape[-1]):
            plt.plot(arr[:, turb], label=f"Turbine {turb+1}")
        plt.legend()
        plt.savefig(os.path.dirname(filename) + f'/action_angle_plot_ENV{env_number}_EP{episode_number}.png')
        print(os.path.dirname(filename) + f'/action_angle_plot_ENV{env_number}_EP{episode_number}.png')
        plt.close()

    


