import numpy as np
import pickle
import sys
import os
import matplotlib.pyplot as plt
import re
import scipy.signal as signal


def low_pass_filter(signal_data, cutoff_freq, sampling_rate, order=5):
    """
    Apply a low-pass Butterworth filter to remove high-frequency oscillations.

    Parameters:
    - signal_data: 1D numpy array, the signal to be filtered
    - cutoff_freq: float, the cutoff frequency for the filter (Hz)
    - sampling_rate: float, the sampling rate of the signal (Hz)
    - order: int, order of the Butterworth filter (default is 5)

    Returns:
    - filtered_signal: 1D numpy array, the filtered signal
    """
    nyquist = 0.5 * sampling_rate  # Nyquist frequency
    normal_cutoff = cutoff_freq / nyquist  # Normalized cutoff frequency
    b, a = signal.butter(order, normal_cutoff, btype='low', analog=False)
    filtered_signal = signal.filtfilt(b, a, signal_data)
    return filtered_signal


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
    pic_path = dir_name + '/yaw_action_plots_smoothed/'
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

    colours = ['red', 'green', 'blue']

    for episode_number in range(1, num_episodes+1):
        for env_number in range(1, num_envs+1):
            alpha_arr = data[f'alphas_ENV_{env_number}_EPISODE_{episode_number}']
            fig, axes = plt.subplots(1, 1, figsize=(12, 5))
            cutoff_freq = 0.003
            for turb in range(alpha_arr.shape[-1]):
                axes.plot(alpha_arr[:, turb], label=f"Turbine {turb+1}", alpha=0.5, color=colours[turb])
                alpha_filtered = low_pass_filter(alpha_arr[:, turb], sampling_rate=0.1, cutoff_freq=cutoff_freq)
                axes.plot(alpha_filtered, label=f"Turbine {turb + 1} filtered", color=colours[turb])
            axes.legend()
            axes.set_xlabel('RL frames')
            axes.grid(True)
            axes.set_ylabel('Turbine yaw angles (degrees)')
            fig.suptitle(f"Environment {env_number} | Episode {episode_number} | Sampling freq 0.1Hz | Filter freq {cutoff_freq}Hz")
            plt.tight_layout()
            plt.savefig(pic_path + f'action_angle_plot_ENV{env_number}_EP{episode_number}.png')
            plt.close()

    print(f"Saved plots to {os.path.abspath(pic_path)}.")

    


