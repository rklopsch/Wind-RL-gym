import numpy as np
import pickle
import sys
import os
import matplotlib.pyplot as plt
import re


# We define the base power output here
BASE_POWER = 3.21075


def find_num_eps_envs(data):
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
    return num_episodes, num_envs


def extract_mean_powers_test(data):
    mean_powers = []
    n_envs = 16
    for ev in range(1, n_envs+1):
        powers = data[f"power_ENV_{ev}"]
        ep_power = powers.mean()
        ep_power *= 3
        mean_powers.append(ep_power)
    return np.asarray(mean_powers)


if __name__ == '__main__':
    with open('active_test/fixed_actions_logs.pkl', 'rb') as f:
        test_data = pickle.load(f)
    test_powers = extract_mean_powers_test(test_data)


    # absolute
    m = test_powers.mean()
    stde = test_powers.std() / np.sqrt(test_powers.shape[0])
    l = m - 1.96 * stde
    u = m + 1.96 * stde
    print(f"Absolute power (MW) active TEST: MEAN {m:.5f} ({l:.5f}-{u:.5f})")

    # relative
    print(f"active TEST: {100*(m/BASE_POWER-1):.2f}% ({100*(l/BASE_POWER-1):.2f}% - {100*(u/BASE_POWER-1):.2f}%)")


