import numpy as np
import pickle
import sys
import os
import matplotlib.pyplot as plt
import re


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


def extract_mean_powers(data):
    mean_powers = []
    n_eps, n_envs = find_num_eps_envs(data)
    print(f"Found {n_eps} episodes and {n_envs} envs")
    for ep in range(1, n_eps+1):
        for ev in range(1, n_envs+1):
            powers = data[f"power_ENV_{ev}_EPISODE_{ep}"]
            ep_power = powers.mean()
            ep_power *= 3
            mean_powers.append(ep_power)
    return np.asarray(mean_powers)


if __name__ == '__main__':
    # Load best training run data
    with open('long_evals_final/RL/eval_logs.pkl', 'rb') as f:
        rl_data = pickle.load(f)
    rl_powers = extract_mean_powers(rl_data)

    with open('long_evals_final/BO/eval_logs.pkl', 'rb') as f:
        bo_data = pickle.load(f)
    bo_powers = extract_mean_powers(bo_data)

    with open('long_evals_final/zero/eval_logs.pkl', 'rb') as f:
        zero_data = pickle.load(f)
    zero_powers = extract_mean_powers(zero_data)


    # absolute
    ps = [rl_powers, bo_powers, zero_powers]
    names = ["RL", "BO", "greedy"]
    means = {}
    ls = {}
    us = {}
    for p,name in zip(ps,names):
        m = p.mean()
        stde = p.std() / np.sqrt(p.shape[0])
        l = m - 1.96 * stde
        u = m + 1.96 * stde
        print(f"Absolute power (MW) {name}: MEAN {m:.5f} ({l:.5f}-{u:.5f})")
        means[name] = m
        ls[name] = l
        us[name] = u

    # relative
    print(f"BO: {100*(means['BO']/means['greedy']-1):.2f}% ({100*(ls['BO']/means['greedy']-1):.2f}%-{100*(us['BO']/means['greedy']-1):.2f}%)")
    print(f"RL: {100*(means['RL'] / means['greedy']-1):.2f}% ({100*(ls['RL'] / means['greedy']-1):.2f}%-{100*(us['RL'] / means['greedy']-1):.2f}%)")


