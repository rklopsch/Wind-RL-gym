#!/usr/bin/env python3
"""Plot mean episode farm power across parallel training environments.

Usage:
    python plotting/plot_episode_power.py training_sac
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

# this is the greedy baseline value for 3 turbines. If we change the configuration we need to rerun a greedy baseline to get this.
BASE_POWER = 3.21075e6

run_dir = Path(sys.argv[1]).resolve()
config = yaml.safe_load((run_dir / "sac" / "config_sac.yaml").read_text())
n_turbines = config["env"]["turbines"]
n_parallel = config["env"]["n_parallel"]
reset_frames = config["env"]["reset_frames"]
initial_reset_frames = config["env"]["initial_reset_frames"]
episode_length = config["collector"]["max_episode_length"]

# sac.py does one reset, then initial_reset_frames // reset_frames more resets.
# Training then alternates: episode, reset, episode, reset, ...
start = (1 + initial_reset_frames // reset_frames) * reset_frames
stride = episode_length + reset_frames

curves = []
# iterate over each parallel environment
for env_dir in run_dir.glob("WindFarm_*"):
    # read power data for each turbine in this environment
    powers = [
        pd.read_csv(env_dir / f"disc{t}.adm", sep=r"\s+|,\s*", engine="python")["Power"].to_numpy()
        for t in range(1, n_turbines + 1)
    ]
    # truncate all power arrays to the same length
    n_rows = min(len(p) for p in powers)
    powers = [p[:n_rows] for p in powers]
    # compute total farm power for this environment
    farm_power = np.sum(powers, axis=0)
    # identify the starting indices for each episode within this environment
    starts = range(start, n_rows - episode_length + 1, stride)
    curves.append([farm_power[s : s + episode_length].mean() for s in starts])

n_episodes = min(len(c) for c in curves)
power = np.array([c[:n_episodes] for c in curves])
n_envs = len(curves)

episodes = np.arange(1, n_episodes + 1)
# compute mean and confidence intervals across environments
mean = power.mean(axis=0)
std = power.std(axis=0, ddof=1)
stderr = std / np.sqrt(n_envs)
# 95% confidence interval for the mean across environments
ci = 1.96 * stderr

fig, ax = plt.subplots(figsize=(8, 4.5), constrained_layout=True)
ax.plot(episodes * n_parallel, (mean / BASE_POWER - 1) * 100, color="tab:blue", linewidth=2.0,
        label=f"Mean across {n_envs} environments")
ax.fill_between(episodes * n_parallel, ((mean - ci) / BASE_POWER - 1) * 100, ((mean + ci) / BASE_POWER - 1) * 100,
                color="tab:blue", alpha=0.2, label="95% confidence interval")
ax.set_xlabel("Episode")
ax.set_ylabel("power gain (%)")
ax.set_title("Mean episode power production")
ax.grid(True, alpha=0.3)
ax.legend()
fig.savefig(run_dir / "mean_episode_power.png", dpi=200)

pd.DataFrame({
    "episode": episodes,
    "mean_farm_power_mw": mean,
    "std_across_environments_mw": std,
    "stderr_across_environments_mw": stderr,
    "environment_count": n_envs,
}).to_csv(run_dir / "mean_episode_power.csv", index=False)
