#!/usr/bin/env python3
"""Plot how the turbine yaw angles evolve over training across parallel environments.

Produces a four-panel figure:
  (a) env-mean yaw trajectory of every training episode, concatenated
  (b) settled yaw per episode (mean over the second half of the episode),
      with bars showing the within-episode temporal std (actuation jitter)
  (c) within-episode yaw profiles of the last few episodes (settling behaviour)
  (d) mean farm power gain per episode, relative to the greedy baseline

Usage:
    python plotting/plot_yaw_evolution.py training_sac
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

# greedy (zero-yaw) baseline for 3 turbines: mean farm power over frames 3000-5100 of the initial
# zero-yaw warm-up in training_sac, averaged over all 32 environments. Recompute if the configuration changes.
BASE_POWER = 3.38031e6
# column indices in disc*.adm: i, Time, UF, Power, Thrust, Udisc_ave, Power_ave, Thrust_ave, alpha, YawAng, TiltAng
POWER_COL, YAW_COL = 3, 9
# number of most recent episodes shown in the within-episode panel
N_RECENT = 3
COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]

run_dir = Path(sys.argv[1]).resolve()
config = yaml.safe_load((run_dir / "sac" / "config_sac.yaml").read_text())
n_turbines = config["env"]["turbines"]
n_parallel = config["env"]["n_parallel"]
reset_frames = config["env"]["reset_frames"]
initial_reset_frames = config["env"]["initial_reset_frames"]
max_yaw_angle = config["env"]["max_yaw_angle"]
episode_length = config["collector"]["max_episode_length"]
init_random_frames = config["collector"]["init_random_frames"]

# sac.py does one reset, then initial_reset_frames // reset_frames more resets.
# Training then alternates: episode, reset, episode, reset, ...
start = (1 + initial_reset_frames // reset_frames) * reset_frames
stride = episode_length + reset_frames
# second half of each episode is used as the "settled" window
settle = episode_length // 2

yaws, powers = [], []
# iterate over each parallel environment
for env_dir in sorted(run_dir.glob("WindFarm_*"), key=lambda p: int(p.name.split("_")[1])):
    data = [
        pd.read_csv(env_dir / f"disc{t}.adm", sep=r"\s+", skiprows=1, header=None, usecols=[POWER_COL, YAW_COL])
        for t in range(1, n_turbines + 1)
    ]
    # truncate all turbines to the same length
    n_rows = min(len(d) for d in data)
    yaw = np.array([d[YAW_COL].to_numpy()[:n_rows] for d in data])
    farm_power = np.sum([d[POWER_COL].to_numpy()[:n_rows] for d in data], axis=0)
    # slice out each episode: (episode, turbine, frame)
    starts = range(start, n_rows - episode_length + 1, stride)
    yaws.append(np.array([yaw[:, s : s + episode_length] for s in starts]))
    powers.append(np.array([farm_power[s : s + episode_length].mean() for s in starts]))

n_episodes = min(len(y) for y in yaws)
yaw = np.array([y[:n_episodes] for y in yaws])  # (env, episode, turbine, frame)
power = np.array([p[:n_episodes] for p in powers])  # (env, episode)
n_envs = len(yaws)

episodes = np.arange(1, n_episodes + 1)
# episodes before this were collected with uniformly random actions
n_random = init_random_frames / (n_parallel * episode_length)

settled = yaw[..., settle:]
settled_mean = settled.mean(axis=-1)  # (env, episode, turbine)
mean_yaw = settled_mean.mean(axis=0)
# spread of the settled yaw across environments
env_std = settled_mean.std(axis=0, ddof=1)
# temporal fluctuation of yaw within the settled window
jitter = settled.std(axis=-1).mean(axis=0)
# mean absolute yaw change per frame
activity = np.abs(np.diff(settled, axis=-1)).mean(axis=(0, -1))
gain = (power.mean(axis=0) / BASE_POWER - 1) * 100
gain_ci = 1.96 * power.std(axis=0, ddof=1) / np.sqrt(n_envs) / BASE_POWER * 100

fig, axes = plt.subplots(2, 2, figsize=(13, 8), constrained_layout=True)
ax_traj, ax_settled, ax_recent, ax_power = axes.flat

# (a) concatenated env-mean trajectories, x in units of episodes across all environments
frame_x = (np.arange(n_episodes)[:, None] + np.arange(episode_length)[None, :] / episode_length) * n_parallel
for t in range(n_turbines):
    traj = yaw[:, :, t, :].mean(axis=0)
    for k in range(n_episodes):
        ax_traj.plot(frame_x[k], traj[k], color=COLORS[t], linewidth=0.8,
                     label=f"Turbine {t + 1}" if k == 0 else None)
ax_traj.axvspan(0, n_random * n_parallel, color="0.5", alpha=0.12, label="Random actions")
ax_traj.set_title(f"(a) Yaw trajectories, mean across {n_envs} environments")
ax_traj.set_xlabel("Episode")

# (b) settled yaw per episode
offsets = (np.arange(n_turbines) - (n_turbines - 1) / 2) * 0.12 * n_parallel
for t in range(n_turbines):
    ax_settled.errorbar(episodes * n_parallel + offsets[t], mean_yaw[:, t], yerr=jitter[:, t], color=COLORS[t],
                        marker="o", markersize=5, linewidth=2.0, capsize=3, label=f"Turbine {t + 1}")
ax_settled.axvspan(0, (n_random + 0.5) * n_parallel, color="0.5", alpha=0.12)
ax_settled.set_title("(b) Settled yaw (second half of episode), bars: within-episode std")
ax_settled.set_xlabel("Episode")

# (c) within-episode profiles of the most recent episodes
recent = range(max(0, n_episodes - N_RECENT), n_episodes)
styles = [":", "--", "-"][-len(recent):]
for t in range(n_turbines):
    for k, ls in zip(recent, styles):
        ax_recent.plot(np.arange(episode_length), yaw[:, k, t, :].mean(axis=0), color=COLORS[t], linestyle=ls,
                       linewidth=1.2, label=f"Episode {(k + 1) * n_parallel}" if t == 0 else None)
ax_recent.axvline(settle, color="0.5", linewidth=1.0)
ax_recent.set_title(f"(c) Within-episode yaw, last {len(recent)} episodes (colour = turbine)")
ax_recent.set_xlabel("Frame within episode")

for ax in (ax_traj, ax_settled, ax_recent):
    ax.axhline(0, color="0.3", linewidth=0.8)
    ax.set_ylim(-max_yaw_angle - 3, max_yaw_angle + 3)
    ax.set_ylabel("Yaw angle (deg)")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, loc="lower left")

# (d) power gain per episode
ax_power.plot(episodes * n_parallel, gain, color="#2a78d6", marker="o", markersize=5, linewidth=2.0,
              label=f"Mean across {n_envs} environments")
ax_power.fill_between(episodes * n_parallel, gain - gain_ci, gain + gain_ci, color="#2a78d6", alpha=0.2,
                      label="95% confidence interval")
ax_power.axhline(0, color="0.3", linewidth=0.8)
ax_power.set_title("(d) Mean episode power gain over greedy baseline")
ax_power.set_xlabel("Episode")
ax_power.set_ylabel("Power gain (%)")
ax_power.grid(True, alpha=0.3)
ax_power.legend(fontsize=8)

fig.savefig(run_dir / "yaw_evolution.png", dpi=200)

stats = {"episode": episodes * n_parallel}
for t in range(n_turbines):
    stats[f"settled_yaw_t{t + 1}_deg"] = mean_yaw[:, t]
    stats[f"env_std_t{t + 1}_deg"] = env_std[:, t]
    stats[f"jitter_std_t{t + 1}_deg"] = jitter[:, t]
    stats[f"mean_abs_dyaw_t{t + 1}_deg"] = activity[:, t]
stats["power_gain_pct"] = gain
pd.DataFrame(stats).to_csv(run_dir / "yaw_episode_stats.csv", index=False, float_format="%.3f")
