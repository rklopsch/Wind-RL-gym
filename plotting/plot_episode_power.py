#!/usr/bin/env python3
"""Plot mean episode farm power across parallel training environments.

The script reads ``WindFarm_*/disc*.adm`` files from a SmartSim run directory,
splits the training portion into episodes using ``sac/config_sac.yaml``, sums the
turbine powers to obtain farm power, and averages episode means over all
parallel environments.

Example:
    python plotting/plot_episode_power.py training_sac
"""

import argparse
import csv
import os
import re
import tempfile
from pathlib import Path

# Matplotlib must see a writable cache directory before it is imported. This is
# useful on systems where $HOME is read-only or unavailable on compute nodes.
os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / f"matplotlib-{os.getuid()}"))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot mean episode farm power averaged over parallel environments."
    )
    parser.add_argument(
        "run_directory",
        type=Path,
        help="SmartSim run directory containing WindFarm_* directories (for example, training_sac).",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="SAC config file. Defaults to <run_directory>/sac/config_sac.yaml.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output plot path. Defaults to <run_directory>/mean_episode_power.png.",
    )
    parser.add_argument(
        "--csv-output",
        type=Path,
        default=None,
        help="Output CSV path. Defaults to <run_directory>/mean_episode_power.csv.",
    )
    parser.add_argument(
        "--show-environments",
        action="store_true",
        help="Also draw each environment's episode-power curve with low opacity.",
    )
    parser.add_argument(
        "--no-confidence-band",
        action="store_true",
        help="Do not draw the 95%% confidence interval across environments.",
    )
    return parser.parse_args()


def environment_number(path):
    match = re.fullmatch(r"WindFarm_(\d+)", path.name)
    return int(match.group(1)) if match else None


def find_environment_directories(run_directory):
    environments = []
    for path in run_directory.glob("WindFarm_*"):
        number = environment_number(path)
        if number is not None and path.is_dir():
            environments.append((number, path))
    environments.sort(key=lambda item: item[0])
    if not environments:
        raise FileNotFoundError(f"No WindFarm_* directories found in {run_directory}")
    return environments


def load_config(config_path):
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    try:
        env = config["env"]
        collector = config["collector"]
        return {
            "n_turbines": int(env["turbines"]),
            "n_parallel": int(env["n_parallel"]),
            "initial_reset_frames": int(env["initial_reset_frames"]),
            "reset_frames": int(env["reset_frames"]),
            "episode_length": int(collector["max_episode_length"]),
        }
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"Could not read required settings from {config_path}: {exc}") from exc


def read_turbine_power(path):
    data = pd.read_csv(path, sep=r"\s+|,\s*", engine="python")
    if "Power_ave" not in data.columns:
        raise ValueError(f"{path} does not contain a Power_ave column")
    return data["Power_ave"].to_numpy(dtype=np.float64) / 1e6


def load_farm_power(environment_directory, n_turbines):
    turbine_power = []
    for turbine in range(1, n_turbines + 1):
        path = environment_directory / f"disc{turbine}.adm"
        if not path.exists():
            raise FileNotFoundError(f"Missing turbine power file: {path}")
        turbine_power.append(read_turbine_power(path))

    shortest = min(len(values) for values in turbine_power)
    if shortest == 0:
        raise ValueError(f"Empty turbine power data in {environment_directory}")
    if any(len(values) != shortest for values in turbine_power):
        print(
            f"Warning: turbine files in {environment_directory} have different lengths; "
            f"truncating to {shortest} rows."
        )

    return np.sum(np.stack([values[:shortest] for values in turbine_power]), axis=0)


def extract_episode_means(farm_power, initial_reset_frames, reset_frames, episode_length):
    # sac.py performs one reset before the initial-reset loop, followed by
    # floor(initial_reset_frames / reset_frames) additional resets.
    initial_reset_total = (1 + initial_reset_frames // reset_frames) * reset_frames
    cursor = initial_reset_total
    episode_means = []

    # Training order is: episode, automatic reset, episode, automatic reset, ...
    while cursor + episode_length <= len(farm_power):
        episode = farm_power[cursor : cursor + episode_length]
        episode_means.append(float(np.mean(episode)))
        cursor += episode_length + reset_frames

    return np.asarray(episode_means, dtype=np.float64)


def save_csv(path, episode_numbers, mean_power, std_power, stderr_power, environment_count):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "episode",
                "mean_farm_power_mw",
                "std_across_environments_mw",
                "stderr_across_environments_mw",
                "environment_count",
            ]
        )
        for episode, mean, std, stderr in zip(
            episode_numbers, mean_power, std_power, stderr_power
        ):
            writer.writerow([episode, mean, std, stderr, environment_count])


def main():
    args = parse_args()
    run_directory = args.run_directory.resolve()
    config_path = args.config or run_directory / "sac" / "config_sac.yaml"
    output_path = args.output or run_directory / "mean_episode_power.png"
    csv_path = args.csv_output or run_directory / "mean_episode_power.csv"

    config = load_config(config_path)
    environments = find_environment_directories(run_directory)
    if len(environments) != config["n_parallel"]:
        print(
            f"Warning: config specifies {config['n_parallel']} environments, "
            f"but found {len(environments)} WindFarm_* directories."
        )

    curves = []
    labels = []
    for number, directory in environments:
        farm_power = load_farm_power(directory, config["n_turbines"])
        episode_means = extract_episode_means(
            farm_power=farm_power,
            initial_reset_frames=config["initial_reset_frames"],
            reset_frames=config["reset_frames"],
            episode_length=config["episode_length"],
        )
        if episode_means.size == 0:
            print(f"Warning: no complete training episodes found for WindFarm_{number}; skipping it.")
            continue
        curves.append(episode_means)
        labels.append(f"Environment {number}")

    if not curves:
        raise RuntimeError("No complete episodes were found in any environment.")

    common_episode_count = min(len(curve) for curve in curves)
    if any(len(curve) != common_episode_count for curve in curves):
        print(
            "Warning: environments contain different numbers of complete episodes; "
            f"plotting the first {common_episode_count} episodes shared by all environments."
        )

    power_by_environment = np.stack(
        [curve[:common_episode_count] for curve in curves], axis=0
    )
    episode_numbers = np.arange(1, common_episode_count + 1)
    mean_power = np.mean(power_by_environment, axis=0)
    std_power = np.std(power_by_environment, axis=0, ddof=1) if len(curves) > 1 else np.zeros_like(mean_power)
    stderr_power = std_power / np.sqrt(len(curves))

    fig, ax = plt.subplots(figsize=(8, 4.5), constrained_layout=True)
    if args.show_environments:
        for label, curve in zip(labels, power_by_environment):
            ax.plot(episode_numbers, curve, color="tab:blue", alpha=0.15, linewidth=0.8, label="_nolegend_")

    ax.plot(
        episode_numbers,
        mean_power,
        color="tab:blue",
        linewidth=2.0,
        label=f"Mean across {len(curves)} environments",
    )
    if not args.no_confidence_band and len(curves) > 1:
        confidence = 1.96 * stderr_power
        ax.fill_between(
            episode_numbers,
            mean_power - confidence,
            mean_power + confidence,
            color="tab:blue",
            alpha=0.2,
            label="95% confidence interval",
        )

    ax.set_xlabel("Episode")
    ax.set_ylabel("Mean farm power production (MW)")
    ax.set_title("Mean episode power production")
    ax.grid(True, alpha=0.3)
    ax.legend()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)

    save_csv(
        csv_path,
        episode_numbers,
        mean_power,
        std_power,
        stderr_power,
        len(curves),
    )
    print(f"Plotted {common_episode_count} complete episodes from {len(curves)} environments.")
    print(f"Saved plot to {output_path}")
    print(f"Saved data to {csv_path}")


if __name__ == "__main__":
    main()
