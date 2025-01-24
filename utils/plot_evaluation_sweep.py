import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import fonts
import os
import sys
import glob
from pprint import pprint as pp


# field = 'gamma'

defaults = {'gamma': 0.95,
            'alpha_init': 1,
            # 'max_yaw_speed': 1.0,
            # 'frame_stack': 5,
            'max_episode_length': 500,
            'lr': 3e-4,
            'frames_per_batch': 256,
            'probes_per_turbine': 60}
            # 'entropy_lr': 60}

for field in defaults.keys():
    print(f'Plotting sweep of {field}')

    fig, axs = plt.subplots(2, 1,
                            figsize=(6, 8),
                            constrained_layout=True,
                            sharex='col')

    cases = [1]
    values=[defaults[field]]

    # load arguments list
    file_path = "../sa_sac_array/array_arguments.txt"
    with open(file_path, "r") as file:
        for line_n, line in enumerate(file):
            line = line.strip()
            if line and "=" in line:  # Ensure the line is not empty and contains '='
                key, value = line.split("=", 1)  # Split at the first '='
                try:
                    value = int(value)
                except ValueError:
                    try:
                        value = float(value)
                    except ValueError:
                        print(f'{value} is undertermined argument value type')

                stripped_key = key.split(".", 1)[-1]  # Keep only the part after the first '.'
                if stripped_key == field:
                    values.append(value)
                    cases.append(line_n+1)

    # plots
    # base case
    pp(cases)
    results = []
    for case in cases:
        file_path = f"../sa_sac_array/evaluation/training_sasac_array_{case}_eval_*/mean_error.csv"
        try:
            file = glob.glob(file_path)[0]
            df = pd.read_csv(file)
            data = dict(zip(df["Label"], df["Value"]))
        except:
            data = ('EMPTY', 'EMPTY')

        results.append(data)

    base_file = '../base_zero/mean_error.csv'
    df = pd.read_csv(base_file)
    base_data = dict(zip(df["Label"], df["Value"]))
    base_mean = base_data["mean of power"]
    base_std = base_data["std of power"]

    means = [item["mean of power"]/base_mean for item in results]
    mean_error = [item["stderror of mean"]/base_mean for item in results]
    stds = [item["std of power"]/base_std for item in results]
    stds_error = [item["stderror of std"]/base_std for item in results]

    axs[0].errorbar(values, means, mean_error,
                    fmt='o',  # Marker style
                    color='gray',  # Line and marker color
                    ecolor='k',  # Error bar color
                    elinewidth=1,  # Width of error bar lines
                    capsize=5,  # Error bar cap size
                    capthick=1,  # Cap thickness
                    markersize=8,  # Marker size
                    )
    axs[0].set_xlabel(field)
    axs[0].set_ylabel('Mean Power / Base Power')

    axs[1].errorbar(values, stds, stds_error,
                    fmt='o',  # Marker style
                    color='gray',  # Line and marker color
                    ecolor='k',  # Error bar color
                    elinewidth=1,  # Width of error bar lines
                    capsize=5,  # Error bar cap size
                    capthick=1,  # Cap thickness
                    markersize=8,  # Marker size
                    )
    axs[1].set_xlabel(field)
    axs[1].set_ylabel('Std Power / Base Std Power')
    fig.savefig(f"../sa_sac_array/evaluation/{field}.png")
    axs[0].set_xscale('log')
    axs[1].set_xscale('log')
    fig.savefig(f"../sa_sac_array/evaluation/{field}_log.png")
