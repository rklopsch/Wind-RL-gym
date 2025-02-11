import os
import numpy as np
import re
from pathlib import Path
import matplotlib.pyplot as plt

BASE_POWER = 3.223854866218793
STD_POWER = 0.49171201813667925


def parse_arguments(line):
    """Parses a line from array_arguments.txt into a dictionary."""
    return dict(item.split("=") for item in line.strip().split())


def find_matching_directory(base_path, identifier):
    """Finds the directory that contains 'array_X' where X matches the identifier."""
    for directory in base_path.iterdir():
        if directory.is_dir() and f"array_{identifier}" in directory.name:
            return directory
    return None


def load_mean_error_csv(directory):
    """Loads mean_error.csv and returns its contents as a dictionary, ignoring the first line."""
    csv_path = directory / "mean_error.csv"
    if not csv_path.exists():
        return None

    error_data = {}
    with open(csv_path, "r") as f:
        next(f)  # Skip the first line (header)
        for line in f:
            label, value = line.strip().split(",", 1)
            error_data[label] = float(value)  # Convert value to float

    return error_data


def process_lines(file_path, base_path, start_line, end_line):
    """Processes specified lines from array_arguments.txt."""
    results = []

    with open(file_path, "r") as f:
        lines = f.readlines()

    for i in range(start_line, min(end_line, len(lines))):
        line = lines[i]
        arg_dict = parse_arguments(line)
        identifier = i+1  # Adjust identifier key if needed
        matching_dir = find_matching_directory(base_path, identifier)
        if matching_dir:
            mean_error_data = load_mean_error_csv(matching_dir)
            results.append({**arg_dict, **mean_error_data})
        else:
            print(f"Could not find array_{identifier} in {base_path}")

    return results


# Example usage:
base_directory = Path("../outputs/hyperparam_entropylr_alphainit/computed_means")  # Change this to the actual base directory if needed
array_arguments_file = base_directory / "array_arguments_2.txt"
start_line, end_line = 0, 14  # Adjust range as needed

results = process_lines(array_arguments_file, base_directory, start_line, end_line)

# Output results
max_power = -1.
params = None
for res in results:
    if res['mean of power']/BASE_POWER > max_power:
        max_power = res['mean of power']/BASE_POWER
        params = res
print(f"Maximum relative power {max_power:.3f} for params entrop lr = {params['optim.entropy_lr']}, alpha init = {params['optim.alpha_init']}")

# For each alpha_init, plot the mean of power as fn of entropy_lr
alpha_inits = set()
lrs = set()
for res in results:
    alpha_inits.add(res['optim.alpha_init'])
    lrs.add(float(res['optim.entropy_lr']))
alpha_inits = list(sorted(alpha_inits))
lrs = list(sorted(lrs))

print(alpha_inits, lrs)

fig, axes = plt.subplots(1,2,figsize=(10,5))
for ainit in alpha_inits:
    # find all relevant mean powers
    label = f"alpha_init={ainit}"
    x = []
    y = []
    yerr = []
    y_std = []
    y_std_err = []
    for lr in lrs:
        for res in results:
            if float(res['optim.entropy_lr']) == lr and res['optim.alpha_init'] == ainit:
                x.append(lr)
                mp = res['mean of power']/BASE_POWER
                stde = res['stderror of mean']/BASE_POWER
                y.append(mp)
                yerr.append(stde)
                y_std.append(res['std of power']/STD_POWER)
                y_std_err.append(res['stderror of std']/STD_POWER)

    axes[0].plot(x,y, label=label)
    axes[0].fill_between(x, y-1.96*np.asarray(yerr), y+1.96*np.asarray(yerr), alpha=0.3)
    axes[1].plot(x,y_std, label=label)
    axes[1].fill_between(x, y_std-1.96*np.asarray(y_std_err), y_std+1.96*np.asarray(y_std_err), alpha=0.3)
axes[0].set_xticks(lrs, lrs)
axes[0].set_xscale('log')
axes[0].legend()
axes[0].set_xlabel(f"entropy_lr")
axes[0].set_ylabel(f"normalised mean power")
axes[0].grid(True)
axes[1].set_xticks(lrs, lrs)
axes[1].set_xscale('log')
axes[1].legend()
axes[1].set_xlabel(f"entropy_lr")
axes[1].set_ylabel(f"normalised std of power")
axes[1].grid(True)
plt.savefig(f'entropy_lr_alpha_init.png')
plt.close()

