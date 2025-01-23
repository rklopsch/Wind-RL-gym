import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import sys


def compute_ema(data, alpha):
    ema = np.zeros_like(data)
    ema[0] = data[0]  # Initialize EMA with the first value
    for t in range(1, len(data)):
        ema[t] = alpha * data[t] + (1 - alpha) * ema[t - 1]
    return ema


def load_data_single_environment(path, burnin):
    # Important: burnin is measured in RL frames, not seconds.
    means = {}
    stds = {}
    farm_power = []
    colors = ['red', 'green', 'blue']
    for turbine in range(3):
        file = path + f'/disc{turbine + 1}.adm'
        dat = pd.read_csv(file, sep="\s+|, ", engine='python')
        power = dat.Power_ave / 1e6
        power = np.asarray(power[burnin:])
        times = np.asarray(dat.Time[burnin:])
        means[f'Turbine {turbine + 1}'] = power.mean()
        stds[f'Turbine {turbine + 1}'] = power.std()
        farm_power.append(power)

        # Plot power signal just for testing
        plt.plot(times, power, alpha=0.2, color=colors[turbine])
        plt.plot(times, compute_ema(power, alpha=0.01), color=colors[turbine], label=f'Turbine {turbine+1}')

    farm_power = np.asarray(farm_power)
    farm_power = np.sum(farm_power, axis=0)
    means['Farm'] = farm_power.mean()
    stds['Farm'] = farm_power.std()

    plt.ylabel('Power in MW')
    plt.xlabel('Time in seconds (from start of solver)')
    plt.legend()
    plt.savefig(path + f'/powers.png')
    plt.close()

    return means, stds


def compute_stderror(k, arr, iterations=10000):
    l = []
    for rep in range(iterations):
        l.append(np.mean(np.random.choice(arr, size=k, replace=True)))
    std_k = np.std(l)
    return std_k


def compute_power_mean(base_directory, burnin):
    farm_means = []
    farm_stds = []
    fluctuations = []
    # Iterate over all directories in the given directory
    for entry in os.listdir(base_directory):
        entry_path = os.path.join(base_directory, entry)
        if entry.startswith('Wind') and os.path.isdir(entry_path):  # Check if the entry is a directory
            print(f"Getting data from directory: {entry_path}")
            means, stds = load_data_single_environment(entry_path, burnin)
            farm_means.append(means['Farm'])
            farm_stds.append(stds['Farm'])
            fluctuations.append(2*stds['Farm']/means['Farm'])

    overall_mean = np.mean(farm_means)
    overall_std = np.mean(farm_stds)
    overall_stderror_mean = np.std(farm_means) / np.sqrt(len(farm_means))
    overall_stderror_std = compute_stderror(len(farm_stds), farm_stds)

    ks = []
    std_k_list = []
    for k in range(1, len(farm_means) + 1):
        std_k = compute_stderror(k, farm_means)
        normalised_std_k = std_k / np.mean(farm_means)
        ks.append(k)
        std_k_list.append(normalised_std_k)
    std_k_list = np.asarray(std_k_list)
    plt.plot(ks, 100 * std_k_list)
    plt.ylabel(f"Standard deviation of mean power normalised by mean in %")
    plt.xlabel(f"Number of environments used to compute mean")
    plt.savefig(base_directory + '/std_by_num_envs.png')

    return overall_mean, overall_std, overall_stderror_mean, overall_stderror_std


if __name__ == '__main__':
    # Check if any command-line arguments are provided
    if len(sys.argv) > 1:
        path = sys.argv[1]  # The first command-line argument
    else:
        raise Exception("No directory to run data provided. Provide this as the first argument to the script.")

    # for testing
    # path = '../outputs/base_zero_yaw/eval_zero_24-11-24'
    burnin = 5100  # in RL frames

    # Compute metrics
    mean, std, mean_stderr, std_stderr = compute_power_mean(path, burnin)

    # Output and write to disk
    print(f"Using {burnin} RL frames as burn in. Check that this is correct!")
    print(f"Mean power {mean:.5f} MW (standard error {100*mean_stderr/mean:.2f}%) | Std of power {std:.5f} MW (standard error {100*std_stderr/std:.2f}%)")
    df = pd.DataFrame({'Label': ['mean of power', 'std of power', 'stderror of mean', 'stderror of std'], 'Value': [mean, std, mean_stderr, std_stderr]})
    df.to_csv(path + '/mean_error.csv', index=False)
