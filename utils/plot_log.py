from matplotlib import pyplot as plt
import hydra
import pickle
import os
import sys
# import fonts


def plot_keys(keys, data, path='./'):
    n_plots = len(keys)
    # steps = data['step']
    fig, axes = plt.subplots(n_plots, 1, figsize=(4, 2 * n_plots),
                             sharex=True, constrained_layout=True, squeeze=False)

    # Plot each field in a separate subplot
    for i, key in enumerate(keys):
        axes[i][0].plot(data[key])
        label = key.replace('train/', '')
        axes[i][0].set_ylabel(label.replace('_', ' '))
        # if 'loss_critic' in key:
        if 'loss' in key:
            axes[i][0].set_yscale('symlog', linthresh=1)

    # Set the label for the common x-axis
    axes[-1][0].set_xlabel('Step')
    # axes[-1][0].set_xlim(steps[0], steps[-1])
    fig.savefig(os.path.join(path, f'./{label}.pdf'))


# @hydra.main(config_path="../sa_sac_array/training/training_sasac_array_17-01-24/training_sasac_array_30/sac/outputs/hydra_logs", config_name="config", version_base="1.2")
# def main(cfg: "DictConfig"):

def main():


    # Check if an argument is provided
    if len(sys.argv) < 2:
        print("Usage: python shorten_replay_buffer.py <filename_of_buffers>")
        sys.exit(1)

    for i in range(len(sys.argv)-1):

        path = sys.argv[i+1]


        logfile = os.path.join(path, 'sac/outputs/logs.pkl')
        with open(logfile, 'rb') as file:
            data = pickle.load(file)

        print(len(data.keys()))
        print(data.keys())

        # loss_keys = [key for key in data if key.startswith('train/loss_')]
        loss_keys = [key for key in data if 'loss' in key]
        time_keys = [key for key in data if key.endswith('_time')]
        episode_keys = [key for key in data if 'episode' in key]
        # lr_keys = [key for key in data if 'lr' in key]

        plot_keys(loss_keys, data, path)
        plot_keys(time_keys, data, path)
        plot_keys(episode_keys, data, path)
        # plot_keys(lr_keys, data)


if __name__ == "__main__":
    main()
