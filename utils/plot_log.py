from matplotlib import pyplot as plt
import hydra
import pickle
import fonts


def plot_keys(keys, data):
    n_plots = len(keys)
    steps = data['step']
    fig, axes = plt.subplots(n_plots, 1, figsize=(4, 2 * n_plots),
                             sharex=True, constrained_layout=True, squeeze=False)

    # Plot each field in a separate subplot
    for i, key in enumerate(keys):
        axes[i][0].plot(steps, data[key])
        label = key.replace('train/', '')
        axes[i][0].set_ylabel(label.replace('_', ' '))

    # Set the label for the common x-axis
    axes[-1][0].set_xlabel('Step')
    axes[-1][0].set_xlim(steps[0], steps[-1])
    fig.savefig(f'./training_ppo/ppo/{label}.pdf')


@hydra.main(config_path="../training_ppo/ppo/outputs/hydra_logs", config_name="config", version_base="1.2")
def main(cfg: "DictConfig"):

    logfile = './training_ppo/ppo/outputs/logs.pkl'
    with open(logfile, 'rb') as file:
        data = pickle.load(file)

    print(len(data.keys()))
    print(data.keys())

    loss_keys = [key for key in data if key.startswith('train/loss_')]
    time_keys = [key for key in data if key.endswith('_time')]
    episode_keys = [key for key in data if 'episode' in key]
    lr_keys = [key for key in data if 'lr' in key]

    plot_keys(loss_keys, data)
    plot_keys(time_keys, data)
    plot_keys(episode_keys, data)
    plot_keys(lr_keys, data)


if __name__ == "__main__":
    main()
