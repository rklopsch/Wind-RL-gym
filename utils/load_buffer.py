import torch
import matplotlib.pyplot as plt

def load_buffer(path):
    return torch.load(path)


if __name__ == '__main__':
    td = load_buffer('../launch_dummy_run/ppo/checkpoints/replay_buffer_checkpoint.pt')
    # print(td)

    actions = td.get(("_data", "agents", "action")).squeeze()
    alphas = td.get(("_data", "agents", "alpha")).squeeze()
    observation = td.get(("_data", "agents", "observation"))
    rewards = td.get(("_data", "next", "agents", "reward")).squeeze()

    print(rewards.shape)
    for i in range(rewards.shape[1]):
        plt.plot(rewards[:, i, :].mean(dim=-1), label=f"Env {i+1}")

