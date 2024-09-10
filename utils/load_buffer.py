import torch

def load_buffer(path):
    return torch.load(path)


if __name__ == '__main__':
    td = load_buffer('../launch_dummy_run/ppo/checkpoints/replay_buffer_checkpoint.pt')
    print(td)

    actions = td.get(("_data", "agents", "action"))
    alphas = td.get(("_data", "agents", "alpha"))
    observation = td.get(("_data", "agents", "observation"))
    alphas = td.get(("_data", "next", "agents", "reward"))
