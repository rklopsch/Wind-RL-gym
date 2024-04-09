import pickle
import torch
from utils.vecnorm_fixed import VecNorm


def save_model(env, actor, critic, filepath, id):
    # Extract the VecNorm from the 1st environment
    # Note this is very suboptimal code and hacky
    _sum = env.state_dict()["worker0"]["transforms.3._extra_state"]["td"]["agents_observation_sum"]
    _count = env.state_dict()["worker0"]["transforms.3._extra_state"]["td"]["agents_observation_count"]
    _ssq = env.state_dict()["worker0"]["transforms.3._extra_state"]["td"]["agents_observation_ssq"]

    norm_dict = {}
    mean = _sum / _count
    std = (_ssq / _count - mean.pow(2)).clamp_min(1e-4).sqrt()
    norm_dict[("agents", "observation")] = {'loc': mean, 'scale': std}

    # Save env transforms
    with open(filepath+'env_transforms' + f"_{id}" + '.pkl', 'wb') as file:
        pickle.dump(norm_dict, file)
    # Save model
    with open(filepath+'actor' + f"_{id}" + '.pkl', 'wb') as file:
        torch.save(actor.state_dict(), file)
    with open(filepath+'critic' + f"_{id}" + '.pkl', 'wb') as file:
        torch.save(critic.state_dict(), file)

    return True

