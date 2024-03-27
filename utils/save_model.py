import pickle
import torch
from torchrl.envs.transforms import VecNorm


def save_model(env, actor, critic, filepath, id):
    # Get the env transforms
    transforms = env.transform
    # The only thing we need to remember is the VecNorm
    norm_dict = {}
    for t in transforms:
        if isinstance(t, VecNorm):
            obs_norm = t.to_observation_norm()
            assert len(obs_norm.in_keys) == 1
            key = obs_norm.in_keys[0]
            loc = obs_norm.loc
            scale = obs_norm.scale
            norm_dict[key] = {'loc': loc, 'scale': scale}
    
    # Save env transforms
    with open(filepath+'env_transforms' + f"_{id}" + '.pkl', 'wb') as file:
        pickle.dump(norm_dict, file)
    # Save model
    with open(filepath+'actor' + f"_{id}" + '.pkl', 'wb') as file:
        torch.save(actor.state_dict(), file)
    with open(filepath+'critic' + f"_{id}" + '.pkl', 'wb') as file:
        torch.save(critic.state_dict(), file)

    return True

