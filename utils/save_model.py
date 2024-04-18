import pickle
import torch


def save_model(env, actor, critic, filepath, id):
    # Read out loc and scale used in ObservationNorm
    norm_dict = {
        'loc': env.transform[-1].loc,
        'scale': env.transform[-1].scale,
    }
    # Save env transforms
    with open(filepath + 'env_transforms' + f"_{id}" + '.pkl', 'wb') as file:
        pickle.dump(norm_dict, file)
    # Save model
    with open(filepath + 'actor' + f"_{id}" + '.pkl', 'wb') as file:
        torch.save(actor.state_dict(), file)
    with open(filepath + 'critic' + f"_{id}" + '.pkl', 'wb') as file:
        torch.save(critic.state_dict(), file)

    return True
