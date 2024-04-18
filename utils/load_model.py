import pickle
from hydra import compose, initialize
import torch
from utils.save_model import save_model
from ppo.utils_ppo import make_env, make_parallel_env
from ppo.utils_ppo import make_ma_ppo_models
from ppo.utils_ppo import add_env_transforms


def compare_model_parameters(model1, model2):
    result = True
    for a, b in zip(model1.parameters(), model2.parameters()):
        if not torch.all(a == b):
            result = False
    return result


def load_model(env_params, filepath, id, dummy_update=False):
    try:
        # Load env transforms
        with open(filepath + 'env_transforms' + f"_{id}" + '.pkl', 'rb') as file:
            transforms_params = pickle.load(file)
        # Load model parameters
        with open(filepath + 'actor' + f"_{id}" + '.pkl', 'rb') as file:
            actor_params = torch.load(file)
        with open(filepath + 'critic' + f"_{id}" + '.pkl', 'rb') as file:
            critic_params = torch.load(file)
    except FileNotFoundError:
        print(f"File {filepath}env_transforms_{id}.pkl or {filepath}model_{id}.pkl has not been found.")
        return False

    device = "cpu" if not torch.cuda.device_count() else "cuda"
    # Build the env without transforms
    # Since the purpose of loading a trained model is to test, we only build a single env
    env = make_env(
        env_params,
        instance='TestEnv',
        save=True,
        device=device,
        dummy_update=dummy_update,
        add_transforms=False,
    )

    # Rebuild the Transforms, but replacing the VecNorm with an ObservationNorm
    env = add_env_transforms(env, obs_norm_params=transforms_params)

    # Instantiating the model with random params
    actor, critic = make_ma_ppo_models(env_params, dummy_update=True)
    actor, critic = actor.to(device), critic.to(device)
    # Inserting the loaded parameters
    actor.load_state_dict(actor_params)
    critic.load_state_dict(critic_params)

    return env, actor, critic


if __name__ == '__main__':
    # Test the save model function
    env_params = None
    env = make_parallel_env(env_params, 3, dummy_update=True)
    test_env = make_env(env_params, dummy_update=True)
    actor, critic = make_ma_ppo_models(env_params, dummy_update=True)
    env.reset()
    env.rollout(100)

    save_model(test_env, actor, critic, './testing/', 0)

    loaded_env, loaded_actor, loaded_critic = load_model(None, './testing/', 0, dummy_update=True)

    # Test if loaded model is identical
    actor_the_same = compare_model_parameters(actor, loaded_actor)
    critic_the_same = compare_model_parameters(critic, loaded_critic)
    print(f"Loaded model and model are identical? Actor: {actor_the_same}, Critic: {critic_the_same} \n")

    print("Environment transforms of original env:")
    print(env.transform)
    print("Environment transforms of loaded env:")
    print(loaded_env.transform)

