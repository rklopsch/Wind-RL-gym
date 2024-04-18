import pickle
from hydra import compose, initialize
import torch
from utils.save_model import save_model
from ppo.utils_ppo import load_model
from ppo.utils_ppo import make_env, make_parallel_env
from ppo.utils_ppo import make_ma_ppo_models


def compare_model_parameters(model1, model2):
    result = True
    for a, b in zip(model1.parameters(), model2.parameters()):
        if not torch.all(a == b):
            result = False
    return result


if __name__ == '__main__':
    # Test the save model function
    env_params = None
    env = make_parallel_env(env_params, 3, dummy_update=True)
    test_env = make_env(env_params, dummy_update=True)
    actor, critic = make_ma_ppo_models(env_params)
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

