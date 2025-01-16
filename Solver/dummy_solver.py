from smartredis import Client
import numpy as np
import time
from hydra import initialize, compose
import os
import logging


# Check if we are running in training or eval mode
# If in training mode, there will be a "ppo" or "sac" directory
# If in eval mode, there will be a "eval" directory (and no "ppo" or "sac" directory)
if os.path.isdir('../eval'):
    config_dir = '../eval'
    training = False
    algo = 'eval'
    logging.info('Running dummy solver in evaluation mode.')
elif os.path.isdir('../ppo'):
    config_dir = '../ppo'
    training = True
    algo = 'ppo'
    logging.info('Running dummy solver in training mode (PPO).')
elif os.path.isdir('../sac'):
    config_dir = '../sac'
    training = True
    algo = 'sac'
    logging.info('Running dummy solver in training mode (SAC).')
else:
    raise RuntimeError(f"Did not find either ppo, sac or eval directory.")

initialize(config_path=config_dir, version_base="1.2")
cfg = compose(config_name=f"config_{algo}.yaml")

n_turbines = cfg.env.turbines
total_probes = cfg.env.turbines * cfg.env.probes_per_turbine
n_envs = cfg.env.n_parallel if training else cfg.eval.n_parallel  # use correct env number

client = Client(address=None, cluster=False)
instances = [i+1 for i in range(n_envs)]

print(f"Creating {n_envs} dummy solvers.")

while True:
    # only for testing this file on its own
    # client.put_tensor(f'{instance}_yaws_done', np.array([1]))
    # Check if the file exists - idle until it does
    while not all([client.poll_key(f'{i}_yaws_done', 100, 10) for i in instances]):
        # print(f"Waiting for yaws done to be created")
        continue

    # Check if the yaws have been written by the agent
    while not all([client.get_tensor(f'{i}_yaws_done')[0] for i in instances]):
        # print(f"Waiting for yaws to be done. Time {time.time()}")
        continue

    # Read yaws determined by agent
    yaws = client.get_tensor(f'{instances[0]}_yaws')
    # yaws = np.zeros([3])

    # print(f"Yaws are {yaws}")

    # Immediately reset all yaws_done to False after reading the yaws.
    for i in instances:
        client.put_tensor(f'{i}_yaws_done', np.array([0]))
    
    # Store the new outputs of the simulation on the database
    for i in instances:
        client.put_tensor(f'{i}_turbine_powers', 1e6 * np.random.randn(n_turbines))
        # print(f"Put tensor for key {i}_turbine_powers succeeded.")
        for j in range(total_probes):
            client.put_tensor(f"{i}_probe_{j+1}", np.random.randn(3))

    # Reset the yaw done to False
    # We might have to have a flag here that checks if ALL the solvers have
    # had a chance to read the yaws, and only then reset the yaw done flag to False.
    # Otherwise, the first solver reads the yaws, sets the flag to False and therefore
    # all other solvers will wait perpetually until the flag gets set to True again.

    # Set the sim_done flag to be true
    for i in instances:
        client.put_tensor(f'{i}_sim_done', np.array([1]))

    # Simulated wait time for computation to finish
    time.sleep(.01)

