from smartredis import Client
import numpy as np
import time


def dummy_solve(yaws):
    pows = np.ones(3)
    obs = np.zeros([3, 53])
    return pows, obs 

client = Client(address=None, cluster=False)
instances = [1,2]

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
    
    # Do the numerical magic
    pows_list = []
    obs_list = []
    for _ in range(50):
        pows, obs = dummy_solve(yaws)
        pows_list.append(pows)
        obs_list.append(obs)

    pows = np.stack(pows_list, axis=-1)
    obs = np.stack(obs_list, axis=-1)
    
    # Store the new outputs of the simulation on the database
    for i in instances:
        client.put_tensor(f'{i}_turbine_powers', pows)
        for j in range(75):
            client.put_tensor(f"{i}_probe_{j+1}", np.ones(3))

    # Reset the yaw done to False
    # We might have to have a flag here that checks if ALL the solvers have
    # had a chance to read the yaws, and only then reset the yaw done flag to False.
    # Otherwise, the first solver reads the yaws, sets the flag to False and therefore
    # all other solvers will wait perpetually until the flag gets set to True again.
    for i in instances:
        client.put_tensor(f'{i}_yaws_done', np.array([0]))

    # Set the sim_done flag to be true
    for i in instances:
        client.put_tensor(f'{i}_sim_done', np.array([1]))

