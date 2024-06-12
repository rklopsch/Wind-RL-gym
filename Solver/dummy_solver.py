from smartredis import Client
import numpy as np
import time


def dummy_solve(yaws):
    pows = np.ones(3)
    obs = np.zeros([3, 5])
    return pows, obs 

client = Client(address=None, cluster=False)
instance = 0

while True:
    # only for testing this file on its own
    # client.put_tensor(f'{instance}_yaws_done', np.array([1]))
    
    # Check if the file exists - idle until it does
    while not client.poll_key(f'{instance}_yaws_done', 100, 10):
        time.sleep(0.1)
        print(f"Waiting for yaws done to be created")
        continue

    # Check if the yaws have been written by the agent
    while not client.get_tensor(f'{instance}_yaws_done')[0]:
        time.sleep(0.1)
        print(f"Waiting for yaws to be done. Time {time.time()}")
        continue

    # Reset the yaw done to False
    # We might have to have a flag here that checks if ALL the solvers have
    # had a chance to read the yaws, and only then reset the yaw done flag to False.
    # Otherwise, the first solver reads the yaws, sets the flag to False and therefore
    # all other solvers will wait perpetually until the flag gets set to True again.
    client.put_tensor(f'{instance}_yaws_done', np.array([0]))

    # Read yaws determined by agent
    yaws = client.get_tensor(f'{instance}_yaws')
    # yaws = np.zeros([3])

    print(f"Yaws are {yaws}")
    
    # Do the numerical magic
    for _ in range(50):
        pows, obs = dummy_solve(yaws)
    
    # Store the new outputs of the simulation on the database
    client.put_tensor(f'{instance}_turbine_powers', pows)
    client.put_tensor(f'{instance}_probe_data', obs)

    # Set the sim_done flag to be true
    client.put_tensor(f'{instance}_sim_done', np.array([1]))

