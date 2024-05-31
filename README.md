# Wind-RL
Reinforcement Learning for Wind Farm Control

## How to install


### v1: using mpi run to restart the simulation at every agent interaction
Python requirements are available in requirements.txt
[XCompact3D](https://github.com/admole/Incompact3d/tree/my_dev) is required to run the wind farm simulations.

Xcompact should be compiled following its instructions and added to a known path with
```bash
sudo ln -s /path/to/Incompact3D/xcompact3d /usr/bin/xcompact3d
```

### v2: using SmartSim to couple xcompact3d and RL code
The following approach seems to be working:

1. Install SmartSim with `pip install smartsim`.
2. Build SmartSim by running
```bash
smart build --device cpu --no_tf
```
This will build SmartSim and request the PyTorch backend and RedisAI. Note that currently, this will install PyTorch 2.0.1 (CPU only), but we need a newer version of PyTorch.
3. Avoid dependency conflicts by unsintalling torchvision (which is not used anyway)
```bash
pip uninstall torchvision
```
4. Install the rest of the requirements by running 
```bash
pip install -r requirements.txt
```
Note that this will install PyTorch 2.2.1 (GPU version). However we did not install GPU support for SmartSim so that we can only run on CPU. This is okay since the time to train the RL models is a tiny fraction of time required for the wind farm simulations.


## Running

### v1

The reinforcement learning can be run with the ppo algorithm from the main directory using
```bash
python3 ./ppo/ppo.py
```
This will log the results to the Wind-RL project on weights and biases.

The run configurations can be modified in ./ppo/config_ppo.yaml

For a more verbose output set the enviroment variable
```bash
export WINDRL_VERBOSE=true
```


### v2

Coming soon!
