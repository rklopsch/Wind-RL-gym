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
These instructions will get SmartSim installed together with TorchRL:

1. Install torchrl with
```bash
pip install torchrl==0.2.1
```

2. Install the correct version of tensordict with 
```bash
pip uninstall tensordict
pip install tensordict==0.2.0
```

3. Uninstall the PyTorch build with
```bash
`pip uninstall torch`
```

4. Build SmartSim, requesting PyTorch as an ML backend
```bash
smart build --device cpu --no_tf
```
This will install torch 2.0.1 (CPU only).

5. Install the remaining requirements by running
```bash
pip install tqdm matplotlib hydra-core wandb f90nml
```

Happy times.


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
