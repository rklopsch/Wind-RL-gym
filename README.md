# Wind-RL
Reinforcement Learning for Wind Farm Control

## Requirements

Python requirements are available in requirements.txt
[XCompact3D](https://github.com/admole/Incompact3d/tree/my_dev) is required to run the wind farm simulations.

Xcompact should be compiled following its instructions and added to a known path with
```bash
sudo ln -s /path/to/Incompact3D/xcompact3d /usr/bin/xcompact3d
```

## Running

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
