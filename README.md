# Learning Successor Features with Distributed Hebbian Temporal Memory

## Installation
1. Install `htm.core` from the [repository](https://anonymous.4open.science/r/htm_core-DD2D) according to its instructions.
2. Install `requirements.txt`, `python>=3.9` is required.
3. Run `python install -e .` in the root of this repository containing `setup.py`.
4. Setup [wandb](https://wandb.ai/site) logging system.
5. Install [AnimalAI](https://github.com/Kinds-of-Intelligence-CFI/animal-ai)
6. Define environment with the following variables:

```
ANIMALAI_EXE=path/to/animal-ai/exe/file
ANIMALAI_ROOT=path/to/animal-ai/project/root
GRIDWORLD_ROOT=path/to/this/repository/dhtm/experiments/successor_representations/configs/environment/gridworld/setups
OPENBLAS_NUM_THREADS=1
```

## Running experiments
To run an experiment from the paper: 
1. specify a path to a corresponding config `RUN_CONF=path/to/config.yaml` in `dhtm/experiments/successor_representations/configs`.
2. run command: `cd this_repo/dhtm/experiments/successor_representations`
3. run command: `python runners/test.py`

All configs are in the folder `him-agent/hima/experiments/successor_representations/configs/runner`.
The folder contains configs for the following experiments:
```    
# Gridworld experiments with changing partially observable environment (Figure 2):
girdworld_dhtm.yaml
gridworld_cscg.yaml
gridworld_lstm.yaml
gridworld_rwkv.yaml
girdworld_ec.yaml

# AnimalAI experiment (Figure 3)
animalai_dhtm.yaml
animalai_dhtm_vae.yaml  # DHTM with Categorical VAE encoder
animalai_ec.yaml
animalai_cscg.yaml
animalai_lstm.yaml
animalai_rwkv.yaml

# Additional experiments (Appendix H)
## H.2 Scalability
gridworld_dhtm_scale.yaml
gridworld_ec_scale.yaml

## H.3 Noise tolerance
animalai_dhtm_distraction.yaml
animalai_ec_distraction.yaml

```
To reproduce figures from the paper, run each config at least 5 times with wandb logging enabled.
