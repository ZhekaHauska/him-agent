# A Biologically Interpretable Cognitive Architecture for Online Structuring of Episodic Memories into Cognitive Maps
This is experiments code for "A Biologically Interpretable Cognitive Architecture for Online Structuring of Episodic Memories into Cognitive Maps" paper.

## Installation
1. Install `requirements.txt`, `python>=3.9` is required.
2. Run `python install -e .` in the root of this repository containing `setup.py`.
3. Setup [wandb](https://wandb.ai/site) or [aim](https://aimstack.readthedocs.io/en/latest/) logging system.
4. Define environment with the following variables:

```
GRIDWORLD_ROOT=path/to/this/repository/dhtm/experiments/successor_representations/configs/environment/gridworld/setups
OPENBLAS_NUM_THREADS=1
RUN_CONF=path/to/config.yaml
```

## Reproducing experiments
### Section 3.1 (Rationale)
1. Set `export RUN_CONF=configs/runner/gridworld_gather_data.yaml`
2. run command: `cd this_repo/epm/experiments/successor_representations`
3. run command: `mkdir logs`
4. run command: `python runners/test.py`

* A file `logs/prefix_*.pkl` will be created, which stores episodic memory of 1000 episodes for one 10x10 environment.
* To gather experience from more environments (it was 3 in the paper) change seed value in `configs/environment/gridworld/pomdp.yaml` and 
file prefix in `configs/scenario/gridworld_gather_data.yaml`

5. Follow `results/sf_tests.ipynb` to process episodic memory and plot the results.

### Section 4 (Main experiments)
1. Set `export RUN_CONF=configs/runner/gridworld.yaml`
2. Change `configs/agent/ec/gridworld_merge_test.yaml`. 
Important parameters:
   * min_cluster_size: 5, 10, 20, 30, 50, 100
   * merge_iterations: 0 (no merge) or 5
   * merge_mode: random_sf (random merge), sf

For each combination run 5 seed values commented in `configs/runner/gridworld.yaml`
and 3 seed values in `configs/environment/gridworld/pomdp.yaml` (which specify colouring)
    
3. run command: `cd this_repo/epm/experiments/successor_representations`
4. run command: `python runners/test.py` (to run one parameter combination)
5. Follow `results/sf_tests.ipynb` to further process and plot the results (aim needed) (Section Main Experiments)

