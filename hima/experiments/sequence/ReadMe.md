### Experiments for paper: Learning Hidden Markov Model of Stochastic Environment with Bio-Inspired Probabilistic Temporal Memory

This ReadMe provides instructions for installing and running code for experiments described in the Main Text.

#### Installation
1. Install [htm.core](https://github.com/ZhekaHauska/htm.core), see instuctions in corresponding repository.
2. Install [pinball](https://github.com/ZhekaHauska/pinball), see instuctions in corresponding repository.
3. Install modules for our Temporal Memory model, see [instructions](https://github.com/AIRI-Institute/him-agent).
#### Running experiments

##### Configure environment variables:

To run experiments in the Pinball environment you should specify path to executable and its root folder.
If you would like to use logging, you also need sign up to [wandb.ai](wandb.ai) and specify your login in `WANDB_ENTITY` variable.

```
PINBALL_EXE=/your/path/to/exe;
PINBALL_ROOT=/your/path/to/pinball;
WANDB_ENTITY=your_profile_in_wandb
```

if you run one experiment, specify config of the experiment via variable `RUN_CONF=configs/runner/belieftm/pinball.yaml` Instead of specifying this variable you can pass config directly to the runner as an inline argument. 

##### Run one experiment

To run experiment just type in console `python runners/$RUNNER.py [config_path]`, where `$RUNNER` is one of the following:  

`belieftm_runner` --- our probabilistic temporal memory, its working name is BeliefTM.  
`hmm_runner` --- CHMM baseline.  
`htm_runner` --- HTM baseline.
`lstm` --- HTM baseline.
`uniform` --- Uniform baseline.

feel free to adjust configs accroding to your research needs.

##### Run series of experiments

To run a series of experiments use `wandb sweep path_to_sweep_config`. All sweep configs are placed in `sweeps/bica`. Read more about sweeps [here](https://docs.wandb.ai/guides/sweeps).
