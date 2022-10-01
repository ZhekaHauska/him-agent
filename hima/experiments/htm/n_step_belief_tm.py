#  Copyright (c) 2022 Autonomous Non-Profit Organization "Artificial Intelligence Research
#  Institute" (AIRI); Moscow Institute of Physics and Technology (National Research University).
#  All rights reserved.
#
#  Licensed under the AGPLv3 license. See LICENSE in the project root for license information.
import pickle
import yaml
import numpy as np
from scipy.special import rel_entr
import wandb
import os
import matplotlib.pyplot as plt
from functools import reduce


def main(
        config_path
):
    with open(config_path, 'r') as file:
        config = yaml.load(file, Loader=yaml.Loader)

    with open(config['model'], 'rb') as file:
        tm, mpg, obs_encoder, policy_encoder, ptsm = pickle.load(file)

    if config['log']:
        if config['logger'] == 'wandb':
            logger = wandb.init(
                project=config['project_name'], entity=os.environ['WANDB_ENTITY'],
                config=config
            )
        else:
            logger = config['logger']
    else:
        logger = None

    # stats
    kl_divs = list()
    tm_dists = list()
    true_dists = list()

    rng = np.random.default_rng(config['seed'])

    tm.reset()
    mpg.set_policy(config['policy'])
    mpg.set_current_state(config['state'])

    # choose random letter leading to the state
    tl = mpg.transition_letters
    letters = [tl[i][config['state']] for i in mpg.states if tl[i][config['state']] != 0]

    if len(letters) > 0:
        letter = rng.choice(letters)
        encoded_letter = obs_encoder.encode(mpg.char_to_num[letter])
    else:
        letter = '∅'
        encoded_letter = np.empty(0, dtype=np.dtype('int32'))

    tm.set_active_feedback_cells(policy_encoder.encode(config['policy']))

    if not config['use_ptsm']:
        tm.set_active_columns(encoded_letter)
        tm.activate_apical_dendrites(learn=False)
        tm.predict_cells()
        tm.activate_cells(learn=False)
        active_cells = tm.get_active_cells()
    else:
        active_cells = reduce(
            np.union1d,
            ptsm.state_to_patterns(config['state'], config['policy'])
        )

    tm.set_active_context_cells(active_cells)
    tm.predict_columns_density(update_receptive_fields=False)

    letter_dist = np.mean(
        tm.column_probs.reshape((-1, obs_encoder.n_active_bits)).T, axis=0
    )

    letter_dist = np.append(letter_dist, np.clip(1 - letter_dist.sum(), 0, 1))
    step = 0

    true_let_dist = mpg.predict_letters(from_state=config['state'], steps=step)
    true_let_dist = np.append(true_let_dist, np.clip(1 - true_let_dist.sum(), 0, 1))

    tm_dists.append(letter_dist)
    true_dists.append(true_let_dist)

    kl_div = rel_entr(true_let_dist, letter_dist).sum()
    kl_divs.append(kl_div)

    # logging
    if logger is not None:
        if logger != 'local':
            logger.log(
                {
                    'main_metrics/dkl': kl_div
                },
                step=0
            )

        tick_labels = mpg.alphabet.copy()
        tick_labels.append('∅')

        k = int(np.ceil(np.sqrt(config['n_steps'])))
        fig, axs = plt.subplots(k, k, figsize=(10, 10))
        fig.tight_layout(pad=3.0)

        ax = axs[step // k][step % k]
        ax.grid()
        ax.set_ylim(0, 1)
        ax.bar(
            np.arange(letter_dist.shape[0]),
            letter_dist,
            tick_label=tick_labels,
            color=(0.7, 1.0, 0.3),
            label='TM'
        )

        ax.bar(
            np.arange(true_let_dist.shape[0]),
            true_let_dist,
            tick_label=tick_labels,
            color=(0.8, 0.5, 0.5),
            alpha=0.6,
            label='True'
        )

        ax.set_title(f'steps: {step + 1}; KL: {np.round(kl_div, 2)}')

    for step in range(1, config['n_steps']):
        tm.predict_n_step_density(1, mc_iterations=config['mc_iterations'])

        let_dist = np.mean(
            tm.column_probs.reshape((-1, obs_encoder.n_active_bits)).T, axis=0
        )
        let_dist = np.append(let_dist, np.clip(1 - let_dist.sum(), 0, 1))

        # ground truth
        true_let_dist = mpg.predict_letters(from_state=config['state'], steps=step)
        true_let_dist = np.append(true_let_dist, np.clip(1 - true_let_dist.sum(), 0, 1))

        tm_dists.append(let_dist)
        true_dists.append(true_let_dist)

        kl_div = rel_entr(true_let_dist, let_dist).sum()
        kl_divs.append(kl_div)

        # logging
        if logger is not None:
            if logger != 'local':
                logger.log(
                    {
                        'main_metrics/dkl': kl_div
                    },
                    step=step
                )
            ax = axs[step // k][step % k]
            ax.grid()
            ax.set_ylim(0, 1)

            ax.bar(
                np.arange(let_dist.shape[0]),
                let_dist,
                tick_label=tick_labels,
                color=(0.7, 1.0, 0.3)
            )

            ax.bar(
                np.arange(true_let_dist.shape[0]),
                true_let_dist,
                tick_label=tick_labels,
                color=(0.8, 0.5, 0.5),
                alpha=0.6
            )

            ax.set_title(f'steps: {step + 1}; KL: {np.round(kl_div, 2)}')

    if logger is not None:
        fig.legend(loc=7)
        fig.suptitle(f'From state: {config["state"]} First letter: {letter}')

        if logger != 'local':
            logger.log({f'density/n_step_letter_predictions': wandb.Image(fig)})
        else:
            plt.show()


if __name__ == '__main__':
    main('configs/n_step_belief_tm_runner.yaml')
