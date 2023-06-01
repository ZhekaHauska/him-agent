#  Copyright (c) 2022 Autonomous Non-Profit Organization "Artificial Intelligence Research
#  Institute" (AIRI); Moscow Institute of Physics and Technology (National Research University).
#  All rights reserved.
#
#  Licensed under the AGPLv3 license. See LICENSE in the project root for license information.

import numpy as np
from hima.envs.mpg import MultiMarkovProcessGrammar
from hima.common.sdr_encoders import IntBucketEncoder
from hima.modules.belief.belief_tm import HybridNaiveBayesTM
from hima.modules.htm.pattern_sorter import PatternToStateMemory
from hima.modules.htm.tm_writer import HTMWriter

from scipy.special import rel_entr
import pickle

import wandb
import matplotlib.pyplot as plt
import yaml
import os


def main(config_path):
    config = dict()

    with open(config_path, 'r') as file:
        config['run'] = yaml.load(file, Loader=yaml.Loader)

    with open(config['run']['tm_conf'], 'r') as file:
        config['tm'] = yaml.load(file, Loader=yaml.Loader)

    with open(config['run']['mpg_conf'], 'r') as file:
        config['mpg'] = yaml.load(file, Loader=yaml.Loader)

    mpg = MultiMarkovProcessGrammar(
        seed=config['run']['seed'],
        **config['mpg']
    )

    obs_encoder = IntBucketEncoder(len(mpg.alphabet), config['run']['bucket_size'])
    policy_encoder = IntBucketEncoder(len(mpg.policy_transition_probs), config['run']['bucket_size'])

    print(f'n_columns: {obs_encoder.output_sdr_size}')

    if config['run']['log']:
        logger = wandb.init(
            project=config['run']['project_name'], entity=os.environ['WANDB_ENTITY'],
            config=config
        )
    else:
        logger = None

    if config['run']['tm_type'] == 'hybrid_naive_bayes':
        run_hybrid_naive_bayes_tm(config, mpg, obs_encoder, policy_encoder, logger)
    else:
        raise ValueError


def run_hybrid_naive_bayes_tm(config, mpg, obs_encoder, policy_encoder, logger):
    activation_threshold = int(config['run']['bucket_size']*config['tm'].pop('activation_threshold'))
    learning_threshold = int(config['run']['bucket_size']*config['tm'].pop('learning_threshold'))

    tm = HybridNaiveBayesTM(
        columns=obs_encoder.output_sdr_size,
        context_cells=obs_encoder.output_sdr_size * config['tm']['cells_per_column'],
        feedback_cells=policy_encoder.output_sdr_size,
        activation_threshold_basal=activation_threshold,
        learning_threshold_basal=learning_threshold,
        learning_threshold_inhib=learning_threshold,
        activation_threshold_apical=activation_threshold,
        learning_threshold_apical=learning_threshold,
        max_synapses_per_segment_apical=config['run']['bucket_size'],
        max_synapses_per_segment_basal=config['run']['bucket_size'],
        seed=config['run']['seed'],
        **config['tm']
    )

    ptsm = PatternToStateMemory(
        obs_encoder.n_values,
        policy_encoder.n_values,
        config['run']['min_pattern_distance']
    )

    if config['run']['debug_tm_state']:
        state_logger = HTMWriter(
            'belief_tm',
            config['run']['state_log_path'],
            tm,
            config['run']['chunk_size']
        )
    else:
        state_logger = None

    density = np.zeros((policy_encoder.n_values, len(mpg.states), len(mpg.alphabet)))
    hist_dist = np.zeros((policy_encoder.n_values, len(mpg.states), len(mpg.alphabet)))
    lr = 0.02

    true_densities = np.zeros((policy_encoder.n_values, len(mpg.states), len(mpg.alphabet)))
    for pol in range(policy_encoder.n_values):
        mpg.set_policy(pol)
        true_densities[pol] = np.array([mpg.predict_letters(from_state=i) for i in mpg.states])

    true_hist = np.copy(true_densities)
    true_hist[true_hist > 0] = 1.0

    for i in range(config['run']['epochs']):
        mpg.reset()
        tm.reset()

        word = []
        anomaly = []
        confidence = []
        surprise = []
        dkl = []
        iou = []

        policy = mpg.rng.integers(policy_encoder.n_values)
        mpg.set_policy(policy)

        while True:
            letter = mpg.next_state()

            if letter:
                word.append(letter)
            else:
                break

            # set winner cells from previous step
            tm.set_active_context_cells(tm.get_winner_cells())
            tm.set_active_columns(obs_encoder.encode(mpg.char_to_num[letter]))
            if config['run']['use_feedback']:
                tm.set_active_feedback_cells(policy_encoder.encode(policy))
                tm.activate_apical_dendrites(learn=True)
                if len(tm.active_cells_context.sparse) == 0:
                    tm.predict_cells()
            tm.activate_cells(learn=True)

            # connect active pattern to state
            ptsm.connect(tm.winner_cells, mpg.current_state, mpg.current_policy)

            surprise.append(min(200.0, tm.surprise))
            anomaly.append(tm.anomaly[-1])

            conf = len(tm.get_predicted_columns()) / len(tm.get_active_columns())
            confidence.append(conf)

            tm.set_active_context_cells(tm.get_active_cells())
            tm.activate_basal_dendrites(learn=True)
            tm.predict_cells()
            tm.predict_columns_density()

            if (
                    state_logger is not None
                    and
                    (config['run']['debug_range'][0] <= i <= config['run']['debug_range'][1])
            ):
                if len(word) > 1:
                    prev_letter = word[-2]
                else:
                    prev_letter = None

                state_logger.write(letter, prev_letter, str(policy))

            letter_dist = np.mean(
                tm.column_probs.reshape((-1, config['run']['bucket_size'])).T, axis=0
            )
            density[policy][mpg.current_state] += lr * (letter_dist - density[policy][mpg.current_state])

            dkl.append(min(rel_entr(true_densities[policy][mpg.current_state], letter_dist).sum(), 200.0))

            pred_columns_dense = np.zeros(tm.columns)
            pred_columns_dense[tm.get_predicted_columns()] = 1
            predicted_letters = np.mean(
                pred_columns_dense.reshape((-1, config['run']['bucket_size'])).T, axis=0
            )
            hist_dist[policy][mpg.current_state] += lr * (predicted_letters - hist_dist[policy][mpg.current_state])

            iou.append(
                np.logical_and(predicted_letters, true_hist[policy][mpg.current_state]).sum() / np.logical_or(predicted_letters, true_hist[policy][mpg.current_state]).sum()
            )

        if logger is not None:
            logger.log(
                {
                    'main_metrics/surprise': np.array(surprise)[1:].mean(),
                    'main_metrics/anomaly': np.array(anomaly)[1:].mean(),
                    'main_metrics/confidence': np.array(confidence)[1:].mean(),
                    'main_metrics/dkl': np.array(np.abs(dkl))[1:].mean(),
                    'main_metrics/iou': np.nanmean(np.array(iou)[1:]),
                    'segments/apical': tm.apical_connections.numSegments(),
                    'segments/basal': tm.basal_connections.numSegments()
                }, step=i
            )

            if i % config['run']['update_rate'] == 0:
                kl_divs = rel_entr(true_densities, density).sum(axis=-1)

                n_states = len(mpg.states)
                k = int(np.ceil(np.sqrt(n_states)))
                for pol in range(density.shape[0]):
                    fig, axs = plt.subplots(k, k)
                    fig.tight_layout(pad=3.0)

                    for n in range(n_states):
                        ax = axs[n // k][n % k]
                        ax.grid()
                        ax.set_ylim(0, 1)
                        ax.set_title(
                            f's: {n}; ' + '$D_{KL}$: ' + f'{np.round(kl_divs[pol][n], 2)}'
                        )
                        ax.bar(
                            np.arange(density[pol][n].shape[0]),
                            density[pol][n],
                            tick_label=mpg.alphabet,
                            label='TM',
                            color=(0.7, 1.0, 0.3),
                            capsize=4,
                            ecolor='#2b4162'
                        )
                        ax.bar(
                            np.arange(density[pol][n].shape[0]),
                            true_densities[pol][n],
                            tick_label=mpg.alphabet,
                            color='#8F754F',
                            alpha=0.6,
                            label='True'
                        )

                    fig.legend(['TM', 'True'], loc=7)

                    logger.log({f'density/letter_predictions_policy_{pol}': wandb.Image(fig)}, step=i)

                    plt.close(fig)

                for pol in range(density.shape[0]):
                    fig, axs = plt.subplots(k, k)
                    fig.tight_layout(pad=3.0)

                    for n in range(n_states):
                        intersection = (hist_dist[pol][n] * true_hist[pol][n]).sum()
                        union = hist_dist[pol][n].sum() + true_hist[pol][n].sum() - intersection
                        iou = intersection / union

                        ax = axs[n // k][n % k]
                        ax.grid()
                        ax.set_ylim(0, 1)
                        ax.set_title(
                            f's: {n}; iou {np.round(iou, 2)}'
                        )
                        ax.bar(
                            np.arange(density[pol][n].shape[0]),
                            hist_dist[pol][n],
                            tick_label=mpg.alphabet,
                            label='TM',
                            color=(0.7, 1.0, 0.3),
                            capsize=4,
                            ecolor='#2b4162'
                        )
                        ax.bar(
                            np.arange(density[pol][n].shape[0]),
                            true_hist[pol][n],
                            tick_label=mpg.alphabet,
                            color='#8F754F',
                            alpha=0.6,
                            label='True'
                        )

                    fig.legend(['TM', 'True'], loc=7)

                    logger.log({f'deterministic/letter_predictions_policy_{pol}': wandb.Image(fig)}, step=i)

                    plt.close(fig)
    else:
        if state_logger is not None:
            state_logger.save()

    if logger is not None:
        name = logger.name

        np.save(f'logs/density_{name}.npy', density)
        np.save(f'logs/hist_dist_{name}.npy', hist_dist)

        if config['run']['save_model']:
            with open(f"logs/model_env_{name}_{config['run']['epochs']}.pkl", 'wb') as file:
                pickle.dump((tm, mpg, obs_encoder, policy_encoder, ptsm), file)


if __name__ == '__main__':
    main('configs/mpg_runner.yaml')
