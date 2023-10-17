#  Copyright (c) 2022 Autonomous Non-Profit Organization "Artificial Intelligence Research
#  Institute" (AIRI); Moscow Institute of Physics and Technology (National Research University).
#  All rights reserved.
#
#  Licensed under the AGPLv3 license. See LICENSE in the project root for license information.
from hima.common.sdr_encoders import IntBucketEncoder
from hima.modules.belief.belief_tm import HybridNaiveBayesTM
from hima.modules.htm.tm_writer import HTMWriter
from hima.envs.mpg.mpg import MultiMarkovProcessGrammar, draw_mpg
from hima.modules.htm.spatial_pooler import SPDecoder, HtmSpatialPooler
from htm.bindings.sdr import SDR
from hima.experiments.sequence.runners.utils import get_surprise

try:
    from pinball import Pinball
except ModuleNotFoundError:
    Pinball = None

import numpy as np
from scipy.special import rel_entr
import matplotlib.pyplot as plt
from pathlib import Path
import wandb
import yaml
import os
import sys
import ast
import pickle
import imageio
from copy import copy


class MPGTest:
    def __init__(self, logger, conf):
        self.seed = conf['run']['seed']

        conf['hmm']['seed'] = self.seed
        conf['env']['seed'] = self.seed

        self.mpg = MultiMarkovProcessGrammar(**conf['env'])

        self.n_obs_states = len(self.mpg.alphabet)
        self.bucket_size = conf['run']['bucket_size']

        self.encoder = IntBucketEncoder(self.n_obs_states, self.bucket_size)

        conf['hmm']['columns'] = self.encoder.output_sdr_size
        conf['hmm']['context_cells'] = conf['hmm']['columns'] * conf['hmm']['cells_per_column']

        self.hmm = HybridNaiveBayesTM(**conf['hmm'])

        self.n_episodes = conf['run']['n_episodes']
        self.smf_dist = conf['run']['smf_dist']
        self.log_update_rate = conf['run']['update_rate']
        self.max_steps = conf['run']['max_steps']
        self.save_model = conf['run']['save_model']
        self.log_path = conf['run']['log_path']
        self.n_steps = conf['run'].get('n_step_test', None)
        if self.n_steps is not None:
            self.mc_iterations = conf['run']['mc_iterations']

        self.logger = logger

        if conf['run']['debug_tm_state']:
            self.state_logger = HTMWriter(
                'belief_tm',
                os.environ.get('TM_STATE_PATH', 'tm_state_log'),
                self.hmm,
                conf['run']['chunk_size']
            )
        else:
            self.state_logger = None

        self.tm_state_debug_range = conf['run'].get('debug_range', [0, self.n_episodes])

        if self.logger is not None:
            if self.n_steps is not None:
                self.logger.define_metric(
                    name='main_metrics/n_step_dkl',
                    step_metric='prediction_step'
                )

            im_name = f'/tmp/mpg_{self.logger.name}.png'
            draw_mpg(
                im_name,
                self.mpg.transition_probs,
                self.mpg.transition_letters
            )

            self.logger.log({'mpg': wandb.Image(im_name)})

    def run(self):
        dist = np.zeros((len(self.mpg.states), len(self.mpg.alphabet) + 1))
        dist_disp = np.zeros((len(self.mpg.states), len(self.mpg.alphabet) + 1))

        true_dist = np.array([self.mpg.predict_letters(from_state=i) for i in self.mpg.states])
        norm = true_dist.sum(axis=-1)
        empty_prob = np.clip(1 - norm, 0, 1)
        true_dist = np.hstack([true_dist, empty_prob.reshape(-1, 1)])

        total_surprise = 0
        total_dkl = 0

        for i in range(self.n_episodes):
            self.mpg.reset()
            self.hmm.reset()

            dkls = []
            surprises = []
            anomalies = []
            confidences = []
            word = []

            steps = 0

            while True:
                prev_state = self.mpg.current_state

                letter = self.mpg.next_state()

                word.append(letter)

                if letter is None:
                    obs_state = []
                else:
                    obs_state = self.encoder.encode(self.mpg.char_to_num[letter])

                self.hmm.set_active_context_cells(self.hmm.get_active_cells())
                self.hmm.activate_basal_dendrites(learn=True)
                self.hmm.predict_cells()
                self.hmm.predict_columns_density()

                column_probs = np.mean(
                    self.hmm.column_probs.reshape(
                        (-1, self.bucket_size)
                    ).T, axis=0
                )

                # set winner cells from previous step
                self.hmm.set_active_context_cells(self.hmm.get_winner_cells())
                self.hmm.set_active_columns(obs_state)
                self.hmm.activate_cells(learn=True)

                # debug
                if (
                        self.state_logger is not None
                        and
                        (self.tm_state_debug_range[0] <= i <= self.tm_state_debug_range[1])
                ):
                    if len(word) > 1:
                        prev_letter = word[-2]
                    else:
                        prev_letter = None

                    self.state_logger.write(letter, prev_letter)
                    
                # metrics
                if (letter is not None) and (prev_state != 0):
                    # 1. surprise
                    surprise = self.get_surprise(self.hmm.column_probs, obs_state)
                    surprises.append(surprise)
                    total_surprise += surprise

                # 2. distribution
                column_probs = np.append(
                    column_probs, np.clip(1 - column_probs.sum(), 0, 1)
                )

                delta = column_probs - dist[prev_state]
                dist_disp[prev_state] += self.smf_dist * (
                        np.power(delta, 2) - dist_disp[prev_state])
                dist[prev_state] += self.smf_dist * delta

                # 3. Kl distance
                if prev_state != 0:
                    dkl = min(
                            rel_entr(true_dist[prev_state], column_probs).sum(),
                            200.0
                        )
                    dkls.append(dkl)
                    total_dkl += dkl

                    # 4. anomaly
                    anomalies.append(self.hmm.anomaly[-1])

                    # 5. confidence
                    confidences.append(self.hmm.confidence[-1])

                steps += 1

                if letter is None:
                    break

                if steps >= self.max_steps:
                    break

            if self.logger is not None:
                self.logger.log(
                    {
                        'main_metrics/surprise': np.array(surprises).mean(),
                        'spec_metrics/anomaly': np.array(anomalies).mean(),
                        'spec_metrics/confidence': np.array(confidences).mean(),
                        'main_metrics/dkl': np.abs(dkls).mean(),
                        'main_metrics/total_surprise': total_surprise,
                        'main_metrics/total_dkl': total_dkl,
                        'connections/basal_segments': self.hmm.basal_connections.numSegments(),
                        'connections/inhib_segments': self.hmm.inhib_connections.numSegments(),
                        'connections/apical_segments': self.hmm.apical_connections.numSegments()
                    }, step=i
                )

                if (self.log_update_rate is not None) and (i % self.log_update_rate == 0):
                    kl_divs = rel_entr(true_dist, dist).sum(axis=-1)

                    n_states = len(self.mpg.states)
                    k = int(np.ceil(np.sqrt(n_states)))

                    tick_labels = self.mpg.alphabet.copy()
                    tick_labels.append('∅')

                    fig, axs = plt.subplots(k, k)
                    fig.tight_layout(pad=3.0)

                    for n in range(n_states):
                        ax = axs[n // k][n % k]
                        ax.grid()
                        ax.set_ylim(0, 1)
                        ax.set_title(
                            f's: {n}; ' + '$D_{KL}$: ' + f'{np.round(kl_divs[n], 2)}'
                        )
                        ax.bar(
                            np.arange(dist[n].shape[0]),
                            dist[n],
                            tick_label=tick_labels,
                            label='TM',
                            color=(0.7, 1.0, 0.3),
                            capsize=4,
                            ecolor='#2b4162',
                            yerr=np.sqrt(dist_disp[n])
                        )
                        ax.bar(
                            np.arange(dist[n].shape[0]),
                            true_dist[n],
                            tick_label=tick_labels,
                            color='#8F754F',
                            alpha=0.6,
                            label='True'
                        )

                        fig.legend(['Predicted', 'True'], loc=8)

                        self.logger.log(
                            {'density/letter_predictions': wandb.Image(fig)}, step=i
                        )

                        plt.close(fig)

        if self.state_logger is not None:
            self.state_logger.save()

        if self.logger is not None and self.save_model:
            name = self.logger.name

            path = Path(self.log_path)
            if not path.exists():
                path.mkdir()

            np.save(f'{self.log_path}/dist_{name}.npy', dist)

            with open(f"{self.log_path}/model_{name}.pkl", 'wb') as file:
                pickle.dump((self.mpg, self.hmm), file)

        if self.n_steps is not None:
            self.run_n_step()

    def run_n_step(self):
        self.hmm.reset()
        self.mpg.reset()

        k = int(np.ceil(np.sqrt(self.n_steps)))
        fig, axs = plt.subplots(k, k, figsize=(10, 10))
        fig.tight_layout(pad=3.0)

        self.hmm.set_active_columns([])
        self.hmm.activate_basal_dendrites(learn=False)
        self.hmm.predict_cells()
        self.hmm.set_active_context_cells(self.hmm.get_active_cells())
        self.hmm.predict_columns_density(update_receptive_fields=False)

        dkls = []
        n_step_dists = []

        for step in range(self.n_steps):
            predicted_dist = np.mean(
                    self.hmm.column_probs.reshape(
                        (-1, self.bucket_size)
                    ).T, axis=0
                )

            if step == 0:
                labels = ['Predicted', 'True']
                letter = self.mpg.next_state()

                obs_state = self.encoder.encode(self.mpg.char_to_num[letter])

                self.hmm.set_active_columns(obs_state)
                self.hmm.activate_apical_dendrites(learn=False)
                self.hmm.predict_cells()
                self.hmm.activate_cells(learn=False)
                active_cells = self.hmm.get_active_cells()
                self.hmm.set_active_context_cells(active_cells)
                self.hmm.predict_columns_density(update_receptive_fields=False)
            else:
                labels = [None, None]
                self.hmm.predict_n_step_density(1, mc_iterations=self.mc_iterations)

            true_dist = self.mpg.predict_letters(from_state=0, steps=step)

            true_dist = np.append(
                true_dist,
                np.clip(1 - true_dist.sum(), 0, 1)
            )

            predicted_dist = np.append(
                predicted_dist,
                np.clip(1 - predicted_dist.sum(), 0, 1)
            )

            n_step_dists.append(predicted_dist)

            kl_div = rel_entr(true_dist, predicted_dist).sum()
            dkls.append(kl_div)

            if self.logger is not None:
                self.logger.log(
                    {
                        'main_metrics/n_step_dkl': kl_div,
                        'prediction_step': step
                    }
                )

            tick_labels = self.mpg.alphabet.copy()
            tick_labels.append('∅')

            ax = axs[step // k][step % k]
            ax.grid()
            ax.set_ylim(0, 1)
            ax.bar(
                np.arange(predicted_dist.shape[0]),
                predicted_dist,
                tick_label=tick_labels,
                color=(0.7, 1.0, 0.3),
                label=labels[0]
            )

            ax.bar(
                np.arange(true_dist.shape[0]),
                true_dist,
                tick_label=tick_labels,
                color=(0.8, 0.5, 0.5),
                alpha=0.6,
                label=labels[1]
            )

            ax.set_title(f'steps: {step + 1}; KL: {np.round(kl_div, 2)}')

        fig.legend(loc=7)

        if self.logger is not None:
            dkls = np.array(dkls)
            n_step_dists = np.vstack(n_step_dists)

            name = self.logger.name

            if self.save_model:
                np.save(f'logs/n_step_dist_{name}.npy', n_step_dists)

            self.logger.log({f'density/n_step_letter_predictions': wandb.Image(fig)})
            self.logger.log(
                {
                    f'main_metrics/average_nstep_dkl': np.abs(dkls).mean(where=~np.isinf(dkls))
                }
            )

    @staticmethod
    def get_surprise(probs, obs):
        is_coincide = np.isin(
            np.arange(len(probs)), obs
        )
        surprise = - np.sum(
            np.log(
                np.clip(probs[is_coincide], 1e-7, 1)
            )
        )
        surprise += - np.sum(
            np.log(
                np.clip(1 - probs[~is_coincide], 1e-7, 1)
            )
        )

        return surprise


class MMPGTest:
    def __init__(self, logger, conf):
        self.seed = conf['run']['seed']

        conf['hmm']['seed'] = self.seed
        conf['env']['seed'] = self.seed

        self.mpg = MultiMarkovProcessGrammar(**conf['env'])

        self.n_policies = self.mpg.policy_transition_probs.shape[0]

        if None in self.mpg.alphabet:
            self.n_obs_states = len(self.mpg.alphabet) - 1
        else:
            self.n_obs_states = len(self.mpg.alphabet)

        conf['hmm']['columns'] = self.n_obs_states
        conf['hmm']['context_cells'] = conf['hmm']['columns'] * conf['hmm']['cells_per_column']

        self.use_feedback = conf['run']['use_feedback']

        if self.use_feedback:
            conf['hmm']['feedback_cells'] = self.n_policies
        else:
            conf['hmm']['context_cells'] += self.n_policies
            conf['hmm']['feedback_cells'] = 0

        self.hmm = HybridNaiveBayesTM(**conf['hmm'])

        self.n_episodes = conf['run']['n_episodes']
        self.smf_dist = conf['run']['smf_dist']
        self.log_update_rate = conf['run']['update_rate']
        self.max_steps = conf['run']['max_steps']
        self.save_model = conf['run']['save_model']
        self.logger = logger

        if conf['run']['debug_tm_state']:
            self.state_logger = HTMWriter(
                'belief_tm',
                os.environ.get('TM_STATE_PATH', 'tm_state_log'),
                self.hmm,
                conf['run']['chunk_size']
            )
        else:
            self.state_logger = None

        self.tm_state_debug_range = conf['run'].get('debug_range', [0, self.n_episodes])

        if self.logger is not None:
            for i in range(self.n_policies):
                self.mpg.set_policy(i)
                im_name = f'/tmp/policy_{i}_{self.logger.name}.png'
                draw_mpg(
                    im_name,
                    self.mpg.transition_probs,
                    self.mpg.transition_letters
                )
                self.logger.log({f'mpg_policy_{i}': wandb.Image(im_name)})

    def run(self):
        dist = np.zeros((self.n_policies, len(self.mpg.states), len(self.mpg.alphabet)))
        dist_disp = np.zeros((self.n_policies, len(self.mpg.states), len(self.mpg.alphabet)))
        true_dist = np.zeros((self.n_policies, len(self.mpg.states), len(self.mpg.alphabet)))

        for pol in range(self.n_policies):
            self.mpg.set_policy(pol)
            true_dist[pol] = np.array(
                [self.mpg.predict_letters(from_state=i) for i in self.mpg.states]
            )

        total_surprise = 0
        total_dkl = 0

        for i in range(self.n_episodes):
            self.mpg.reset()
            self.hmm.reset()

            dkls = []
            surprises = []
            anomalies = []
            confidences = []
            word = []

            policy = self.mpg.rng.integers(self.n_policies)
            self.mpg.set_policy(policy)

            steps = 0

            while True:
                prev_state = self.mpg.current_state

                letter = self.mpg.next_state()

                word.append(letter)

                if letter is None:
                    obs_state = []
                else:
                    obs_state = [self.mpg.char_to_num[letter]]

                prediction_context = self.hmm.get_active_cells()
                if not self.use_feedback:
                    prediction_context = np.append(
                        prediction_context,
                        self.hmm.columns * self.hmm.cells_per_column + policy
                    )

                self.hmm.set_active_context_cells(prediction_context)
                self.hmm.activate_basal_dendrites(learn=True)

                if self.use_feedback:
                    self.hmm.set_active_feedback_cells([policy])
                    self.hmm.activate_apical_dendrites(learn=True)

                self.hmm.predict_cells()
                self.hmm.predict_columns_density()

                column_probs = self.hmm.column_probs[:self.n_obs_states]

                # set winner cells from the previous step
                learning_context = self.hmm.get_winner_cells()
                if not self.use_feedback:
                    learning_context = np.append(
                        learning_context,
                        self.hmm.columns * self.hmm.cells_per_column + policy
                    )

                self.hmm.set_active_context_cells(learning_context)
                self.hmm.set_active_columns(obs_state)
                self.hmm.activate_cells(learn=True)

                # debug
                if (
                        self.state_logger is not None
                        and
                        (self.tm_state_debug_range[0] <= i <= self.tm_state_debug_range[1])
                ):
                    if len(word) > 1:
                        prev_letter = word[-2]
                    else:
                        prev_letter = None

                    self.state_logger.write(letter, prev_letter, str(policy))

                # metrics
                if letter is not None:
                    # 1. surprise
                    surprise = self.get_surprise(column_probs, obs_state)
                    surprises.append(surprise)
                    total_surprise += surprise

                    # 2. anomaly
                    anomalies.append(self.hmm.anomaly[-1])

                # 3. distribution
                if None in self.mpg.alphabet:
                    column_probs = np.append(column_probs, 1 - column_probs.sum())

                delta = column_probs - dist[policy][prev_state]
                dist_disp[policy][prev_state] += self.smf_dist * (
                        np.power(delta, 2) - dist_disp[policy][prev_state])
                dist[policy][prev_state] += self.smf_dist * delta

                # 4. Kl distance
                dkl = min(
                        rel_entr(true_dist[policy][prev_state], column_probs).sum(),
                        200.0
                    )
                dkls.append(dkl)
                total_dkl += dkl

                # 5. confidence
                confidences.append(self.hmm.confidence[-1])

                steps += 1

                if self.use_feedback and (steps == 0):
                    self.hmm.set_active_feedback_cells([])
                    self.hmm.activate_apical_dendrites(learn=False)

                if letter is None:
                    break

                if steps >= self.max_steps:
                    break

            if self.logger is not None:
                self.logger.log(
                    {
                        'main_metrics/surprise': np.array(surprises[1:]).mean(),
                        'main_metrics/anomaly': np.array(anomalies[1:]).mean(),
                        'main_metrics/confidence': np.array(confidences[1:]).mean(),
                        'main_metrics/dkl': np.abs(dkls[1:]).mean(),
                        'main_metrics/total_surprise': total_surprise,
                        'main_metrics/total_dkl': total_dkl,
                        'main_metrics/steps': steps,
                        'connections/basal_segments': self.hmm.basal_connections.numSegments(),
                        'connections/inhib_segments': self.hmm.inhib_connections.numSegments(),
                        'connections/apical_segments': self.hmm.apical_connections.numSegments()
                    }, step=i
                )

                if (self.log_update_rate is not None) and (i % self.log_update_rate == 0):
                    # distributions
                    kl_divs = rel_entr(true_dist, dist).sum(axis=-1)

                    n_states = len(self.mpg.states)
                    k = int(np.ceil(np.sqrt(n_states)))

                    tick_labels = self.mpg.alphabet.copy()
                    tick_labels = ['∅' if x is None else x for x in tick_labels]

                    for pol in range(self.n_policies):
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
                                np.arange(dist[pol][n].shape[0]),
                                dist[pol][n],
                                tick_label=tick_labels,
                                label='TM',
                                color=(0.7, 1.0, 0.3),
                                capsize=4,
                                ecolor='#2b4162',
                                yerr=np.sqrt(dist_disp[pol][n])
                            )
                            ax.bar(
                                np.arange(dist[pol][n].shape[0]),
                                true_dist[pol][n],
                                tick_label=tick_labels,
                                color='#8F754F',
                                alpha=0.6,
                                label='True'
                            )

                            fig.legend(['Predicted', 'True'], loc=8)

                            self.logger.log(
                                {f'density/letter_predictions_policy_{pol}': wandb.Image(fig)},
                                step=i
                            )

                            plt.close(fig)

        if self.state_logger is not None:
            self.state_logger.save()

        if self.logger is not None and self.save_model:
            name = self.logger.name

            path = Path('logs')
            if not path.exists():
                path.mkdir()

            np.save(f'logs/dist_{name}.npy', dist)

            with open(f"logs/model_{name}.pkl", 'wb') as file:
                pickle.dump((self.mpg, self.hmm), file)

    @staticmethod
    def get_surprise(probs, obs):
        is_coincide = np.isin(
            np.arange(len(probs)), obs
        )
        surprise = - np.sum(
            np.log(
                np.clip(probs[is_coincide], 1e-7, 1)
            )
        )
        surprise += - np.sum(
            np.log(
                np.clip(1 - probs[~is_coincide], 1e-7, 1)
            )
        )

        return surprise


class NStepTest:
    def __init__(self, logger, conf):
        with open(conf['run']['model_path'], 'rb') as file:
            self.mpg, self.hmm = pickle.load(file)

        self.policy = conf['run'].get('policy', None)

        if self.policy is not None:
            self.mpg.initial_policy = self.policy
        else:
            self.policy = self.mpg.initial_policy

        self.n_obs_states = len(self.mpg.alphabet)

        self.mc_iterations = conf['run']['mc_iterations']
        self.n_steps = conf['run']['n_steps']
        self.logger = logger
        self._rng = np.random.default_rng(conf['run']['seed'])
        self.mpg.reset()

        if self.logger is not None:
            im_name = f'/tmp/mpg_{self.logger.name}.png'
            draw_mpg(
                im_name,
                self.mpg.transition_probs,
                self.mpg.transition_letters
            )

            self.logger.log({'mpg': wandb.Image(im_name)})

    def run(self):
        self.hmm.reset()
        self.mpg.reset()

        k = int(np.ceil(np.sqrt(self.n_steps)))
        fig, axs = plt.subplots(k, k, figsize=(10, 10))
        fig.tight_layout(pad=3.0)

        self.hmm.set_active_columns([])
        self.hmm.activate_basal_dendrites(learn=False)
        self.hmm.predict_cells()
        self.hmm.set_active_context_cells(self.hmm.get_active_cells())
        self.hmm.predict_columns_density(update_receptive_fields=False)

        dkls = []

        for step in range(self.n_steps):
            predicted_dist = self.hmm.column_probs[:self.n_obs_states]

            if step == 0:
                labels = ['Predicted', 'True']
                letter = self.mpg.next_state()

                obs_state = [
                        self.mpg.char_to_num[letter]
                    ]

                self.hmm.set_active_columns(obs_state)
                self.hmm.activate_apical_dendrites(learn=False)
                self.hmm.predict_cells()
                self.hmm.activate_cells(learn=False)
                active_cells = self.hmm.get_active_cells()
                self.hmm.set_active_context_cells(active_cells)
                self.hmm.predict_columns_density(update_receptive_fields=False)
            else:
                labels = [None, None]
                self.hmm.predict_n_step_density(1, mc_iterations=self.mc_iterations)

            true_dist = self.mpg.predict_letters(from_state=0, steps=step)

            true_dist = np.append(
                true_dist,
                np.clip(1 - true_dist.sum(), 0, 1)
            )

            predicted_dist = np.append(
                predicted_dist,
                np.clip(1 - predicted_dist.sum(), 0, 1)
            )

            kl_div = rel_entr(true_dist, predicted_dist).sum()
            dkls.append(kl_div)
            
            if self.logger is not None:
                self.logger.log(
                    {
                        'main_metrics/dkl': kl_div
                    },
                    step=step
                )

            tick_labels = self.mpg.alphabet.copy()
            tick_labels.append('∅')

            ax = axs[step // k][step % k]
            ax.grid()
            ax.set_ylim(0, 1)
            ax.bar(
                np.arange(predicted_dist.shape[0]),
                predicted_dist,
                tick_label=tick_labels,
                color=(0.7, 1.0, 0.3),
                label=labels[0]
            )

            ax.bar(
                np.arange(true_dist.shape[0]),
                true_dist,
                tick_label=tick_labels,
                color=(0.8, 0.5, 0.5),
                alpha=0.6,
                label=labels[1]
            )

            ax.set_title(f'steps: {step + 1}; KL: {np.round(kl_div, 2)}')

        fig.legend(loc=7)

        if self.logger is not None:
            dkls = np.array(dkls)
            self.logger.log({f'density/n_step_letter_predictions': wandb.Image(fig)})
            self.logger.log(
                {
                    f'main_metrics/average_dkl': np.abs(dkls).mean(where=~np.isinf(dkls))
                }
            )
        else:
            plt.show()


class PinballTest:
    def __init__(self, logger, conf):
        self.seed = conf['run']['seed']

        conf['hmm']['seed'] = self.seed
        conf['env']['seed'] = self.seed
        conf['env']['exe_path'] = os.environ.get('PINBALL_EXE', None)
        conf['env']['config_path'] = os.path.join(
            os.environ.get('PINBALL_ROOT', None),
            'configs',
            f"{conf['run']['setup']}.json"
        )

        self.env = Pinball(**conf['env'])

        obs = self.env.obs()
        self.obs_shape = (obs.shape[0], obs.shape[1])

        sp_conf = conf.get('sp', None)
        if sp_conf is not None:
            sp_conf['seed'] = self.seed
            self.encoder = HtmSpatialPooler(
                self.obs_shape,
                **sp_conf
            )
            self.obs_shape = self.encoder.getColumnDimensions()
            self.sp_input = SDR(self.encoder.getInputDimensions())
            self.sp_output = SDR(self.encoder.getColumnDimensions())

            self.decoder = SPDecoder(self.encoder)
        else:
            self.encoder = None
            self.sp_input = None
            self.sp_output = None
            self.decoder = None

        self.n_obs_vars = self.obs_shape[0] * self.obs_shape[1]
        self.n_obs_states = 1

        conf['hmm']['columns'] = self.n_obs_vars
        conf['hmm']['context_cells'] = conf['hmm']['columns'] * conf['hmm']['cells_per_column']

        self.hmm = HybridNaiveBayesTM(**conf['hmm'])

        self.actions = conf['run']['actions']
        self.positions = conf['run']['positions']
        self.prediction_steps = conf['run']['prediction_steps']
        self.n_episodes = conf['run']['n_episodes']
        self.log_update_rate = conf['run']['update_rate']
        self.max_steps = conf['run']['max_steps']
        self.save_model = conf['run']['save_model']
        self.log_fps = conf['run']['log_gif_fps']
        self.mc_iterations = conf['run']['mc_iterations']

        self._rng = np.random.default_rng(self.seed)

        self.logger = logger

        if self.logger is not None:
            self.logger.log(
                {
                    'setting': wandb.Image(
                        plt.imshow(self.env.obs())
                    )
                },
                step=0
            )

    def run(self):
        total_surprise = 0
        total_surprise_decoder = 0

        for i in range(self.n_episodes):
            surprises = []
            anomalies = []
            surprises_decoder = []

            obs_probs_stack = []
            hidden_probs_stack = []
            n_step_surprise_obs = [list() for t in range(self.prediction_steps)]
            n_step_surprise_hid = [list() for t in range(self.prediction_steps)]

            steps = 0

            if self.encoder is not None:
                prev_latent = np.zeros(self.encoder.getColumnDimensions())
            else:
                prev_latent = None

            if (self.logger is not None) and (i % self.log_update_rate == 0):
                writer_raw = imageio.get_writer(
                    f'/tmp/{self.logger.name}_raw_ep{i}.gif',
                    mode='I',
                    fps=self.log_fps
                )
                if self.encoder is not None:
                    writer_hidden = imageio.get_writer(
                        f'/tmp/{self.logger.name}_hidden_ep{i}.gif',
                        mode='I',
                        fps=self.log_fps
                    )
                else:
                    writer_hidden = None
            else:
                writer_raw = None
                writer_hidden = None

            init_i = self._rng.integers(0, len(self.actions), 1)
            action = self.actions[init_i[0]]
            position = self.positions[init_i[0]]
            self.env.reset(position)
            self.env.act(action)

            self.hmm.reset()

            self.env.step()
            prev_im = self.preprocess(self.env.obs())
            prev_diff = np.zeros_like(prev_im)

            while True:
                self.env.step()
                raw_im = self.preprocess(self.env.obs())
                thresh = raw_im.mean()
                diff = np.abs(raw_im - prev_im) >= thresh
                prev_im = raw_im.copy()

                obs_state = np.flatnonzero(diff)

                if self.encoder is not None:
                    self.sp_input.sparse = obs_state
                    self.encoder.compute(self.sp_input, True, self.sp_output)
                    obs_state = self.sp_output.sparse

                self.hmm.set_active_context_cells(self.hmm.get_active_cells())
                self.hmm.activate_basal_dendrites(learn=True)
                self.hmm.predict_cells()
                self.hmm.predict_columns_density()

                column_probs = self.hmm.column_probs

                # set winner cells from previous step
                self.hmm.set_active_context_cells(self.hmm.get_winner_cells())
                self.hmm.set_active_columns(obs_state)
                self.hmm.activate_cells(learn=True)

                if steps > 0:
                    # metrics
                    # 0. anomaly
                    anomalies.append(self.hmm.anomaly[-1])

                    # 1. surprise
                    surprise = get_surprise(column_probs, obs_state)
                    surprises.append(surprise)
                    total_surprise += surprise

                    if self.decoder is not None:
                        decoded_probs = self.decoder.decode(column_probs, update=True)

                        surprise_decoder = get_surprise(decoded_probs, self.sp_input.sparse)

                        surprises_decoder.append(surprise_decoder)
                        total_surprise_decoder += surprise_decoder

                    # 2. image
                    if (writer_raw is not None) and (i % self.log_update_rate == 0):
                        obs_probs = []
                        hidden_probs = []

                        if self.decoder is not None:
                            hidden_prediction = column_probs.reshape(self.obs_shape)
                            decoded_probs = self.decoder.decode(column_probs, update=True)
                            decoded_probs = decoded_probs.reshape(self.encoder.getInputDimensions())
                        else:
                            decoded_probs = column_probs.reshape(self.obs_shape)
                            hidden_prediction = None

                        raw_predictions = [(decoded_probs * 255).astype(np.uint8)]

                        if hidden_prediction is not None:
                            hidden_predictions = [(hidden_prediction * 255).astype(np.uint8)]
                        else:
                            hidden_predictions = None

                        obs_probs.append(decoded_probs.copy())
                        hidden_probs.append(hidden_prediction.copy())

                        for j in range(self.prediction_steps - 1):
                            self.hmm.predict_n_step_density(1, mc_iterations=self.mc_iterations)
                            column_probs = self.hmm.column_probs

                            if self.decoder is not None:
                                hidden_prediction = column_probs.reshape(self.obs_shape)
                                decoded_probs = self.decoder.decode(column_probs)
                                decoded_probs = decoded_probs.reshape(self.encoder.getInputDimensions())
                            else:
                                decoded_probs = column_probs.reshape(self.obs_shape)
                                hidden_prediction = None

                            obs_probs.append(decoded_probs.copy())
                            hidden_probs.append(hidden_prediction.copy())

                            raw_predictions.append(
                                (decoded_probs * 255).astype(np.uint8)
                            )

                            if hidden_predictions is not None:
                                hidden_predictions.append(
                                    (hidden_prediction * 255).astype(np.uint8)
                                )

                        obs_probs_stack.append(copy(obs_probs))
                        hidden_probs_stack.append(copy(hidden_probs))

                        # remove empty lists
                        obs_probs_stack = [x for x in obs_probs_stack if len(x) > 0]
                        hidden_probs_stack = [x for x in hidden_probs_stack if len(x) > 0]

                        pred_horizon = [self.prediction_steps - len(x) for x in obs_probs_stack]
                        current_predictions_obs = [x.pop(0) for x in obs_probs_stack]
                        current_predictions_hid = [x.pop(0) for x in hidden_probs_stack]

                        for p_obs, p_hid, s in zip(current_predictions_obs, current_predictions_hid, pred_horizon):
                            surp_obs = get_surprise(p_obs.flatten(), self.sp_input.sparse)
                            surp_hid = get_surprise(p_hid.flatten(), self.sp_output.sparse)
                            n_step_surprise_obs[s].append(surp_obs)
                            n_step_surprise_hid[s].append(surp_hid)

                        raw_im = [prev_diff.astype(np.uint8)*255]
                        raw_im.extend(raw_predictions)
                        raw_im = np.hstack(raw_im)
                        writer_raw.append_data(raw_im)

                        if hidden_predictions is not None:
                            hid_im = [prev_latent.astype(np.uint8) * 255]
                            hid_im.extend(hidden_predictions)
                            hid_im = np.hstack(hid_im)
                            writer_hidden.append_data(hid_im)

                steps += 1
                prev_diff = diff.copy()
                prev_latent = self.sp_output.dense.copy()

                if steps >= self.max_steps:
                    if writer_raw is not None:
                        writer_raw.close()

                    if writer_hidden is not None:
                        writer_hidden.close()

                    break

            if self.logger is not None:
                self.logger.log(
                    {
                        'main_metrics/surprise': np.array(surprises).mean(),
                        'spec_metrics/anomaly': np.array(anomalies).mean(),
                        'main_metrics/total_surprise': total_surprise,
                        'main_metrics/steps': steps,
                        'connections/basal_segments': self.hmm.basal_connections.numSegments(),
                        'connections/inhib_segments': self.hmm.inhib_connections.numSegments(),
                        'connections/apical_segments': self.hmm.apical_connections.numSegments()
                    }, step=i
                )

                if self.decoder is not None:
                    self.logger.log(
                        {
                            'main_metrics/surprise_decoder': np.array(surprises_decoder).mean(),
                            'main_metrics/total_surprise_decoder': total_surprise_decoder,
                        }, step=i
                    )

                if (self.log_update_rate is not None) and (i % self.log_update_rate == 0):
                    n_step_surprises_hid = {
                        f'n_step_hidden/surprise_step_{s+1}': np.mean(x) for s, x in enumerate(n_step_surprise_hid)
                    }
                    n_step_surprises_obs = {
                        f'n_step_raw/surprise_step_{s+1}': np.mean(x) for s, x in enumerate(n_step_surprise_obs)
                    }

                    self.logger.log(
                        n_step_surprises_obs,
                        step=i
                    )

                    self.logger.log(
                        n_step_surprises_hid,
                        step=i
                    )

                    self.logger.log(
                        {
                            'gifs/raw_prediction': wandb.Video(
                                f'/tmp/{self.logger.name}_raw_ep{i}.gif'
                            )
                        },
                        step=i
                    )
                    if writer_hidden is not None:
                        self.logger.log(
                            {
                                'gifs/hidden_prediction': wandb.Video(
                                    f'/tmp/{self.logger.name}_hidden_ep{i}.gif'
                                )
                            },
                            step=i
                        )

        if self.logger is not None and self.save_model:
            name = self.logger.name

            path = Path('logs')
            if not path.exists():
                path.mkdir()

            with open(f"logs/model_{name}.pkl", 'wb') as file:
                pickle.dump(self.hmm, file)

    @staticmethod
    def preprocess(image):
        gray_im = image.sum(axis=-1)
        gray_im /= gray_im.max()

        return gray_im


def main(config_path):
    if len(sys.argv) > 1:
        config_path = sys.argv[1]

    config = dict()

    with open(config_path, 'r') as file:
        config['run'] = yaml.load(file, Loader=yaml.Loader)

    with open(config['run']['hmm_conf'], 'r') as file:
        config['hmm'] = yaml.load(file, Loader=yaml.Loader)

    with open(config['run']['env_conf'], 'r') as file:
        config['env'] = yaml.load(file, Loader=yaml.Loader)

    sp_conf = config['run'].get('sp_conf', None)
    if sp_conf is not None:
        with open(sp_conf, 'r') as file:
            config['sp'] = yaml.load(file, Loader=yaml.Loader)

    for arg in sys.argv[2:]:
        key, value = arg.split('=')

        try:
            value = ast.literal_eval(value)
        except ValueError:
            ...

        key = key.lstrip('-')
        if key.endswith('.'):
            # a trick that allow distinguishing sweep params from config params
            # by adding a suffix `.` to sweep param - now we should ignore it
            key = key[:-1]
        tokens = key.split('.')
        c = config
        for k in tokens[:-1]:
            if not k:
                # a trick that allow distinguishing sweep params from config params
                # by inserting additional dots `.` to sweep param - we just ignore it
                continue
            if 0 in c:
                k = int(k)
            c = c[k]
        c[tokens[-1]] = value

    if config['run']['seed'] is None:
        config['run']['seed'] = np.random.randint(0, np.iinfo(np.int32).max)

    if config['run']['log']:
        logger = wandb.init(
            project=config['run']['project_name'], entity=os.environ.get('WANDB_ENTITY', None),
            config=config
        )
    else:
        logger = None

    experiment = config['run']['experiment']

    if experiment == 'mpg':
        runner = MPGTest(logger, config)
    elif experiment == 'mmpg':
        runner = MMPGTest(logger, config)
    elif experiment == 'nstep':
        runner = NStepTest(logger, config)
    elif experiment == 'pinball':
        runner = PinballTest(logger, config)
    else:
        raise ValueError

    runner.run()


if __name__ == '__main__':
    default_config = 'configs/belieftm/mpg_single.yaml'
    main(os.environ.get('RUN_CONF', default_config))
