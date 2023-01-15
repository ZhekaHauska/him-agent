#  Copyright (c) 2022 Autonomous Non-Profit Organization "Artificial Intelligence Research
#  Institute" (AIRI); Moscow Institute of Physics and Technology (National Research University).
#  All rights reserved.
#
#  Licensed under the AGPLv3 license. See LICENSE in the project root for license information.

from hima.modules.hmm import CHMMBasic
from hima.envs.mpg.mpg import MultiMarkovProcessGrammar, draw_mpg
import numpy as np
from scipy.special import rel_entr
from pathlib import Path
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import wandb
import yaml
import os
import sys
import ast


class HMMRunner:
    def __init__(self, logger, conf):
        self.seed = conf['run']['seed']

        conf['hmm']['seed'] = self.seed
        conf['mpg']['seed'] = self.seed

        self.mpg = MultiMarkovProcessGrammar(**conf['mpg'])

        conf['hmm']['n_columns'] = len(self.mpg.alphabet)
        self.hmm = CHMMBasic(**conf['hmm'])

        self.n_episodes = conf['run']['n_episodes']
        self.smf_dist = conf['run']['smf_dist']
        self.log_update_rate = conf['run']['update_rate']
        self.max_steps = conf['run']['max_steps']
        self.save_model = conf['run']['save_model']
        self.n_steps = conf['run'].get('n_step_test', None)
        self.logger = logger

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

            steps = 0

            while True:
                prev_state = self.mpg.current_state

                letter = self.mpg.next_state()

                if letter is None:
                    break
                else:
                    obs_state = self.mpg.char_to_num[letter]

                column_probs = self.hmm.predict_columns()
                self.hmm.observe(obs_state, learn=True)

                # metrics
                # 1. surprise
                if prev_state != 0:
                    active_columns = np.arange(self.hmm.n_columns) == obs_state
                    surprise = - np.sum(np.log(column_probs[active_columns]))
                    surprise += - np.sum(np.log(1 - column_probs[~active_columns]))

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

                steps += 1

                if steps > self.max_steps:
                    break

            if self.logger is not None:
                self.logger.log(
                    {
                        'main_metrics/surprise': np.array(surprises).mean(),
                        'main_metrics/dkl': np.array(np.abs(dkls)).mean(),
                        'main_metrics/total_surprise': total_surprise,
                        'main_metrics/total_dkl': total_dkl,
                    }, step=i
                )

                if (self.log_update_rate is not None) and (i % self.log_update_rate == 0):
                    kl_divs = rel_entr(true_dist, dist).sum(axis=-1)

                    n_states = len(self.mpg.states)
                    k = int(np.ceil(np.sqrt(n_states)))
                    fig, axs = plt.subplots(k, k)
                    fig.tight_layout(pad=3.0)

                    tick_labels = self.mpg.alphabet.copy()
                    tick_labels.append('∅')

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

                    self.logger.log(
                        {
                            'weights/priors': wandb.Image(
                                sns.heatmap(
                                    self.hmm.log_state_prior.reshape((1, -1)),
                                    cmap='coolwarm'
                                )
                            )
                        },
                        step=i
                    )
                    plt.close('all')

                    self.logger.log(
                        {
                            'weights/prior_probs': wandb.Image(
                                sns.heatmap(
                                    self.hmm.state_prior.reshape((1, -1)),
                                    cmap='coolwarm'
                                )
                            )
                        },
                        step=i
                    )
                    plt.close('all')

                    self.logger.log(
                        {
                            'weights/transitions': wandb.Image(
                                sns.heatmap(self.hmm.log_transition_factors,
                                    cmap='coolwarm')
                            )
                        },
                        step=i
                    )
                    plt.close('all')

                    self.logger.log(
                        {
                            'weights/transition_probs': wandb.Image(
                                sns.heatmap(self.hmm.transition_probs,
                                    cmap='coolwarm')
                            )
                        },
                        step=i
                    )
                    plt.close('all')

        if self.logger is not None and self.save_model:
            name = self.logger.name

            path = Path('logs')
            if not path.exists():
                path.mkdir()

            np.save(f'logs/dist_{name}.npy', dist)

            with open(f"logs/model_{name}.pkl", 'wb') as file:
                pickle.dump((self.mpg, self.hmm), file)

        if self.n_steps is not None:
            self.run_n_step()

    def run_n_step(self):
        self.hmm.reset()
        self.mpg.reset()

        k = int(np.ceil(np.sqrt(self.n_steps)))
        fig, axs = plt.subplots(k, k, figsize=(10, 10))
        fig.tight_layout(pad=3.0)

        # super-fast workaround for terminal state without observation
        # TODO do it fairly with additional terminal state
        transition_matrix = self.hmm.transition_probs
        states_for_terminal_obs_state = self.hmm._obs_state_to_hidden(
            self.mpg.char_to_num['E']
        )
        transition_matrix[states_for_terminal_obs_state] = 0

        dkls = []
        n_step_dists = []

        forward_message = self.hmm.state_prior
        predicted_dist = np.reshape(
            forward_message, (self.hmm.n_columns, self.hmm.cells_per_column)
        ).sum(axis=-1)

        for step in range(self.n_steps):

            if step == 0:
                labels = ['Predicted', 'True']
            else:
                labels = [None, None]
                forward_message = np.dot(forward_message, transition_matrix)
                predicted_dist = np.reshape(
                    forward_message, (self.hmm.n_columns, self.hmm.cells_per_column)
                ).sum(axis=-1)

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


def main(config_path):
    if len(sys.argv) > 1:
        config_path = sys.argv[1]

    config = dict()

    with open(config_path, 'r') as file:
        config['run'] = yaml.load(file, Loader=yaml.Loader)

    with open(config['run']['hmm_conf'], 'r') as file:
        config['hmm'] = yaml.load(file, Loader=yaml.Loader)

    with open(config['run']['mpg_conf'], 'r') as file:
        config['mpg'] = yaml.load(file, Loader=yaml.Loader)

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

    if config['run']['log']:
        logger = wandb.init(
            project=config['run']['project_name'], entity=os.environ['WANDB_ENTITY'],
            config=config
        )
    else:
        logger = None

    runner = HMMRunner(logger, config)
    runner.run()


if __name__ == '__main__':
    default_config = 'configs/runner/hmm/mpg_single.yaml'
    main(os.environ.get('RUN_CONF', default_config))
