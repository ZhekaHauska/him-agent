#  Copyright (c) 2022 Autonomous Non-Profit Organization "Artificial Intelligence Research
#  Institute" (AIRI); Moscow Institute of Physics and Technology (National Research University).
#  All rights reserved.
#
#  Licensed under the AGPLv3 license. See LICENSE in the project root for license information.

from hima.modules.baselines.hmm import CHMMBasic
from hima.experiments.temporal_pooling.stp.sp_ensemble import SpatialPoolerGroupedWrapper
from hima.envs.mpg.mpg import MultiMarkovProcessGrammar, draw_mpg
from hima.common.run.argparse import parse_arg_list
from hima.common.config.base import read_config, override_config
import numpy as np
from scipy.special import rel_entr
from pathlib import Path
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import wandb
import os
import sys
import igraph


def plot_graph(chmm, x, a, output_file, cmap=cm.Spectral, multiple_episodes=False, vertex_size=30):
    """
    Restore markov chain graph from transition matrix.
    """

    states = chmm.decode(x, a)[1]
    v = np.unique(states)
    if multiple_episodes:
        T = chmm.C[:, v][:, :, v][:-1, 1:, 1:]
        v = v[1:]
    else:
        T = chmm.C[:, v][:, :, v]

    A = T.sum(0)
    A /= A.sum(1, keepdims=True)

    g = igraph.Graph.Adjacency((A > 0).tolist())
    node_labels = np.arange(x.max() + 1).repeat(n_clones)[v]
    if multiple_episodes:
        node_labels -= 1

    colors = [cmap(nl)[:3] for nl in node_labels / node_labels.max()]
    out = igraph.plot(
        g,
        output_file,
        layout=g.layout("kamada_kawai"),
        vertex_color=colors,
        vertex_label=v,
        vertex_size=vertex_size,
        margin=50,
    )

    return out


def generate_sr(hmm: CHMMBasic, mpg: MultiMarkovProcessGrammar, steps, gamma):
    forward_message = hmm.forward_message
    prediction = np.reshape(
        forward_message, (hmm.n_columns, hmm.cells_per_column)
    ).sum(axis=-1)

    transition_matrix = hmm.transition_probs
    states_for_terminal_obs_state = hmm._obs_state_to_hidden(
        mpg.char_to_num['E']
    )
    transition_matrix[states_for_terminal_obs_state] = 0

    sr = np.zeros(hmm.n_columns)
    discount = 1.0
    for step in range(steps):
        sr += discount * prediction

        forward_message = np.dot(forward_message, transition_matrix)
        prediction = np.reshape(
            forward_message, (hmm.n_columns, hmm.cells_per_column)
        ).sum(axis=-1)

        discount *= gamma

    return sr


class GridworldTest:
    def __init__(self, logger, conf):
        self.seed = conf['run']['seed']

        conf['hmm']['seed'] = self.seed
        conf['env']['seed'] = self.seed

        self.env = self.setup_environment(conf['']['env_conf'])

        conf['hmm']['n_columns'] = len(self.mpg.alphabet)
        self.hmm = CHMMBasic(**conf['hmm'])

        external_hmm = conf['run'].get('external_prior_hmm', None)
        if external_hmm is not None:
            with open(external_hmm, 'rb') as file:
                self.ext_hmm, _ = pickle.load(file)
        else:
            self.ext_hmm = None

        encoder_conf = conf['encoder']
        encoder_conf['seed'] = self.seed
        encoder_conf['feedforward_sds'] = [[self.ext_hmm.n_columns, 1], 1]
        encoder_conf['output_sds'] = [
            [self.hmm.cells_per_column, self.hmm.n_columns],
            self.hmm.n_columns
        ]

        self.encoder = SpatialPoolerGroupedWrapper(**encoder_conf)

        self.n_episodes = conf['run']['n_episodes']
        self.smf_dist = conf['run']['smf_dist']
        self.log_update_rate = conf['run']['update_rate']
        self.max_steps = conf['run']['max_steps']
        self.save_model = conf['run']['save_model']
        self.log_path = conf['run']['log_path']
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
            if self.ext_hmm is not None:
                self.ext_hmm.reset()

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

                if self.ext_hmm is not None:
                    self.ext_hmm.observe(obs_state, learn=False)
                    sr = generate_sr(self.ext_hmm)

                    probs = self.encoder.compute(sr)
                    self.hmm.set_external_hidden_state_prior(probs)

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

            path = Path(self.log_path)
            if not path.exists():
                path.mkdir()

            np.save(f'{self.log_path}/dist_{name}.npy', dist)

            with open(f"{self.log_path}/model_{name}.pkl", 'wb') as file:
                pickle.dump((self.mpg, self.hmm), file)

        if self.n_steps is not None:
            self.run_n_step()

    def setup_environment(self, setup: str):
        from hima.envs.gridworld import GridWorld

        config = read_config(os.path.join(
            os.environ.get('GRIDWORLD_ROOT', None),
            f"{setup}.yaml"
        ))

        env = GridWorld(
            **{
                'room': np.array(config['room']),
                'default_reward': config['default_reward'],
                'seed': self.seed
            }
        )

        raw_obs_shape = (np.max(env.colors) + 1, 1)
        actions = list(env.actions)
        n_actions = len(actions)

        return env, raw_obs_shape, actions, n_actions


def main(config_path):
    if len(sys.argv) > 1:
        config_path = sys.argv[1]

    config = dict()

    config['run'] = read_config(config_path)
    if 'env_conf' in config['run']:
        config['env'] = read_config(config['run']['env_conf'])

    config['hmm'] = read_config(config['run']['hmm_conf'])

    overrides = parse_arg_list(sys.argv[2:])
    override_config(config, overrides)

    if config['run']['seed'] is None:
        config['run']['seed'] = np.random.randint(0, np.iinfo(np.int32).max)

    if config['run']['log']:
        import wandb
        logger = wandb.init(
            project=config['run']['project_name'], entity=os.environ.get('WANDB_ENTITY', None),
            config=config
        )
    else:
        logger = None

    experiment = config['run']['experiment']

    if experiment == 'mpg':
        runner = MPGTest(logger, config)
    else:
        raise ValueError

    runner.run()


if __name__ == '__main__':
    default_config = 'configs/runner/mpg_single.yaml'
    main(os.environ.get('RUN_CONF', default_config))
