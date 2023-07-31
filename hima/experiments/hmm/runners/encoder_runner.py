#  Copyright (c) 2022 Autonomous Non-Profit Organization "Artificial Intelligence Research
#  Institute" (AIRI); Moscow Institute of Physics and Technology (National Research University).
#  All rights reserved.
#
#  Licensed under the AGPLv3 license. See LICENSE in the project root for license information.
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import seaborn as sns
import wandb
import yaml
import os
import sys
import ast
import pickle
import imageio
from copy import copy

from hima.common.utils import prepend_dict_keys
from hima.modules.belief.cortial_column.layer import Layer, REAL_DTYPE, UINT_DTYPE
from hima.modules.belief.cortial_column.input_layer import Encoder, Decoder
from htm.bindings.sdr import SDR
from hima.experiments.hmm.runners.utils import get_surprise

try:
    from pinball import Pinball
except ModuleNotFoundError:
    Pinball = None


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
        self.raw_obs_shape = (obs.shape[0], obs.shape[1])

        self.encoder_type = conf['run']['encoder']
        sp_conf = conf.get('sp', None)
        input_layer_conf = conf.get('input_layer', None)
        decoder_conf = conf.get('decoder', None)

        if self.encoder_type == 'one_sp':
            from hima.modules.htm.spatial_pooler import SPDecoder, HtmSpatialPooler
            sp_conf['seed'] = self.seed
            self.encoder = HtmSpatialPooler(self.raw_obs_shape, **sp_conf)
            self.obs_shape = self.encoder.getColumnDimensions()
            self.sp_input = SDR(self.encoder.getInputDimensions())
            self.sp_output = SDR(self.encoder.getColumnDimensions())

            self.decoder = SPDecoder(self.encoder)

            self.n_obs_vars = self.obs_shape[0] * self.obs_shape[1]
            self.n_obs_states = 1

            self.surprise_mode = 'bernoulli'
        elif self.encoder_type == 'sp_ensemble':
            from hima.modules.htm.spatial_pooler import SPDecoder, HtmSpatialPooler, SPEnsemble
            sp_conf['seed'] = self.seed
            sp_conf['inputDimensions'] = list(self.raw_obs_shape)
            n_sp = sp_conf.pop('n_sp')
            self.encoder = SPEnsemble(n_sp, **sp_conf)
            shape = self.encoder.sps[0].getColumnDimensions()
            self.obs_shape = (shape[0] * self.encoder.n_sp, shape[1])
            self.sp_input = SDR(self.encoder.getNumInputs())
            self.sp_output = SDR(self.encoder.getNumColumns())

            if decoder_conf is not None:
                decoder_conf['n_context_vars'] = n_sp
                decoder_conf['n_context_states'] = shape[0]*shape[1]
                decoder_conf['n_obs_vars'] = self.raw_obs_shape[0] * self.raw_obs_shape[1]
                decoder_conf['n_obs_states'] = 2
                self.decoder = Decoder(**decoder_conf)
            else:
                self.decoder = SPDecoder(self.encoder)

            self.n_obs_vars = self.encoder.n_sp
            self.n_obs_states = self.encoder.sps[0].getNumColumns()

            self.surprise_mode = 'categorical'
        elif self.encoder_type == 'new_sp_ensemble':
            from hima.experiments.temporal_pooling.stp.sp_ensemble import SpatialPoolerEnsemble
            sp_conf['seed'] = self.seed
            sp_conf['feedforward_sds'] = [self.raw_obs_shape, 0.1]
            self.encoder = SpatialPoolerEnsemble(**sp_conf)
            shape = self.encoder.getSingleColumnsDimensions()
            self.obs_shape = (shape[0] * self.encoder.n_sp, shape[1])
            self.sp_input = SDR(self.encoder.getNumInputs())
            self.sp_output = SDR(self.encoder.getNumColumns())

            if decoder_conf is not None:
                decoder_conf['n_context_vars'] = self.encoder.n_sp
                decoder_conf['n_context_states'] = shape[0]*shape[1]
                decoder_conf['n_obs_vars'] = self.raw_obs_shape[0] * self.raw_obs_shape[1]
                decoder_conf['n_obs_states'] = 2
                self.decoder = Decoder(**decoder_conf)
            else:
                from hima.experiments.temporal_pooling.stp.sp_decoder import SpatialPoolerDecoder
                self.decoder = SpatialPoolerDecoder(self.encoder)

            self.n_obs_vars = self.encoder.n_sp
            self.n_obs_states = self.encoder.getSingleNumColumns()

            self.surprise_mode = 'categorical'
        elif self.encoder_type == 'input_layer':
            assert input_layer_conf is not None
            encoder_conf = input_layer_conf.get('encoder')
            decoder_conf = input_layer_conf.get('decoder')
            encoder_conf['seed'] = self.seed
            decoder_conf['seed'] = self.seed

            encoder_conf['n_context_vars'] = self.raw_obs_shape[0] * self.raw_obs_shape[1]
            encoder_conf['n_context_states'] = 2

            self.encoder = Encoder(**encoder_conf)
            self.n_obs_vars = self.encoder.n_hidden_vars
            self.n_obs_states = self.encoder.n_hidden_states
            self.obs_shape = (self.n_obs_vars, self.n_obs_states)

            decoder_conf['n_context_vars'] = self.encoder.n_hidden_vars
            decoder_conf['n_context_states'] = self.encoder.n_hidden_states
            decoder_conf['n_obs_vars'] = self.encoder.n_context_vars
            decoder_conf['n_obs_states'] = self.encoder.n_context_states

            self.decoder = Decoder(**decoder_conf)

            self.surprise_mode = 'categorical'
            self.sp_input = None
            self.sp_output = None
        else:
            self.encoder = None
            self.sp_input = None
            self.sp_output = None
            self.decoder = None

            self.n_obs_vars = self.raw_obs_shape[0] * self.raw_obs_shape[1]
            self.n_obs_states = 1
            self.obs_shape = self.raw_obs_shape

            self.surprise_mode = 'bernoulli'

        conf['hmm']['n_obs_states'] = self.n_obs_states
        conf['hmm']['n_obs_vars'] = self.n_obs_vars
        conf['hmm']['n_context_states'] = self.n_obs_states * conf['hmm']['cells_per_column']
        conf['hmm']['n_context_vars'] = self.n_obs_vars

        if 'actions' in conf['run']:
            self.actions = conf['run']['actions']
            # add idle action
            self.actions.append([0.0, 0.0])

            conf['hmm']['n_external_vars'] = 1
            conf['hmm']['n_external_states'] = len(self.actions)

            self.idle_action = len(self.actions) - 1
            self.action = len(self.actions)
            self.is_action_observable = conf['run']['action_observable']
            self.action_delay = conf['run']['action_delay']
        else:
            conf['hmm']['n_external_vars'] = 0
            conf['hmm']['n_external_states'] = 0
            self.actions = None
            self.action = None

        self.hmm = Layer(**conf['hmm'])

        self.start_actions = conf['run']['start_actions']
        self.start_positions = conf['run']['start_positions']
        self.prediction_steps = conf['run']['prediction_steps']
        self.n_episodes = conf['run']['n_episodes']
        self.log_update_rate = conf['run']['update_rate']
        self.max_steps = conf['run']['max_steps']
        self.save_model = conf['run']['save_model']
        self.log_fps = conf['run']['log_gif_fps']
        self.internal_dependence_1step = conf['run']['internal_dependence_1step']

        self.rng = np.random.default_rng(self.seed)
        self.logger = logger
        if self.logger is not None:
            self.logger.log({'setting': wandb.Image(self.env.obs())}, step=0)

        # Stats
        self.total_surprise = 0
        self.total_surprise_decoder = 0

    def log_scheduled(self, step):
        return (self.logger is not None) and (step % self.log_update_rate == 0)

    def run(self):
        for episode in range(self.n_episodes):
            self.run_episode(episode)
        self.dump_model()

    def get_gif_writers(self, episode):
        writer_raw = None
        writer_hidden = None
        if self.log_scheduled(episode):
            writer_raw = imageio.get_writer(
                f'/tmp/{self.logger.name}_raw_ep{episode}.gif',
                mode='I',
                duration=1000 / self.log_fps,
            )
            if self.encoder is not None:
                writer_hidden = imageio.get_writer(
                    f'/tmp/{self.logger.name}_hidden_ep{episode}.gif',
                    mode='I',
                    duration=1000 / self.log_fps,
                )
        return writer_raw, writer_hidden

    def run_episode(self, episode):
        surprises = []
        surprises_decoder = []

        obs_probs_stack = []
        hidden_probs_stack = []
        n_step_surprise_obs = [[] for _ in range(self.prediction_steps)]
        n_step_surprise_hid = [[] for _ in range(self.prediction_steps)]
        step = 0

        writer_raw, writer_hidden = self.get_gif_writers(episode)

        init_index = self.rng.choice(len(self.start_actions))
        action = self.start_actions[init_index]
        position = self.start_positions[init_index]
        self.env.reset(position)
        self.env.act(action)

        self.hmm.reset()

        self.env.step()
        prev_im = self.preprocess(self.env.obs())
        prev_diff = np.zeros_like(prev_im)

        prev_latent = None
        if self.encoder is not None:
            prev_latent = np.zeros_like(prev_im)

        initial_context = np.zeros_like(self.hmm.context_messages)
        initial_context[
            np.arange(self.hmm.n_hidden_vars) * self.hmm.n_hidden_states
        ] = 1
        self.hmm.set_context_messages(initial_context)

        while True:
            self.env.step()
            raw_im = self.preprocess(self.env.obs())
            diff = np.abs(raw_im - prev_im) >= raw_im.mean()
            prev_im = raw_im.copy()

            raw_obs_state = np.flatnonzero(diff)

            action_probs = None
            if self.actions is not None:
                if step == self.action_delay:
                    # choose between non-idle actions
                    self.action = self.rng.choice(len(self.actions) - 1)
                    self.env.act(self.actions[self.action])
                else:
                    # choose idle action
                    self.action = self.idle_action

                action = self.action if self.is_action_observable else self.idle_action
                action_probs = one_hot(action, len(self.actions))

            self.hmm.set_external_messages(action_probs)
            self.hmm.predict(
                include_internal_connections=(
                    self.hmm.enable_internal_connections and self.internal_dependence_1step
                )
            )
            column_probs = self.hmm.prediction_columns

            if self.decoder is not None:
                if isinstance(self.decoder, Decoder):
                    dense_raw_obs = np.asarray(diff, dtype=REAL_DTYPE).flatten()
                    observation = np.arange(
                        0,
                        self.decoder.n_obs_vars * self.decoder.n_obs_states,
                        self.decoder.n_obs_states
                    ) + dense_raw_obs

                    decoded_probs = self.decoder.decode(
                        column_probs, observation.astype(UINT_DTYPE)
                    )
                    decoded_probs = decoded_probs[1::2]
                else:
                    decoded_probs = self.decoder.decode(column_probs, learn=True)

            obs_state = raw_obs_state
            if self.encoder is not None:
                if self.encoder_type == 'input_layer':
                    dense_raw_obs = np.asarray(diff, dtype=REAL_DTYPE).flatten()
                    observation = np.zeros(self.encoder.context_input_size)
                    observation[1::2] = dense_raw_obs
                    observation[::2] = 1 - observation[1::2]

                    obs_state = self.encoder.encode(
                        observation,
                        # column_probs,
                        # decoded_probs
                    )
                else:
                    self.sp_input.sparse = raw_obs_state
                    self.encoder.compute(self.sp_input, True, self.sp_output)
                    obs_state = self.sp_output.sparse

            self.hmm.observe(obs_state, learn=True)
            self.hmm.set_context_messages(self.hmm.internal_forward_messages)

            if step > 0:
                # metrics
                # 1. surprise
                surprise = get_surprise(column_probs, obs_state, mode=self.surprise_mode)
                surprises.append(surprise)
                self.total_surprise += surprise

                if self.decoder is not None:
                    surprise_decoder = get_surprise(decoded_probs, raw_obs_state)
                    surprises_decoder.append(surprise_decoder)
                    self.total_surprise_decoder += surprise_decoder

                # 2. image
                if (writer_raw is not None) and (episode % self.log_update_rate == 0):
                    obs_probs = []
                    hidden_probs = []

                    if self.prediction_steps > 1:
                        back_up_massages = self.hmm.context_messages.copy()
                        self.hmm.set_context_messages(self.hmm.prediction_cells)

                    if self.decoder is not None:
                        hidden_prediction = column_probs.reshape(self.obs_shape)
                        decoded_probs = decoded_probs.reshape(self.raw_obs_shape)
                    else:
                        hidden_prediction = None
                        decoded_probs = column_probs.reshape(self.obs_shape)

                    raw_predictions = [(decoded_probs * 255).astype(np.uint8)]

                    if hidden_prediction is not None:
                        hidden_predictions = [(hidden_prediction * 255).astype(np.uint8)]

                    obs_probs.append(decoded_probs.copy())
                    hidden_probs.append(hidden_prediction.copy())

                    for j in range(self.prediction_steps - 1):
                        self.hmm.predict(
                            include_internal_connections=self.hmm.enable_internal_connections
                        )
                        column_probs = self.hmm.prediction_columns

                        if self.decoder is not None:
                            hidden_prediction = column_probs.reshape(self.obs_shape)
                            decoded_probs = self.decoder.decode(column_probs, learn=False)

                            if type(self.decoder) is Decoder:
                                decoded_probs = decoded_probs[1::2]

                            decoded_probs = decoded_probs.reshape(
                                self.raw_obs_shape
                            )
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

                        self.hmm.set_context_messages(self.hmm.internal_forward_messages)

                    if self.prediction_steps > 1:
                        self.hmm.set_context_messages(back_up_massages)

                    obs_probs_stack.append(copy(obs_probs))
                    hidden_probs_stack.append(copy(hidden_probs))

                    # remove empty lists
                    obs_probs_stack = [x for x in obs_probs_stack if len(x) > 0]
                    hidden_probs_stack = [x for x in hidden_probs_stack if len(x) > 0]

                    pred_horizon = [self.prediction_steps - len(x) for x in obs_probs_stack]
                    current_predictions_obs = [x.pop(0) for x in obs_probs_stack]
                    current_predictions_hid = [x.pop(0) for x in hidden_probs_stack]

                    for p_obs, p_hid, s in zip(
                            current_predictions_obs, current_predictions_hid, pred_horizon
                    ):
                        surp_obs = get_surprise(
                            p_obs.flatten(),
                            raw_obs_state
                        )
                        surp_hid = get_surprise(
                            p_hid.flatten(),
                            obs_state,
                            mode=self.surprise_mode
                        )
                        n_step_surprise_obs[s].append(surp_obs)
                        n_step_surprise_hid[s].append(surp_hid)

                    raw_im = [(prev_diff * 255).astype(np.uint8)]
                    raw_im.extend(raw_predictions)
                    raw_im = np.hstack(raw_im)
                    writer_raw.append_data(raw_im)

                    if hidden_predictions is not None:
                        hid_im = [(prev_latent * 255).astype(np.uint8)]
                        hid_im.extend(hidden_predictions)
                        hid_im = np.hstack(hid_im)
                        writer_hidden.append_data(hid_im)

            step += 1
            prev_diff = diff.copy()

            if self.encoder is not None:
                prev_latent = np.zeros(np.prod(self.obs_shape))
                prev_latent[obs_state] = 1
                prev_latent = prev_latent.reshape(self.obs_shape)
            else:
                prev_latent = None

            if step >= self.max_steps:
                if writer_raw is not None:
                    writer_raw.close()

                if writer_hidden is not None:
                    writer_hidden.close()

                break

        # AFTER EPISODE IS DONE
        if self.logger is not None:
            log_metrics = {}

            # main metrics
            main_metrics = dict(
                steps=step,
                surprise=np.mean(surprises),
                total_surprise=self.total_surprise,
            )
            if self.decoder is not None:
                main_metrics |= dict(
                    surprise_decoder=np.mean(surprises_decoder),
                    total_surprise_decoder=self.total_surprise_decoder,
                )
            log_metrics |= prepend_dict_keys(main_metrics, 'main_metrics')

            if self.log_scheduled(episode):
                for s, x in enumerate(n_step_surprise_hid):
                    log_metrics[f'n_step_hidden/surprise_step_{s + 1}'] = np.mean(x)
                for s, x in enumerate(n_step_surprise_obs):
                    log_metrics[f'n_step_raw/surprise_step_{s + 1}'] = np.mean(x)

                # noinspection PyDictCreation
                gifs = {}
                gifs['raw_prediction'] = wandb.Video(
                    f'/tmp/{self.logger.name}_raw_ep{episode}.gif'
                )
                if writer_hidden is not None:
                    gifs['hidden_prediction'] = wandb.Video(
                        f'/tmp/{self.logger.name}_hidden_ep{episode}.gif'
                    )

                log_metrics |= prepend_dict_keys(gifs, 'gifs')

                # factors and segments
                self.hmm.draw_factor_graph(
                    f'/tmp/{self.logger.name}_factor_graph_ep{episode}.png'
                )
                log_metrics['factors/graph'] = wandb.Image(
                    f'/tmp/{self.logger.name}_factor_graph_ep{episode}.png'
                )
                for factors, f_type in zip(
                        (self.hmm.context_factors, self.hmm.internal_factors),
                        ('context', 'internal')
                ):
                    if factors is not None:
                        log_metrics |= self.log_factors(factors, f_type, episode)

            self.logger.log(log_metrics, step=episode)

    @staticmethod
    def preprocess(image):
        gray_im = image.sum(axis=-1)
        gray_im /= gray_im.max()

        return gray_im

    def log_factors(self, factors, f_type, step):
        metrics = {
            f'connections/n_{f_type}_segments': factors.connections.numSegments(),
            f'connections/n_{f_type}_factors': factors.factor_connections.numSegments()
        }

        n_segments = np.zeros(self.hmm.internal_cells)
        sum_factor_value = np.zeros(self.hmm.internal_cells)
        for cell in range(self.hmm.internal_cells):
            segments = factors.connections.segmentsForCell(cell)

            value = 0
            if len(segments) > 0:
                value = np.exp(factors.log_factor_values_per_segment[segments]).sum()

            n_segments[cell] = len(segments)
            sum_factor_value[cell] = value

        n_segments = n_segments.reshape((-1, self.hmm.cells_per_column)).T
        sum_factor_value = sum_factor_value.reshape((-1, self.hmm.cells_per_column)).T

        # noinspection PyDictCreation
        factor_metrics = {}
        factor_metrics['n_segments'] = wandb.Image(sns.heatmap(n_segments))
        plt.close('all')
        factor_metrics['sum_factor_value'] = wandb.Image(sns.heatmap(sum_factor_value))
        plt.close('all')
        factor_metrics['var_score'] = wandb.Image(sns.scatterplot(factors.var_score))
        plt.close('all')

        if len(factors.segments_in_use) > 0:
            segment_activity = factors.segment_activity[factors.segments_in_use]
            segment_log_values = factors.log_factor_values_per_segment[factors.segments_in_use]

            factor_metrics['segment_activity'] = wandb.Image(sns.histplot(segment_activity))
            plt.close('all')
            factor_metrics['segment_log_values'] = wandb.Image(sns.histplot(segment_log_values))
            plt.close('all')
            factor_metrics['segment_score'] = wandb.Image(sns.histplot(
                np.exp(segment_log_values) * segment_activity
            ))
            plt.close('all')

        metrics |= prepend_dict_keys(factor_metrics, f'{f_type}_factors')
        return metrics

    def dump_model(self):
        if self.logger is None or not self.save_model:
            return

        path = Path('logs')
        if not path.exists():
            path.mkdir()

        name = self.logger.name
        with open(f"logs/models/model_{name}.pkl", 'wb') as file:
            pickle.dump(self.hmm, file)


def one_hot(i, size):
    arr = np.zeros(size)
    arr[i] = 1.
    return arr


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

    input_layer_conf = config['run'].get('input_layer_conf', None)
    if input_layer_conf is not None:
        with open(input_layer_conf, 'r') as file:
            config['input_layer'] = yaml.load(file, Loader=yaml.Loader)

    decoder_conf = config['run'].get('decoder_conf', None)
    if decoder_conf is not None:
        with open(decoder_conf, 'r') as file:
            config['decoder'] = yaml.load(file, Loader=yaml.Loader)

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

    if experiment == 'pinball':
        runner = PinballTest(logger, config)
    else:
        raise ValueError

    runner.run()


if __name__ == '__main__':
    default_config = 'configs/runner/layer/pinball.yaml'
    main(os.environ.get('RUN_CONF', default_config))
