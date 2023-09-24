#  Copyright (c) 2022 Autonomous Non-Profit Organization "Artificial Intelligence Research
#  Institute" (AIRI); Moscow Institute of Physics and Technology (National Research University).
#  All rights reserved.
#
#  Licensed under the AGPLv3 license. See LICENSE in the project root for license information.

import os
import pickle
import sys
from copy import copy
from pathlib import Path

import numpy as np
from htm.bindings.sdr import SDR

from hima.common.config.base import override_config, read_config
from hima.common.lazy_imports import lazy_import
from hima.common.run.argparse import parse_arg_list
from hima.common.sdr import sparse_to_dense
from hima.common.utils import prepend_dict_keys, to_gray_img
from hima.experiments.hmm.runners.utils import get_surprise_2
from hima.modules.belief.cortial_column.layer import Layer

try:
    from pinball import Pinball
except ModuleNotFoundError:
    Pinball = None


wandb = lazy_import('wandb')
imageio = lazy_import('imageio')


class TotalStats:
    state_surprise: float
    obs_surprise: float

    def __init__(self):
        self.state_surprise = 0.
        self.obs_surprise = 0.


class EpisodeStats:
    def __init__(self, n_prediction_steps):
        self.state_surprise = []
        self.state_mae = []

        self.obs_surprise = []
        self.obs_mae = []

        self.obs_probs_stack = []
        self.hidden_probs_stack = []
        self.n_step_surprise_obs = [[] for _ in range(n_prediction_steps)]
        self.n_step_surprise_hid = [[] for _ in range(n_prediction_steps)]


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
        # all dims except color channels
        self.obs_shape = obs.shape[:2]

        self.encoder_type = conf['run']['encoder']
        self.decoder_type = conf['run'].get('decoder', None)
        sp_conf = conf.get('sp', None)
        input_layer_conf = conf.get('input_layer', None)
        decoder_conf = conf.get('decoder', None)

        if self.encoder_type == 'one_sp':
            from hima.modules.htm.spatial_pooler import SPDecoder, HtmSpatialPooler
            sp_conf['seed'] = self.seed
            sp_conf['inputDimensions'] = list(self.obs_shape)
            self.encoder = HtmSpatialPooler(**sp_conf)
            self.decoder = SPDecoder(self.encoder)

            self.state_shape = self.encoder.getColumnDimensions()
            self.sp_input = SDR(self.obs_shape)
            self.sp_output = SDR(self.state_shape)

            self.n_hmm_obs_vars = self.state_shape[0] * self.state_shape[1]
            self.n_hmm_obs_states = 1
            self.surprise_mode = 'bernoulli'

        elif self.encoder_type == 'sp_ensemble':
            from hima.modules.htm.spatial_pooler import SPDecoder, SPEnsemble
            sp_conf['seed'] = self.seed
            sp_conf['inputDimensions'] = list(self.obs_shape)
            self.encoder = SPEnsemble(**sp_conf)
            self.decoder = SPDecoder(self.encoder)

            shape = self.encoder.sps[0].getColumnDimensions()
            self.state_shape = (shape[0] * self.encoder.n_groups, shape[1])
            self.sp_input = SDR(self.encoder.getNumInputs())
            self.sp_output = SDR(self.encoder.getNumColumns())

            self.n_hmm_obs_vars = self.encoder.n_groups
            self.n_hmm_obs_states = self.encoder.sps[0].getNumColumns()
            self.surprise_mode = 'categorical'

        elif self.encoder_type == 'new_sp_ensemble':
            from hima.experiments.temporal_pooling.stp.sp_ensemble import SpatialPoolerEnsemble
            from hima.experiments.temporal_pooling.stp.sp_decoder import SpatialPoolerDecoder
            sp_conf['seed'] = self.seed
            sp_conf['feedforward_sds'] = [self.obs_shape, 0.1]
            self.encoder = SpatialPoolerEnsemble(**sp_conf)
            self.decoder = make_decoder(self.encoder, self.decoder_type)

            self.state_shape = self.encoder.getColumnsDimensions()
            self.sp_input = SDR(self.encoder.getNumInputs())
            self.sp_output = SDR(self.encoder.getNumColumns())

            self.n_hmm_obs_vars = self.encoder.n_sp
            self.n_hmm_obs_states = self.encoder.getSingleNumColumns()
            self.surprise_mode = 'categorical'

        elif self.encoder_type == 'new_sp_grouped':
            from hima.experiments.temporal_pooling.stp.sp_ensemble import (
                SpatialPoolerGroupedWrapper
            )
            sp_conf['seed'] = self.seed
            sp_conf['feedforward_sds'] = [self.obs_shape, 0.1]
            self.encoder = SpatialPoolerGroupedWrapper(**sp_conf)
            self.decoder = make_decoder(self.encoder, self.decoder_type)

            self.state_shape = self.encoder.getColumnsDimensions()
            self.sp_input = SDR(self.encoder.getNumInputs())
            self.sp_output = SDR(self.encoder.getNumColumns())

            self.n_hmm_obs_vars = self.encoder.n_groups
            self.n_hmm_obs_states = self.encoder.getSingleNumColumns()
            self.surprise_mode = 'categorical'

        elif self.encoder_type == 'input_layer':
            from hima.modules.belief.cortial_column.input_layer import Encoder, Decoder
            assert input_layer_conf is not None
            encoder_conf = input_layer_conf.get('encoder')
            decoder_conf = input_layer_conf.get('decoder')
            encoder_conf['seed'] = self.seed
            decoder_conf['seed'] = self.seed

            encoder_conf['n_context_vars'] = self.obs_shape[0] * self.obs_shape[1]
            encoder_conf['n_context_states'] = 2

            self.encoder = Encoder(**encoder_conf)
            self.n_hmm_obs_vars = self.encoder.n_hidden_vars
            self.n_hmm_obs_states = self.encoder.n_hidden_states
            self.state_shape = (self.n_hmm_obs_vars, self.n_hmm_obs_states)

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

            self.n_hmm_obs_vars = self.obs_shape[0] * self.obs_shape[1]
            self.n_hmm_obs_states = 1
            self.state_shape = self.obs_shape
            self.surprise_mode = 'bernoulli'

        conf['hmm']['n_obs_vars'] = self.n_hmm_obs_vars
        conf['hmm']['n_obs_states'] = self.n_hmm_obs_states
        conf['hmm']['n_context_states'] = self.n_hmm_obs_states * conf['hmm']['cells_per_column']
        conf['hmm']['n_context_vars'] = self.n_hmm_obs_vars

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

        self.total_stats = TotalStats()

        assert (self.encoder is None) == (self.decoder is None)
        self.all_observations = []

    def run(self):
        for episode in range(self.n_episodes):
            self.run_episode(episode)
        self.dump_model()

        # pinball_root_path = Path('/Users/kuderov/data/pinball/')
        # pinball_root_path.mkdir(parents=True, exist_ok=True)
        # np.save(pinball_root_path / 'pinball', np.array(self.all_observations))

    def run_episode(self, episode):
        episode_stats = EpisodeStats(n_prediction_steps=self.prediction_steps)
        writer_raw, writer_hidden = self.get_gif_writers(episode)

        init_index = self.rng.choice(len(self.start_actions))
        action = self.start_actions[init_index]
        position = self.start_positions[init_index]
        self.env.reset(position)
        self.env.act(action)

        self.hmm.reset()

        prev_raw_img = raw_img = self.step_observe()
        prev_obs = obs = np.zeros(self.obs_shape)
        prev_state = np.zeros(self.state_shape)

        initial_context = np.zeros_like(self.hmm.context_messages)
        initial_context[
            np.arange(self.hmm.n_hidden_vars) * self.hmm.n_hidden_states
        ] = 1
        self.hmm.set_context_messages(initial_context)

        for step in range(self.max_steps):
            raw_img = self.step_observe()
            # noinspection PyTypeChecker
            obs: np.ndarray = np.abs(raw_img - prev_raw_img) >= raw_img.mean()
            # self.all_observations.append(obs)
            obs_sdr = np.flatnonzero(obs)

            self.act(step)
            state_prediction = self.predict_state()

            self.sp_input.sparse = obs_sdr
            self.encoder.compute(self.sp_input, True, self.sp_output)
            state_sdr = self.sp_output.sparse

            obs_prediction = self.decoder.decode(state_prediction, learn=True, correct_obs=obs)

            # metrics for encoder/decoder
            in_avg_sparsity = self.encoder.ff_avg_active_size / self.encoder.feedforward_sds.size
            obs_mae = np.abs(obs_prediction - obs.flatten()).mean() / in_avg_sparsity
            episode_stats.obs_mae.append(obs_mae)

            out_avg_sparsity = self.encoder.output_sds.sparsity
            _state_prediction_error = state_prediction.flatten().copy()
            _state_prediction_error[state_sdr] -= 1.0
            state_mae = np.abs(_state_prediction_error).mean() / out_avg_sparsity
            episode_stats.state_mae.append(state_mae)

            # Feedback Encoder
            feedback_sdr = np.flatnonzero(np.abs(_state_prediction_error) > .4)
            modulation = 0.01 * np.expand_dims(_state_prediction_error[feedback_sdr], -1)
            self.encoder.process_feedback(feedback_sdr, modulation)

            self.hmm.observe(state_sdr, learn=True)
            self.hmm.set_context_messages(self.hmm.internal_forward_messages)

            # metrics
            # 1. surprise
            state_surprise = get_surprise_2(state_prediction, state_sdr, mode=self.surprise_mode)
            episode_stats.state_surprise.append(state_surprise)
            self.total_stats.state_surprise += state_surprise

            obs_surprise = get_surprise_2(obs_prediction, obs_sdr)
            episode_stats.obs_surprise.append(obs_surprise)
            self.total_stats.obs_surprise += obs_surprise

            # 2. image
            if writer_raw is not None:
                self.predict_images(
                    state_prediction=state_prediction, obs_prediction=obs_prediction,
                    episode_stats=episode_stats, obs_sdr=obs_sdr, state_sdr=state_sdr,
                    prev_obs=prev_obs, prev_state=prev_state,
                    writer_raw=writer_raw, writer_hidden=writer_hidden,
                )

            prev_obs = obs.copy()
            prev_raw_img = raw_img.copy()
            prev_state = sparse_to_dense(state_sdr, like=prev_state)

        # AFTER EPISODE IS DONE
        if writer_raw is not None:
            writer_raw.close()
        if writer_hidden is not None:
            writer_hidden.close()

        if self.logger is None:
            return

        # LOG METRICS
        log_metrics = {}

        main_metrics = dict(
            surprise=np.mean(episode_stats.state_surprise),
            surprise_decoder=np.mean(episode_stats.obs_surprise),
            obs_mae=np.mean(episode_stats.obs_mae),
            state_mae=np.mean(episode_stats.state_mae),
            total_surprise=self.total_stats.state_surprise,
            total_surprise_decoder=self.total_stats.obs_surprise,
        )
        log_metrics |= prepend_dict_keys(main_metrics, 'main_metrics')

        if self.log_scheduled(episode):
            for s, x in enumerate(episode_stats.n_step_surprise_obs):
                log_metrics[f'n_step_raw/surprise_step_{s + 1}'] = np.mean(x)
            for s, x in enumerate(episode_stats.n_step_surprise_hid):
                log_metrics[f'n_step_hidden/surprise_step_{s + 1}'] = np.mean(x)

            # noinspection PyDictCreation
            gifs = {}
            raw_prediction_path = f'/tmp/{self.logger.name}_raw_ep{episode}.gif'
            gifs['raw_prediction'] = wandb.Video(raw_prediction_path)
            if writer_hidden is not None:
                hidden_prediction_path = f'/tmp/{self.logger.name}_hidden_ep{episode}.gif'
                gifs['hidden_prediction'] = wandb.Video(hidden_prediction_path)
            log_metrics |= prepend_dict_keys(gifs, 'gifs')

            # factors and segments
            factors_graph_path = f'/tmp/{self.logger.name}_factor_graph_ep{episode}.png'
            self.hmm.draw_factor_graph(factors_graph_path)
            log_metrics['factors/graph'] = wandb.Image(factors_graph_path)

            for factors, f_type in zip(
                (self.hmm.context_factors, self.hmm.internal_factors), ('context', 'internal')
            ):
                if factors is not None:
                    log_metrics |= self.log_factors(factors, f_type, episode)

        self.logger.log(log_metrics, step=episode)

    def predict_images(
            self, state_prediction, obs_prediction, episode_stats,
            obs_sdr, state_sdr, prev_obs, prev_state, writer_raw, writer_hidden,
    ):
        if self.prediction_steps > 1:
            context_backup = self.hmm.context_messages.copy()
            self.hmm.set_context_messages(self.hmm.prediction_cells)

        obs_predictions, obs_img_predictions = [], []
        state_predictions, state_img_predictions = [], []

        for j in range(self.prediction_steps):
            if j > 0:
                state_prediction = self.predict_state()
                obs_prediction = self.decoder.decode(state_prediction, learn=False)

            obs_prediction = obs_prediction.reshape(self.obs_shape)
            state_prediction = state_prediction.reshape(self.state_shape)

            obs_predictions.append(obs_prediction)
            state_predictions.append(state_prediction)

            obs_img_predictions.append(to_gray_img(obs_prediction))
            if state_img_predictions is not None:
                state_img_predictions.append(to_gray_img(state_prediction))

            self.hmm.set_context_messages(self.hmm.internal_forward_messages)

        if self.prediction_steps > 1:
            self.hmm.set_context_messages(context_backup)

        self.make_smth_with_predictions(
            episode_stats, obs_predictions, state_predictions, obs_sdr, state_sdr
        )

        writer_raw.append_data(np.hstack([to_gray_img(prev_obs)] + obs_img_predictions))
        if state_img_predictions is not None:
            writer_hidden.append_data(np.hstack([to_gray_img(prev_state)] + state_img_predictions))

    def make_smth_with_predictions(
            self, episode_stats, obs_predictions, hidden_probs, obs_sdr, state_sdr
    ):
        episode_stats.obs_probs_stack.append(copy(obs_predictions))
        episode_stats.hidden_probs_stack.append(copy(hidden_probs))

        # remove empty lists
        episode_stats.obs_probs_stack = [
            x for x in episode_stats.obs_probs_stack if len(x) > 0
        ]
        episode_stats.hidden_probs_stack = [
            x for x in episode_stats.hidden_probs_stack if len(x) > 0
        ]

        pred_horizon = [self.prediction_steps - len(x) for x in episode_stats.obs_probs_stack]
        current_predictions_obs = [x.pop(0) for x in episode_stats.obs_probs_stack]
        current_predictions_hid = [x.pop(0) for x in episode_stats.hidden_probs_stack]

        for p_obs, p_hid, s in zip(
                current_predictions_obs, current_predictions_hid, pred_horizon
        ):
            surp_obs = get_surprise_2(p_obs.flatten(), obs_sdr)
            surp_hid = get_surprise_2(p_hid.flatten(), state_sdr, mode=self.surprise_mode)
            episode_stats.n_step_surprise_obs[s].append(surp_obs)
            episode_stats.n_step_surprise_hid[s].append(surp_hid)

    def step_observe(self):
        self.env.step()

        img = self.env.obs()
        gray_im = img.sum(axis=-1)
        gray_im /= gray_im.max()
        return gray_im

    def act(self, step):
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
            action_probs = to_dense(action, len(self.actions))

        self.hmm.set_external_messages(action_probs)

    def predict_state(self):
        include_internal_connections = (
                self.hmm.enable_internal_connections and self.internal_dependence_1step
        )
        self.hmm.predict(include_internal_connections=include_internal_connections)
        return self.hmm.prediction_columns.copy()

    def log_scheduled(self, step):
        return (self.logger is not None) and (step % self.log_update_rate == 0)

    def get_gif_writers(self, episode):
        writer_raw = None
        writer_hidden = None
        if self.log_scheduled(episode):
            writer_raw = imageio.get_writer(
                f'/tmp/{self.logger.name}_raw_ep{episode}.gif',
                mode='I',
                duration=1000 / self.log_fps,
            )
            writer_hidden = imageio.get_writer(
                f'/tmp/{self.logger.name}_hidden_ep{episode}.gif',
                mode='I',
                duration=1000 / self.log_fps,
            )
        return writer_raw, writer_hidden

    def log_factors(self, factors, f_type, step):
        metrics = {
            f'connections/n_{f_type}_segments': factors.connections.numSegments(),
            f'connections/n_{f_type}_factors': factors.factor_connections.numSegments()
        }

        # n_segments = np.zeros(self.hmm.internal_cells)
        # sum_factor_value = np.zeros(self.hmm.internal_cells)
        # for cell in range(self.hmm.internal_cells):
        #     segments = factors.connections.segmentsForCell(cell)
        #
        #     value = 0
        #     if len(segments) > 0:
        #         value = np.exp(factors.log_factor_values_per_segment[segments]).sum()
        #
        #     n_segments[cell] = len(segments)
        #     sum_factor_value[cell] = value
        #
        # n_segments = n_segments.reshape((-1, self.hmm.cells_per_column)).T
        # sum_factor_value = sum_factor_value.reshape((-1, self.hmm.cells_per_column)).T

        # noinspection PyDictCreation
        factor_metrics = {}
        # factor_metrics['n_segments'] = wandb.Image(sns.heatmap(n_segments))
        # plt.close('all')
        # factor_metrics['sum_factor_value'] = wandb.Image(sns.heatmap(sum_factor_value))
        # plt.close('all')
        # factor_metrics['var_score'] = wandb.Image(sns.scatterplot(factors.var_score))
        # plt.close('all')
        #
        # if len(factors.segments_in_use) > 0:
        #     segment_activity = factors.segment_activity[factors.segments_in_use]
        #     segment_log_values = factors.log_factor_values_per_segment[factors.segments_in_use]
        #
        #     factor_metrics['segment_activity'] = wandb.Image(sns.histplot(segment_activity))
        #     plt.close('all')
        #     factor_metrics['segment_log_values'] = wandb.Image(sns.histplot(segment_log_values))
        #     plt.close('all')
        #     factor_metrics['segment_score'] = wandb.Image(sns.histplot(
        #         np.exp(segment_log_values) * segment_activity
        #     ))
        #     plt.close('all')

        if len(factor_metrics) > 0:
            metrics |= prepend_dict_keys(factor_metrics, f'{f_type}_factors')
        return metrics

    def dump_model(self):
        if self.logger is None or not self.save_model:
            return

        path = Path('logs')
        path.mkdir(parents=True, exist_ok=True)

        name = self.logger.name
        with open(f"logs/models/model_{name}.pkl", 'wb') as file:
            pickle.dump(self.hmm, file)


def to_dense(i, size):
    arr = np.zeros(size).flatten()
    arr[i] = 1.
    return arr


def make_decoder(encoder, decoder):
    if decoder == 'naive':
        from hima.experiments.temporal_pooling.stp.sp_decoder import SpatialPoolerDecoder
        return SpatialPoolerDecoder(encoder)
    elif decoder == 'learned':
        from hima.experiments.temporal_pooling.stp.sp_decoder import SpatialPoolerLearnedDecoder
        return SpatialPoolerLearnedDecoder(encoder, hidden_dims=(64, 64))
    elif decoder == 'learned_input':
        from hima.experiments.temporal_pooling.stp.sp_decoder import (
            SpatialPoolerLearnedOverNaiveDecoder
        )
        return SpatialPoolerLearnedOverNaiveDecoder(encoder)
    else:
        raise ValueError(f'Decoder {decoder} is not supported')


def main(config_path):
    if len(sys.argv) > 1:
        config_path = sys.argv[1]

    config = dict()

    config['run'] = read_config(config_path)
    config['hmm'] = read_config(config['run']['hmm_conf'])
    config['env'] = read_config(config['run']['env_conf'])

    sp_conf = config['run'].get('sp_conf', None)
    if sp_conf is not None:
        config['sp'] = read_config(sp_conf)

    input_layer_conf = config['run'].get('input_layer_conf', None)
    if input_layer_conf is not None:
        config['input_layer'] = read_config(input_layer_conf)

    decoder_conf = config['run'].get('decoder_conf', None)
    if decoder_conf is not None:
        config['decoder'] = read_config(decoder_conf)

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

    if experiment == 'pinball':
        runner = PinballTest(logger, config)
    else:
        raise ValueError

    runner.run()


if __name__ == '__main__':
    default_config = 'configs/runner/layer/pinball.yaml'
    main(os.environ.get('RUN_CONF', default_config))
