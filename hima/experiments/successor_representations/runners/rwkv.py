#  Copyright (c) 2023 Autonomous Non-Profit Organization "Artificial Intelligence Research
#  Institute" (AIRI); Moscow Institute of Physics and Technology (National Research University).
#  All rights reserved.
#
#  Licensed under the AGPLv3 license. See LICENSE in the project root for license information.
import os
import sys
from typing import Literal

import numpy as np

from hima.agents.succesor_representations.agent import LstmBioHima
from hima.common.config.base import read_config, override_config
from hima.common.lazy_imports import lazy_import
from hima.common.run.argparse import parse_arg_list
from hima.common.run.wandb import turn_off_gui_for_matplotlib
from hima.common.sdr import sparse_to_dense
from hima.common.utils import to_gray_img, isnone
from hima.experiments.successor_representations.runners.utils import print_digest, make_decoder
from hima.modules.baselines.lstm import to_numpy, LstmLayer
from hima.modules.baselines.rwkv import RwkvLayer
from hima.modules.belief.cortial_column.cortical_column import CorticalColumn
from hima.modules.belief.utils import normalize

wandb = lazy_import('wandb')


class AnimalAITest:
    def __init__(self, logger, conf, max_workers=10):
        from animalai.envs.actions import AAIActions
        from animalai.envs.environment import AnimalAIEnvironment
        from mlagents_envs.exception import UnityWorkerInUseException

        self.logger = logger
        self.seed = conf['run'].get('seed')
        self._rng = np.random.default_rng(self.seed)

        conf['env']['seed'] = self.seed
        conf['env']['file_name'] = os.environ.get('ANIMALAI_EXE', None)
        conf['env']['arenas_configurations'] = os.path.join(
            os.environ.get('ANIMALAI_ROOT', None),
            'configs',
            f"{conf['run']['setup']}"
        )

        worker_id = 0
        while worker_id < max_workers:
            try:
                self.environment = AnimalAIEnvironment(
                    worker_id=worker_id,
                    **conf['env']
                )
                break
            except UnityWorkerInUseException:
                worker_id += 1
        else:
            raise Exception('Too many workers.')

        # get agent proxi in unity
        self.behavior = list(self.environment.behavior_specs.keys())[0]
        self.raw_obs_shape = self.environment.behavior_specs[self.behavior].observation_specs[
            0].shape[:2]
        self.actions = [
            AAIActions().LEFT,
            AAIActions().FORWARDS,
            AAIActions().RIGHT,
            AAIActions().BACKWARDS
        ]
        self.n_actions = len(self.actions)

        # assembly agent
        if 'encoder' in conf:
            encoder_type = conf['run']['encoder']
            encoder_conf = conf['encoder']
        else:
            encoder_type = None
            encoder_conf = None
        layer_conf = conf['layer']

        if encoder_type == 'sp_ensemble':
            from hima.modules.htm.spatial_pooler import SPDecoder, SPEnsemble

            encoder_conf['seed'] = self.seed
            encoder_conf['inputDimensions'] = list(self.raw_obs_shape)

            encoder = SPEnsemble(**encoder_conf)
            decoder = SPDecoder(encoder)

            layer_conf['n_obs_vars'] = encoder.n_sp
            layer_conf['n_obs_states'] = encoder.sps[0].getNumColumns()
        elif encoder_type == 'sp_grouped':
            from hima.experiments.temporal_pooling.stp.sp_ensemble import (
                SpatialPoolerGroupedWrapper
            )
            encoder_conf['seed'] = self.seed
            encoder_conf['feedforward_sds'] = [self.raw_obs_shape, 0.1]

            decoder_type = conf['run'].get('decoder', None)
            decoder_conf = conf['decoder']

            encoder = SpatialPoolerGroupedWrapper(**encoder_conf)
            decoder = make_decoder(encoder, decoder_type, decoder_conf)

            layer_conf['n_obs_vars'] = encoder.n_groups
            layer_conf['n_obs_states'] = encoder.getSingleNumColumns()
        else:
            encoder = None
            decoder = None
            layer_conf['n_obs_vars'] = np.prod(self.raw_obs_shape)
            layer_conf['n_obs_states'] = 1

        layer_conf['n_external_vars'] = 1
        layer_conf['n_external_states'] = self.n_actions
        layer_conf['seed'] = self.seed

        layer = RwkvLayer(**layer_conf)

        # noinspection PyTypeChecker
        cortical_column = CorticalColumn(layer, encoder, decoder)
        self.agent = LstmBioHima(cortical_column, **conf['agent'])

        self.n_episodes = conf['run']['n_episodes']
        self.max_steps = conf['run']['max_steps']
        self.update_rate = conf['run']['update_rate']
        self.reset_context_period = conf['run'].get('reset_context_period', 0)
        self.action_inertia = conf['run'].get('action_inertia', 1)

        self.initial_previous_image = self._rng.random(self.raw_obs_shape)
        self.prev_image = self.initial_previous_image
        self.initial_context = layer.context_messages

        if self.logger is not None:
            from hima.common.metrics import ScalarMetrics, HeatmapMetrics, ImageMetrics
            # define metrics
            self.scalar_metrics = ScalarMetrics(
                {
                    'main_metrics/reward': np.sum,
                    'main_metrics/steps': np.mean,
                    'layer/surprise_hidden': np.mean,
                    'layer/loss': np.mean,
                    'agent/td_error': np.mean
                },
                self.logger
            )
            self.heatmap_metrics = HeatmapMetrics(
                {
                    'agent/obs_rewards': np.mean,
                    'agent/striatum_weights': np.mean
                },
                self.logger
            )
            self.image_metrics = ImageMetrics(
                [
                    'agent/behavior',
                    'agent/sr',
                    'layer/predictions'
                ],
                self.logger,
                log_fps=conf['run']['log_gif_fps']
            )
        else:
            from hima.common.metrics import ScalarMetrics
            self.scalar_metrics = ScalarMetrics(
                {
                    'main_metrics/reward': np.sum,
                    'main_metrics/steps': np.mean,
                    'layer/surprise_hidden': np.mean,
                    'layer/loss': np.mean,
                    'agent/td_error': np.mean
                },
                self.logger
            )

    def run(self):
        episode_print_schedule = 25
        decoder = self.agent.cortical_column.decoder

        for episode in range(self.n_episodes):
            if episode % episode_print_schedule == 0:
                print(f'Episode {episode}')

            steps = 0
            running = True
            action = None

            self.prev_image = self.initial_previous_image
            self.environment.reset()
            self.agent.reset(self.initial_context, action)

            while running:
                if (self.reset_context_period > 0) and (steps > 0):
                    if (steps % self.reset_context_period) == 0:
                        self.agent.cortical_column.layer.set_context_messages(
                            self.initial_context
                        )

                self.environment.step()
                dec, term = self.environment.get_steps(self.behavior)

                reward = 0
                if len(dec) > 0:
                    obs = self.environment.get_obs_dict(dec.obs)["camera"]
                    reward += dec.reward

                if len(term) > 0:
                    obs = self.environment.get_obs_dict(term.obs)["camera"]
                    reward += term.reward
                    running = False

                # noinspection PyUnboundLocalVariable
                events = self.preprocess(obs)

                pred_sr, gen_sr = self.agent.observe((events, action), learn=True)
                self.agent.reinforce(reward)

                if running:
                    if (action is None) or ((steps % self.action_inertia) == 0):
                        action = self.agent.sample_action()
                    # convert to AAI action
                    aai_action = self.actions[action]
                    self.environment.set_actions(self.behavior, aai_action.action_tuple)

                # >>> logging
                # noinspection PyUnresolvedReferences
                self.scalar_metrics.update({
                    'main_metrics/reward': reward,
                    'layer/surprise_hidden': self.agent.surprise,
                    'layer/loss': self.agent.cortical_column.layer.last_loss_value,
                    'agent/td_error': self.agent.td_error
                })
                if self.logger is not None:
                    if (episode % self.update_rate) == 0:
                        raw_beh = self.to_img(self.prev_image)
                        proc_beh = self.to_img(sparse_to_dense(events, like=self.prev_image))
                        pred_beh = self.to_img(self.agent.cortical_column.predicted_image)
                        pred_sr = self.to_img(decoder.decode(pred_sr))
                        gen_sr = self.to_img(decoder.decode(gen_sr))

                        self.image_metrics.update({
                            'agent/behavior': np.hstack([
                                raw_beh, proc_beh, pred_beh, pred_sr, gen_sr
                            ])
                        })
                # <<< logging

                steps += 1

                if steps >= self.max_steps:
                    running = False

            # >>> logging
            self.scalar_metrics.update({'main_metrics/steps': steps})
            if self.logger is not None:
                self.scalar_metrics.log(episode)

                if (episode % self.update_rate) == 0:
                    obs_rewards = self.agent.cortical_column.decoder.decode(
                        normalize(
                            self.agent.observation_rewards.reshape(
                                self.agent.cortical_column.layer.n_obs_vars, -1
                            )
                        ).flatten()
                    ).reshape(self.raw_obs_shape)
                    self.heatmap_metrics.update(
                        {
                            'agent/obs_rewards': obs_rewards,
                            'agent/striatum_weights': self.agent.striatum_weights
                        }
                    )
                    self.heatmap_metrics.log(episode)
                    self.image_metrics.log(episode)
            else:
                print_digest(self.scalar_metrics.summarize())
                self.scalar_metrics.reset()
            # <<< logging
        else:
            self.environment.close()

    def to_img(self, x: np.ndarray, shape=None):
        return to_gray_img(x, like=isnone(shape, self.raw_obs_shape))

    def preprocess(self, image):
        gray = np.dot(image, [299 / 1000, 587 / 1000, 114 / 1000])

        diff = np.abs(gray - self.prev_image)

        self.prev_image = gray.copy()

        thresh = diff.mean()
        events = np.flatnonzero(diff > thresh)

        return events


class PinballTest:
    def __init__(self, logger, conf):
        from pinball import Pinball

        self.logger = logger
        self.seed = conf['run'].get('seed')
        self._rng = np.random.default_rng(self.seed)

        self.n_episodes = conf['run']['n_episodes']
        self.max_steps = conf['run']['max_steps']
        self.update_period = conf['run']['update_period']
        self.update_start = conf['run']['update_start']
        self.camera_mode = conf['run']['camera_mode']
        self.reward_free = conf['run'].get('reward_free', False)
        self.test_srs = conf['run'].get('test_srs', False)
        self.test_sr_steps = conf['run'].get('test_sr_steps', 0)
        self.layer_type = conf['run']['layer']
        self.action_inertia = conf['run'].get('action_inertia', 1)

        self.setups = conf['run']['setup']
        self.setup_period = conf['run'].get('setup_period', None)

        if self.setup_period is None:
            self.setup_period = [self.n_episodes//len(self.setups)]*len(self.setups)
        elif type(self.setup_period) is int:
            period = self.setup_period
            self.setup_period = [period]*len(self.setups)

        assert len(self.setups) == len(self.setup_period)

        conf['env']['seed'] = self.seed
        conf['env']['exe_path'] = os.environ.get('PINBALL_EXE', None)
        conf['env']['config_path'] = self.get_setup_path(self.setups[0])

        self.environment = Pinball(**conf['env'])
        obs, _, _ = self.environment.obs()
        self.raw_obs_shape = (obs.shape[0], obs.shape[1])
        self.start_position = conf['run']['start_position']
        self.actions = conf['run']['actions']
        self.n_actions = len(self.actions)

        if 'reset_context_period' in conf['layer']:
            self.reset_context_period = conf['layer'].pop(
                'reset_context_period'
            )
        else:
            self.reset_context_period = 0

        self.agent = self.make_agent(conf, conf['run'].get('agent_path', None))

        self.initial_previous_image = self._rng.random(self.raw_obs_shape)
        self.prev_image = self.initial_previous_image

        if self.layer_type == 'fchmm':
            self.initial_action = -1
            self.initial_context = np.empty(0)
            self.initial_external_message = np.empty(0)
        elif self.layer_type == 'dhtm':
            self.initial_action = None
            self.initial_context = sparse_to_dense(
                np.arange(
                    self.agent.cortical_column.layer.n_hidden_vars
                ) * self.agent.cortical_column.layer.n_hidden_states,
                like=self.agent.cortical_column.layer.context_messages
            )
            self.initial_external_message = None
        else:
            self.initial_action = None
            self.initial_context = self.agent.cortical_column.layer.context_messages
            self.initial_external_message = None

        self.predicted_sr_stack = None
        self.predicted_sr_stack_raw = None
        self.generated_sr_stack = None
        self.generated_sr_stack_raw = None
        self.prediction_stack = None

        if self.logger is not None:
            from hima.common.metrics import ScalarMetrics, HeatmapMetrics, ImageMetrics
            # define metrics
            self.logger.define_metric("main_metrics/steps", summary="mean")
            self.logger.define_metric("main_metrics/reward", summary="mean")

            self.scalar_metrics = ScalarMetrics(
                {
                    'main_metrics/reward': np.sum,
                    'main_metrics/steps': np.mean,
                    'layer/surprise_hidden': np.mean,
                    'layer/norm_surprise_hidden': np.mean,
                    'layer/loss': np.mean,
                    'sr/td_error': np.mean,
                    'sr/norm_td_error': np.mean,
                    'sr/test_mse_approx_tail': np.mean,
                    'sr/test_mse': np.mean,
                    'agent/sr_steps': np.mean,
                    'agent/striatum_lr': np.mean
                },
                self.logger
            )
            self.heatmap_metrics = HeatmapMetrics(
                {
                    'agent/obs_rewards': np.mean,
                    'agent/striatum_weights': np.mean,
                    'agent/real_rewards': np.mean
                },
                self.logger
            )
            self.image_metrics = ImageMetrics(
                [
                    'agent/behavior',
                    'agent/hidden',
                    'sr/sr',
                    'layer/predictions'
                ],
                self.logger,
                log_fps=conf['run']['log_gif_fps']
            )

            if self.test_srs:
                from hima.common.metrics import SRStackSurprise
                self.predicted_sr_stack = SRStackSurprise(
                    'sr/pred/hid/surprise',
                    self.logger,
                    self.agent.observation_messages.size,
                    history_length=self.test_sr_steps
                )

                self.generated_sr_stack = SRStackSurprise(
                    'sr/gen/hid/surprise',
                    self.logger,
                    self.agent.observation_messages.size,
                    history_length=self.test_sr_steps
                )

                self.predicted_sr_stack_raw = SRStackSurprise(
                    'sr/pred/raw/surprise',
                    self.logger,
                    self.raw_obs_shape[0] * self.raw_obs_shape[1],
                    history_length=self.test_sr_steps,
                )

                self.generated_sr_stack_raw = SRStackSurprise(
                    'sr/gen/raw/surprise',
                    self.logger,
                    self.raw_obs_shape[0] * self.raw_obs_shape[1],
                    history_length=self.test_sr_steps,
                )

                from hima.common.metrics import PredictionsStackSurprise
                self.prediction_stack = PredictionsStackSurprise(
                    'layer_n_step/hidden_surprise',
                    self.logger,
                    self.test_sr_steps + 1
                )
        else:
            from hima.common.metrics import ScalarMetrics
            self.scalar_metrics = ScalarMetrics(
                {
                    'main_metrics/reward': np.sum,
                    'main_metrics/steps': np.mean,
                    'layer/surprise_hidden': np.mean,
                    'layer/norm_surprise_hidden': np.mean,
                    'layer/loss': np.mean,
                    'sr/td_error': np.mean,
                    'sr/norm_td_error': np.mean,
                    'sr/test_mse_approx_tail': np.mean,
                    'sr/test_mse': np.mean,
                    'agent/sr_steps': np.mean,
                    'agent/striatum_lr': np.mean
                },
                self.logger
            )

    def run(self):
        episode_print_schedule = 50
        decoder = self.agent.cortical_column.decoder
        total_reward = np.zeros(self.raw_obs_shape).flatten()
        setup_episodes = 0
        current_setup_id = 0

        for episode in range(self.n_episodes):
            if episode % episode_print_schedule == 0:
                print(f'Episode {episode}')

            steps = 0
            running = True
            action = self.initial_action

            self.prev_image = self.initial_previous_image

            # change setup
            if setup_episodes >= self.setup_period[current_setup_id]:
                current_setup_id += 1
                current_setup_id = current_setup_id % len(self.setups)
                self.environment.set_config(self.get_setup_path(self.setups[
                    current_setup_id
                ]))
                setup_episodes = 0

            self.environment.reset(self.start_position)
            self.agent.reset(self.initial_context, self.initial_external_message)

            while running:
                if self.reset_context_period > 0 and steps > 0:
                    if steps % self.reset_context_period == 0:
                        self.agent.cortical_column.layer.set_context_messages(self.initial_context)

                self.environment.step()
                obs, reward, is_terminal = self.environment.obs()
                running = not is_terminal

                events = self.preprocess(obs, mode=self.camera_mode)
                # observe events_t and action_{t-1}
                pred_sr, gen_sr = self.agent.observe((events, action), learn=True)
                self.agent.reinforce(reward)

                total_reward[events] += reward

                if running:
                    if steps % self.action_inertia == 0:
                        if self.reward_free:
                            action = self._rng.integers(self.n_actions)
                        else:
                            action = self.agent.sample_action()

                    # convert to Pinball action
                    pinball_action = self.actions[action]
                    self.environment.act(pinball_action)

                # >>> logging
                # noinspection PyUnresolvedReferences
                self.scalar_metrics.update({
                    'main_metrics/reward': reward,
                    'layer/surprise_hidden': self.agent.surprise,
                    'layer/norm_surprise_hidden': self.agent.ss_surprise.norm_value,
                    'layer/loss': self.agent.cortical_column.layer.last_loss_value,
                    'sr/td_error': self.agent.td_error,
                    'sr/norm_td_error': self.agent.ss_td_error.norm_value,
                    'agent/sr_steps': self.agent.sr_steps,
                    'agent/striatum_lr': self.agent.striatum_lr
                })
                if self.logger is not None:
                    if (episode % self.update_period) == 0:
                        raw_beh = self.to_img(self.prev_image)
                        proc_beh = self.to_img(sparse_to_dense(events, like=self.prev_image))
                        pred_beh = self.to_img(self.agent.cortical_column.predicted_image)
                        # FIXME: check normalization in iclr
                        pred_sr = self.to_img(decoder.decode(pred_sr))
                        gen_sr = self.to_img(decoder.decode(gen_sr))

                        actual_state = self.agent.cortical_column.layer.internal_forward_messages
                        predicted_state = self.agent.cortical_column.layer.prediction_cells
                        if type(actual_state) is list:
                            # noinspection PyProtectedMember
                            actual_state = to_numpy(
                                self.agent._extract_state_from_context(actual_state)
                            )
                            # noinspection PyProtectedMember
                            predicted_state = to_numpy(
                                self.agent._extract_state_from_context(predicted_state)
                            )

                        hid_beh = self.to_img(
                            actual_state,
                            shape=(self.agent.cortical_column.layer.n_hidden_vars, -1)
                        )
                        hid_pred_beh = self.to_img(
                            predicted_state,
                            shape=(self.agent.cortical_column.layer.n_hidden_vars, -1)
                        )
                        hid_diff_beh = self.to_img(
                            np.abs(actual_state - predicted_state),
                            shape=(self.agent.cortical_column.layer.n_hidden_vars, -1)
                        )

                        self.image_metrics.update({
                            'agent/behavior': np.hstack([
                                raw_beh, proc_beh, pred_beh, pred_sr, gen_sr
                            ]),
                            'agent/hidden': np.hstack([
                                hid_beh, hid_pred_beh, hid_diff_beh
                            ])
                        })
                # <<< logging

                steps += 1

                if steps >= self.max_steps:
                    running = False

            # >>> logging
            self.scalar_metrics.update({'main_metrics/steps': steps})
            if self.logger is not None:
                self.scalar_metrics.log(episode)

                if self.test_srs:
                    self.predicted_sr_stack.log(episode)
                    self.predicted_sr_stack_raw.log(episode)
                    self.generated_sr_stack.log(episode)
                    self.generated_sr_stack_raw.log(episode)
                    self.prediction_stack.log(episode)

                if episode % self.update_period == 0:
                    obs_rewards = self.agent.cortical_column.decoder.decode(
                        normalize(
                            self.agent.observation_rewards.reshape(
                                self.agent.cortical_column.layer.n_obs_vars, -1
                            )
                        ).flatten()
                    ).reshape(self.raw_obs_shape)
                    self.heatmap_metrics.update(
                        {
                            'agent/obs_rewards': obs_rewards,
                            'agent/striatum_weights': self.agent.striatum_weights,
                            'agent/real_rewards': total_reward.reshape(self.raw_obs_shape)
                        }
                    )
                    self.heatmap_metrics.log(episode)
                    self.image_metrics.log(episode)
            else:
                print_digest(self.scalar_metrics.summarize())
                self.scalar_metrics.reset()
            # <<< logging
            setup_episodes += 1
        else:
            self.environment.close()

    def to_img(self, x: np.ndarray, shape=None):
        return to_gray_img(x, like=isnone(shape, self.raw_obs_shape))

    def preprocess(self, image, mode: Literal['abs', 'clip'] = 'abs'):
        gray = np.dot(image[:, :, :3], [299 / 1000, 587 / 1000, 114 / 1000])

        if mode == 'abs':
            diff = np.abs(gray - self.prev_image)
        elif mode == 'clip':
            diff = np.clip(gray - self.prev_image, 0, None)
        else:
            raise ValueError(f'There is no such mode: "{mode}"!')

        self.prev_image = gray.copy()

        thresh = diff.mean()
        events = np.flatnonzero(diff > thresh)

        return events

    def make_agent(self, conf=None, path=None):
        if path is not None:
            raise NotImplementedError
        elif conf is not None:
            layer_type = conf['run']['layer']
            # assembly agent
            encoder_type = conf['run']['encoder']
            encoder_conf = conf['encoder']
            layer_conf = conf['layer']
            seed = conf['run']['seed']

            if encoder_type == 'sp_ensemble':
                from hima.modules.htm.spatial_pooler import SPDecoder, SPEnsemble

                encoder_conf['seed'] = seed
                encoder_conf['inputDimensions'] = list(self.raw_obs_shape)

                encoder = SPEnsemble(**encoder_conf)
                decoder = SPDecoder(encoder)
            elif encoder_type == 'sp_grouped':
                from hima.experiments.temporal_pooling.stp.sp_ensemble import (
                    SpatialPoolerGroupedWrapper
                )
                encoder_conf['seed'] = seed
                encoder_conf['feedforward_sds'] = [self.raw_obs_shape, 0.1]

                decoder_type = conf['run'].get('decoder', None)
                decoder_conf = conf['decoder']

                encoder = SpatialPoolerGroupedWrapper(**encoder_conf)
                decoder = make_decoder(encoder, decoder_type, decoder_conf)
            else:
                raise ValueError(f'Encoder type {encoder_type} is not supported')

            layer_conf['n_obs_vars'] = encoder.n_groups
            layer_conf['n_obs_states'] = encoder.getSingleNumColumns()
            layer_conf['n_external_states'] = self.n_actions
            layer_conf['seed'] = seed

            if layer_type == 'fchmm':
                from hima.modules.baselines.hmm import FCHMMLayer
                layer = FCHMMLayer(**layer_conf)
            elif layer_type == 'dhtm':
                layer_conf['n_context_states'] = (
                        encoder.getSingleNumColumns() * layer_conf['cells_per_column']
                )
                layer_conf['n_context_vars'] = encoder.n_groups
                layer_conf['n_external_vars'] = 1
                from hima.modules.belief.cortial_column.layer import Layer
                layer = Layer(**layer_conf)
            elif layer_type == 'lstm':
                layer_conf['n_external_vars'] = 1
                layer = LstmLayer(**layer_conf)
            elif layer_type == 'rwkv':
                layer_conf['n_external_vars'] = 1
                from hima.modules.baselines.rwkv import RwkvLayer
                layer = RwkvLayer(**layer_conf)
            else:
                raise NotImplementedError

            cortical_column = CorticalColumn(
                layer,
                encoder,
                decoder
            )

            conf['agent']['seed'] = seed

            if layer_type in {'lstm', 'rwkv'}:
                agent = LstmBioHima(cortical_column, **conf['agent'])
            elif layer_type == 'fchmm':
                from hima.agents.succesor_representations.agent import FCHMMBioHima
                agent = FCHMMBioHima(cortical_column, **conf['agent'])
            else:
                from hima.agents.succesor_representations.agent import BioHIMA
                agent = BioHIMA(
                    cortical_column,
                    **conf['agent']
                )
        else:
            raise ValueError

        return agent

    def compare_srs(self, sr_steps, approximate_tail):
        current_state = self.agent.cortical_column.layer.internal_forward_messages
        pred_sr = self.agent.predict_sr(current_state)
        pred_sr = normalize(
                pred_sr.reshape(
                    self.agent.cortical_column.layer.n_obs_vars, -1
                )
            ).flatten()

        gen_sr, predictions = self.agent.generate_sr(
            sr_steps,
            initial_messages=current_state,
            initial_prediction=self.agent.observation_messages,
            approximate_tail=approximate_tail,
            return_predictions=True
        )
        gen_sr = normalize(
                gen_sr.reshape(
                    self.agent.cortical_column.layer.n_obs_vars, -1
                )
            ).flatten()

        pred_sr_raw = self.agent.cortical_column.decoder.decode(
            pred_sr
        )
        gen_sr_raw = self.agent.cortical_column.decoder.decode(
            gen_sr
        )

        mse = np.mean(np.power(pred_sr_raw - gen_sr_raw, 2))

        return mse, pred_sr, gen_sr, pred_sr_raw, gen_sr_raw, predictions

    @staticmethod
    def get_setup_path(setup):
        return os.path.join(
            os.environ.get('PINBALL_ROOT', None),
            'configs',
            f"{setup}.json"
        )


def main(config_path):
    if len(sys.argv) > 1:
        config_path = sys.argv[1]

    config = dict()

    config['run'] = read_config(config_path)
    experiment = config['run']['experiment']

    if experiment == 'animalai':
        runner = AnimalAITest
    elif experiment == 'pinball':
        runner = PinballTest
    else:
        raise ValueError(f'There is no such {experiment=}!')

    layer = config['run'].get('layer', 'rwkv')
    config['run']['layer_conf'] = f'configs/{layer}/{experiment}.yaml'

    config['env'] = read_config(config['run']['env_conf'])
    config['agent'] = read_config(config['run']['agent_conf'])
    config['layer'] = read_config(config['run']['layer_conf'])

    if 'encoder_conf' in config['run']:
        config['encoder'] = read_config(config['run']['encoder_conf'])
    if 'decoder_conf' in config['run']:
        config['decoder'] = read_config(config['run']['decoder_conf'])

    overrides = parse_arg_list(sys.argv[2:])
    override_config(config, overrides)

    if config['run']['seed'] is None:
        config['run']['seed'] = np.random.randint(0, np.iinfo(np.int32).max)

    if config['run']['log']:
        turn_off_gui_for_matplotlib()
        logger = wandb.init(
            project=config['run']['project_name'], entity=os.environ.get('WANDB_ENTITY', None),
            config=config
        )
    else:
        logger = None

    runner = runner(logger, config)
    runner.run()


if __name__ == '__main__':
    default_config = 'configs/runner/animalai.yaml'
    main(os.environ.get('RUN_CONF', default_config))
