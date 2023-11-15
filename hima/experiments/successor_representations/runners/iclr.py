#  Copyright (c) 2023 Autonomous Non-Profit Organization "Artificial Intelligence Research
#  Institute" (AIRI); Moscow Institute of Physics and Technology (National Research University).
#  All rights reserved.
#
#  Licensed under the AGPLv3 license. See LICENSE in the project root for license information.

import os
import sys

import numpy as np
from scipy.ndimage import gaussian_filter

from hima.agents.succesor_representations.agent import BioHIMA, LstmBioHima, FCHMMBioHima
from hima.common.config.base import read_config, override_config
from hima.common.lazy_imports import lazy_import
from hima.common.run.argparse import parse_arg_list
from hima.modules.belief.cortial_column.cortical_column import CorticalColumn, Layer
from hima.modules.baselines.lstm import LstmLayer
from hima.modules.baselines.rwkv import RwkvLayer
from hima.modules.belief.utils import normalize
from hima.modules.baselines.hmm import FCHMMLayer
from hima.experiments.successor_representations.runners.utils import make_decoder
from hima.common.utils import to_gray_img, isnone
from hima.common.sdr import sparse_to_dense

from typing import Literal

wandb = lazy_import('wandb')


def compare_srs(agent, sr_steps, approximate_tail):
    current_state = agent.cortical_column.layer.internal_forward_messages
    pred_sr = agent.predict_sr(current_state)
    pred_sr = normalize(
        pred_sr.reshape(
            agent.cortical_column.layer.n_obs_vars, -1
        )
    ).flatten()

    gen_sr, predictions = agent.generate_sr(
        sr_steps,
        initial_messages=current_state,
        initial_prediction=agent.observation_messages,
        approximate_tail=approximate_tail,
        return_predictions=True
    )
    gen_sr = normalize(
        gen_sr.reshape(
            agent.cortical_column.layer.n_obs_vars, -1
        )
    ).flatten()

    pred_sr_raw = agent.cortical_column.decoder.decode(
        pred_sr
    )
    gen_sr_raw = agent.cortical_column.decoder.decode(
        gen_sr
    )

    mse = np.mean(np.power(pred_sr_raw - gen_sr_raw, 2))

    return mse, pred_sr, gen_sr, pred_sr_raw, gen_sr_raw, predictions


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
        self.log_value_function = conf['run'].get('log_value_function', False)
        self.value_func_sigma = conf['run'].get('value_func_sigma', 2)

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
        self.start_position = conf['run'].get('start_position', None)
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

        if self.logger is not None:
            from hima.common.metrics import ScalarMetrics, HeatmapMetrics, ImageMetrics, SRStackSurprise, PredictionsStackSurprise
            # define metrics
            self.logger.define_metric("main_metrics/steps", summary="mean")
            self.logger.define_metric("main_metrics/reward", summary="mean")

            basic_scalar_metrics = {
                'main_metrics/reward': np.sum,
                'main_metrics/steps': np.mean,
                'layer/surprise_hidden': np.mean,
                'layer/norm_surprise_hidden': np.mean,
                'sr/td_error': np.mean,
                'sr/norm_td_error': np.mean,
                'sr/test_mse_approx_tail': np.mean,
                'sr/test_mse': np.mean,
                'agent/sr_steps': np.mean,
                'agent/striatum_lr': np.mean
            }

            if self.layer_type == 'dhtm':
                basic_scalar_metrics['layer/n_segments'] = np.mean

            self.scalar_metrics = ScalarMetrics(
                basic_scalar_metrics,
                self.logger
            )

            self.heatmap_metrics = HeatmapMetrics(
                {
                    'agent/obs_rewards': np.mean,
                    'agent/striatum_weights': np.mean,
                    'agent/real_rewards': np.mean,
                    'agent/value_function': np.mean
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

                self.prediction_stack = PredictionsStackSurprise(
                    'layer_n_step/hidden_surprise',
                    self.logger,
                    self.test_sr_steps + 1
                )
            else:
                self.predicted_sr_stack = None
                self.predicted_sr_stack_raw = None
                self.generated_sr_stack = None
                self.generated_sr_stack_raw = None
                self.prediction_stack = None

    def run(self):
        decoder = self.agent.cortical_column.decoder
        total_reward = np.zeros(self.raw_obs_shape).flatten()
        value_function = np.zeros(self.raw_obs_shape).flatten()
        obs_counts = np.zeros(self.raw_obs_shape).flatten()
        setup_episodes = 0
        current_setup_id = 0

        # get setup image
        self.environment.reset(self.start_position)
        setup_im, _, _ = self.environment.obs()

        if self.logger is not None:
            self.logger.log({'setup': wandb.Image(setup_im)}, step=0)

        for i in range(self.n_episodes):
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

                # get setup image
                self.environment.reset(self.start_position)
                setup_im, _, _ = self.environment.obs()

                if self.logger is not None:
                    self.logger.log({'setup': wandb.Image(setup_im)}, step=i)

            self.environment.reset(self.start_position)
            self.agent.reset(self.initial_context, self.initial_external_message)

            while running:
                if (self.reset_context_period > 0) and (steps > 0):
                    if (steps % self.reset_context_period) == 0:
                        self.agent.cortical_column.layer.set_context_messages(
                            self.initial_context
                        )

                self.environment.step()
                obs, reward, is_terminal = self.environment.obs()
                running = not is_terminal

                events = self.preprocess(obs, mode=self.camera_mode)
                # observe events_t and action_{t-1}
                pred_sr, gen_sr = self.agent.observe((events, action), learn=True)
                self.agent.reinforce(reward)

                total_reward[events] += reward

                if running:
                    if (steps % self.action_inertia) == 0:
                        if self.reward_free:
                            action = self._rng.integers(self.n_actions)
                        else:
                            action = self.agent.sample_action()

                    # convert to Pinball action
                    pinball_action = self.actions[action]
                    self.environment.act(pinball_action)

                # >>> logging
                if self.logger is not None:
                    scalar_metrics_update = {
                        'main_metrics/reward': reward,
                        'layer/surprise_hidden': self.agent.surprise,
                        'layer/norm_surprise_hidden': self.agent.ss_surprise.norm_value,
                        'sr/td_error': self.agent.td_error,
                        'sr/norm_td_error': self.agent.ss_td_error.norm_value,
                        'agent/sr_steps': self.agent.sr_steps,
                        'agent/striatum_lr': self.agent.striatum_lr
                    }
                    if self.layer_type == 'dhtm':
                        scalar_metrics_update['layer/n_segments'] = (
                            self.agent.cortical_column.layer.
                            context_factors.connections.numSegments()
                        )
                    self.scalar_metrics.update(
                        scalar_metrics_update
                    )

                    if self.test_srs:
                        (
                            sr_mse_approx_tail,
                            _,
                            gen_sr_test_tail,
                            _,
                            gen_sr_test_tail_raw,
                            predictions
                        ) = compare_srs(
                            self.agent,
                            self.test_sr_steps,
                            True
                        )
                        (
                            sr_mse,
                            pred_sr_test,
                            gen_sr_test,
                            pred_sr_test_raw,
                            gen_sr_test_raw,
                            _
                        ) = compare_srs(
                            self.agent,
                            self.test_sr_steps,
                            False
                        )
                        self.scalar_metrics.update(
                            {
                                'sr/test_mse_approx_tail': sr_mse_approx_tail,
                                'sr/test_mse': sr_mse
                            }
                        )

                        self.predicted_sr_stack.update(
                            pred_sr_test,
                            self.agent.cortical_column.output_sdr.sparse
                        )
                        self.predicted_sr_stack_raw.update(
                            pred_sr_test_raw,
                            self.agent.cortical_column.input_sdr.sparse
                        )
                        self.generated_sr_stack.update(
                            gen_sr_test,
                            self.agent.cortical_column.output_sdr.sparse
                        )
                        self.generated_sr_stack_raw.update(
                            gen_sr_test_raw,
                            self.agent.cortical_column.input_sdr.sparse
                        )

                        preds = [self.agent.cortical_column.layer.prediction_columns.copy()]
                        preds.extend(predictions)
                        self.prediction_stack.update(
                            preds,
                            self.agent.cortical_column.output_sdr.sparse
                        )
                    else:
                        pred_sr_test_raw = None
                        gen_sr_test_raw = None
                        gen_sr_test_tail_raw = None

                    if self.log_value_function and (i >= self.update_start):
                        value = np.sum(self.agent.evaluate_actions(with_planning=True))
                        value_function[events] += value
                        obs_counts[events] += 1

                    if (i >= self.update_start) and (i % self.update_period) == 0:
                        raw_beh = self.to_img(self.prev_image)
                        proc_beh = self.to_img(sparse_to_dense(events, like=self.prev_image))
                        pred_beh = self.to_img(self.agent.cortical_column.predicted_image)

                        actual_state = self.agent.cortical_column.layer.internal_forward_messages
                        predicted_state = self.agent.cortical_column.layer.prediction_cells
                        if type(actual_state) is list:
                            actual_state = self.agent._extract_state_from_context(
                                actual_state
                            ).cpu().numpy()
                            predicted_state = self.agent._extract_state_from_context(
                                predicted_state
                            ).cpu().numpy()

                        hid_beh = self.to_img(
                            actual_state,
                            shape=(self.agent.cortical_column.layer.n_columns, -1)
                        )
                        hid_pred_beh = self.to_img(
                            predicted_state,
                            shape=(self.agent.cortical_column.layer.n_columns, -1)
                        )

                        if pred_sr is not None:
                            pred_sr = self.to_img(
                                    decoder.decode(
                                        normalize(
                                            pred_sr.reshape(
                                                self.agent.cortical_column.layer.n_obs_vars, -1
                                            )
                                        ).flatten()
                                    )
                            )
                        else:
                            pred_sr = np.zeros(self.raw_obs_shape).astype('uint8')

                        if gen_sr is not None:
                            gen_sr = self.to_img(
                                    decoder.decode(
                                        normalize(
                                            gen_sr.reshape(
                                                self.agent.cortical_column.layer.n_obs_vars, -1
                                            )
                                        ).flatten()
                                    )
                            )
                        else:
                            gen_sr = np.zeros(self.raw_obs_shape).astype('uint8')

                        self.image_metrics.update(
                            {
                                'agent/behavior': np.hstack(
                                    [raw_beh, proc_beh, pred_beh, pred_sr, gen_sr]),
                                'agent/hidden': np.hstack(
                                    [hid_beh, hid_pred_beh]
                                )
                            }
                        )

                        if self.test_srs:
                            self.image_metrics.update(
                                {
                                    'sr/sr': np.hstack(
                                        [
                                            raw_beh,
                                            proc_beh,
                                            self.to_img(pred_sr_test_raw),
                                            self.to_img(gen_sr_test_raw),
                                            self.to_img(gen_sr_test_tail_raw)
                                        ]
                                    )
                                }
                            )
                # <<< logging

                steps += 1

                if steps >= self.max_steps:
                    running = False

            # >>> logging
            if self.logger is not None:
                self.scalar_metrics.update({'main_metrics/steps': steps})
                self.scalar_metrics.log(i)

                if self.test_srs:
                    self.predicted_sr_stack.log(i)
                    self.predicted_sr_stack_raw.log(i)
                    self.generated_sr_stack.log(i)
                    self.generated_sr_stack_raw.log(i)
                    self.prediction_stack.log(i)

                if (i >= self.update_start) and (i % self.update_period) == 0:
                    obs_rewards = decoder.decode(
                        normalize(
                            self.agent.observation_rewards.reshape(
                                self.agent.cortical_column.layer.n_obs_vars, -1
                            )
                        ).flatten()
                    ).reshape(self.raw_obs_shape)
                    value_function_im = gaussian_filter(
                                np.divide(
                                        value_function,
                                        obs_counts,
                                        where=obs_counts > 0,
                                        out=np.zeros_like(value_function)
                                ).reshape(self.raw_obs_shape),
                                sigma=self.value_func_sigma
                            )
                    setup_im_gray = setup_im.sum(axis=-1)
                    setup_im_bin = np.flatnonzero(setup_im_gray > setup_im_gray.mean())
                    value_function_im = value_function_im.flatten()
                    value_function_im[setup_im_bin] = np.nan

                    self.heatmap_metrics.update(
                        {
                            'agent/obs_rewards': obs_rewards,
                            'agent/striatum_weights': self.agent.striatum_weights,
                            'agent/real_rewards': total_reward.reshape(self.raw_obs_shape),
                            'agent/value_function': value_function_im.reshape(self.raw_obs_shape)
                        }
                    )
                    self.heatmap_metrics.log(i)
                    self.image_metrics.log(i)
            # <<< logging
            setup_episodes += 1
        else:
            self.environment.close()

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

    def to_img(self, x: np.ndarray, shape=None):
        return to_gray_img(x, like=isnone(shape, self.raw_obs_shape))

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
                layer = FCHMMLayer(**layer_conf)
            elif layer_type == 'dhtm':
                layer_conf['n_context_states'] = (
                        encoder.getSingleNumColumns() * layer_conf['cells_per_column']
                )
                layer_conf['n_context_vars'] = encoder.n_groups
                layer_conf['n_external_vars'] = 1
                layer = Layer(**layer_conf)
            elif layer_type == 'lstm':
                layer_conf['n_external_vars'] = 1
                layer = LstmLayer(**layer_conf)
            elif layer_type == 'rwkv':
                layer_conf['n_external_vars'] = 1
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
                agent = FCHMMBioHima(cortical_column, **conf['agent'])
            else:
                agent = BioHIMA(
                    cortical_column,
                    **conf['agent']
                )
        else:
            raise ValueError

        return agent

    @staticmethod
    def get_setup_path(setup):
        return os.path.join(
            os.environ.get('PINBALL_ROOT', None),
            'configs',
            f"{setup}.json"
        )


class AnimalAITest:
    def __init__(self, logger, conf, max_workers=10):
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
        self.frame_skip = conf['run'].get('frame_skip', 0)
        self.strategies = conf['run'].get('strategies', None)

        assert self.frame_skip >= 0

        self.setups = conf['run']['setup']
        self.setup_period = conf['run'].get('setup_period', None)

        if self.setup_period is None:
            self.setup_period = [self.n_episodes // len(self.setups)] * len(self.setups)
        elif type(self.setup_period) is int:
            period = self.setup_period
            self.setup_period = [period] * len(self.setups)

        assert len(self.setups) == len(self.setup_period)

        if 'reset_context_period' in conf['layer']:
            self.reset_context_period = conf['layer'].pop(
                'reset_context_period'
            )
        else:
            self.reset_context_period = 0

        self.max_workers = max_workers
        self.env_conf = conf['env']
        self.env_conf['seed'] = self.seed
        self.env_conf['file_name'] = os.environ.get('ANIMALAI_EXE', None)

        (
            self.environment,
            self.behavior,
            self.raw_obs_shape,
            self.actions,
            self.n_actions
        ) = self.setup_environment(self.setups[0])

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

        if self.logger is not None:
            from hima.common.metrics import ScalarMetrics, HeatmapMetrics, ImageMetrics, SRStackSurprise, PredictionsStackSurprise
            # define metrics
            basic_scalar_metrics = {
                    'main_metrics/reward': np.sum,
                    'main_metrics/steps': np.mean,
                    'layer/surprise_hidden': np.mean,
                    'layer/norm_surprise_hidden': np.mean,
                    'layer/relative_surprise': np.mean,
                    'sr/td_error': np.mean,
                    'sr/norm_td_error': np.mean,
                    'sr/test_mse_approx_tail': np.mean,
                    'sr/test_mse': np.mean,
                    'agent/sr_steps': np.mean,
                    'agent/striatum_lr': np.mean
                }

            if self.layer_type == 'dhtm':
                basic_scalar_metrics['layer/n_segments'] = np.mean

            self.scalar_metrics = ScalarMetrics(
                basic_scalar_metrics,
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

                self.prediction_stack = PredictionsStackSurprise(
                    'layer_n_step/hidden_surprise',
                    self.logger,
                    self.test_sr_steps + 1
                )
            else:
                self.predicted_sr_stack = None
                self.predicted_sr_stack_raw = None
                self.generated_sr_stack = None
                self.generated_sr_stack_raw = None

    def run(self):
        decoder = self.agent.cortical_column.decoder
        total_reward = np.zeros(self.raw_obs_shape).flatten()
        setup_episodes = 0
        current_setup_id = 0

        for i in range(self.n_episodes):
            steps = 0
            running = True
            action = self.initial_action

            self.prev_image = self.initial_previous_image

            # change setup
            if setup_episodes >= self.setup_period[current_setup_id]:
                self.environment.close()
                current_setup_id += 1
                current_setup_id = current_setup_id % len(self.setups)
                (
                    self.environment,
                    self.behavior,
                    self.raw_obs_shape,
                    self.actions,
                    self.n_actions
                ) = self.setup_environment(self.setups[current_setup_id])
                setup_episodes = 0

            self.environment.reset()
            self.agent.reset(self.initial_context, self.initial_external_message)

            strategy = None
            action_step = 0

            while running:
                if (self.reset_context_period > 0) and (steps > 0):
                    if (steps % self.reset_context_period) == 0:
                        self.agent.cortical_column.layer.set_context_messages(
                            self.initial_context
                        )

                reward = 0
                for frame in range(self.frame_skip + 1):
                    if action is not None:
                        aai_action = self.actions[action]
                        self.environment.set_actions(self.behavior, aai_action.action_tuple)

                    self.environment.step()
                    dec, term = self.environment.get_steps(self.behavior)

                    if len(dec) > 0:
                        obs = self.environment.get_obs_dict(dec.obs)["camera"]
                        reward += dec.reward

                    if len(term):
                        obs = self.environment.get_obs_dict(term.obs)["camera"]
                        reward += term.reward
                        running = False
                        break

                    if action is None:
                        break

                events = self.preprocess(obs, mode=self.camera_mode)
                # observe events_t and action_{t-1}
                pred_sr, gen_sr = self.agent.observe((events, action), learn=True)
                self.agent.reinforce(reward)

                total_reward[events] += reward

                if running:
                    if self.strategies is not None:
                        if steps == 0:
                            strategy = self.strategies[self.agent.sample_action()]

                        if (steps % self.action_inertia) == 0:
                            if action_step < len(strategy):
                                action = strategy[action_step]
                            else:
                                running = False
                            action_step += 1
                    else:
                        if (steps % self.action_inertia) == 0:
                            if self.reward_free:
                                action = self._rng.integers(self.n_actions)
                            else:
                                action = self.agent.sample_action()

                # >>> logging
                if self.logger is not None:
                    scalar_metrics_update = {
                        'main_metrics/reward': reward,
                        'layer/surprise_hidden': self.agent.surprise,
                        'layer/norm_surprise_hidden': self.agent.ss_surprise.norm_value,
                        'layer/relative_surprise': self.agent.relative_log_surprise,
                        'sr/td_error': self.agent.td_error,
                        'sr/norm_td_error': self.agent.ss_td_error.norm_value,
                        'agent/sr_steps': self.agent.sr_steps,
                        'agent/striatum_lr': self.agent.striatum_lr
                    }
                    if self.layer_type == 'dhtm':
                        scalar_metrics_update['layer/n_segments'] = (
                            self.agent.cortical_column.layer.
                            context_factors.connections.numSegments()
                        )
                    self.scalar_metrics.update(
                        scalar_metrics_update
                    )

                    if self.test_srs:
                        (
                            sr_mse_approx_tail,
                            _,
                            gen_sr_test_tail,
                            _,
                            gen_sr_test_tail_raw,
                            predictions
                        ) = compare_srs(
                            self.agent,
                            self.test_sr_steps,
                            True
                        )
                        (
                            sr_mse,
                            pred_sr_test,
                            gen_sr_test,
                            pred_sr_test_raw,
                            gen_sr_test_raw,
                            _
                        ) = compare_srs(
                            self.agent,
                            self.test_sr_steps,
                            False
                        )
                        self.scalar_metrics.update(
                            {
                                'sr/test_mse_approx_tail': sr_mse_approx_tail,
                                'sr/test_mse': sr_mse
                            }
                        )

                        self.predicted_sr_stack.update(
                            pred_sr_test,
                            self.agent.cortical_column.output_sdr.sparse
                        )
                        self.predicted_sr_stack_raw.update(
                            pred_sr_test_raw,
                            self.agent.cortical_column.input_sdr.sparse
                        )
                        self.generated_sr_stack.update(
                            gen_sr_test,
                            self.agent.cortical_column.output_sdr.sparse
                        )
                        self.generated_sr_stack_raw.update(
                            gen_sr_test_raw,
                            self.agent.cortical_column.input_sdr.sparse
                        )

                        preds = [self.agent.cortical_column.layer.prediction_columns.copy()]
                        preds.extend(predictions)
                        self.prediction_stack.update(
                            preds,
                            self.agent.cortical_column.output_sdr.sparse
                        )
                    else:
                        pred_sr_test_raw = None
                        gen_sr_test_raw = None
                        gen_sr_test_tail_raw = None

                    if (i % self.update_period) == 0:
                        raw_beh = self.to_img(self.prev_image)
                        proc_beh = self.to_img(sparse_to_dense(events, like=self.prev_image))
                        pred_beh = self.to_img(self.agent.cortical_column.predicted_image)

                        actual_state = self.agent.cortical_column.layer.internal_forward_messages
                        predicted_state = self.agent.cortical_column.layer.prediction_cells
                        if type(actual_state) is list:
                            actual_state = self.agent._extract_state_from_context(
                                actual_state
                            ).cpu().numpy()
                            predicted_state = self.agent._extract_state_from_context(
                                predicted_state
                            ).cpu().numpy()

                        hid_beh = self.to_img(
                            actual_state,
                            shape=(self.agent.cortical_column.layer.n_columns, -1)
                        )
                        hid_pred_beh = self.to_img(
                            predicted_state,
                            shape=(self.agent.cortical_column.layer.n_columns, -1)
                        )

                        if pred_sr is not None:
                            pred_sr = self.to_img(
                                    decoder.decode(
                                        normalize(
                                            pred_sr.reshape(
                                                self.agent.cortical_column.layer.n_obs_vars, -1
                                            )
                                        ).flatten()
                                    )
                            )
                        else:
                            pred_sr = np.zeros(self.raw_obs_shape).astype('uint8')

                        if gen_sr is not None:
                            gen_sr = self.to_img(
                                    decoder.decode(
                                        normalize(
                                            gen_sr.reshape(
                                                self.agent.cortical_column.layer.n_obs_vars, -1
                                            )
                                        ).flatten()
                                    )
                            )
                        else:
                            gen_sr = np.zeros(self.raw_obs_shape).astype('uint8')

                        self.image_metrics.update(
                            {
                                'agent/behavior': np.hstack(
                                    [raw_beh, proc_beh, pred_beh, pred_sr, gen_sr]),
                                'agent/hidden': np.hstack(
                                    [hid_beh, hid_pred_beh]
                                )
                            }
                        )

                        if self.test_srs:
                            self.image_metrics.update(
                                {
                                    'sr/sr': np.hstack(
                                        [
                                            raw_beh,
                                            proc_beh,
                                            self.to_img(pred_sr_test_raw),
                                            self.to_img(gen_sr_test_raw),
                                            self.to_img(gen_sr_test_tail_raw)
                                        ]
                                    )
                                }
                            )
                # <<< logging

                steps += 1

                if steps >= self.max_steps:
                    running = False

            # >>> logging
            if self.logger is not None:
                self.scalar_metrics.update({'main_metrics/steps': steps})
                self.scalar_metrics.log(i)

                if self.test_srs:
                    self.predicted_sr_stack.log(i)
                    self.predicted_sr_stack_raw.log(i)
                    self.generated_sr_stack.log(i)
                    self.generated_sr_stack_raw.log(i)
                    self.prediction_stack.log(i)

                if (i >= self.update_start) and (i % self.update_period) == 0:
                    obs_rewards = decoder.decode(
                        normalize(
                            self.agent.observation_rewards.reshape(
                                self.agent.cortical_column.layer.n_obs_vars, -1
                            ) - self.agent.observation_rewards.min()
                        ).flatten()
                    ).reshape(self.raw_obs_shape)
                    self.heatmap_metrics.update(
                        {
                            'agent/obs_rewards': obs_rewards,
                            'agent/striatum_weights': self.agent.striatum_weights,
                            'agent/real_rewards': total_reward.reshape(self.raw_obs_shape)
                        }
                    )
                    self.heatmap_metrics.log(i)
                    self.image_metrics.log(i)
            # <<< logging
            setup_episodes += 1
        else:
            self.environment.close()

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

    def to_img(self, x: np.ndarray, shape=None):
        return to_gray_img(x, like=isnone(shape, self.raw_obs_shape))

    def setup_environment(self, setup):
        from animalai.envs.actions import AAIActions
        from animalai.envs.environment import AnimalAIEnvironment
        from mlagents_envs.exception import UnityWorkerInUseException

        self.env_conf['arenas_configurations'] = self.get_setup_path(
            setup
        )
        worker_id = 0
        while worker_id < self.max_workers:
            try:
                environment = AnimalAIEnvironment(
                    worker_id=worker_id,
                    **self.env_conf
                )
                break
            except UnityWorkerInUseException:
                worker_id += 1
        else:
            raise Exception('Too many workers.')

        # get agent proxi in unity
        behavior = list(environment.behavior_specs.keys())[0]
        raw_obs_shape = environment.behavior_specs[behavior].observation_specs[
            0].shape[:2]
        actions = [
            AAIActions().LEFT,
            AAIActions().FORWARDS,
            AAIActions().RIGHT,
            # AAIActions().BACKWARDS
        ]
        n_actions = len(actions)

        return environment, behavior, raw_obs_shape, actions, n_actions

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
                layer = FCHMMLayer(**layer_conf)
            elif layer_type == 'dhtm':
                layer_conf['n_context_states'] = (
                        encoder.getSingleNumColumns() * layer_conf['cells_per_column']
                )
                layer_conf['n_context_vars'] = encoder.n_groups
                layer_conf['n_external_vars'] = 1
                layer = Layer(**layer_conf)
            elif layer_type == 'lstm':
                layer_conf['n_external_vars'] = 1
                layer = LstmLayer(**layer_conf)
            elif layer_type == 'rwkv':
                layer_conf['n_external_vars'] = 1
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
            else:
                agent = BioHIMA(
                    cortical_column,
                    **conf['agent']
                )
        else:
            raise ValueError

        return agent

    @staticmethod
    def get_setup_path(setup):
        return os.path.join(
            os.environ.get('ANIMALAI_ROOT', None),
            'configs',
            f"{setup}"
        )


class GridWorldTest:
    def __init__(self, logger, conf):
        self.logger = logger
        self.seed = conf['run'].get('seed')
        self._rng = np.random.default_rng(self.seed)

        self.setups = conf['run']['setups']
        self.current_setup_id = 0
        self.setup_period = conf['run'].get('setup_period', 0)

        if 'reset_context_period' in conf['layer']:
            self.reset_context_period = conf['layer'].pop(
                'reset_context_period'
            )
        else:
            self.reset_context_period = 0

        (
            self.environment,
            self.raw_obs_shape,
            self.actions,
            self.n_actions
         ) = self.setup_environment(
            self.setups[self.current_setup_id]
        )

        self.start_position = conf['run']['start_position']

        self.agent = self.make_agent(conf, conf['run'].get('agent_path', None))

        self.n_episodes = conf['run']['n_episodes']
        self.max_steps = conf['run']['max_steps']
        self.update_rate = conf['run']['update_rate']
        self.reward_free = conf['run'].get('reward_free', False)
        self.test_srs = conf['run'].get('test_srs', False)
        self.test_sr_steps = conf['run'].get('test_sr_steps', 0)
        self.layer_type = conf['run']['layer']

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

        if self.logger is not None:
            from hima.common.metrics import ScalarMetrics, HeatmapMetrics, ImageMetrics, SRStackSurprise
            # define metrics
            basic_scalar_metrics = {
                'main_metrics/reward': np.sum,
                'main_metrics/steps': np.mean,
                'layer/surprise_hidden': np.mean,
                'sr/td_error': np.mean,
                'sr/test_mse_approx_tail': np.mean,
                'sr/test_mse': np.mean
            }

            if self.layer_type == 'dhtm':
                basic_scalar_metrics['layer/n_segments'] = np.mean

            self.scalar_metrics = ScalarMetrics(
                basic_scalar_metrics,
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
                    'agent/predictions',
                    'sr/sr',
                    'layer/predictions'
                ],
                self.logger,
                log_fps=conf['run']['log_gif_fps']
            )

            if self.test_srs:
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
            else:
                self.predicted_sr_stack = None
                self.generated_sr_stack = None

    def run(self):
        total_reward = np.zeros(self.raw_obs_shape).flatten()
        for i in range(self.n_episodes):
            steps = 0
            running = True
            action = self.initial_action

            # change setup
            if (self.setup_period * i > 0) and (i % self.setup_period == 0):
                self.current_setup_id += 1
                self.current_setup_id = self.current_setup_id % len(self.setups)
                (
                    self.environment,
                    self.raw_obs_shape,
                    self.actions,
                    self.n_actions
                ) = self.setup_environment(
                    self.setups[self.current_setup_id]
                )

            self.environment.reset(*self.start_position)
            self.agent.reset(self.initial_context, self.initial_external_message)

            while running:
                if (self.reset_context_period > 0) and (steps > 0):
                    if (steps % self.reset_context_period) == 0:
                        self.agent.cortical_column.layer.set_context_messages(
                            self.initial_context
                        )

                self.environment.step()
                obs, reward, is_terminal = self.environment.obs()
                events = [obs]
                running = not is_terminal

                # observe events_t and action_{t-1}
                pred_sr, gen_sr = self.agent.observe((events, action), learn=True)
                self.agent.reinforce(reward)

                total_reward[events] += reward

                if running:
                    if self.reward_free:
                        action = self._rng.integers(self.n_actions)
                    else:
                        action = self.agent.sample_action()

                    # convert to AAI action
                    pinball_action = self.actions[action]
                    self.environment.act(pinball_action)

                # >>> logging
                if self.logger is not None:
                    scalar_metrics_update = {
                        'main_metrics/reward': reward,
                        'layer/surprise_hidden': self.agent.surprise,
                        'sr/td_error': self.agent.td_error
                    }
                    if self.layer_type == 'dhtm':
                        scalar_metrics_update['layer/n_segments'] = (
                            self.agent.cortical_column.layer.
                            context_factors.connections.numSegments()
                        )
                    self.scalar_metrics.update(
                        scalar_metrics_update
                    )

                    if self.test_srs:
                        (
                            sr_mse_approx_tail,
                            _,
                            gen_sr_test_tail,
                        ) = self.compare_srs(
                            self.test_sr_steps,
                            True
                        )
                        (
                            sr_mse,
                            pred_sr_test,
                            gen_sr_test
                        ) = self.compare_srs(
                            self.test_sr_steps,
                            False
                        )
                        self.scalar_metrics.update(
                            {
                                'sr/test_mse_approx_tail': sr_mse_approx_tail,
                                'sr/test_mse': sr_mse
                            }
                        )

                        self.predicted_sr_stack.update(
                            pred_sr_test,
                            self.agent.cortical_column.output_sdr.sparse
                        )
                        self.generated_sr_stack.update(
                            gen_sr_test,
                            self.agent.cortical_column.output_sdr.sparse
                        )

                    else:
                        pred_sr_test = None
                        gen_sr_test = None
                        gen_sr_test_tail = None

                    if (i % self.update_rate) == 0:
                        raw_beh = self.environment.colors.astype(np.float64)
                        agent_color = self.agent.cortical_column.layer.n_obs_states + 1
                        raw_beh[self.environment.r, self.environment.c] = agent_color
                        raw_beh += 1
                        raw_beh /= (agent_color + 1)
                        raw_beh = self.to_img(raw_beh, shape=raw_beh.shape)

                        proc_beh = self.to_img(sparse_to_dense(events, shape=self.raw_obs_shape))
                        pred_beh = self.to_img(
                            self.agent.cortical_column.predicted_image,
                            shape=self.raw_obs_shape
                        )

                        if pred_sr is not None:
                            pred_sr = self.to_img(pred_sr, shape=self.raw_obs_shape)
                        else:
                            pred_sr = np.zeros(self.raw_obs_shape).astype('uint8')

                        if gen_sr is not None:
                            gen_sr = self.to_img(gen_sr, shape=self.raw_obs_shape)
                        else:
                            gen_sr = np.zeros(self.raw_obs_shape).astype('uint8')

                        self.image_metrics.update(
                            {
                                'agent/behavior': raw_beh,
                                'agent/predictions': np.hstack(
                                    [proc_beh, pred_beh, pred_sr, gen_sr]
                                )
                            }
                        )

                        if self.test_srs:
                            self.image_metrics.update(
                                {
                                    'sr/sr': np.hstack(
                                        [
                                            proc_beh,
                                            self.to_img(pred_sr_test, shape=self.raw_obs_shape),
                                            self.to_img(gen_sr_test, shape=self.raw_obs_shape),
                                            self.to_img(gen_sr_test_tail, shape=self.raw_obs_shape)
                                        ]
                                    )
                                }
                            )
                # <<< logging

                steps += 1

                if steps >= self.max_steps:
                    running = False

            # >>> logging
            if self.logger is not None:
                self.scalar_metrics.update({'main_metrics/steps': steps})
                self.scalar_metrics.log(i)

                if self.test_srs:
                    self.predicted_sr_stack.log(i)
                    self.generated_sr_stack.log(i)

                if (i % self.update_rate) == 0:
                    obs_rewards = self.agent.observation_rewards.reshape(self.raw_obs_shape)
                    self.heatmap_metrics.update(
                        {
                            'agent/obs_rewards': obs_rewards,
                            'agent/striatum_weights': self.agent.striatum_weights,
                            'agent/real_rewards': total_reward.reshape(self.raw_obs_shape)
                        }
                    )
                    self.heatmap_metrics.log(i)
                    self.image_metrics.log(i)
            # <<< logging

    def to_img(self, x: np.ndarray, shape=None):
        return to_gray_img(x, like=isnone(shape, self.raw_obs_shape))

    def make_agent(self, conf=None, path=None):
        if path is not None:
            raise NotImplementedError
        elif conf is not None:
            layer_type = conf['run']['layer']
            # assembly agent
            layer_conf = conf['layer']
            seed = conf['run']['seed']

            layer_conf['n_obs_vars'] = 1
            layer_conf['n_obs_states'] = self.raw_obs_shape[0]
            layer_conf['n_external_states'] = self.n_actions
            layer_conf['seed'] = seed

            if layer_type == 'fchmm':
                layer = FCHMMLayer(**layer_conf)
            elif layer_type == 'dhtm':
                layer_conf['n_context_states'] = (
                        layer_conf['n_obs_states'] * layer_conf['cells_per_column']
                )
                layer_conf['n_context_vars'] = 1
                layer_conf['n_external_vars'] = 1
                layer = Layer(**layer_conf)
            elif layer_type == 'lstm':
                layer_conf['n_external_vars'] = 1
                layer = LstmLayer(**layer_conf)
            elif layer_type == 'rwkv':
                layer_conf['n_external_vars'] = 1
                layer = RwkvLayer(**layer_conf)
            else:
                raise NotImplementedError

            cortical_column = CorticalColumn(
                layer,
                encoder=None,
                decoder=None
            )

            conf['agent']['seed'] = seed

            if layer_type in {'lstm', 'rwkv'}:
                agent = LstmBioHima(cortical_column, **conf['agent'])
            else:
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

        gen_sr = self.agent.generate_sr(
            sr_steps,
            initial_messages=current_state,
            initial_prediction=self.agent.observation_messages,
            approximate_tail=approximate_tail,
        )
        gen_sr = normalize(
                gen_sr.reshape(
                    self.agent.cortical_column.layer.n_obs_vars, -1
                )
            ).flatten()

        mse = np.mean(np.power(pred_sr - gen_sr, 2))

        return mse, pred_sr, gen_sr

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
    config['agent'] = read_config(config['run']['agent_conf'])

    layer_conf_path = config['run']['layer_conf']
    config['run']['layer'] = layer_conf_path.split('/')[-2]
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
        import wandb
        logger = wandb.init(
            project=config['run']['project_name'], entity=os.environ.get('WANDB_ENTITY', None),
            config=config
        )
    else:
        logger = None

    if config['run']['experiment'] == 'pinball':
        runner = PinballTest(logger, config)
    elif config['run']['experiment'] == 'animalai':
        runner = AnimalAITest(logger, config)
    elif config['run']['experiment'] == 'gridworld':
        runner = GridWorldTest(logger, config)
    else:
        raise ValueError(f'There is no such experiment {config["run"]["experiment"]}!')

    runner.run()


if __name__ == '__main__':
    default_config = 'configs/runner/pinball.yaml'
    main(os.environ.get('RUN_CONF', default_config))
