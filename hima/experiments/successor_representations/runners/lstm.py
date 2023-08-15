#  Copyright (c) 2023 Autonomous Non-Profit Organization "Artificial Intelligence Research
#  Institute" (AIRI); Moscow Institute of Physics and Technology (National Research University).
#  All rights reserved.
#
#  Licensed under the AGPLv3 license. See LICENSE in the project root for license information.
import io
import os
import sys

import numpy as np
from PIL import Image

from hima.agents.succesor_representations.agent import BioHIMA
from hima.common.config.base import read_config, override_config
from hima.common.lazy_imports import lazy_import
from hima.common.run.argparse import parse_arg_list
from hima.modules.belief.cortial_column.cortical_column import CorticalColumn, Layer

wandb = lazy_import('wandb')


class AnimalAITest:
    def __init__(self, logger, conf):
        from animalai.envs.actions import AAIActions
        from animalai.envs.environment import AnimalAIEnvironment

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
        self.environment = AnimalAIEnvironment(**conf['env'])
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

        encoder_type = conf['run']['encoder']
        layer_conf = conf['layer']

        if encoder_type == 'sp_ensemble':
            from hima.modules.htm.spatial_pooler import SPDecoder, SPEnsemble

            # assembly agent
            encoder_conf = conf['encoder']
            encoder_conf['seed'] = self.seed
            encoder_conf['inputDimensions'] = list(self.raw_obs_shape)

            encoder = SPEnsemble(**encoder_conf)
            decoder = SPDecoder(encoder)

            layer_conf['n_obs_states'] = encoder.sps[0].getNumColumns()
            layer_conf['n_obs_vars'] = encoder.n_sp
            layer_conf['n_context_states'] = (
                    encoder.sps[0].getNumColumns() * layer_conf['cells_per_column']
            )
            layer_conf['n_context_vars'] = encoder.n_sp
        elif encoder_type == 'sp_grouped':
            from hima.experiments.temporal_pooling.stp.sp_ensemble import (
                SpatialPoolerGroupedWrapper
            )
            # assembly agent
            encoder_conf = conf['encoder']
            encoder_conf['seed'] = self.seed
            encoder_conf['feedforward_sds'] = [self.raw_obs_shape, 0.1]
            decoder_type = conf['run'].get('decoder', None)

            encoder = SpatialPoolerGroupedWrapper(**encoder_conf)
            decoder = make_decoder(encoder, decoder_type)

            layer_conf['n_obs_states'] = encoder.getSingleNumColumns()
            layer_conf['n_obs_vars'] = encoder.n_groups
            layer_conf['n_context_states'] = (
                    encoder.getSingleNumColumns() * layer_conf['cells_per_column']
            )
            layer_conf['n_context_vars'] = encoder.n_groups
        else:
            raise ValueError(f'Encoder type {encoder_type} is not supported')

        layer_conf['n_external_vars'] = 1
        layer_conf['n_external_states'] = self.n_actions
        layer_conf['seed'] = self.seed

        layer = Layer(**layer_conf)

        cortical_column = CorticalColumn(
            layer,
            encoder,
            decoder
        )

        self.agent = BioHIMA(
            cortical_column,
            **conf['agent']
        )

        self.n_episodes = conf['run']['n_episodes']
        self.max_steps = conf['run']['max_steps']
        self.update_rate = conf['run']['update_rate']
        self.reset_context_period = conf['run'].get('reset_context_period', 0)

        self.initial_previous_image = self._rng.random(self.raw_obs_shape)
        self.prev_image = self.initial_previous_image
        self.initial_context = np.zeros_like(
            self.agent.cortical_column.layer.context_messages
        )
        self.initial_context[
            np.arange(
                self.agent.cortical_column.layer.n_hidden_vars
            ) * self.agent.cortical_column.layer.n_hidden_states
        ] = 1

        if self.logger is not None:
            from metrics import ScalarMetrics, HeatmapMetrics, ImageMetrics
            # define metrics
            self.scalar_metrics = ScalarMetrics(
                {
                    'main_metrics/reward': np.sum,
                    'main_metrics/steps': np.mean,
                    'layer/surprise_hidden': np.mean,
                    'layer/n_segments': np.mean,
                    'layer/n_factors': np.mean,
                    'agent/td_error': np.mean
                },
                self.logger
            )

            self.heatmap_metrics = HeatmapMetrics(
                {
                    'agent/prior': np.mean,
                    'agent/striatum_weights': np.mean
                },
                self.logger
            )

            self.image_metrics = ImageMetrics(
                [
                    'agent/behavior',
                    'agent/sr',
                    'layer/factor_graph',
                    'layer/predictions'
                ],
                self.logger,
                log_fps=conf['run']['log_gif_fps']
            )

    def run(self):
        episode_print_schedule = 50

        for i in range(self.n_episodes):
            if i % episode_print_schedule == 0:
                print(f'Episode {i}')

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

                if len(term):
                    obs = self.environment.get_obs_dict(term.obs)["camera"]
                    reward += term.reward
                    running = False

                events = self.preprocess(obs)

                pred_sr, gen_sr = self.agent.observe(
                    (events, action), learn=True, raw_observation=self.prev_image
                )
                self.agent.reinforce(reward)

                if running:
                    action = self.agent.sample_action()
                    # convert to AAI action
                    aai_action = self.actions[action]
                    self.environment.set_actions(self.behavior, aai_action.action_tuple)

                # >>> logging
                if self.logger is not None:
                    self.scalar_metrics.update(
                        {
                            'main_metrics/reward': reward,
                            'layer/surprise_hidden': self.agent.surprise,
                            'layer/n_segments': self.agent.cortical_column.layer.
                            context_factors.connections.numSegments(),
                            'layer/n_factors': self.agent.cortical_column.layer.
                            context_factors.factor_connections.numSegments(),
                            'agent/td_error': self.agent.td_error
                        }
                    )

                    if (i % self.update_rate) == 0:
                        raw_beh = (self.prev_image * 255).astype('uint8')

                        proc_beh = np.zeros(self.raw_obs_shape).flatten()
                        proc_beh[events] = 1
                        proc_beh = (proc_beh.reshape(self.raw_obs_shape) * 255).astype('uint8')

                        pred_beh = (self.agent.cortical_column.predicted_image.reshape(
                            self.raw_obs_shape
                        ) * 255).astype('uint8')

                        if pred_sr is not None:
                            pred_sr = (
                                    self.agent.cortical_column.decoder.decode(pred_sr)
                                    .reshape(self.raw_obs_shape) * 255
                            ).astype('uint8')
                        else:
                            pred_sr = np.zeros(self.raw_obs_shape).astype('uint8')

                        if gen_sr is not None:
                            gen_sr = (
                                    self.agent.cortical_column.decoder.decode(gen_sr)
                                    .reshape(self.raw_obs_shape) * 255
                            ).astype('uint8')
                        else:
                            gen_sr = np.zeros(self.raw_obs_shape).astype('uint8')

                        self.image_metrics.update(
                            {
                                'agent/behavior': np.hstack(
                                    [raw_beh, proc_beh, pred_beh, pred_sr, gen_sr])
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

                if (i % self.update_rate) == 0:
                    prior_probs = self.agent.cortical_column.decoder.decode(
                        self.agent.observation_prior
                    ).reshape(self.raw_obs_shape)
                    self.heatmap_metrics.update(
                        {
                            'agent/prior': prior_probs,
                            'agent/striatum_weights': self.agent.striatum_weights
                        }
                    )
                    self.heatmap_metrics.log(i)

                    self.image_metrics.update(
                        {
                            'layer/factor_graph': Image.open(
                                io.BytesIO(
                                    self.agent.cortical_column.layer.draw_factor_graph()
                                )
                            )
                        }
                    )
                    self.image_metrics.log(i)
            # <<< logging
        else:
            self.environment.close()

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

        conf['env']['seed'] = self.seed
        conf['env']['exe_path'] = os.environ.get('PINBALL_EXE', None)
        conf['env']['config_path'] = os.path.join(
            os.environ.get('PINBALL_ROOT', None),
            'configs',
            f"{conf['run']['setup']}.json"
        )

        self.environment = Pinball(**conf['env'])
        obs, _, _ = self.environment.obs()
        self.raw_obs_shape = (obs.shape[0], obs.shape[1])
        self.start_position = conf['run']['start_position']
        self.actions = conf['run']['actions']
        self.n_actions = len(self.actions)

        encoder_type = conf['run']['encoder']
        layer_conf = conf['layer']

        if encoder_type == 'sp_ensemble':
            from hima.modules.htm.spatial_pooler import SPDecoder, SPEnsemble

            # assembly agent
            encoder_conf = conf['encoder']
            encoder_conf['seed'] = self.seed
            encoder_conf['inputDimensions'] = list(self.raw_obs_shape)

            encoder = SPEnsemble(**encoder_conf)
            decoder = SPDecoder(encoder)

            layer_conf['n_obs_states'] = encoder.sps[0].getNumColumns()
            layer_conf['n_obs_vars'] = encoder.n_sp
            layer_conf['n_context_states'] = (
                    encoder.sps[0].getNumColumns() * layer_conf['cells_per_column']
            )
            layer_conf['n_context_vars'] = encoder.n_sp
        elif encoder_type == 'sp_grouped':
            from hima.experiments.temporal_pooling.stp.sp_ensemble import (
                SpatialPoolerGroupedWrapper
            )
            # assembly agent
            encoder_conf = conf['encoder']
            encoder_conf['seed'] = self.seed
            encoder_conf['feedforward_sds'] = [self.raw_obs_shape, 0.1]
            decoder_type = conf['run'].get('decoder', None)

            encoder = SpatialPoolerGroupedWrapper(**encoder_conf)
            decoder = make_decoder(encoder, decoder_type)

            layer_conf['n_obs_states'] = encoder.getSingleNumColumns()
            layer_conf['n_obs_vars'] = encoder.n_groups
            layer_conf['n_context_states'] = (
                    encoder.getSingleNumColumns() * layer_conf['cells_per_column']
            )
            layer_conf['n_context_vars'] = encoder.n_groups
        else:
            raise ValueError(f'Encoder type {encoder_type} is not supported')

        layer_conf['n_external_vars'] = 1
        layer_conf['n_external_states'] = self.n_actions
        layer_conf['seed'] = self.seed

        layer = Layer(**layer_conf)

        cortical_column = CorticalColumn(
            layer,
            encoder,
            decoder
        )

        self.agent = BioHIMA(
            cortical_column,
            **conf['agent']
        )

        self.n_episodes = conf['run']['n_episodes']
        self.max_steps = conf['run']['max_steps']
        self.update_rate = conf['run']['update_rate']

        self.initial_previous_image = self._rng.random(self.raw_obs_shape)
        self.prev_image = self.initial_previous_image
        self.initial_context = np.zeros_like(
            self.agent.cortical_column.layer.context_messages
        )
        self.initial_context[
            np.arange(
                self.agent.cortical_column.layer.n_hidden_vars
            ) * self.agent.cortical_column.layer.n_hidden_states
        ] = 1

        if self.logger is not None:
            from metrics import ScalarMetrics, HeatmapMetrics, ImageMetrics
            # define metrics
            self.scalar_metrics = ScalarMetrics(
                {
                    'main_metrics/reward': np.sum,
                    'main_metrics/steps': np.mean,
                    'layer/surprise_hidden': np.mean,
                    'layer/n_segments': np.mean,
                    'layer/n_factors': np.mean,
                    'agent/td_error': np.mean
                },
                self.logger
            )

            self.heatmap_metrics = HeatmapMetrics(
                {
                    'agent/prior': np.mean,
                    'agent/striatum_weights': np.mean
                },
                self.logger
            )

            self.image_metrics = ImageMetrics(
                [
                    'agent/behavior',
                    'agent/sr',
                    'layer/factor_graph',
                    'layer/predictions'
                ],
                self.logger,
                log_fps=conf['run']['log_gif_fps']
            )

    def run(self):
        episode_print_schedule = 50

        for i in range(self.n_episodes):
            if i % episode_print_schedule == 0:
                print(f'Episode {i}')

            steps = 0
            running = True
            action = None

            self.prev_image = self.initial_previous_image
            self.environment.reset(self.start_position)
            self.agent.reset(self.initial_context, action)

            while running:
                self.environment.step()
                obs, reward, is_terminal = self.environment.obs()
                running = not is_terminal

                events = self.preprocess(obs)
                # observe events_t and action_{t-1}
                pred_sr, gen_sr = self.agent.observe(
                    (events, action), learn=True, raw_observation=self.prev_image
                )
                self.agent.reinforce(reward)

                if running:
                    if steps == 0:
                        # action = self._rng.integers(self.n_actions)
                        action = self.agent.sample_action()
                        # convert to AAI action
                        pinball_action = self.actions[action]
                        self.environment.act(pinball_action)

                # >>> logging
                if self.logger is not None:
                    self.scalar_metrics.update(
                        {
                            'main_metrics/reward': reward,
                            'layer/surprise_hidden': self.agent.surprise,
                            'layer/n_segments': self.agent.cortical_column.layer.
                            context_factors.connections.numSegments(),
                            'layer/n_factors': self.agent.cortical_column.layer.
                            context_factors.factor_connections.numSegments(),
                            'agent/td_error': self.agent.td_error
                        }
                    )

                    if (i % self.update_rate) == 0:
                        raw_beh = (self.prev_image * 255).astype('uint8')

                        proc_beh = np.zeros(self.raw_obs_shape).flatten()
                        proc_beh[events] = 1
                        proc_beh = (proc_beh.reshape(self.raw_obs_shape) * 255).astype('uint8')

                        pred_beh = (self.agent.cortical_column.predicted_image.reshape(
                            self.raw_obs_shape
                        ) * 255).astype('uint8')

                        if pred_sr is not None:
                            pred_sr = (
                                    self.agent.cortical_column.decoder.decode(pred_sr)
                                    .reshape(self.raw_obs_shape) * 255
                            ).astype('uint8')
                        else:
                            pred_sr = np.zeros(self.raw_obs_shape).astype('uint8')

                        if gen_sr is not None:
                            gen_sr = (
                                    self.agent.cortical_column.decoder.decode(gen_sr)
                                    .reshape(self.raw_obs_shape) * 255
                            ).astype('uint8')
                        else:
                            gen_sr = np.zeros(self.raw_obs_shape).astype('uint8')

                        self.image_metrics.update(
                            {
                                'agent/behavior': np.hstack(
                                    [raw_beh, proc_beh, pred_beh, pred_sr, gen_sr])
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

                if (i % self.update_rate) == 0:
                    prior_probs = self.agent.cortical_column.decoder.decode(
                        self.agent.observation_prior
                    ).reshape(self.raw_obs_shape)
                    self.heatmap_metrics.update(
                        {
                            'agent/prior': prior_probs,
                            'agent/striatum_weights': self.agent.striatum_weights
                        }
                    )
                    self.heatmap_metrics.log(i)

                    self.image_metrics.update(
                        {
                            'layer/factor_graph': Image.open(
                                io.BytesIO(
                                    self.agent.cortical_column.layer.draw_factor_graph()
                                )
                            )
                        }
                    )
                    self.image_metrics.log(i)
            # <<< logging
        else:
            self.environment.close()

    def preprocess(self, image):
        gray = np.dot(image[:, :, :3], [299 / 1000, 587 / 1000, 114 / 1000])

        diff = np.abs(gray - self.prev_image)

        self.prev_image = gray.copy()

        thresh = diff.mean()
        events = np.flatnonzero(diff > thresh)

        return events


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
    config['env'] = read_config(config['run']['env_conf'])
    config['agent'] = read_config(config['run']['agent_conf'])
    config['layer'] = read_config(config['run']['layer_conf'])
    config['encoder'] = read_config(config['run']['encoder_conf'])

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

    if config['run']['experiment'] == 'animalai':
        runner = AnimalAITest(logger, config)
    elif config['run']['experiment'] == 'pinball':
        runner = PinballTest(logger, config)
    else:
        raise ValueError(f'There is no such experiment {config["run"]["experiment"]}!')

    runner.run()


if __name__ == '__main__':
    default_config = 'configs/runner/animalai.yaml'
    main(os.environ.get('RUN_CONF', default_config))