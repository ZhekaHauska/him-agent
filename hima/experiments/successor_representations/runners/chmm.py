#  Copyright (c) 2023 Autonomous Non-Profit Organization "Artificial Intelligence Research
#  Institute" (AIRI); Moscow Institute of Physics and Technology (National Research University).
#  All rights reserved.
#
#  Licensed under the AGPLv3 license. See LICENSE in the project root for license information.

import os
import sys

import numpy as np

from hima.agents.succesor_representations.agent import BioHIMA
from hima.common.config.base import read_config, override_config
from hima.common.lazy_imports import lazy_import
from hima.common.run.argparse import parse_arg_list
from hima.modules.belief.cortial_column.cortical_column import CorticalColumn
from hima.modules.belief.utils import normalize
from hima.modules.baselines.hmm import CHMMLayer
from hima.experiments.successor_representations.runners.utils import make_decoder
from typing import Literal

wandb = lazy_import('wandb')


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

        self.agent = self.load_agent(conf, conf['run'].get('agent_path', None))

        self.n_episodes = conf['run']['n_episodes']
        self.max_steps = conf['run']['max_steps']
        self.update_rate = conf['run']['update_rate']
        self.camera_mode = conf['run']['camera_mode']
        self.reward_free = conf['run'].get('reward_free', False)
        self.test_srs = conf['run'].get('test_srs', False)
        self.test_sr_steps = conf['run'].get('test_sr_steps', 0)

        self.initial_previous_image = self._rng.random(self.raw_obs_shape)
        self.prev_image = self.initial_previous_image
        self.initial_context = np.empty(0)

        if self.logger is not None:
            from metrics import ScalarMetrics, HeatmapMetrics, ImageMetrics
            # define metrics
            self.scalar_metrics = ScalarMetrics(
                {
                    'main_metrics/reward': np.sum,
                    'main_metrics/steps': np.mean,
                    'layer/surprise_hidden': np.mean,
                    'agent/td_error': np.mean,
                    'agent/test_sr_mse_approx_tail': np.mean,
                    'agent/test_sr_mse': np.mean
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
                    'agent/sr',
                    'layer/predictions'
                ],
                self.logger,
                log_fps=conf['run']['log_gif_fps']
            )

    def run(self):
        total_reward = np.zeros(self.raw_obs_shape).flatten()
        for i in range(self.n_episodes):
            steps = 0
            running = True
            action = -1

            self.prev_image = self.initial_previous_image
            self.environment.reset(self.start_position)
            self.agent.reset(self.initial_context, np.empty(0))

            while running:
                self.environment.step()
                obs, reward, is_terminal = self.environment.obs()
                running = not is_terminal

                events = self.preprocess(obs, mode=self.camera_mode)
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
                else:
                    # additional update for model-free TD
                    if self.agent.sr_steps == 0:
                        self.agent.observe((None, None), learn=True)

                # >>> logging
                if self.logger is not None:
                    self.scalar_metrics.update(
                        {
                            'main_metrics/reward': reward,
                            'layer/surprise_hidden': self.agent.surprise,
                            'agent/td_error': self.agent.td_error
                        }
                    )
                    if self.test_srs:
                        sr_mse_approx_tail, _, gen_sr_test_tail = self.compare_srs(
                            self.test_sr_steps,
                            True
                        )
                        sr_mse, pred_sr_test, gen_sr_test = self.compare_srs(
                            self.test_sr_steps,
                            False
                        )
                        self.scalar_metrics.update(
                            {
                                'agent/test_sr_mse_approx_tail': sr_mse_approx_tail,
                                'agent/test_sr_mse': sr_mse
                            }
                        )
                    else:
                        pred_sr_test = None
                        gen_sr_test = None
                        gen_sr_test_tail = None

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
                                    self.agent.cortical_column.decoder.decode(
                                        normalize(
                                            pred_sr.reshape(
                                                self.agent.cortical_column.layer.n_obs_vars, -1
                                            )
                                        ).flatten()
                                    )
                                    .reshape(self.raw_obs_shape) * 255
                            ).astype('uint8')
                        else:
                            pred_sr = np.zeros(self.raw_obs_shape).astype('uint8')

                        if gen_sr is not None:
                            gen_sr = (
                                    self.agent.cortical_column.decoder.decode(
                                        normalize(
                                            gen_sr.reshape(
                                                self.agent.cortical_column.layer.n_obs_vars, -1
                                            )
                                        ).flatten()
                                    )
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

                        if self.test_srs:
                            self.image_metrics.update(
                                {
                                    'agent/sr': np.hstack(
                                        [
                                            raw_beh,
                                            proc_beh,
                                            (pred_sr_test.reshape(
                                                self.raw_obs_shape
                                            ) * 255).astype('uint8'),
                                            (gen_sr_test.reshape(
                                                self.raw_obs_shape
                                            ) * 255).astype('uint8'),
                                            (gen_sr_test_tail.reshape(
                                                self.raw_obs_shape
                                            ) * 255).astype('uint8')
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

                if (i % self.update_rate) == 0:
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
                    self.heatmap_metrics.log(i)
                    self.image_metrics.log(i)
            # <<< logging
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

    def load_agent(self, conf=None, path=None):
        if path is not None:
            raise NotImplementedError
        elif conf is not None:
            # assembly agent
            encoder_type = conf['run']['encoder']
            encoder_conf = conf['encoder']
            layer_conf = conf['layer']

            if encoder_type == 'sp_ensemble':
                from hima.modules.htm.spatial_pooler import SPDecoder, SPEnsemble

                encoder_conf['seed'] = self.seed
                encoder_conf['inputDimensions'] = list(self.raw_obs_shape)

                encoder = SPEnsemble(**encoder_conf)
                decoder = SPDecoder(encoder)
                assert encoder.n_sp == 1

                layer_conf['n_obs_states'] = encoder.sps[0].getNumColumns()
                layer_conf['n_context_states'] = (
                        encoder.sps[0].getNumColumns() * layer_conf['cells_per_column']
                )
            elif encoder_type == 'sp_grouped':
                from hima.experiments.temporal_pooling.stp.sp_ensemble import (
                    SpatialPoolerGroupedWrapper
                )
                encoder_conf['seed'] = self.seed
                encoder_conf['feedforward_sds'] = [self.raw_obs_shape, 0.1]

                decoder_type = conf['run'].get('decoder', None)
                decoder_conf = conf['decoder']

                encoder = SpatialPoolerGroupedWrapper(**encoder_conf)
                assert encoder.n_groups == 1

                decoder = make_decoder(encoder, decoder_type, decoder_conf)

                layer_conf['n_obs_states'] = encoder.getSingleNumColumns()
                layer_conf['n_context_states'] = (
                        encoder.getSingleNumColumns() * layer_conf['cells_per_column']
                )
            else:
                raise ValueError(f'Encoder type {encoder_type} is not supported')

            layer_conf['n_external_states'] = self.n_actions
            layer_conf['seed'] = self.seed

            layer = CHMMLayer(**layer_conf)

            cortical_column = CorticalColumn(
                layer,
                encoder,
                decoder
            )

            conf['agent']['seed'] = self.seed
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
        gen_sr = self.agent.generate_sr(
            sr_steps,
            initial_messages=current_state,
            initial_prediction=self.agent.observation_messages,
            approximate_tail=approximate_tail,
        )

        pred_sr = self.agent.cortical_column.decoder.decode(
            normalize(
                pred_sr.reshape(
                    self.agent.cortical_column.layer.n_obs_vars, -1
                )
            ).flatten()
        )
        gen_sr = self.agent.cortical_column.decoder.decode(
            normalize(
                gen_sr.reshape(
                    self.agent.cortical_column.layer.n_obs_vars, -1
                )
            ).flatten()
        )

        mse = np.mean(np.power(pred_sr - gen_sr, 2))

        return mse, pred_sr, gen_sr


class GridWorldTest:
    def __init__(self, logger, conf):
        from hima.envs.gridworld import GridWorld

        self.logger = logger
        self.seed = conf['run'].get('seed')
        self._rng = np.random.default_rng(self.seed)

        env_conf = conf['env']
        self.environment = GridWorld(
            room=np.array(env_conf['room']),
            default_reward=env_conf['default_reward'],
            seed=self.seed
        )

        self.start_position = conf['run']['start_position']
        self.actions = conf['run']['actions']
        self.n_actions = len(self.actions)

        # assembly agent
        layer_conf = conf['layer']
        layer_conf['n_external_states'] = self.n_actions
        layer_conf['n_obs_states'] = np.max(self.environment.colors) + 1
        layer_conf['n_context_states'] = (
                layer_conf['n_obs_states'] * layer_conf['cells_per_column']
        )
        layer_conf['seed'] = self.seed

        layer = CHMMLayer(**layer_conf)

        cortical_column = CorticalColumn(
            layer,
            encoder=None,
            decoder=None
        )

        self.agent = BioHIMA(
            cortical_column,
            **conf['agent']
        )

        self.n_episodes = conf['run']['n_episodes']
        self.max_steps = conf['run']['max_steps']
        self.update_rate = conf['run']['update_rate']

        self.initial_context = np.empty(0)

        if self.logger is not None:
            from metrics import ScalarMetrics, HeatmapMetrics, ImageMetrics
            # define metrics
            self.scalar_metrics = ScalarMetrics(
                {
                    'main_metrics/reward': np.sum,
                    'main_metrics/steps': np.mean,
                    'layer/surprise_hidden': np.mean,
                    'agent/td_error': np.mean
                },
                self.logger
            )

            self.heatmap_metrics = HeatmapMetrics(
                {
                    'agent/striatum_weights': np.mean
                },
                self.logger
            )

            self.image_metrics = ImageMetrics(
                [
                    'agent/behavior'
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
            action = -1

            self.environment.reset(*self.start_position)
            self.agent.reset(self.initial_context, np.empty(0))

            while running:
                self.environment.step()
                obs, reward, is_terminal = self.environment.obs()
                running = not is_terminal

                events = [obs]
                # observe events_t and action_{t-1}
                pred_sr, gen_sr = self.agent.observe((events, action), learn=True)
                self.agent.reinforce(reward)

                if running:
                    # action = self._rng.integers(self.n_actions)
                    action = self.agent.sample_action()
                    # convert to AAI action
                    pinball_action = self.actions[action]
                    self.environment.act(pinball_action)

                # >>> logging
                if self.logger is not None:
                    if steps > 0:
                        self.scalar_metrics.update(
                            {
                                'main_metrics/reward': reward,
                                'layer/surprise_hidden': self.agent.surprise,
                                'agent/td_error': self.agent.td_error
                            }
                        )

                    if (i % self.update_rate) == 0:
                        raw_beh = self.environment.colors.astype(np.float64)
                        agent_color = self.agent.cortical_column.layer.n_obs_states + 1
                        raw_beh[self.environment.r, self.environment.c] = agent_color
                        raw_beh += 1
                        raw_beh /= (agent_color + 1)

                        raw_beh = (raw_beh * 255).astype('uint8')

                        self.image_metrics.update(
                            {
                                'agent/behavior': raw_beh
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
                    self.heatmap_metrics.update(
                        {
                            'agent/striatum_weights': self.agent.striatum_weights
                        }
                    )
                    self.heatmap_metrics.log(i)
                    self.image_metrics.log(i)
            # <<< logging


def main(config_path):
    if len(sys.argv) > 1:
        config_path = sys.argv[1]

    config = dict()

    config['run'] = read_config(config_path)
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
        import wandb
        logger = wandb.init(
            project=config['run']['project_name'], entity=os.environ.get('WANDB_ENTITY', None),
            config=config
        )
    else:
        logger = None

    if config['run']['experiment'] == 'pinball':
        runner = PinballTest(logger, config)
    elif config['run']['experiment'] == 'gridworld':
        runner = GridWorldTest(logger, config)
    else:
        raise ValueError(f'There is no such experiment {config["run"]["experiment"]}!')

    runner.run()


if __name__ == '__main__':
    default_config = 'configs/runner/pinball.yaml'
    main(os.environ.get('RUN_CONF', default_config))
