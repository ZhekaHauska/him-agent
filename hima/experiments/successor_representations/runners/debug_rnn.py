#  Copyright (c) 2023 Autonomous Non-Profit Organization "Artificial Intelligence Research
#  Institute" (AIRI); Moscow Institute of Physics and Technology (National Research University).
#  All rights reserved.
#
#  Licensed under the AGPLv3 license. See LICENSE in the project root for license information.
import os
import sys

import numpy as np

from hima.common.config.base import read_config, override_config
from hima.common.lazy_imports import lazy_import
from hima.common.run.argparse import parse_arg_list
from hima.common.sdr import sparse_to_dense
from hima.common.utils import to_gray_img, isnone
from hima.experiments.successor_representations.runners.lstm import LstmBioHima
from hima.experiments.successor_representations.runners.utils import print_digest, make_decoder
from hima.experiments.temporal_pooling.stp.sp_ensemble import SpatialPoolerGroupedWrapper
from hima.modules.belief.cortial_column.cortical_column import CorticalColumn
from hima.modules.belief.utils import normalize

wandb = lazy_import('wandb')


class PinballDebugTest:
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

        # assembly agent
        if 'encoder' in conf:
            encoder_type = conf['run']['encoder']
            encoder_conf = conf['encoder']
        else:
            encoder_type = None
            encoder_conf = None

        assert encoder_type == 'sp_grouped'
        encoder_conf['seed'] = self.seed
        encoder_conf['feedforward_sds'] = [self.raw_obs_shape, 0.1]
        decoder_type = conf['run'].get('decoder', None)

        self.encoder = SpatialPoolerGroupedWrapper(**encoder_conf)
        self.decoder = make_decoder(self.encoder, decoder_type, conf['decoder'])

        layer_conf = conf['layer']
        layer_conf['n_obs_vars'] = self.encoder.n_groups
        layer_conf['n_obs_states'] = self.encoder.getSingleNumColumns()
        layer_conf['n_external_vars'] = 1
        layer_conf['n_external_states'] = self.n_actions
        layer_conf['seed'] = self.seed

        layer = conf['run']['layer']
        if layer == 'lstm':
            from hima.modules.baselines.lstm import LstmLayer
            self.layer = LstmLayer(**layer_conf)
        elif layer == 'rwkv':
            from hima.modules.baselines.rwkv import RwkvLayer
            self.layer = RwkvLayer(**layer_conf)
        else:
            raise ValueError(f'Unsupported layer {layer}')

        self.n_episodes = conf['run']['n_episodes']
        self.max_steps = conf['run']['max_steps']
        self.update_rate = conf['run']['update_rate']

        self.initial_previous_image = np.zeros(self.raw_obs_shape)
        self.prev_image = self.initial_previous_image
        self.initial_context = layer.context_messages

        if self.logger is not None:
            from metrics import ScalarMetrics, HeatmapMetrics, ImageMetrics
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
            from metrics import ScalarMetrics
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

    def reset(self):
        self.layer.reset()
        self.layer.set_context_messages(self.initial_context)
        self.layer.set_external_messages(None)

    def observe(self, obs, action, learn):
        self.surprise = 0

        if obs is None:
            return

        # predict current events using observed action
        self.cortical_column.observe(events, action, learn=learn)

        # predict current local input step
        if action is not None:
            external_messages = np.zeros(self.layer.external_input_size)
            if action >= 0:
                external_messages[action] = 1
            else:
                external_messages = np.empty(0)
        else:
            external_messages = None

        self.layer.set_external_messages(external_messages)
        self.layer.predict(learn=learn)

        self.input_sdr.sparse = local_input

        if self.decoder is not None:
            self.predicted_image = self.decoder.decode(
                self.layer.prediction_columns, learn=learn, correct_obs=self.input_sdr.dense
            )
        else:
            self.predicted_image = self.layer.prediction_columns

        # observe real outcome and optionally learn using prediction error
        if self.encoder is not None:
            self.encoder.compute(self.input_sdr, learn, self.output_sdr)
        else:
            self.output_sdr.sparse = self.input_sdr.sparse

        self.layer.observe(self.output_sdr.sparse, learn=learn)
        self.layer.set_context_messages(self.layer.internal_forward_messages)


        encoded_obs = self.cortical_column.output_sdr.sparse

        if len(encoded_obs) > 0:
            self.surprise = get_surprise(
                self.cortical_column.layer.prediction_columns, encoded_obs, mode='categorical'
            )

        self.observation_messages = sparse_to_dense(encoded_obs, like=self.observation_messages)


    def run(self):
        self.raw_observations = []
        self.observations = []
        self.rewards = []

        episode_print_schedule = 50
        encoder, decoder = self.encoder, self.decoder

        for episode in range(self.n_episodes):
            if episode % episode_print_schedule == 0:
                print(f'Episode {episode}')

            steps = 0
            running = True
            action = None

            self.prev_image = self.initial_previous_image
            self.environment.reset(self.start_position)
            self.reset()

            while running:
                self.environment.step()
                obs, reward, is_terminal = self.environment.obs()
                events = self.preprocess(obs)

                self.raw_observations.append(self.prev_image)
                self.observations.append(events)
                self.rewards.append(reward)

                # observe events_t and action_{t-1}
                pred_sr, gen_sr = self.agent.observe((events, action), learn=True)
                self.agent.reinforce(reward)

                running = not is_terminal
                if running:
                    action = self._rng.choice(self.n_actions)
                    pinball_action = self.actions[action]
                    self.environment.act(pinball_action)

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
        gray = np.dot(image[:, :, :3], [299 / 1000, 587 / 1000, 114 / 1000])

        diff = np.abs(gray - self.prev_image)

        self.prev_image = gray.copy()

        thresh = diff.mean()
        events = np.flatnonzero(diff > thresh)

        return events


def main(config_path):
    if len(sys.argv) > 1:
        config_path = sys.argv[1]

    config = dict()

    config['run'] = read_config(config_path)
    runner = PinballDebugTest
    layer = config['run']['layer']
    config['run']['layer_conf'] = f'configs/{layer}/pinball.yaml'

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
        config['run']['seed'] = np.random.randint(0, 1_000_000)

    if config['run']['log']:
        import wandb
        logger = wandb.init(
            project=config['run']['project_name'], entity=os.environ.get('WANDB_ENTITY', None),
            config=config
        )
    else:
        logger = None

    runner = runner(logger, config)
    runner.run()


if __name__ == '__main__':
    default_config = 'configs/runner/debug_rnn.yaml'
    main(os.environ.get('RUN_CONF', default_config))
