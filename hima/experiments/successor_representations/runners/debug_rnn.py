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
from hima.experiments.successor_representations.runners.utils import print_digest
from hima.experiments.temporal_pooling.stp.sp_ensemble import SpatialPoolerGroupedWrapper

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

        encoder_type = conf['run']['encoder']
        encoder_conf = conf['encoder']
        assert encoder_type == 'sp_grouped'
        encoder_conf['seed'] = self.seed
        encoder_conf['feedforward_sds'] = [self.raw_obs_shape, 0.1]
        self.encoder = SpatialPoolerGroupedWrapper(**encoder_conf)

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
        self.initial_context = self.layer.context_messages

        self.images = []
        self.raw_observations = []
        self.observations = []
        self.predicted_observations = []
        self.rewards = []

        from metrics import ScalarMetrics
        self.scalar_metrics = ScalarMetrics(
            {
                'main_metrics/reward': np.sum,
                'main_metrics/steps': np.mean,
                'layer/loss': np.mean,
            },
            self.logger
        )

    def reset(self):
        self.layer.reset()
        self.layer.set_context_messages(self.initial_context)
        self.layer.set_external_messages(None)

    def observe(self, raw_obs, action, learn):
        if raw_obs is None:
            return

        # predict current local input step
        external_messages = None
        if action is not None:
            external_messages = sparse_to_dense([action], size=self.layer.external_input_size)

        self.layer.set_external_messages(external_messages)
        self.layer.predict(learn=learn)

        # observe real outcome and optionally learn using prediction error
        obs = self.encoder.compute(raw_obs, learn)
        predicted_obs = self.layer.prediction_columns

        self.layer.observe(obs, learn=learn)
        self.layer.set_context_messages(self.layer.internal_forward_messages)
        return obs, predicted_obs

    def run(self):
        episode_print_schedule = 50

        for episode in range(self.n_episodes):
            if episode % episode_print_schedule == 0:
                print(f'Episode {episode}')

            steps = 0
            running = True
            action = None

            self.prev_image = self.initial_previous_image
            self.environment.reset(self.start_position)
            self.reset()

            while running and steps < self.max_steps:
                self.environment.step()
                image, reward, is_terminal = self.environment.obs()
                raw_obs = self.preprocess(image)

                self.images.append(self.prev_image)
                self.raw_observations.append(raw_obs)
                self.rewards.append(reward)

                # observe events_t and action_{t-1}
                obs, predicted_obs = self.observe(raw_obs, action, learn=True)
                self.observations.append(obs)
                self.predicted_observations.append(predicted_obs)

                running = not is_terminal
                if running:
                    action = self._rng.choice(self.n_actions)
                    pinball_action = self.actions[action]
                    self.environment.act(pinball_action)

                # >>> logging
                # noinspection PyUnresolvedReferences
                self.scalar_metrics.update({
                    'main_metrics/reward': reward,
                    'layer/loss': self.layer.last_loss_value,
                })
                steps += 1

            # >>> logging
            self.scalar_metrics.update({'main_metrics/steps': steps})
            if self.logger is not None:
                self.scalar_metrics.log(episode)
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
