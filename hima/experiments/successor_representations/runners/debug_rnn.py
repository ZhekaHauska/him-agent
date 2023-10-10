#  Copyright (c) 2023 Autonomous Non-Profit Organization "Artificial Intelligence Research
#  Institute" (AIRI); Moscow Institute of Physics and Technology (National Research University).
#  All rights reserved.
#
#  Licensed under the AGPLv3 license. See LICENSE in the project root for license information.
import os
import pickle
import sys
from pathlib import Path

import numpy as np
from tqdm import trange

from hima.common.config.base import read_config, override_config
from hima.common.lazy_imports import lazy_import
from hima.common.run.argparse import parse_arg_list
from hima.common.sdr import sparse_to_dense
from hima.common.utils import to_gray_img, isnone
from hima.experiments.successor_representations.runners.utils import print_digest
from hima.experiments.temporal_pooling.stp.sp_ensemble import SpatialPoolerGroupedWrapper

wandb = lazy_import('wandb')


class CachedRunData:
    def __init__(self, encoder):
        self.images = []
        self.raw_observations = []
        self.observations = []
        self.predicted_observations = []
        self.actions = []
        self.rewards = []
        self.dones = []

        self.encoder = encoder


class PinballDebugTest:
    def __init__(self, logger, conf):
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

        self.env_conf = conf['env']
        self.raw_obs_shape = (50, 36)
        self.start_position = conf['run']['start_position']
        self.pinball_actions = conf['run']['actions']
        self.n_actions = len(self.pinball_actions)

        encoder_type = conf['run']['encoder']
        encoder_conf = conf['encoder']
        assert encoder_type == 'sp_grouped'
        encoder_conf['seed'] = self.seed
        encoder_conf['feedforward_sds'] = [self.raw_obs_shape, 0.1]

        self.use_cache = conf['run']['use_cache']
        self.cache_dir = Path(conf['run']['cache_dir']).expanduser().resolve()
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache_data_path = self.cache_dir / 'data.pkl'
        if self.use_cache and self.cache_data_path.exists():
            with self.cache_data_path.open('rb') as _f:
                self.data = pickle.load(_f)
            self.encoder = self.data.encoder
        else:
            self.encoder = SpatialPoolerGroupedWrapper(**encoder_conf)
            self.data = CachedRunData(self.encoder)

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

        self.n_encoder_updates = conf['run']['n_encoder_updates']
        self.n_rnn_updates = conf['run']['n_rnn_updates']

        self.initial_previous_image = np.zeros(self.raw_obs_shape)
        self.prev_image = self.initial_previous_image
        self.initial_context = self.layer.context_messages

        from metrics import ScalarMetrics
        self.scalar_metrics = ScalarMetrics(
            {
                'main_metrics/steps': np.mean,
                'layer/loss': np.mean,
            },
            self.logger
        )

    def reset(self):
        self.layer.reset()
        self.layer.set_context_messages(self.initial_context)
        self.layer.set_external_messages(None)

    def observe(self, obs, action, learn):
        # predict current local input step
        external_messages = None
        if action is not None:
            external_messages = sparse_to_dense([action], size=self.layer.external_input_size)

        self.layer.set_external_messages(external_messages)
        self.layer.predict(learn=learn)

        # observe real outcome and optionally learn using prediction error
        predicted_obs = self.layer.prediction_columns

        self.layer.observe(obs, learn=learn)
        self.layer.set_context_messages(self.layer.internal_forward_messages)
        return predicted_obs

    def run(self):
        self.collect_trajectories()
        self.train_encoder()
        self.encode_observations()

        print_schedule = 5000
        n_steps = self.n_rnn_updates
        n_raw_obs = len(self.data.raw_observations)

        self.reset()
        action = None

        for step in range(n_steps):
            if step % print_schedule == 0:
                self.scalar_metrics.update({'main_metrics/steps': step})
                if self.logger is not None:
                    self.scalar_metrics.log(step)
                else:
                    print_digest(self.scalar_metrics.summarize())
                    self.scalar_metrics.reset()

            i = step % n_raw_obs
            obs = self.data.observations[i]
            done = self.data.dones[i]

            # observe events_t and action_{t-1}
            predicted_obs = self.observe(obs, action, learn=True)

            action = self.data.actions[i]
            if done:
                action = None
                self.reset()

            self.scalar_metrics.update({'layer/loss': self.layer.last_loss_value})

    def collect_trajectories(self):
        if len(self.data.raw_observations) > 0:
            return

        print('==> Collect trajectories')
        from pinball import Pinball
        environment = Pinball(**self.env_conf)

        episode_print_schedule = 50

        for episode in range(self.n_episodes):
            if episode % episode_print_schedule == 0:
                print(f'Episode {episode}')

            steps = 0
            running = True

            self.prev_image = self.initial_previous_image
            environment.reset(self.start_position)
            self.reset()

            while running and steps < self.max_steps:
                environment.step()
                image, reward, is_terminal = environment.obs()
                raw_obs = self.preprocess(image)

                self.data.images.append(self.prev_image)
                self.data.raw_observations.append(raw_obs)
                self.data.rewards.append(reward)
                self.data.dones.append(is_terminal)

                action = self._rng.choice(self.n_actions)
                self.data.actions.append(action)
                if not is_terminal:
                    environment.act(self.pinball_actions[action])
                steps += 1

        environment.close()
        self.save_data()
        print('<== Collect trajectories')

    def train_encoder(self):
        if self.encoder.newborn_pruning_stage != 0:
            return

        print('==> Learn encoder')
        n_observations = len(self.data.raw_observations)

        for _ in trange(self.n_encoder_updates):
            ind = self._rng.choice(n_observations)
            raw_obs = self.data.raw_observations[ind]
            self.encoder.compute(raw_obs, learn=True)

        self.save_data()
        print('<== Learn encoder')

    def encode_observations(self):
        if len(self.data.observations) > 0:
            return

        print('==> Encode observations')
        n_observations = len(self.data.raw_observations)

        for ind in trange(n_observations):
            raw_obs = self.data.raw_observations[ind]
            obs = self.encoder.compute(raw_obs, learn=False)
            self.data.observations.append(obs)

        self.save_data()
        print('<== Encode observations')

    def save_data(self):
        with self.cache_data_path.open('wb') as _f:
            pickle.dump(self.data, _f)
            print('Pickled!')

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
