#  Copyright (c) 2023 Autonomous Non-Profit Organization "Artificial Intelligence Research
#  Institute" (AIRI); Moscow Institute of Physics and Technology (National Research University).
#  All rights reserved.
#
#  Licensed under the AGPLv3 license. See LICENSE in the project root for license information.
import io

from animalai.envs.environment import AnimalAIEnvironment
from animalai.envs.actions import AAIActions
from hima.agents.succesor_representations.agent import BioHIMA
from hima.modules.belief.cortial_column.cortical_column import CorticalColumn, Layer
from hima.modules.htm.spatial_pooler import SPEnsemble, SPDecoder
from metrics import ScalarMetrics, HeatmapMetrics, ImageMetrics
from PIL import Image

import wandb
import numpy as np
import yaml
import sys
import os
import ast


class AnimalAITest:
    def __init__(self, logger, conf):
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
        self.actions = AAIActions().allActions
        self.n_actions = len(self.actions)

        # assembly agent
        encoder_conf = conf['encoder']
        encoder_conf['seed'] = self.seed
        encoder_conf['inputDimensions'] = list(self.raw_obs_shape)

        encoder = SPEnsemble(**encoder_conf)

        decoder = SPDecoder(encoder)

        layer_conf = conf['layer']
        layer_conf['n_obs_states'] = encoder.sps[0].getNumColumns()
        layer_conf['n_obs_vars'] = encoder.n_sp
        layer_conf['n_context_states'] = (
                encoder.sps[0].getNumColumns() * layer_conf['cells_per_column']
        )
        layer_conf['n_context_vars'] = encoder.n_sp
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

        self.prev_image = np.zeros(self.raw_obs_shape)
        self.initial_context = np.zeros_like(
            self.agent.cortical_column.layer.context_messages
        )
        self.initial_context[
            np.arange(
                self.agent.cortical_column.layer.n_hidden_vars
            ) * self.agent.cortical_column.layer.n_hidden_states
        ] = 1

        # define metrics
        self.scalar_metrics = ScalarMetrics(
            {
                'main_metrics/reward': np.sum,
                'main_metrics/steps': np.mean,
                'layer/surprise_hidden': np.mean,
                'layer/n_segments': np.mean,
                'layer/n_factors': np.mean
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
        for i in range(self.n_episodes):
            steps = 0
            running = True
            action = None

            self.prev_image = self._rng.random(self.raw_obs_shape)
            self.environment.reset()
            self.agent.reset(self.initial_context, action)

            steps = 0
            running = True
            while running:
                self.environment.step()
                dec, term = self.environment.get_steps(self.behavior)

                if len(dec) > 0:
                    obs = self.environment.get_obs_dict(dec.obs)["camera"]
                    events = self.preprocess(obs)

                reward = 0
                if len(dec.reward) > 0:
                    reward = dec.reward
                if len(term.reward) > 0:
                    reward += term.reward
                    running = False

                self.agent.reinforce(np.clip(reward, 0, None))

                pred_sr = None
                gen_sr = None
                if running:
                    action = self.agent.sample_action()
                    pred_sr, gen_sr = self.agent.observe((events, action), learn=True)

                    # convert to AAI action
                    action = self.actions[action]
                    self.environment.set_actions(self.behavior, action.action_tuple)

                if self.logger is not None:
                    self.scalar_metrics.update(
                        {
                            'main_metrics/reward': reward,
                            'layer/surprise_hidden': self.agent.surprise,
                            'layer/n_segments': self.agent.cortical_column.layer.
                            context_factors.connections.numSegments(),
                            'layer/n_factors': self.agent.cortical_column.layer.
                            context_factors.factor_connections.numSegments()
                        }
                    )

                steps += 1

                if steps >= self.max_steps:
                    running = False

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
        else:
            self.environment.close()

    def preprocess(self, image):
        gray = np.dot(image, [299 / 1000, 587 / 1000, 114 / 1000])

        diff = np.abs(gray - self.prev_image)

        self.prev_image = gray.copy()

        thresh = diff.mean()
        events = np.flatnonzero(diff > thresh)

        return events


def main(config_path):
    if len(sys.argv) > 1:
        config_path = sys.argv[1]

    config = dict()

    with open(config_path, 'r') as file:
        config['run'] = yaml.load(file, Loader=yaml.Loader)

    with open(config['run']['env_conf'], 'r') as file:
        config['env'] = yaml.load(file, Loader=yaml.Loader)
    with open(config['run']['agent_conf'], 'r') as file:
        config['agent'] = yaml.load(file, Loader=yaml.Loader)
    with open(config['run']['layer_conf'], 'r') as file:
        config['layer'] = yaml.load(file, Loader=yaml.Loader)
    with open(config['run']['encoder_conf'], 'r') as file:
        config['encoder'] = yaml.load(file, Loader=yaml.Loader)

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

    runner = AnimalAITest(logger, config)
    runner.run()


if __name__ == '__main__':
    default_config = 'configs/runner/animalai.yaml'
    main(os.environ.get('RUN_CONF', default_config))
