#  Copyright (c) 2023 Autonomous Non-Profit Organization "Artificial Intelligence Research
#  Institute" (AIRI); Moscow Institute of Physics and Technology (National Research University).
#  All rights reserved.
#
#  Licensed under the AGPLv3 license. See LICENSE in the project root for license information.
from animalai.envs.environment import AnimalAIEnvironment
from animalai.envs.actions import AAIActions
from hima.agents.succesor_representations.agent import BioHIMA
from hima.modules.belief.cortial_column.cortical_column import CorticalColumn, Layer
from hima.modules.htm.spatial_pooler import SPEnsemble, SPDecoder
from hima.experiments.hmm.runners.utils import get_surprise


import wandb
import numpy as np
import yaml
import sys
import os
import ast
import imageio


class ScalarMetrics:
    def __init__(self, metrics, logger):
        self.logger = logger
        self.metrics = {metric: [] for metric in metrics.keys()}
        self.agg_func = {metric: func for metric, func in metrics.items()}

    def update(self, metric_values):
        for key, value in metric_values.items():
            self.metrics[key].append(value)

    def log(self, step):
        average_metrics = {
            key: self.agg_func[key](values)
            for key, values in self.metrics.items()
            if len(values) > 0
        }

        self.logger.log(
            average_metrics,
            step=step
        )

        self.metrics = {metric: [] for metric in self.metrics.keys()}


class GifMetric:
    def __init__(self, name, logger, log_fps, log_dir='/tmp'):
        self.name = name
        self.logger = logger
        self.log_fps = log_fps
        self.log_dir = log_dir
        self.images = []

    def update(self, image):
        self.images.append(image)

    def log(self, step):
        with imageio.get_writer(
            os.path.join(self.log_dir, f'{self.logger.name}_{self.name}_{step}.gif'),
            mode='I',
            fps=self.log_fps
        ) as writer:
            for image in self.images:
                writer.append_data(image)

        self.logger.log(
            {
                self.name: wandb.Video(
                    os.path.join(self.log_dir, f'{self.logger.name}_{self.name}_{step}.gif')
                )
            },
            step=step
        )


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
        self.prev_image = np.zeros(self.raw_obs_shape)
        self.initial_context = np.zeros_like(
            self.agent.cortical_column.layer.context_messages
        )
        self.initial_context[
                np.arange(
                    self.agent.cortical_column.layer.n_hidden_vars
                ) * self.agent.cortical_column.layer.n_hidden_states
            ] = 1

        self.scalar_metrics = ScalarMetrics(
            {
                'main_metrics/reward': np.sum,
                'main_metrics/steps': np.mean,
                'main_metrics/surprise': np.mean
            },
            self.logger
        )

    def run(self):
        for i in range(self.n_episodes):
            self.environment.reset()
            # TODO do we need initial context? why not just prior?
            self.agent.reset(self.initial_context)

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

                self.agent.reinforce(reward)

                if running:
                    action = self.agent.sample_action()
                    self.agent.observe((events, action), learn=True)

                    # convert to AAI action
                    action = self.actions[action]
                    self.environment.set_actions(self.behavior, action.action_tuple)

                if self.logger is not None:
                    self.scalar_metrics.update({'main_metrics/reward': reward})

                steps += 1

                if steps >= self.max_steps:
                    running = False

            if self.logger is not None:
                self.scalar_metrics.update({'main_metrics/steps': steps})
                self.scalar_metrics.log(i)
        else:
            self.environment.close()

    def preprocess(self, image):
        gray = np.dot(image, [299/1000, 587/1000, 114/1000])

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
