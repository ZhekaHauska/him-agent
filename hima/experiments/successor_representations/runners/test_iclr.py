#  Copyright (c) 2023 Autonomous Non-Profit Organization "Artificial Intelligence Research
#  Institute" (AIRI); Moscow Institute of Physics and Technology (National Research University).
#  All rights reserved.
#
#  Licensed under the AGPLv3 license. See LICENSE in the project root for license information.
import sys
import os
from typing import Union, Any

import numpy as np
from hima.common.config.base import read_config, override_config
from hima.common.run.argparse import parse_arg_list
from hima.experiments.successor_representations.runners.base import BaseRunner
from hima.modules.belief.utils import normalize


class ICLRunner(BaseRunner):
    @staticmethod
    def make_agent(agent_type, conf):
        from hima.experiments.successor_representations.runners.agents import BioAgentWrapper
        if agent_type == 'bio':
            agent = BioAgentWrapper(conf)
        else:
            raise NotImplementedError

        return agent

    @staticmethod
    def make_environment(env_type, conf, setup):
        if env_type == 'pinball':
            from hima.experiments.successor_representations.runners.envs import PinballWrapper
            env = PinballWrapper(conf, setup)
        elif env_type == 'animalai':
            from hima.experiments.successor_representations.runners.envs import AnimalAIWrapper
            env = AnimalAIWrapper(conf, setup)
        elif env_type == 'gridworld':
            from hima.experiments.successor_representations.runners.envs import GridWorldWrapper
            env = GridWorldWrapper(conf, setup)
        else:
            raise NotImplementedError
        return env

    @property
    def obs_reward(self):
        agent = self.agent.agent
        decoder = agent.cortical_column.decoder
        if decoder is not None:
            obs_rewards = decoder.decode(
                normalize(
                    agent.observation_rewards.reshape(
                        agent.cortical_column.layer.n_obs_vars, -1
                    )
                ).flatten()
            ).reshape(self.environment.raw_obs_shape)
        else:
            obs_rewards = agent.observation_rewards.reshape(
                agent.cortical_column.layer.n_obs_vars, -1
            )
        return obs_rewards


def main(config_path):
    if len(sys.argv) > 1:
        config_path = sys.argv[1]

    config: dict[str, Union[Union[dict[str, Any], list[Any]], Any]] = dict()

    # main part
    config['run'] = read_config(config_path)

    env_conf_path = config['run'].pop('env_conf')
    config['env_type'] = env_conf_path.split('/')[-2]
    config['env'] = read_config(env_conf_path)

    agent_conf_path = config['run'].pop('agent_conf')
    config['agent_type'] = agent_conf_path.split('/')[-2]
    config['agent'] = read_config(agent_conf_path)

    config['metrics'] = read_config(config['run'].pop('metrics_conf'))
    if config['run']['seed'] is None:
        config['run']['seed'] = np.random.randint(0, np.iinfo(np.int32).max)

    # unfolding subconfigs
    layer_conf_path = config['agent'].pop('layer_conf')
    config['agent']['layer_type'] = layer_conf_path.split('/')[-2]
    config['agent']['layer'] = read_config(layer_conf_path)

    if 'encoder_conf' in config['agent']:
        encoder_conf_path = config['agent'].pop('encoder_conf')
        config['agent']['encoder_type'] = encoder_conf_path.split('/')[-2]
        config['agent']['encoder'] = read_config(encoder_conf_path)
    else:
        config['agent']['encoder_type'] = None

    if 'decoder_conf' in config['agent']:
        decoder_conf_path = config['agent'].pop('decoder_conf')
        config['agent']['decoder_type'] = decoder_conf_path.split('/')[-2]
        config['agent']['decoder'] = read_config(decoder_conf_path)
    else:
        config['agent']['decoder_type'] = None

    # override some values
    overrides = parse_arg_list(sys.argv[2:])
    override_config(config, overrides)

    if config['run'].pop('log'):
        import wandb
        logger = wandb.init(
            project=config['run'].pop('project_name'), entity=os.environ.get('WANDB_ENTITY', None),
            config=config
        )
    else:
        logger = None

    runner = ICLRunner(logger, config)
    runner.run()


if __name__ == '__main__':
    default_config = 'configs/runner/pinball.yaml'
    main(os.environ.get('RUN_CONF', default_config))
