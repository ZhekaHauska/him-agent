#  Copyright (c) 2023 Autonomous Non-Profit Organization "Artificial Intelligence Research
#  Institute" (AIRI); Moscow Institute of Physics and Technology (National Research University).
#  All rights reserved.
#
#  Licensed under the AGPLv3 license. See LICENSE in the project root for license information.
import pickle
import sys
import os
from typing import Union, Any

import numpy as np

import epm.envs.gridworld
from epm.common.config.base import read_config, override_config
from epm.common.metrics import WandbLogger, AimLogger
from epm.common.utils import parse_arg_list
from epm.experiments.successor_representations.runners.base import BaseRunner


class ICMLRunner(BaseRunner):
    def make_agent(self, agent_type, conf):
        if agent_type == 'ec':
            from epm.experiments.successor_representations.runners.agents\
                import ECAgentWrapper
            agent = ECAgentWrapper(conf, logger=self.logger)
            if hasattr(self.environment, 'get_true_matrices'):
                T, E = self.environment.get_true_matrices()
                agent.agent.true_transition_matrix, agent.agent.true_emission_matrix = T, E
        else:
            raise NotImplementedError

        return agent

    @staticmethod
    def make_environment(env_type, conf, setup):
        if env_type == 'gridworld':
            from epm.experiments.successor_representations.runners.envs import GridWorldWrapper
            env = GridWorldWrapper(conf, setup)
        else:
            raise NotImplementedError
        return env

    def switch_strategy(self, strategy):
        if strategy == 'random':
            self.reward_free = True
        elif strategy == 'non-random':
            self.reward_free = False

    @property
    def encoded_reward(self):
        im = self.encoder_output
        return im * self.reward

    @property
    def state(self):
        env = self.environment.environment
        assert isinstance(env, epm.envs.gridworld.GridWorld)
        r, c = env.r, env.c
        return r * env.w + c

    @property
    def state_visited(self):
        env = self.environment.environment
        assert isinstance(env, epm.envs.gridworld.GridWorld)
        r, c = env.r, env.c
        values = np.zeros((env.h, env.w))
        values[r, c] = 1

        return values, 1

    @property
    def state_value(self):
        env = self.environment.environment
        assert isinstance(env, epm.envs.gridworld.GridWorld)
        r, c = env.r, env.c
        values = np.zeros((env.h, env.w))
        state_value = self.agent.state_value
        values[r, c] = state_value

        counts = np.zeros_like(values)
        counts[r, c] = 1
        return values, counts

    @property
    def state_size(self):
        env = self.environment.environment
        assert isinstance(env, epm.envs.gridworld.GridWorld)
        r, c = env.r, env.c
        values = np.zeros((env.h, env.w))
        state_size = len(self.agent.agent.cluster)
        values[r, c] = state_size

        counts = np.zeros_like(values)
        counts[r, c] = 1
        return values, counts

    @property
    def state_error(self):
        env = self.environment.environment
        assert isinstance(env, epm.envs.gridworld.GridWorld)
        r, c = env.r, env.c
        values = np.zeros((env.h, env.w))
        state_error = self.agent.agent.second_level_error
        values[r, c] = state_error

        counts = np.zeros_like(values)
        counts[r, c] = 1
        return values, counts

    @property
    def state_prediction(self):
        env = self.environment.environment
        assert isinstance(env, epm.envs.gridworld.GridWorld)
        r, c = env.r, env.c
        values = np.zeros((env.h, env.w))
        state_prediction = self.agent.agent.second_level_none
        values[r, c] = state_prediction

        counts = np.zeros_like(values)
        counts[r, c] = 1
        return values, counts

    @property
    def q_value(self):
        env = self.environment.environment
        assert isinstance(env, epm.envs.gridworld.GridWorld)
        # left, right, up, down
        actions = self.environment.actions
        shifts = np.array([[0, 0], [0, env.w], [env.h, 0], [env.h, env.w]])

        r, c = env.r, env.c
        values = np.zeros((env.h * 2, env.w * 2))
        action_values = self.agent.action_values
        counts = np.zeros_like(values)

        for value, shift in zip(action_values, shifts):
            x, y = r + shift[0], c + shift[1]
            values[x, y] = value
            counts[x, y] = 1

        return values, counts

    @property
    def rewards(self):
        agent = self.agent.agent
        return agent.rewards.reshape(1, -1)


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

    metrics_conf = config['run'].pop('metrics_conf')
    if metrics_conf is not None:
        config['metrics'] = read_config(metrics_conf)

    if config['run']['seed'] is None:
        config['run']['seed'] = int.from_bytes(os.urandom(4), 'big')

    # unfolding subconfigs
    def load_subconfig(entity, conf):
        if f'{entity}_conf' in conf['agent']:
            conf_path = config['agent'].pop(f'{entity}_conf')
            conf['agent'][f'{entity}_type'] = conf_path.split('/')[-2]
            conf['agent'][entity] = read_config(conf_path)
        else:
            conf['agent'][f'{entity}_type'] = None

    load_subconfig('layer', config)
    load_subconfig('encoder', config)

    # override some values
    overrides = parse_arg_list(sys.argv[2:])
    override_config(config, overrides)

    logger = config['run'].pop('logger')
    if logger is not None:
        if logger == 'wandb':
            logger = WandbLogger(config)
        elif logger == 'aim':
            logger = AimLogger(config)
        else:
            raise NotImplementedError

    runner = ICMLRunner(logger, config)
    runner.run()


if __name__ == '__main__':
    default_config = 'configs/runner/pinball.yaml'
    main(os.environ.get('RUN_CONF', default_config))
