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

import hima.envs.gridworld
from hima.common.config.base import read_config, override_config
from hima.common.run.argparse import parse_arg_list
from hima.common.sdr import sparse_to_dense
from hima.experiments.successor_representations.runners.base import BaseRunner
from hima.experiments.successor_representations.runners.utils import to_gray_image
from hima.modules.baselines.lstm import LstmLayer
from hima.modules.baselines.hmm import FCHMMLayer


class ICMLRunner(BaseRunner):
    def make_agent(self, agent_type, conf):
        if agent_type == 'bio':
            from hima.experiments.successor_representations.runners.agents\
                import BioAgentWrapper
            agent = BioAgentWrapper(conf)
        elif agent_type == 'data':
            from hima.experiments.successor_representations.runners.agents\
                import DatasetCreatorAgent
            agent = DatasetCreatorAgent(**conf)
        elif agent_type == 'ec':
            from hima.experiments.successor_representations.runners.agents\
                import ECAgentWrapper
            agent = ECAgentWrapper(conf)
            if hasattr(self.environment, 'get_true_matrices'):
                T, E = self.environment.get_true_matrices()
                agent.agent.true_transition_matrix, agent.agent.true_emission_matrix = T, E
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

    def switch_strategy(self, strategy):
        if strategy == 'random':
            self.reward_free = True
        elif strategy == 'non-random':
            self.reward_free = False

    def reset_buffer(self):
        layer = self.agent.agent.cortical_column.layer
        if isinstance(layer, LstmLayer):
            layer.trajectories.clear()
        elif isinstance(layer, FCHMMLayer):
            layer.reset_buffer()

    def reset_model(self, checkpoint_path=None):
        layer = self.agent.agent.cortical_column.layer
        if isinstance(layer, LstmLayer):
            layer.reset_model(checkpoint_path)
        elif isinstance(layer, FCHMMLayer):
            layer.reset_model()

    def save_model(self, dir_path):
        layer = self.agent.agent.cortical_column.layer
        if isinstance(layer, LstmLayer):
            layer.save_model(os.path.join(dir_path, f'{self.logger.name}_{self.episodes}.pt'))

    def save_agent(self, dir_path):
        if self.logger is not None:
            name = self.logger.name
        else:
            from names_generator import generate_name
            name = generate_name()

        with open(os.path.join(dir_path, f'agent_{name}.pkl'), 'wb') as file:
            pickle.dump(self.agent.agent, file)

    @property
    def real_reward(self):
        if self.agent.camera is not None:
            im = sparse_to_dense(self.agent.events, shape=self.environment.raw_obs_shape)
        else:
            im = to_gray_image(self.agent.events)
        real_reward = im * self.reward
        return real_reward

    @property
    def encoded_reward(self):
        im = self.encoder_output
        return im * self.reward

    @property
    def state(self):
        env = self.environment.environment
        assert isinstance(env, hima.envs.gridworld.GridWorld)
        r, c = env.r, env.c
        return r * env.w + c

    @property
    def state_visited(self):
        env = self.environment.environment
        assert isinstance(env, hima.envs.gridworld.GridWorld)
        r, c = env.r, env.c
        values = np.zeros((env.h, env.w))
        values[r, c] = 1

        return values, 1

    @property
    def state_value(self):
        env = self.environment.environment
        assert isinstance(env, hima.envs.gridworld.GridWorld)
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
        assert isinstance(env, hima.envs.gridworld.GridWorld)
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
        assert isinstance(env, hima.envs.gridworld.GridWorld)
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
        assert isinstance(env, hima.envs.gridworld.GridWorld)
        r, c = env.r, env.c
        values = np.zeros((env.h, env.w))
        state_prediction = self.agent.agent.second_level_none
        values[r, c] = state_prediction

        counts = np.zeros_like(values)
        counts[r, c] = 1
        return values, counts

    @property
    def state_representation(self):
        internal_messages = self.agent.agent.cortical_column.layer.internal_messages
        cells_per_column = self.agent.agent.cortical_column.layer.cells_per_column
        return internal_messages.reshape(-1, cells_per_column).T

    @property
    def q_value(self):
        env = self.environment.environment
        assert isinstance(env, hima.envs.gridworld.GridWorld)
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

    @property
    def raw_observation(self):
        return to_gray_image(self.obs)

    @property
    def camera_output(self):
        return sparse_to_dense(self.agent.events, shape=self.environment.raw_obs_shape)

    @property
    def encoder_output(self):
        return self.agent.agent.observation_messages.reshape(
            self.agent.agent.cortical_column.layer.n_obs_vars,
            -1
        )

    def save_encoder(self, path):
        with open(
            os.path.join(path,
             f'{self.logger.name}_{self.episodes}episodes_sp.pkl'),
            'wb'
        ) as file:
            pickle.dump(
            {
                    'encoder': self.agent.agent.cortical_column.encoder,
                    'camera': self.agent.camera,
                    'decoder': self.agent.agent.cortical_column.decoder
                },
                file=file
            )

    def load_encoder_state(self, file_path):
        with open(file_path, 'rb') as file:
            state_dict = pickle.load(file)

        self.agent.agent.cortical_column.encoder = state_dict['encoder']
        self.agent.agent.cortical_column.decoder = state_dict['decoder']
        self.agent.camera = state_dict['camera']

    def save_buffer(self, path):
        layer = self.agent.agent.cortical_column.layer
        cells_per_column = layer.cells_per_column

        if len(layer.observation_messages_buffer) == 0:
            return

        obs_buffer = np.array(layer.observation_messages_buffer)
        ext_buffer = np.array(layer.external_messages_buffer)
        fwd_buffer = np.array(layer.forward_messages_buffer)
        bwd_buffer = np.array(layer.backward_messages_buffer)
        prior_buffer = np.array([layer.forward_messages_buffer[0]] + layer.prediction_buffer)
        trajectory = np.array(
            [np.zeros_like(self.environment.trajectory[0])] + self.environment.trajectory
        )

        fwd_buffer = np.transpose(
            fwd_buffer.reshape(fwd_buffer.shape[0], -1, cells_per_column),
            (0, 2, 1)
        )
        bwd_buffer = np.transpose(
            bwd_buffer.reshape(bwd_buffer.shape[0], -1, cells_per_column),
            (0, 2, 1)
        )
        prior_buffer = np.transpose(
            prior_buffer.reshape(prior_buffer.shape[0], -1, cells_per_column),
            (0, 2, 1)
        )
        ext_buffer = ext_buffer.reshape(ext_buffer.shape[0], 1, -1)
        obs_buffer = obs_buffer.reshape(obs_buffer.shape[0], 1, -1)

        if self.logger is not None:
            run_name = self.logger.name
        else:
            run_name = str(self.seed)

        for name, array in zip(
                ['ext', 'obs', 'fwd', 'bwd', 'prior', 'traj'],
                [ext_buffer, obs_buffer, fwd_buffer, bwd_buffer, prior_buffer, trajectory]
        ):
            path_name = os.path.join(
                path,
                f'{run_name}_{self.episodes}_{name}.npy'
            )
            np.save(path_name, array)


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
        config['run']['seed'] = int.from_bytes(os.urandom(8), 'big')

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

    if config['run'].pop('log'):
        import wandb
        logger = wandb.init(
            project=config['run'].pop('project_name'), entity=os.environ.get('WANDB_ENTITY', None),
            config=config
        )
    else:
        logger = None

    runner = ICMLRunner(logger, config)
    runner.run()


if __name__ == '__main__':
    default_config = 'configs/runner/pinball.yaml'
    main(os.environ.get('RUN_CONF', default_config))
