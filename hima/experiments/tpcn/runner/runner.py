import os
import numpy as np
import pickle as pkl
import torch
import sys
from typing import Union, Any

from hima.envs.gridworld import GridWorld
from hima.experiments.successor_representations.runners.base import BaseRunner
from hima.common.config.base import read_config, override_config
from hima.common.run.argparse import parse_arg_list

class TPCNRunner(BaseRunner):
    
    @staticmethod
    def make_environment(env_type, conf, setup):
        if env_type == 'gridworld':
            from hima.experiments.successor_representations.runners.envs import GridWorldWrapper
            env = GridWorldWrapper(conf, setup)
        else:
            raise NotImplementedError
        
        return env
    
    @staticmethod
    def make_agent(agent_type, conf):
        if agent_type == 'tpcn':
            from hima.experiments.tpcn.runner.agent import TPCNWrapper
            agent = TPCNWrapper(conf)
        if agent_type == 'eprop_rnn':
            from hima.experiments.tpcn.runner.agent import RNNWithEPropWrapper
            agent = RNNWithEPropWrapper(conf)
        
        return agent

    def switch_strategy(self, strategy):
        if strategy == 'random':
            self.reward_free = True
        elif strategy == 'non-random':
            self.reward_free = False
    
    def save_agent(self, dir_path):
        if self.logger is not None:
            name = self.logger.name
        else:
            from names_generator import generate_name
            name = generate_name()

        with open(os.path.join(dir_path, f'agent_{name}.pkl'), 'wb') as file:
            pkl.dump(self.agent.agent, file)
        
    def save_model(self, dir_path):
        
        model = self.agent.agent.model
        
        if model.name == 'tpcn':
            torch.save(model, os.path.join(dir_path, f'{self.logger.name}_{self.episodes}.pt'))
        elif model.name == 'eprop_rnn':
            import pickle
            with open(os.path.join(dir_path, f'{self.logger.name}_{self.episodes}.pkl', 'wb')) as f:
                pickle.dump(model, f)
    
    @property
    def state(self):
        env = self.environment.environment
        assert isinstance(env, GridWorld)
        r, c = env.r, env.c
        return r * env.w + c
    
    @property
    def state_visited(self):
        env = self.environment.environment
        assert isinstance(env, GridWorld)
        r, c = env.r, env.c
        values = np.zeros((env.h, env.w))
        values[r, c] = 1

        return values, 1
    
    @property
    def state_value(self):
        env = self.environment.environment
        assert isinstance(env, GridWorld)
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
        assert isinstance(env, GridWorld)
        r, c = env.r, env.c
        values = np.zeros((env.h, env.w))
        state_size = len(self.agent.agent.cluster)
        values[r, c] = state_size

        counts = np.zeros_like(values)
        counts[r, c] = 1
        return values, counts
    
    @property
    def state_representation(self):
        internal_messages = self.agent.agent.get_state_representation()
        return internal_messages
    
    @property
    def q_value(self):
        env = self.environment.environment
        assert isinstance(env, GridWorld)
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
    
    def save_encoder(self, path):
        with open(
            os.path.join(path,
             f'{self.logger.name}_{self.episodes}episodes_sp.pkl'),
            'wb'
        ) as file:
            pkl.dump(self.agent.encoder, file=file)

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

    model_config = config['agent']['agent'].pop('model_config')
    if model_config is not None:
        config['agent']['agent']['model_config'] = read_config(model_config)
    
    if config['agent_type'] == 'tpcn':
        config['agent']['agent']['model_config']['n_obs_states'] = config['agent']['agent']['n_obs_states']
        config['agent']['agent']['model_config']['hidden_size'] = config['agent']['agent']['hidden_size']
        config['agent']['agent']['model_config']['n_actions'] = config['agent']['agent']['n_actions']
    
    if config['agent']['encoder_type'] == 'pcn':
        config['agent']['encoder'] = read_config(config['agent']['encoder'])

    if config['run']['seed'] is None:
        config['run']['seed'] = int.from_bytes(os.urandom(4), 'big')


    os.environ["GRIDWORLD_ROOT"] = "configs/environment/gridworld/"
    # override some values
    overrides = parse_arg_list(sys.argv[2:])
    override_config(config, overrides)

    logger = config['run'].pop('logger')
    if logger is not None:
        if logger == 'aim':
            # logger = AimLogger(config)
            pass
        else:
            raise NotImplementedError

    runner = TPCNRunner(logger, config)
    runner.run()


if __name__ == '__main__':
    default_config = 'configs/runner/eprop_rnn.yaml'
    main(os.environ.get('RUN_CONF', default_config))