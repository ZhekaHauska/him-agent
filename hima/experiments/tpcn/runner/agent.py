from hima.agents.tpcn.agent import tPCNAgent
from hima.experiments.successor_representations.runners.base import BaseAgent
import numpy as np
from hima.agents.eprop_rnn.agent import RNNWithEPropAgent

class TPCNWrapper(BaseAgent):

    def __init__(self, conf):
        
        self.conf = conf
        self.encoder_type = conf['encoder_type']

        self.initial_action = 0
        self.logger = None
        self.n_actions = conf['agent']['n_actions']
        self.n_obs_states = conf['agent']['n_obs_states']
        
        self.agent = tPCNAgent(**self.conf['agent'])
        
        self.encoder = self._make_encoder()

    def observe(self, obs, action, reward=0):
        encoded_action = np.zeros(self.n_actions)
        encoded_action[action] = 1
        return self.agent.observe(
            self.encoder.encode(obs), 
            encoded_action,
            reward
        )
    
    def sample_action(self):
        return self.agent.sample_action()

    def reinforce(self, reward):
        return self.agent.reinforce(reward)
    
    def reset(self):
        return self.agent.reset()

    
    @property
    def state_value(self):
        action_values = self.agent.action_values
        if action_values is None:
            action_values = self.agent.evaluate_actions()
        state_value = np.sum(action_values)
        return state_value

    @property
    def action_values(self):
        action_values = self.agent.action_values
        if action_values is None:
            action_values = self.agent.evaluate_actions()
        return action_values
    
    @property
    def state_repr(self):
        return self.agent.prev_hidden.detach().numpy().squueze()

    def _make_encoder(self):
            if self.encoder_type == 'sp_grouped':
                from hima.modules.belief.cortial_column.encoders.sp import (
                    SpatialPooler
                )

                encoder_conf = self.conf['encoder']
                encoder_conf['seed'] = self.seed
                encoder_conf['feedforward_sds'] = [self.n_obs_states, 0.1]

                encoder = SpatialPooler(encoder_conf)
            elif self.encoder_type == 'pcn':
                from encoders import HierarchicalPCN
                encoder_conf = self.conf['encoder']
                encoder_conf['n_obs_states'] = self.n_obs_states
                encoder_conf['hidden_size'] = self.conf['agent']['hidden_size']
                encoder = HierarchicalPCN(**encoder_conf)
            elif self.encoder_type == 'one-hot':
                from encoders import SimpleOneHotEncoder
                encoder = SimpleOneHotEncoder(max_categories=self.n_obs_states)
            elif self.encoder_type == 'vae':
                from hima.modules.belief.cortial_column.encoders.vae import CatVAE
                encoder = CatVAE(**self.conf['encoder'])
            elif self.encoder_type == 'kmeans':
                from hima.modules.belief.cortial_column.encoders.kmeans import KMeansEncoder
                encoder = KMeansEncoder(**self.conf['encoder'])
            elif self.encoder_type is None:
                encoder = None
            else:
                raise ValueError(f'Encoder type {self.encoder_type} is not supported')
            
            return encoder
    
class RNNWithEPropWrapper(BaseAgent):
    def __init__(self, conf):
        
        self.conf = conf
        self.encoder_type = conf['encoder_type']
        

        self.initial_action = 0
        self.logger = None
        self.n_actions = conf['agent']['n_actions']
        self.n_obs_states = conf['agent']['n_obs_states']
        self.agent = RNNWithEPropAgent(**self.conf['agent'])

        self.encoder = self._make_encoder()

    def observe(self, obs, action, reward=0):
        encoded_action = np.zeros(self.n_actions)
        encoded_action[action] = 1
        return self.agent.observe(
            self.encoder.encode(obs), 
            action, 
            reward
        )
    
    def sample_action(self):
        return self.agent.sample_action()

    def reinforce(self, reward):
        return self.agent.reinforce(reward)
    
    def reset(self):
        return self.agent.reset()

    
    @property
    def state_value(self):
        action_values = self.agent.action_values
        if action_values is None:
            action_values = self.agent.evaluate_actions()
        state_value = np.sum(action_values)
        return state_value

    @property
    def action_values(self):
        action_values = self.agent.action_values
        if action_values is None:
            action_values = self.agent.evaluate_actions()
        return action_values
    
    @property
    def state_repr(self):
        return self.agent.prev_hidden.squueze()

    def _make_encoder(self):
        
            if self.encoder_type == 'sp_grouped':
                from hima.modules.belief.cortial_column.encoders.sp import (
                    SpatialPooler
                )

                encoder_conf = self.conf['encoder']
                encoder_conf['seed'] = self.seed
                encoder_conf['feedforward_sds'] = [self.n_obs_states, 0.1]

                encoder = SpatialPooler(encoder_conf)
            elif self.encoder_type == 'pcn':
                from encoders import HierarchicalPCN
                encoder_conf = self.conf['encoder']
                encoder_conf['n_obs_states'] = self.n_obs_states
                encoder_conf['hidden_size'] = self.conf['agent']['hidden_size']
                encoder = HierarchicalPCN(**encoder_conf)
            elif self.encoder_type == 'onehot':
                from encoders import SimpleOneHotEncoder
                encoder = SimpleOneHotEncoder(max_categories=self.n_obs_states)
            elif self.encoder_type == 'vae':
                from hima.modules.belief.cortial_column.encoders.vae import CatVAE
                encoder = CatVAE(**self.conf['encoder'])
            elif self.encoder_type == 'kmeans':
                from hima.modules.belief.cortial_column.encoders.kmeans import KMeansEncoder
                encoder = KMeansEncoder(**self.conf['encoder'])
            elif self.encoder_type is None:
                encoder = None
            else:
                raise ValueError(f'Encoder type {self.encoder_type} is not supported')
            
            return encoder
    
