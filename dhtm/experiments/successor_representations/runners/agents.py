#  Copyright (c) 2023 Autonomous Non-Profit Organization "Artificial Intelligence Research
#  Institute" (AIRI); Moscow Institute of Physics and Technology (National Research University).
#  All rights reserved.
#
#  Licensed under the AGPLv3 license. See LICENSE in the project root for license information.
from __future__ import annotations

from dhtm.experiments.successor_representations.runners.base import BaseAgent
from dhtm.common.sdr import sparse_to_dense
from dhtm.agents.succesor_representations.agent import BioHIMA, LstmBioHima
from dhtm.modules.belief.cortial_column.cortical_column import CorticalColumn
from dhtm.modules.belief.dhtm import BioDHTM, DHTM
from dhtm.modules.baselines.hmm import FCHMMLayer
from dhtm.modules.baselines.lstm import LstmLayer
from dhtm.modules.baselines.rwkv import RwkvLayer
from dhtm.modules.dvs import DVS
from dhtm.agents.episodic_control.agent import ECAgent
import os

import numpy as np
from typing import Literal


class DatasetCreatorAgent(BaseAgent):
    camera: DVS | None

    def __init__(
            self,
            dataset_path,
            raw_obs_shape,
            reward_boost: int = 0,
            camera_mode=None,
            seed=None,
            **_
    ):
        self.initial_action = None
        self.state_value = 0
        self.reward_boost = reward_boost
        self.episodes = 0

        self.seed = seed
        self.dataset_path = dataset_path
        try:
            os.mkdir(self.dataset_path)
        except OSError as error:
            print(error)

        self.raw_obs_shape = raw_obs_shape
        self.camera_mode = camera_mode

        if self.camera_mode is not None:
            self.camera = DVS(self.raw_obs_shape, self.camera_mode, self.seed)
        else:
            self.camera = None

        self.images = []
        self.rewards = []
        self.actions = []
        self.chunk_id = 0

    def observe(self, events, action, reward=0):
        if self.camera is not None:
            events = self.camera.capture(events)
            im = sparse_to_dense(events, shape=self.raw_obs_shape)
        else:
            im = events

        self.images.append(im.copy())
        self.rewards.append(reward)
        self.actions.append(action)
        if reward > 0:
            for _ in range(self.reward_boost):
                self.images.append(im.copy())
                self.rewards.append(reward)
                self.actions.append(action)

    def sample_action(self):
        return None

    def reinforce(self, reward):
        return None

    def reset(self):
        self.episodes += 1
        return None

    def save(self):
        self.actions.append(0)
        np.savez(
            os.path.join(
                self.dataset_path,
                f'data{self.chunk_id}'
            ),
            s=np.array(self.images),
            a=np.array(self.actions[1:]),
            r=np.array(self.rewards)
        )
        self.chunk_id += 1
        self.images.clear()
        self.rewards.clear()
        self.actions.clear()

    def print_digest(self):
        print(f'{self.episodes=}')
        print(f'{self.dataset_path=}')
        print(f'{self.raw_obs_shape=}; {self.camera_mode=}')
        print(f'{self.chunk_id=}; images: {len(self.images)}')


class BioAgentWrapper(BaseAgent):
    agent: BioHIMA
    camera: DVS | None
    layer_type: Literal['fchmm', 'dhtm', 'biodhtm', 'lstm', 'rwkv']
    encoder_type: Literal['sp_grouped', 'vae', 'kmeans']

    def __init__(self, conf):
        """
        config structure:
            agent:
                ...
            layer_type
            layer:
                ...
            encoder_type
            encoder:
                ...
        """
        self.conf = conf
        self.layer_type = conf['layer_type']
        self.encoder_type = conf['encoder_type']
        self.seed = conf['seed']
        self.camera_mode = conf['camera_mode']
        self.events = None
        self.steps = 0

        if self.camera_mode is not None:
            self.camera = DVS(conf['raw_obs_shape'], self.camera_mode, self.seed)
        else:
            self.camera = None

        encoder, n_obs_vars, n_obs_states = self._make_encoder()

        layer_conf = self.conf['layer']
        layer_conf['n_obs_vars'] = n_obs_vars
        layer_conf['n_obs_states'] = n_obs_states
        layer_conf['n_external_states'] = conf['n_actions']
        layer_conf['seed'] = self.seed

        if self.layer_type == 'fchmm':
            layer = FCHMMLayer(**layer_conf)
        elif self.layer_type in {'dhtm', 'biodhtm'}:
            layer_conf['n_context_states'] = (
                    n_obs_states * layer_conf['cells_per_column']
            )
            layer_conf['n_context_vars'] = n_obs_vars * layer_conf['n_hidden_vars_per_obs_var']
            layer_conf['n_external_vars'] = 1
            if self.layer_type == 'dhtm':
                layer = DHTM(**layer_conf)
            else:
                layer = BioDHTM(**layer_conf)
        elif self.layer_type == 'lstm':
            layer_conf['n_external_vars'] = 1
            layer = LstmLayer(**layer_conf)
        elif self.layer_type == 'rwkv':
            layer_conf['n_external_vars'] = 1
            layer = RwkvLayer(**layer_conf)
        else:
            raise NotImplementedError

        cortical_column = CorticalColumn(
            layer,
            encoder
        )

        conf['agent']['seed'] = self.seed

        if self.layer_type in {'lstm', 'rwkv'}:
            self.agent = LstmBioHima(cortical_column, **conf['agent'])
        else:
            self.agent = BioHIMA(
                cortical_column,
                **conf['agent']
            )

        if self.layer_type == 'fchmm':
            self.initial_action = -1
        elif self.layer_type in {'dhtm', 'biodhtm'}:
            self.initial_action = 0
        else:
            self.initial_action = None

    def observe(self, obs, action, reward=0):
        self.steps += 1

        if self.camera is not None:
            self.events = self.camera.capture(obs)
        else:
            self.events = obs

        return self.agent.observe((self.events, action), reward, learn=True)

    def sample_action(self):
        return self.agent.sample_action()

    def reinforce(self, reward):
        return self.agent.reinforce(reward)

    def reset(self):
        self.steps = 0

        if self.camera is not None:
            self.camera.reset()

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
        return self.agent.cortical_column.layer.internal_messages.reshape(
            self.agent.cortical_column.layer.n_hidden_vars,
            -1
        )

    @property
    def num_segments_forward(self):
        if hasattr(self.agent.cortical_column.layer, 'forward_factors'):
            return self.agent.cortical_column.layer.forward_factors.connections.numSegments()
        elif hasattr(self.agent.cortical_column.layer, 'context_factors'):
            return self.agent.cortical_column.layer.context_factors.connections.numSegments()
        else:
            return 0

    @property
    def num_segments_backward(self):
        if hasattr(self.agent.cortical_column.layer, 'backward_factors'):
            return self.agent.cortical_column.layer.backward_factors.connections.numSegments()
        else:
            return 0

    def _make_encoder(self):
        if self.encoder_type == 'sp_grouped':
            from dhtm.modules.belief.cortial_column.encoders.sp import (
                SpatialPooler
            )

            encoder_conf = self.conf['encoder']
            encoder_conf['seed'] = self.seed
            encoder_conf['feedforward_sds'] = [self.conf['raw_obs_shape'], 0.1]

            encoder = SpatialPooler(encoder_conf)
            n_vars = encoder.n_vars
            n_states = encoder.n_states
        elif self.encoder_type == 'vae':
            from dhtm.modules.belief.cortial_column.encoders.vae import CatVAE
            encoder = CatVAE(**self.conf['encoder'])
            n_vars = encoder.n_vars
            n_states = encoder.n_states
        elif self.encoder_type == 'kmeans':
            from dhtm.modules.belief.cortial_column.encoders.kmeans import KMeansEncoder
            encoder = KMeansEncoder(**self.conf['encoder'])
            n_vars = encoder.n_vars
            n_states = encoder.n_states
        elif self.encoder_type is None:
            encoder = None
            n_vars, n_states = self.conf['raw_obs_shape']
        else:
            raise ValueError(f'Encoder type {self.encoder_type} is not supported')

        return encoder, n_vars, n_states


class ECAgentWrapper(BaseAgent):
    agent: ECAgent

    def __init__(self, conf):
        self.seed = conf['seed']
        self.initial_action = 0
        self.conf = conf
        self.encoder_type = conf['encoder_type']

        if self.encoder_type is not None:
            self.encoder, n_obs_vars, n_obs_states = self._make_encoder()
            assert n_obs_vars == 1
            conf['agent']['n_obs_states'] = n_obs_states
        else:
            self.encoder = None
            raw_obs_shape = conf['raw_obs_shape']
            assert raw_obs_shape[0] == 1
            conf['agent']['n_obs_states'] = raw_obs_shape[1]

        conf['agent']['seed'] = conf['seed']
        conf['agent']['n_actions'] = conf['n_actions']

        self.agent = ECAgent(**conf['agent'])

    def observe(self, events, action, reward=0):
        if self.encoder is not None:
            events = self.encoder.encode(events, learn=True)

        self.agent.observe((events, action), reward)

    def sample_action(self):
        return self.agent.sample_action()

    def reinforce(self, reward):
        self.agent.reinforce(reward)

    def reset(self):
        self.agent.reset()

    @property
    def state_value(self):
        action_values = self.agent.action_values
        if action_values is None:
            action_values = self.agent.evaluate_actions()
        state_value = np.sum(action_values)
        return state_value

    @property
    def goal_found(self):
        return float(self.agent.goal_found)

    def _make_encoder(self):
        if self.encoder_type == 'sp_grouped':
            from dhtm.modules.belief.cortial_column.encoders.sp import (
                SpatialPooler
            )

            encoder_conf = self.conf['encoder']
            encoder_conf['seed'] = self.seed
            encoder_conf['feedforward_sds'] = [self.conf['raw_obs_shape'], 0.1]

            encoder = SpatialPooler(encoder_conf)
            n_vars = encoder.n_vars
            n_states = encoder.n_states
        elif self.encoder_type == 'vae':
            from dhtm.modules.belief.cortial_column.encoders.vae import CatVAE
            encoder = CatVAE(**self.conf['encoder'])
            n_vars = encoder.n_vars
            n_states = encoder.n_states
        elif self.encoder_type == 'kmeans':
            from dhtm.modules.belief.cortial_column.encoders.kmeans import KMeansEncoder
            encoder = KMeansEncoder(**self.conf['encoder'])
            n_vars = encoder.n_vars
            n_states = encoder.n_states
        elif self.encoder_type is None:
            encoder = None
            n_vars, n_states = self.conf['raw_obs_shape']
        else:
            raise ValueError(f'Encoder type {self.encoder_type} is not supported')

        return encoder, n_vars, n_states
