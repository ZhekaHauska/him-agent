#  Copyright (c) 2023 Autonomous Non-Profit Organization "Artificial Intelligence Research
#  Institute" (AIRI); Moscow Institute of Physics and Technology (National Research University).
#  All rights reserved.
#
#  Licensed under the AGPLv3 license. See LICENSE in the project root for license information.
from __future__ import annotations

from hima.experiments.successor_representations.runners.base import BaseAgent
from hima.common.sdr import sparse_to_dense
from hima.agents.succesor_representations.agent import BioHIMA
from hima.modules.belief.cortial_column.cortical_column import CorticalColumn
from hima.modules.belief.dhtm import BioDHTM, DHTM
from hima.modules.baselines.hmm import FCHMMLayer
from hima.modules.dvs import DVS
from hima.agents.q.agent import QAgent
from hima.agents.sr.table import SRAgent
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
            **kwargs
    ):
        self.initial_action = None
        self.state_value = 0
        self.reward_boost = reward_boost

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
        self.chunk_id = 0

    def observe(self, events, action, reward=0):
        if self.camera is not None:
            events = self.camera.capture(events)
            im = sparse_to_dense(events, shape=self.raw_obs_shape)
        else:
            im = events

        self.images.append(im.copy())
        if reward > 0:
            for _ in range(self.reward_boost):
                self.images.append(im.copy())

    def sample_action(self):
        return None

    def reinforce(self, reward):
        return None

    def reset(self):
        return None

    def save(self):
        np.save(
            os.path.join(
                self.dataset_path,
                f'data{self.chunk_id}.npy'
            ),
            np.array(self.images)
        )
        self.chunk_id += 1
        self.images.clear()

    def print_digest(self):
        print(f'dataset path: {self.dataset_path}')
        print(f'image size: {self.raw_obs_shape}; camera mode: {self.camera_mode}')
        print(f'chunk: {self.chunk_id}; images: {len(self.images)}')


class BioAgentWrapper(BaseAgent):
    agent: BioHIMA
    camera: DVS | None
    layer_type: Literal['fchmm', 'dhtm', 'lstm', 'rwkv']
    encoder_type: Literal['sp_ensemble', 'sp_grouped']

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
            decoder_type
            decoder:
                ...
        """
        self.conf = conf
        self.layer_type = conf['layer_type']
        self.encoder_type = conf['encoder_type']
        self.seed = conf['seed']
        self.camera_mode = conf['camera_mode']
        self.events = None
        self.steps = 0
        self.reward_attention = conf.get('reward_attention', None)

        if self.camera_mode is not None:
            self.camera = DVS(conf['raw_obs_shape'], self.camera_mode, self.seed)
        else:
            self.camera = None

        encoder, n_obs_vars, n_obs_states, decoder = self._make_encoder()

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
        else:
            raise NotImplementedError

        cortical_column = CorticalColumn(
            layer,
            encoder,
            decoder
        )

        conf['agent']['seed'] = self.seed

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

        if self.reward_attention is not None:
            modulation = max(
                self.reward_attention * int(reward > 0), 1
            )
            self.agent.cortical_column.encoder.modulation = modulation

        return self.agent.observe((self.events, action), learn=True)

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
    def generated_sf(self):
        return self.agent.sf.reshape(
            self.agent.cortical_column.layer.n_obs_vars,
            -1
        )

    @property
    def planned_sf(self):
        sf, steps = self.agent.generate_sf(
            self.agent.max_plan_steps,
            initial_messages=self.agent.cortical_column.layer.internal_messages,
            initial_prediction=self.agent.observation_messages,
        )
        return (sf.reshape(
            self.agent.cortical_column.layer.n_obs_vars,
            -1
        ), steps / self.agent.max_plan_steps)

    @property
    def planned_sr(self):
        _, steps, sr = self.agent.generate_sf(
            self.agent.max_plan_steps,
            initial_messages=self.agent.cortical_column.layer.internal_messages,
            initial_prediction=self.agent.observation_messages,
            return_sr=True
        )
        return (sr.reshape(
            self.agent.cortical_column.layer.n_hidden_vars,
            -1
        ), steps / self.agent.max_plan_steps)

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
        if self.encoder_type == 'sp_ensemble':
            from hima.modules.htm.spatial_pooler import SPDecoder, SPEnsemble

            encoder_conf = self.conf['encoder']
            encoder_conf['seed'] = self.seed
            encoder_conf['inputDimensions'] = list(self.conf['raw_obs_shape'])

            encoder = SPEnsemble(**encoder_conf)
            decoder = SPDecoder(encoder)
            n_groups = encoder.n_groups
            n_states = encoder.getSingleNumColumns()
        elif self.encoder_type == 'sp_grouped':
            from hima.experiments.temporal_pooling.stp.sp_ensemble import (
                SpatialPoolerGroupedWrapper
            )

            encoder_conf = self.conf['encoder']
            encoder_conf['seed'] = self.seed
            encoder_conf['feedforward_sds'] = [self.conf['raw_obs_shape'], 0.1]

            decoder_type = self.conf['decoder_type']
            decoder_conf = self.conf.get('decoder', None)

            encoder = SpatialPoolerGroupedWrapper(**encoder_conf)
            decoder = self._make_decoder(encoder, decoder_type, decoder_conf)
            n_groups = encoder.n_groups
            n_states = encoder.getSingleNumColumns()
        elif self.encoder_type == 'vae':
            from hima.modules.vae import CatVAE
            encoder = CatVAE(**self.conf['encoder'])
            # TODO add decoder
            decoder = None
            n_groups = encoder.model.latent_dim
            n_states = encoder.model.categorical_dim
        elif self.encoder_type is None:
            encoder = None
            decoder = None
            n_groups, n_states = self.conf['raw_obs_shape']
        else:
            raise ValueError(f'Encoder type {self.encoder_type} is not supported')

        return encoder, n_groups, n_states, decoder

    @staticmethod
    def _make_decoder(encoder, decoder_type, decoder_conf):
        if decoder_type == 'naive':
            from hima.experiments.temporal_pooling.stp.sp_decoder import SpatialPoolerDecoder
            return SpatialPoolerDecoder(encoder)
        elif decoder_type == 'learned':
            from hima.experiments.temporal_pooling.stp.sp_decoder import SpatialPoolerLearnedDecoder
            return SpatialPoolerLearnedDecoder(encoder, **decoder_conf)
        elif decoder_type is None:
            return None
        else:
            raise ValueError(f'Decoder {decoder_type} is not supported')


class QTableAgentWrapper(BaseAgent):
    agent: QAgent

    def __init__(self, conf):
        self.seed = conf['seed']
        self.initial_action = None
        ucb_estimate_conf = conf['ucb_estimate']
        eligibility_traces_conf = conf['eligibility_traces']
        qvn_conf = conf['qvn']
        agent_conf = conf['agent']
        agent_conf['seed'] = self.seed
        raw_obs_shape = conf['raw_obs_shape']
        # TODO add SP encoder
        assert raw_obs_shape[0] == 1
        agent_conf['n_states'] = raw_obs_shape[1]
        agent_conf['n_actions'] = conf['n_actions']

        self.agent = QAgent(
            qvn=qvn_conf,
            eligibility_traces=eligibility_traces_conf,
            ucb_estimate=ucb_estimate_conf,
            **agent_conf
        )

        self.observation = None
        self.reward = None
        self.is_first = True

    def observe(self, events, action):
        self.observation = events

    def sample_action(self):
        action = self.agent.act(self.reward, self.observation, self.is_first)
        self.is_first = False
        return action

    def reinforce(self, reward):
        self.reward = reward

    def reset(self):
        self.is_first = True
        self.agent.on_new_episode()

    @property
    def state_value(self):
        actions_sa_sdr = self.agent._encode_s_actions(self.observation)
        action_values = self.agent.Q.values(actions_sa_sdr)
        return np.sum(action_values)

    @property
    def action_values(self):
        actions_sa_sdr = self.agent._encode_s_actions(self.observation)
        action_values = self.agent.Q.values(actions_sa_sdr)
        return action_values


class SRTableAgentWrapper(BaseAgent):
    agent: SRAgent

    def __init__(self, conf):
        self.seed = conf['seed']
        self.initial_action = None
        agent_conf = conf
        agent_conf['seed'] = self.seed
        raw_obs_shape = conf.pop('raw_obs_shape')
        assert raw_obs_shape[0] == 1
        agent_conf['n_states'] = raw_obs_shape[1]
        agent_conf['n_actions'] = conf['n_actions']

        self.agent = SRAgent(**agent_conf)
        self.current_state = None
        self.previous_state = None
        self.previous_action = None
        self.current_reward = None
        self.previous_reward = None

    def observe(self, events, action):
        self.previous_action = action
        self.previous_state = self.current_state
        self.current_state = events[0]

        if self.previous_state is not None:
            self.agent.observe(self.previous_state, self.previous_action)

    def sample_action(self):
        return self.agent.act(self.current_state)

    def reinforce(self, reward):
        self.previous_reward = self.current_reward
        self.current_reward = reward
        if self.previous_reward is not None:
            self.agent.reinforce(self.previous_reward)

    def reset(self):
        if self.current_state is not None:
            self.agent.observe(self.current_state, None)
            self.agent.reinforce(self.current_reward)

        self.current_state = None
        self.previous_state = None
        self.previous_action = None
        self.current_reward = None
        self.previous_reward = None

        self.agent.reset()

    @property
    def state_value(self):
        action_values = np.dot(
            self.agent.sr[:, self.current_state],
            self.agent.rewards
        )
        return np.sum(action_values)

    @property
    def action_values(self):
        action_values = np.dot(
            self.agent.sr[:, self.current_state],
            self.agent.rewards
        )
        return action_values

    @property
    def sr(self):
        return self.agent.sr
