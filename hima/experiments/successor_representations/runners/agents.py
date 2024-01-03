#  Copyright (c) 2023 Autonomous Non-Profit Organization "Artificial Intelligence Research
#  Institute" (AIRI); Moscow Institute of Physics and Technology (National Research University).
#  All rights reserved.
#
#  Licensed under the AGPLv3 license. See LICENSE in the project root for license information.
from __future__ import annotations

from hima.experiments.successor_representations.runners.base import BaseAgent
from hima.common.sdr import sparse_to_dense
from hima.agents.succesor_representations.agent import BioHIMA, LstmBioHima
from hima.modules.belief.cortial_column.cortical_column import CorticalColumn, Layer
from hima.modules.baselines.lstm import LstmLayer
from hima.modules.baselines.rwkv import RwkvLayer
from hima.modules.baselines.hmm import FCHMMLayer
from hima.modules.baselines.srtd import SRTD
from hima.modules.dvs import DVS
from hima.agents.q.agent import QAgent
from hima.agents.sr.table import SRAgent

import numpy as np
from typing import Literal


class BioAgentWrapper(BaseAgent):
    agent: BioHIMA | LstmBioHima
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
        elif self.layer_type == 'dhtm':
            layer_conf['n_context_states'] = (
                    n_obs_states * layer_conf['cells_per_column']
            )
            layer_conf['n_context_vars'] = n_obs_vars
            layer_conf['n_external_vars'] = 1
            layer = Layer(**layer_conf)
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
            encoder,
            decoder
        )

        if conf['srtd_type'] is None:
            srtd = None
        else:
            srtd = SRTD(
                cortical_column.layer.context_input_size,
                cortical_column.layer.input_sdr_size,
                **conf['srtd']
            )

        conf['agent']['seed'] = self.seed

        if self.layer_type in {'lstm', 'rwkv'}:
            self.agent = LstmBioHima(
                cortical_column,
                srtd=srtd,
                **conf['agent']
            )
        else:
            self.agent = BioHIMA(
                cortical_column,
                srtd=srtd,
                **conf['agent']
            )

        if self.layer_type == 'fchmm':
            self.initial_action = -1
            self.initial_context = np.empty(0)
            self.initial_external_message = np.empty(0)
        elif self.layer_type == 'dhtm':
            self.initial_action = None
            self.initial_context = sparse_to_dense(
                np.arange(
                    self.agent.cortical_column.layer.n_hidden_vars
                ) * self.agent.cortical_column.layer.n_hidden_states,
                like=self.agent.cortical_column.layer.context_messages
            )
            self.initial_external_message = None
        else:
            self.initial_action = None
            self.initial_context = self.agent.cortical_column.layer.context_messages
            self.initial_external_message = None

    def observe(self, obs, action):
        if self.camera is not None:
            self.events = self.camera.capture(obs)
        else:
            self.events = obs

        return self.agent.observe((self.events, action), learn=True)

    def sample_action(self):
        return self.agent.sample_action()

    def reinforce(self, reward):
        return self.agent.reinforce(reward)

    def reset(self):
        if self.camera is not None:
            self.camera.reset()

        return self.agent.reset(self.initial_context, self.initial_external_message)

    @property
    def state_value(self):
        action_values = self.agent.evaluate_actions(with_planning=True)
        state_value = np.sum(action_values)
        return state_value

    @property
    def action_values(self):
        return self.agent.evaluate_actions(with_planning=True)

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
            decoder_conf = self.conf['decoder']

            encoder = SpatialPoolerGroupedWrapper(**encoder_conf)
            decoder = self._make_decoder(encoder, decoder_type, decoder_conf)
            n_groups = encoder.n_groups
            n_states = encoder.getSingleNumColumns()
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
