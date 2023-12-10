#  Copyright (c) 2023 Autonomous Non-Profit Organization "Artificial Intelligence Research
#  Institute" (AIRI); Moscow Institute of Physics and Technology (National Research University).
#  All rights reserved.
#
#  Licensed under the AGPLv3 license. See LICENSE in the project root for license information.
from __future__ import annotations

from hima.experiments.successor_representations.runners.base import BaseAgent
from hima.common.sdr import sparse_to_dense
from hima.agents.succesor_representations.agent import BioHIMA, LstmBioHima, FCHMMBioHima
from hima.modules.belief.cortial_column.cortical_column import CorticalColumn, Layer
from hima.modules.baselines.lstm import LstmLayer
from hima.modules.baselines.rwkv import RwkvLayer
from hima.modules.baselines.hmm import FCHMMLayer
from hima.modules.dvs import DVS

import numpy as np
from typing import Literal


class BioAgentWrapper(BaseAgent):
    agent: BioHIMA | LstmBioHima | FCHMMBioHima
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

        conf['agent']['seed'] = self.seed

        if self.layer_type in {'lstm', 'rwkv'}:
            self.agent = LstmBioHima(cortical_column, **conf['agent'])
        elif self.layer_type == 'fchmm':
            self.agent = FCHMMBioHima(cortical_column, **conf['agent'])
        else:
            self.agent = BioHIMA(
                cortical_column,
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

