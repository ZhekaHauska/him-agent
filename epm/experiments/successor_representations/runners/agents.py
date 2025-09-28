#  Copyright (c) 2023 Autonomous Non-Profit Organization "Artificial Intelligence Research
#  Institute" (AIRI); Moscow Institute of Physics and Technology (National Research University).
#  All rights reserved.
#
#  Licensed under the AGPLv3 license. See LICENSE in the project root for license information.
from __future__ import annotations

import pickle

from epm.experiments.successor_representations.runners.base import BaseAgent
from epm.agents.episodic_control.agent import ECAgent
import os
import numpy as np


class ECAgentWrapper(BaseAgent):
    agent: ECAgent

    def __init__(self, conf, logger=None):
        self.seed = conf['seed']
        self.initial_action = 0
        self.conf = conf
        self.encoder_type = conf['encoder_type']
        self.learn = True

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

        self.agent = ECAgent(**conf['agent'], logger=logger)

    def observe(self, events, action, reward=0):
        if self.encoder is not None:
            events = self.encoder.encode(events, learn=True)

        self.agent.observe((events, action), reward, learn=self.learn)

    def sample_action(self):
        return self.agent.sample_action()

    def reinforce(self, reward):
        self.agent.reinforce(reward)

    def reset(self):
        self.agent.reset()

    @property
    def true_state(self):
        if hasattr(self.agent, 'true_state'):
            return self.agent.true_state
        else:
            return None

    @true_state.setter
    def true_state(self, value):
        if hasattr(self.agent, 'true_state'):
            self.agent.true_state = value

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
        if self.encoder_type is None:
            encoder = None
            n_vars, n_states = self.conf['raw_obs_shape']
        else:
            raise ValueError(f'Encoder type {self.encoder_type} is not supported')

        return encoder, n_vars, n_states

    def save_experience(self, path, prefix):
        self.agent._update_second_level()
        self.agent._update_third_level(
            self.agent.third_level_mode,
            self.agent.clamp_transitions,
            self.agent.clamp_labels,
            self.agent.normalise_labels,
        )

        label_to_obs = dict()
        for label in range(self.agent.true_emission_matrix.shape[0]):
            label_to_obs[label] = np.flatnonzero(self.agent.true_emission_matrix[label])[0]

        with open(os.path.join(path, prefix + 'agent_experience.pkl'), 'wb') as file:
            pickle.dump(
                {
                    'first_level': self.agent.first_level_transitions,
                    'state_labels': self.agent.state_labels,
                    'label_to_obs': label_to_obs,
                    'true_transition': self.agent.true_transition_matrix,
                    'true_emission': self.agent.true_emission_matrix,
                    'third_level': self.agent.third_level_transitions
                },
                file
            )
