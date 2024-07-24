#  Copyright (c) 2024 Autonomous Non-Profit Organization "Artificial Intelligence Research
#  Institute" (AIRI); Moscow Institute of Physics and Technology (National Research University).
#  All rights reserved.
#
#  Licensed under the AGPLv3 license. See LICENSE in the project root for license information.
import numpy as np


class ToyDHTM:
    """
        Simplified, fully deterministic DHTM
        for one hidden variable with visualizations.
        Stores transition matrix explicitly.
    """
    def __init__(
            self,
            n_obs_states,
            n_actions,
            n_clones,
            consolidation_threshold: int = 1  # controls noise tolerance?
    ):
        self.n_clones = n_clones
        self.n_obs_states = n_obs_states
        self.n_actions = n_actions
        self.n_hidden_states = self.n_clones * self.n_obs_states
        self.transition_counts = np.zeros(
            (self.n_actions, self.n_hidden_states, self.n_hidden_states)
        )
        self.activation_counts = np.zeros(self.n_hidden_states)
        # determines, how many counts we need to get for a transition to make it permanent
        self.consolidation_threshold = consolidation_threshold

        self.observation_buffer = list()
        self.action_buffer = list()
        self.state_buffer = list()

    def reset(self):
        self.clear_buffers()

    def clear_buffers(self):
        self.observation_buffer.clear()
        self.action_buffer.clear()
        self.state_buffer.clear()

    def observe(self, obs_state, action):
        self.observation_buffer.append(obs_state)
        self.action_buffer.append(action)
        # state to be defined
        self.state_buffer.append(None)

        step = len(self.observation_buffer) - 1

        if step == 0:
            # initial step
            column_states = np.arange(self.n_clones) + obs_state
            state = column_states[np.argmax(self.activation_counts[column_states])]
            self.state_buffer.append(state)
        else:
            pos = step
            resolved = False
            # resolve conflicts
            while not resolved:
                # input variables
                obs_state = self.observation_buffer[pos]
                column_states = np.arange(self.n_clones) + obs_state
                state = self.state_buffer[pos]

                prev_state = self.state_buffer[pos - 1]
                prev_action = self.action_buffer[pos - 1]
                prediction = self.transition_counts[prev_action, prev_state].flatten()
                sparse_prediction = np.flatnonzero(prediction)

                if state is None:
                    coincide = np.isin(sparse_prediction, column_states)
                else:
                    coincide = np.isin(sparse_prediction, state)

                correct_prediction = sparse_prediction[coincide]
                wrong_prediction = sparse_prediction[~coincide]

                permanence_mask = prediction[wrong_prediction] >= self.consolidation_threshold
                wrong_perm = wrong_prediction[
                    permanence_mask
                ]
                wrong_temp = wrong_prediction[
                    ~permanence_mask
                ]

                # cases:
                # 1. correct set is not empty
                if len(correct_prediction) > 0:
                    state = correct_prediction[
                        np.argmax(
                            prediction[correct_prediction] +
                            self.activation_counts[correct_prediction]
                        )
                    ]
                    resolved = True
                # 2. correct set is empty
                else:
                    if state is None:
                        state = column_states[np.argmax(self.activation_counts[column_states])]

                    if len(wrong_perm) == 0:
                        resolved = True
                    else:
                        # resampling previous clone
                        # try to use backward connections first
                        prediction = self.transition_counts[prev_action, :, state].flatten()
                        sparse_prediction = np.flatnonzero(prediction)

                        column_states = np.arange(self.n_clones) + self.observation_buffer[pos - 1]
                        coincide = np.isin(sparse_prediction, column_states)
                        correct_prediction = sparse_prediction[coincide]

                        if len(correct_prediction) > 0:
                            prev_state = correct_prediction[
                                np.argmax(
                                    prediction[correct_prediction] +
                                    self.activation_counts[correct_prediction]
                                )
                            ]
                        else:
                            # choose the least used clone
                            # (presumably with minimum outward connections)
                            prev_state = column_states[
                                np.argmin(
                                    self.activation_counts[column_states]
                                )
                            ]

                        self.state_buffer[pos - 1] = prev_state

                        # move to previous position
                        pos -= 1
                        if pos == 0:
                            resolved = True

                # in any case
                self.state_buffer[pos] = state
                self.transition_counts[prev_action, prev_state, wrong_temp] = 0
                self.transition_counts[prev_action, prev_state, state] += 1
                self.activation_counts[state] += 1
                self.activation_counts[wrong_temp] -= 1
