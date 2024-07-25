#  Copyright (c) 2024 Autonomous Non-Profit Organization "Artificial Intelligence Research
#  Institute" (AIRI); Moscow Institute of Physics and Technology (National Research University).
#  All rights reserved.
#
#  Licensed under the AGPLv3 license. See LICENSE in the project root for license information.
import numpy as np
import socket
import json
import atexit
from hima.modules.belief.utils import get_data, send_string, NumpyEncoder

HOST = "127.0.0.1"
PORT = 5555


class ToyDHTM:
    """
        Simplified, fully deterministic DHTM
        for one hidden variable with visualizations.
        Stores transition matrix explicitly.
    """
    vis_server: socket.socket

    def __init__(
            self,
            n_obs_states,
            n_actions,
            n_clones,
            consolidation_threshold: int = 1,  # controls noise tolerance?
            visualize: bool = False,
            visualization_server=(HOST, PORT)
    ):
        self.n_clones = n_clones
        self.n_obs_states = n_obs_states
        self.n_actions = n_actions
        self.n_hidden_states = self.n_clones * self.n_obs_states
        self.visualize = visualize
        self.vis_server_address = visualization_server

        self.transition_counts = np.zeros(
            (self.n_actions, self.n_hidden_states, self.n_hidden_states)
        )
        self.activation_counts = np.zeros(self.n_hidden_states)
        # determines, how many counts we need to get for a transition to make it permanent
        self.consolidation_threshold = consolidation_threshold

        self.observation_buffer = list()
        self.action_buffer = list()
        self.state_buffer = list()

        if self.visualize:
            self.connect_to_vis_server()
            if self.vis_server is not None:
                atexit.register(self.close)

    def reset(self):
        self.clear_buffers()

    def clear_buffers(self):
        self.observation_buffer.clear()
        self.action_buffer.clear()
        self.state_buffer.clear()

    def observe(self, obs_state, action):
        # for debugging
        # event type: (name: str, data: tuple)
        events = dict()

        self.observation_buffer.append(obs_state)
        self.action_buffer.append(action)
        # state to be defined
        self.state_buffer.append(None)

        step = len(self.observation_buffer) - 1
        pos = step
        resolved = False

        events.update({'new_obs': (pos, obs_state, action)})

        while not resolved:
            if step == 0:
                # initial step
                column_states = np.arange(self.n_clones) + obs_state
                state = column_states[np.argmax(self.activation_counts[column_states])]
                self.state_buffer[pos] = state
                resolved = True

                events.update({'set_state': state})
            else:
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

                events.update(
                    {'predict_forward': (correct_prediction, wrong_perm, wrong_temp)}
                )
                # cases:
                # 1. correct set is not empty
                if len(correct_prediction) > 0:
                    state = correct_prediction[
                        np.argmax(
                            prediction[correct_prediction] +
                            self.activation_counts[correct_prediction]
                        )
                    ]
                    self.state_buffer[pos] = state
                    resolved = True

                    events.update({'set_state': state})
                # 2. correct set is empty
                else:
                    if state is None:
                        state = column_states[np.argmax(self.activation_counts[column_states])]
                        self.state_buffer[pos] = state

                        events.update({'set_state': state})

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

                        events.update(
                            {
                                'predict_backward':
                                (correct_prediction, sparse_prediction[~coincide])
                            }
                        )

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

                        events.update({'set_prev_state': prev_state})

                # in any case
                if len(wrong_temp) > 0:
                    self.transition_counts[prev_action, prev_state, wrong_temp] = 0
                    self.activation_counts[wrong_temp] -= 1

                    events.update({'remove_con': (prev_state, wrong_temp)})

                self.transition_counts[prev_action, prev_state, state] += 1
                self.activation_counts[state] += 1

                events.update({'reinforce_con': (prev_state, state)})
                # move to previous position
                if not resolved:
                    pos -= 1

                    events.update({'move': pos})

                    if pos == 0:
                        resolved = True

            if self.vis_server is not None:
                self._send_events(events)

            events.clear()

    def connect_to_vis_server(self):
        self.vis_server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

        try:
            self.vis_server.connect(self.vis_server_address)
            # handshake
            self._send_json_dict({'type': 'hello'})
            data = get_data(self.vis_server)
            print(data)

            if data != 'toy_dhtm':
                raise socket.error(
                    f'Handshake failed {self.vis_server_address}: It is not ToyDHTM vis server!'
                )
            print(f'Connected to visualization server {self.vis_server_address}!')
        except socket.error as msg:
            self.vis_server.close()
            self.vis_server = None
            print(f'Failed to connect to the visualization server: {msg}. Proceed.')

    def close(self):
        if self.vis_server is not None:
            self._send_json_dict({'type': 'close'})
            self.vis_server.close()
            print('Connection closed.')
        try:
            atexit.unregister(self.close)
        except Exception as e:
            print("exception unregistering close method", e)

    def _send_events(self, events):
        data = get_data(self.vis_server)
        if data == 'skip':
            self._send_json_dict({'type': 'skip'})
        elif data == 'close':
            self.vis_server.close()
            self.vis_server = None
            print('Server shutdown. Proceed.')
        elif data == 'step':
            data_dict = {'type': 'events'}
            data_dict.update(events)
            self._send_json_dict(data_dict)

    def _send_json_dict(self, data_dict):
        send_string(json.dumps(data_dict, cls=NumpyEncoder), self.vis_server)
