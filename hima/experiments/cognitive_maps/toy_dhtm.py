#  Copyright (c) 2024 Autonomous Non-Profit Organization "Artificial Intelligence Research
#  Institute" (AIRI); Moscow Institute of Physics and Technology (National Research University).
#  All rights reserved.
#
#  Licensed under the AGPLv3 license. See LICENSE in the project root for license information.
import numpy as np
import socket
import json
import atexit
import pygraphviz as pgv
import colormap
from hima.modules.belief.utils import get_data, send_string, NumpyEncoder

HOST = "127.0.0.1"
PORT = 5555
EPS = 1e-24


class ToyDHTM:
    """
        Simplified, fully deterministic DHTM
        for one hidden variable with visualizations.
        Stores transition matrix explicitly.
    """
    vis_server: socket.socket = None

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
            (self.n_actions, self.n_hidden_states, self.n_hidden_states),
            dtype=np.int64
        )
        self.activation_counts = np.zeros(self.n_hidden_states, dtype=np.int64)
        # determines, how many counts we need to get for a transition to make it permanent
        self.consolidation_threshold = consolidation_threshold

        self.observation_buffer = list()
        self.action_buffer = list()
        self.state_buffer = list()

        self.vis_server = None
        if self.visualize:
            self.connect_to_vis_server()
            if self.vis_server is not None:
                atexit.register(self.close)

    def reset(self, gridworld_map):
        self.clear_buffers()
        if self.vis_server is not None:
            self._send_events([('reset', {'gridworld_map': gridworld_map})])

    def clear_buffers(self):
        self.observation_buffer.clear()
        self.action_buffer.clear()
        self.state_buffer.clear()

    def observe(self, obs_state, action, true_pos=None):
        # for debugging
        # event type: (name: str, data: tuple)
        events = list()

        self.observation_buffer.append(obs_state)
        self.action_buffer.append(action)
        # state to be defined
        self.state_buffer.append(None)

        step = len(self.observation_buffer) - 1
        pos = step
        resolved = False

        events.append(('new_true_pos', true_pos))
        events.append(('new_obs', pos, obs_state, action))

        while not resolved:
            if step == 0:
                # initial step
                column_states = self._get_column_states(obs_state)
                state = column_states[np.argmax(self.activation_counts[column_states])]
                self.state_buffer[pos] = state
                resolved = True

                events.append(('set_state', self._state_to_clone(state)))
            else:
                # input variables
                obs_state = self.observation_buffer[pos]
                column_states = self._get_column_states(obs_state)
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

                permanence_mask = prediction[wrong_prediction] > self.consolidation_threshold
                wrong_perm = wrong_prediction[
                    permanence_mask
                ]
                wrong_temp = wrong_prediction[
                    ~permanence_mask
                ]

                events.append(
                    (
                        'predict_forward',
                        [
                            self._state_to_clone(x, return_obs_state=True) + (w,)
                            for x, w in
                            zip(correct_prediction, prediction[correct_prediction])
                        ],
                        [
                            self._state_to_clone(x, return_obs_state=True) + (w,)
                            for x, w in
                            zip(wrong_perm, prediction[wrong_perm])
                        ],
                        [
                            self._state_to_clone(x, return_obs_state=True) + (w,)
                            for x, w in
                            zip(wrong_temp, prediction[wrong_temp])
                        ]
                    )
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

                    events.append(('set_state', self._state_to_clone(state)))
                # 2. correct set is empty
                else:
                    if len(wrong_perm) == 0:
                        if state is None:
                            state = column_states[np.argmax(self.activation_counts[column_states])]
                            self.state_buffer[pos] = state

                            events.append(('set_state', self._state_to_clone(state)))

                        resolved = True
                    else:
                        # resampling previous clone
                        # try to use backward connections first
                        if state is None:
                            prediction = self.transition_counts[
                                prev_action, :, column_states
                            ].sum(axis=0).flatten()
                        else:
                            prediction = self.transition_counts[
                                prev_action, :, state
                            ].flatten()

                        sparse_prediction = np.flatnonzero(prediction)
                        prev_obs_state = self.observation_buffer[pos - 1]

                        column_states = self._get_column_states(prev_obs_state)
                        coincide = np.isin(sparse_prediction, column_states)
                        correct_prediction = sparse_prediction[coincide]
                        wrong_prediction = sparse_prediction[~coincide]

                        events.append(
                            (
                                'predict_backward',
                                [
                                    self._state_to_clone(x, return_obs_state=True) + (w,)
                                    for x, w in
                                    zip(correct_prediction, prediction[correct_prediction])
                                ],
                                [
                                    self._state_to_clone(x, return_obs_state=True) + (w,)
                                    for x, w in
                                    zip(wrong_prediction, prediction[wrong_prediction])
                                ]
                            )
                        )

                        if len(correct_prediction) > 0:
                            prev_state = correct_prediction[
                                np.argmax(
                                    prediction[correct_prediction] +
                                    self.activation_counts[correct_prediction]
                                )
                            ]
                            if state is None:
                                prediction = self.transition_counts[prev_action, prev_state].flatten()
                                sparse_prediction = np.flatnonzero(prediction)
                                coincide = np.isin(sparse_prediction, column_states)
                                correct_prediction = sparse_prediction[coincide]
                                state = correct_prediction[
                                    np.argmax(
                                        prediction[correct_prediction] +
                                        self.activation_counts[correct_prediction]
                                    )
                                ]
                                self.state_buffer[pos] = state

                                events.append(('set_state', self._state_to_clone(state)))
                        else:
                            # choose the least used clone
                            # (presumably with minimum outward connections)
                            prev_state = column_states[
                                np.argmin(
                                    self.activation_counts[column_states]
                                )
                            ]
                            if state is None:
                                state = column_states[
                                    np.argmax(self.activation_counts[column_states])]
                                self.state_buffer[pos] = state

                                events.append(('set_state', self._state_to_clone(state)))

                        self.state_buffer[pos - 1] = prev_state

                        events.append(
                            ('set_prev_state', self._state_to_clone(prev_state))
                        )

                # in any case
                if len(wrong_temp) > 0:
                    self.transition_counts[prev_action, prev_state, wrong_temp] = 0
                    self.activation_counts[wrong_temp] -= 1

                    events.append(
                        (
                            'remove_con',
                            prev_action,
                            self._state_to_clone(prev_state, return_obs_state=True),
                            [self._state_to_clone(x, return_obs_state=True) for x in wrong_temp]
                        )
                    )

                self.transition_counts[prev_action, prev_state, state] += 1
                self.activation_counts[state] += 1

                events.append(
                    (
                        'reinforce_con',
                        prev_action,
                        self._state_to_clone(prev_state, return_obs_state=True),
                        self._state_to_clone(state, return_obs_state=True)
                    )
                )
                # move to previous position
                if not resolved:
                    pos -= 1

                    events.append(('move', pos))

                    if pos == 0:
                        resolved = True

            if self.vis_server is not None:
                self._send_events(events)

            events.clear()

    def _state_to_clone(self, state, return_obs_state=False):
        obs_state = state // self.n_clones
        clone = state - self.n_clones * obs_state
        if return_obs_state:
            return clone, obs_state
        else:
            return clone

    def _get_column_states(self, obs_state):
        return np.arange(self.n_clones) + obs_state * self.n_clones

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
            data_dict = {
                'type': 'events',
                'events': events
            }
            self._send_json_dict(data_dict)

    def _send_json_dict(self, data_dict):
        send_string(json.dumps(data_dict, cls=NumpyEncoder), self.vis_server)

    def draw_graph(self, path=None, connection_threshold=0, activation_threshold=0, labels=None):
        g = pgv.AGraph(strict=False, directed=True)
        outline_color = '#3655b3'
        nonzero_states = np.flatnonzero(self.activation_counts > activation_threshold)
        node_cmap = colormap.cmap_builder('Pastel1')
        edge_cmap = colormap.Colormap().cmap_bicolor('white', 'blue')

        for state in nonzero_states:
            if labels is not None:
                label = str(labels[state]) + '_'
            else:
                label = ''
            clone, obs_state = self._state_to_clone(state, return_obs_state=True)
            g.add_node(
                f'{label}{obs_state}({clone})',
                style='filled',
                fillcolor=colormap.rgb2hex(
                    *(node_cmap(obs_state / self.n_obs_states)[:-1]),
                    normalised=True
                ),
                color=outline_color
            )

        for u in nonzero_states:
            if labels is not None:
                u_label = str(labels[u]) + '_'
            else:
                u_label = ''

            u_clone, u_obs_state = self._state_to_clone(u, return_obs_state=True)
            for action in range(self.n_actions):
                transitions = self.transition_counts[action, u]
                weights = transitions / (transitions.sum() + EPS)
                nonzero_transitions = np.flatnonzero(weights > connection_threshold)

                for v, weight in zip(nonzero_transitions, weights[nonzero_transitions]):
                    if labels is not None:
                        v_label = str(labels[v]) + '_'
                    else:
                        v_label = ''
                    v_clone, v_obs_state = self._state_to_clone(v, return_obs_state=True)
                    line_color = colormap.rgb2hex(
                        *(edge_cmap(int(255 * weight))[:-1]),
                        normalised=True
                    )
                    g.add_edge(
                        f'{u_label}{u_obs_state}({u_clone})', f'{v_label}{v_obs_state}({v_clone})',
                        color=line_color,
                        label=str(action)
                    )

        g.layout(prog='dot')
        return g.draw(path, format='png')
