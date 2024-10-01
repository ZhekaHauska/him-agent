#  Copyright (c) 2023 Autonomous Non-Profit Organization "Artificial Intelligence Research
#  Institute" (AIRI); Moscow Institute of Physics and Technology (National Research University).
#  All rights reserved.
#
#  Licensed under the AGPLv3 license. See LICENSE in the project root for license information.
import numpy as np
EPS = 1e-12


class Striatum:
    def __init__(
            self,
            input_size,
            output_sizes,
            n_areas: int = 1,
            max_states: int = 1000,
            state_detection_threshold: float = EPS,
            activity_lr: float = 0.01,
            lr: float = 0.1,
            seed: int = 0
    ):
        """
        Decode cortex messages and associate them with patterns
            input_size: size of input from the cortex
            output_size: output size of each individual area
            n_areas: specify number of areas
                Each area has the same input, but have separate weights
            max_states: maximum cortex states we can decode and store
            state_detection_threshold: minimum state activation probability
        """
        self.seed = seed
        self.input_size = input_size
        self.output_sizes = output_sizes
        self.n_areas = n_areas
        self.max_states = max_states
        self.lr = lr
        self.activity_lr = activity_lr
        self.state_detection_threshold = state_detection_threshold

        self.receptive_fields = np.full((max_states, input_size), fill_value=-1)

        self.states_in_use_mask = np.full(self.max_states, fill_value=False)
        self.states_in_use = np.empty(0)

        self.state_activity = np.zeros(max_states)
        self.weights = [np.zeros((max_states, output_size)) for output_size in self.output_sizes]

        self.active_states = None
        self.probs = None
        self.prediction = None

    def predict(self, messages: np.ndarray, area: int = 0, learn=True):
        self.active_states = np.empty(0)
        self.probs = np.empty(0)
        self.states_in_use = np.flatnonzero(self.state_activity > 0)

        if len(self.states_in_use) > 0:
            probs = messages[self.receptive_fields[self.states_in_use]]

            probs = np.min(probs, axis=-1)
            active_states_mask = probs > self.state_detection_threshold

            self.active_states = self.states_in_use[active_states_mask]
            self.probs = probs[active_states_mask].reshape(-1, 1)

        if len(self.active_states) == 0 and learn:
            messages = messages.reshape(self.input_size, -1)
            n_states = messages.shape[-1]
            cells = np.argmax(messages, axis=-1) + np.arange(self.input_size) * n_states
            probs = np.array([np.min(messages.flatten()[cells])])

            if probs[0] > self.state_detection_threshold:
                new_state = np.argmin(self.state_activity)
                self.state_activity[new_state] = 1
                self.weights[area][new_state] = np.zeros(self.output_sizes[area])

                self.receptive_fields[new_state] = cells

                self.active_states = np.array([new_state])
                self.probs = probs

        if len(self.active_states) > 0:
            if learn:
                self.state_activity -= self.activity_lr * self.state_activity
                self.state_activity[self.active_states] += self.activity_lr * (
                        1 - self.state_activity[self.active_states]
                )

            self.prediction = np.sum(
                self.weights[area][self.active_states] * self.probs.reshape(-1, 1), axis=0
            )
        else:
            self.prediction = np.zeros(self.output_sizes[area])

        return self.prediction

    def update_weights(self, target: np.ndarray, area: int = 0):
        error = target - self.prediction

        if len(self.active_states) > 0:
            self.weights[area][self.active_states] += self.lr * error.reshape(1, -1) * self.probs
            self.weights[area] = np.clip(self.weights[area], 0, None)

        return np.sum(np.power(error, 2))