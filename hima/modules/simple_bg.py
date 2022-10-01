#  Copyright (c) 2022 Autonomous Non-Profit Organization "Artificial Intelligence Research
#  Institute" (AIRI); Moscow Institute of Physics and Technology (National Research University).
#  All rights reserved.
#
#  Licensed under the AGPLv3 license. See LICENSE in the project root for license information.

import copy
import numpy as np
from hima.common.sdr import SparseSdr


def softmax(x):
    e_x = np.exp(x - np.max(x, axis=-1))
    return e_x / np.sum(e_x)


class Striatum:
    def __init__(
            self, input_size: int, output_size: int, discount_factor: float,
            alpha: float, beta: float
    ):
        self._input_size = input_size
        self._output_size = output_size

        self.w_d1 = np.zeros((output_size, input_size))
        self.w_d2 = np.zeros((output_size, input_size))
        self.discount_factor = discount_factor
        self.alpha = alpha
        self.beta = beta

        self.previous_stimulus = None
        self.previous_response = None
        self.current_stimulus = None
        self.current_response = None
        self.current_max_response = None

    def compute(self, exc_input: SparseSdr) -> (np.ndarray, np.ndarray):
        if len(exc_input) > 0:
            d1 = np.mean(self.w_d1[:, exc_input], axis=-1)
            d2 = np.mean(self.w_d2[:, exc_input], axis=-1)
        else:
            d1 = np.zeros(self.w_d1.shape[0])
            d2 = np.zeros(self.w_d1.shape[0])
        return d1, d2

    def update_response(self, response: SparseSdr):
        self.previous_response = copy.deepcopy(self.current_response)
        self.current_response = copy.deepcopy(response)

    def update_stimulus(self, stimulus: SparseSdr):
        self.previous_stimulus = copy.deepcopy(self.current_stimulus)
        self.current_stimulus = copy.deepcopy(stimulus)

    def learn(self, reward, off_policy=False):
        learn_condition = self.previous_response is not None
        learn_condition &= self.previous_stimulus is not None
        learn_condition &= self.current_response is not None
        learn_condition &= self.current_stimulus is not None
        if not learn_condition:
            return
        learn_condition &= len(self.previous_response) > 0
        learn_condition &= len(self.previous_stimulus) > 0
        learn_condition &= len(self.current_response) > 0
        learn_condition &= len(self.current_stimulus) > 0
        if learn_condition:
            d1_d2 = self.w_d1[self.previous_response] - self.w_d2[self.previous_response]
            prev_values = np.mean(d1_d2[:, self.previous_stimulus], axis=-1)

            if off_policy:
                response = self.current_max_response
            else:
                response = self.current_response

            d1_d2 = self.w_d1[response] - self.w_d2[response]
            values = np.mean(d1_d2[:, self.current_stimulus], axis=-1)
            value = np.median(values)

            scaled_r = reward / len(self.previous_response)
            deltas = (scaled_r + self.discount_factor * value) - prev_values

            prev_a = self.previous_response.reshape((-1, 1))
            prev_s = self.previous_stimulus
            self.w_d1[prev_a, prev_s] += self.alpha * deltas.reshape((-1, 1))
            self.w_d2[prev_a, prev_s] -= self.beta * deltas.reshape((-1, 1))

    def reset(self):
        self.previous_response = None
        self.previous_stimulus = None
        self.current_response = None
        self.current_stimulus = None
        self.current_max_response = None


class BasalGanglia:
    alpha: float
    beta: float
    discount_factor: float
    _rng: np.random.Generator

    def __init__(
            self, seed: int, input_size: int, output_size: int, alpha: float, beta: float,
            discount_factor: float, off_policy: bool, softmax_beta: float, epsilon_noise: float,
    ):
        self._input_size = input_size
        self._output_size = output_size

        self.stri = Striatum(input_size, output_size, discount_factor, alpha, beta)

        self.off_policy = off_policy
        self.softmax_beta = softmax_beta
        self.epsilon_noise = epsilon_noise
        self._rng = np.random.default_rng(seed)

    def reset(self):
        self.stri.reset()

    def compute(self, stimulus, responses: list[SparseSdr]):
        d1, d2 = self.stri.compute(stimulus)
        gpi = - d1 + d2
        probs = (gpi - gpi.min()) / (gpi.max() - gpi.min() + 1e-12)
        gpi = self._rng.random(self._output_size) < probs
        bs = ~gpi

        activity = np.zeros(len(responses))
        for ind, response in enumerate(responses):
            activity[ind] = np.sum(bs[response])

        probs = softmax(self.softmax_beta * activity)
        probs = self.epsilon_noise / probs.size + (1 - self.epsilon_noise) * probs
        response_index = self._rng.choice(len(activity), 1, p=probs)[0]

        self.stri.current_max_response = responses[np.nanargmax(activity)]
        self.stri.update_response(responses[response_index])
        self.stri.update_stimulus(stimulus)
        return response_index

    def update(self, reward: float):
        self.stri.learn(reward, self.off_policy)
