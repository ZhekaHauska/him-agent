#  Copyright (c) 2022 Autonomous Non-Profit Organization "Artificial Intelligence Research
#  Institute" (AIRI); Moscow Institute of Physics and Technology (National Research University).
#  All rights reserved.
#
#  Licensed under the AGPLv3 license. See LICENSE in the project root for license information.

import numpy as np
from hima.common.sdr import SparseSdr
from hima.common.utils import update_exp_trace


class Values:
    def __init__(
            self, seed: int, sdr_size: int, discount_factor: float, learning_rate: float,
    ):
        self._rng = np.random.default_rng(seed)
        self.discount_factor = discount_factor
        self.learning_rate = learning_rate
        self.values = self._rng.uniform(-1e-5, 1e-5, size=sdr_size)

    def update(
            self, sdr: SparseSdr, reward: float, next_sdr: SparseSdr, e_traces: np.ndarray
    ):
        lr = self.learning_rate
        v = self.values
        td_err = self.td_error(sdr, reward, next_sdr)
        v += lr * td_err * e_traces

    def td_error(self, sdr: SparseSdr, reward: float, next_sdr: SparseSdr):
        gamma = self.discount_factor
        r = reward
        v_sdr = self.value(sdr)
        next_v_sdr = self.value(next_sdr)
        return r + gamma * next_v_sdr - v_sdr

    def value(self, x: SparseSdr) -> float:
        # x = [] is terminal state
        if len(x) == 0:
            return 0
        return np.sum(self.values[x])


class EligibilityTraces:
    def __init__(
            self, sdr_size: int, trace_decay: float,
            discount_factor: float, with_reset: bool = False
    ):
        self.trace_decay = trace_decay
        self.discount_factor = discount_factor
        self.with_reset = with_reset
        self.traces = np.zeros(sdr_size, dtype=np.float)

    def update(self, sdr: SparseSdr):
        lambda_, gamma = self.trace_decay, self.discount_factor
        update_exp_trace(
            self.traces, sdr,
            decay=lambda_ * gamma,
            with_reset=self.with_reset
        )

    def reset(self):
        self.traces.fill(0.)


class TDLambda:
    def __init__(
            self, seed: int, sdr_size: int, gamma: float,
            alpha: float, lambda_: float, with_reset: bool,
    ):
        self.sdr_size = sdr_size
        self.gamma = gamma
        self.alpha = alpha
        self.synapse_values = Values(seed, sdr_size, gamma, alpha)
        self.synapse_traces = EligibilityTraces(sdr_size, lambda_, gamma, with_reset)

    def get_value(self, sdr: SparseSdr) -> float:
        value = self.synapse_values.value(sdr)
        return value

    def update(self, sdr: SparseSdr, reward: float, next_sdr: SparseSdr):
        """"
        Update rules:
        z_{t+1} <- \gamma \lambda z_t + x_t
        \delta = r_{t+1} + \gamma v_t x_{t+1} - v_t x_t
        v_{t+1} = v_t + \alpha \delta z_{t+1}
        """
        self.synapse_traces.update(sdr)
        self.synapse_values.update(sdr, reward, next_sdr, self.synapse_traces.traces)

    def reset(self):
        self.synapse_traces.reset()