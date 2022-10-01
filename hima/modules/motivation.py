#  Copyright (c) 2022 Autonomous Non-Profit Organization "Artificial Intelligence Research
#  Institute" (AIRI); Moscow Institute of Physics and Technology (National Research University).
#  All rights reserved.
#
#  Licensed under the AGPLv3 license. See LICENSE in the project root for license information.

import numpy as np
from hima.common.sdr import SparseSdr
from hima.modules.td_lambda import TDLambda


class SparseDopamineWeights:
    def __init__(
            self, seed: int, input_size: int, output_size: int, potential_fraction: float,
            discount_factor: float, trace_factor: float, learning_rate: float, trace_reset: bool
    ):
        self._rng = np.random.default_rng(seed)
        self.potential_size = int(input_size * potential_fraction)
        self.pre_cells = np.empty((output_size, self.potential_size), dtype=int)
        self.size = output_size * self.potential_size
        self.dopa_weights = TDLambda(
            seed, self.size, discount_factor, learning_rate, trace_factor, trace_reset
        )

        self.output_size = output_size
        for cell in range(output_size):
            potential_cells = self._rng.permutation(input_size)[:self.potential_size]
            potential_cells.sort()
            self.pre_cells[cell] = potential_cells

    def __mul__(self, other: SparseSdr) -> np.ndarray:
        other_dense = np.isin(self.pre_cells, other)
        other_norm = np.sqrt(np.sum(other_dense * other_dense, axis=1))
        w = self.dopa_weights.synapse_values.values.reshape((self.output_size, self.potential_size))
        this_norm = np.sqrt(np.sum(w * w, axis=1))
        res = (w * other_dense).sum(axis=1) / (other_norm * this_norm)
        return res

    def selective_mult(self, other: SparseSdr, condition: SparseSdr):
        other_dense = np.isin(self.pre_cells, other)[condition, :]
        other_norm = np.sqrt(np.sum(other_dense * other_dense, axis=1))
        w = self.dopa_weights.synapse_values.values.reshape((self.output_size, self.potential_size))
        w = w[condition, :]
        this_norm = np.sqrt(np.sum(w * w, axis=1))
        res = (w * other_dense).sum(axis=1) / (other_norm * this_norm)
        return res

    def update(self, sdr: SparseSdr, reward: float, next_sdr: SparseSdr):
        self.dopa_weights.update(sdr, reward, next_sdr)

    def reset(self):
        self.dopa_weights.reset()

    def xa2sdr(self, x: SparseSdr, a: SparseSdr) -> SparseSdr:
        s = np.nonzero(np.isin(self.pre_cells[a, :], x))
        s = a[s[0]] * self.potential_size + s[1]
        return s


class Striatum:
    def __init__(
            self, seed: int, state_size: int, motiv_size: int, action_size: int, trace_reset: bool,
            motiv_fraction: float, state_fraction: float, potential_fraction: float,
            discount_factor: float, trace_factor: float, boost_strength: float,
            learning_rate: float, activity_factor: float
    ):
        self.state_weights = SparseDopamineWeights(
            seed, state_size, action_size, potential_fraction,
            discount_factor, trace_factor, learning_rate, trace_reset
        )
        self.motiv_weights = SparseDopamineWeights(
            seed, motiv_size, action_size, potential_fraction,
            discount_factor, trace_factor, learning_rate, trace_reset
        )
        self.boost_factors = np.ones(action_size)
        self.mean_activity = np.zeros(action_size)

        self.activity_factor = activity_factor
        self.boost_strength = boost_strength

        self.action_size = action_size
        self.motiv_no_active_size = int(action_size * (1 - motiv_fraction))
        motiv_active_size = action_size - self.motiv_no_active_size
        self.state_no_active_size = int(motiv_active_size * (1 - state_fraction))

        self.last_sa = None
        self.last_ma = None
        self.last_reward = None
        self.is_first = True
        self.preactivation_field = None

    def compute(self, state: SparseSdr, motiv: SparseSdr) -> SparseSdr:
        # p is top under motiv
        bwm = self.boost_factors * (self.motiv_weights * motiv)
        k = self.motiv_no_active_size
        p = np.argpartition(bwm, k)[k:]
        self.preactivation_field = np.copy(p)

        # a is top under p conditioned by state
        ws = self.state_weights.selective_mult(state, p)
        bws = self.boost_factors[p] * ws
        k = self.state_no_active_size
        pa = np.argpartition(bws, k)[k:]
        a = p[pa]
        return a

    def update(self, state: SparseSdr, motiv: SparseSdr, action: SparseSdr, reward: float):
        # update boosting
        self.mean_activity[action] += 1
        self.mean_activity *= self.activity_factor
        a_mean = np.mean(self.mean_activity)
        self.boost_factors = np.exp(-self.boost_strength * (self.mean_activity - a_mean))

        # update values
        sa = self.state_weights.xa2sdr(state, action)
        ma = self.motiv_weights.xa2sdr(motiv, action)
        if self.is_first:
            self.is_first = False
        else:
            self.state_weights.update(self.last_sa, self.last_reward, sa)
            self.motiv_weights.update(self.last_ma, self.last_reward, ma)

        self.last_reward = reward
        self.last_sa = sa
        self.last_ma = ma

    def reset(self):
        self.mean_activity.fill(0)
        self.boost_factors.fill(1)
        self.state_weights.update(self.last_sa, self.last_reward, [])
        self.motiv_weights.update(self.last_ma, self.last_reward, [])
        self.state_weights.reset()
        self.motiv_weights.reset()
        self.last_sa = None
        self.last_ma = None
        self.last_reward = None
        self.is_first = True

