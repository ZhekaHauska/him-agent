#  Copyright (c) 2022 Autonomous Non-Profit Organization "Artificial Intelligence Research
#  Institute" (AIRI); Moscow Institute of Physics and Technology (National Research University).
#  All rights reserved.
#
#  Licensed under the AGPLv3 license. See LICENSE in the project root for license information.

import numpy as np
from htm.bindings.algorithms import SpatialPooler
from htm.bindings.sdr import SDR


class SandwichTp:
    def __init__(
            self, seed: int,
            pooling_decay: float,
            only_upper: bool,
            max_intermediate_used: int = None,
            pooling_decay_r: float = 1.,
            **kwargs
    ):
        rng = np.random.default_rng(seed)

        self.initial_pooling = kwargs['initial_pooling']
        self.pooling_decay = pooling_decay
        self.pooling_decay_r = pooling_decay_r
        self.max_intermediate_used = max_intermediate_used

        self.only_upper = only_upper
        if not self.only_upper:
            self.lower_sp = SpatialPooler(seed=rng.integers(100000), **kwargs['lower_sp_conf'])

        self.upper_sp = SpatialPooler(seed=rng.integers(100000), **kwargs['upper_sp_conf'])

        self._unionSDR = SDR(kwargs['upper_sp_conf']['columnDimensions'])
        self._unionSDR.dense = np.zeros(kwargs['upper_sp_conf']['columnDimensions'])
        self._pooling_activations = np.zeros(kwargs['upper_sp_conf']['inputDimensions'])

        if self.pooling_decay_r > 1:
            self.pooling_decay_mx = self._make_decay_matrix(rng)

        # FIXME: hack to get SandwichTp compatible with other TPs
        # maximum TP active output cells
        self._maxUnionCells = int(
            self.upper_sp.getNumColumns() * self.upper_sp.getLocalAreaDensity()
        )
        self._input_representation = SDR(self._pooling_activations.shape)

    def _pooling_decay_step(self):
        active_pooling_mask = self._pooling_activations != 0
        if self.pooling_decay_r > 1:
            decay_mx = self.pooling_decay_mx
            self._pooling_activations[active_pooling_mask] -= decay_mx[active_pooling_mask]
        else:
            self._pooling_activations[active_pooling_mask] -= self.pooling_decay

        np.clip(self._pooling_activations, 0, 1, out=self._pooling_activations)

    def compute(self, active_neurons: SDR, predicted_neurons: SDR, learn: bool = True) -> SDR:
        # alias for cached SDR obj
        interim_sdr = self._input_representation

        self._pooling_decay_step()

        # determine TP input: predicted are preferred
        # NB: `or True` turns off using active_neurons
        predicted_input_non_empty = len(predicted_neurons.sparse) > 0 or True
        pooling_scale = 1.0 if predicted_input_non_empty else 0.5
        input_sdr = predicted_neurons if predicted_input_non_empty else active_neurons

        # compute pre-pooling step
        if not self.only_upper:
            self.lower_sp.compute(input_sdr, learn=learn, output=interim_sdr)
        else:
            interim_sdr.sparse = np.copy(input_sdr.sparse)

        # increase pooling for current input; added part is scaled depending on
        # whether the input was predicted
        self._pooling_activations[interim_sdr.sparse] = (
                self._pooling_activations[interim_sdr.sparse] + self.initial_pooling * pooling_scale
        ).clip(0, 1)

        # calculate what should be propagated up from the pooling to the Upper SP
        if self.max_intermediate_used is None:
            # usual behavior:
            sparse_sdr_to_upper = np.flatnonzero(self._pooling_activations)
        else:
            # restrict used active pooled cells
            top_k = self.max_intermediate_used
            threshold = np.partition(self._pooling_activations, kth=-top_k)[-top_k]
            if threshold == 0:
                sparse_sdr_to_upper = np.flatnonzero(self._pooling_activations)
            else:
                sparse_sdr_to_upper = np.flatnonzero(self._pooling_activations >= threshold)

        # calculate the output: union SDR
        interim_sdr.sparse = sparse_sdr_to_upper
        self.upper_sp.compute(interim_sdr, learn=learn, output=self._unionSDR)

        return self.getUnionSDR()

    def getUnionSDR(self):
        # ---- middle layer --------

        # res = SDR(self._pooling_activations.shape)
        # res.dense = self._pooling_activations != 0
        # return res
        # --------------------------
        return self._unionSDR

    def getNumInputs(self):
        return self.lower_sp.getNumInputs()

    def getNumColumns(self):
        return self.upper_sp.getNumColumns()

    def reset(self):
        self._pooling_activations = np.zeros(self._pooling_activations.shape)
        self._unionSDR = SDR(self._unionSDR.dense.shape)
        self._unionSDR.dense = np.zeros(self._unionSDR.dense.shape)

    @property
    def output_sdr_size(self):
        return self.upper_sp.getNumColumns()

    @property
    def n_active_bits(self):
        return self._maxUnionCells

    def _make_decay_matrix(self, rng):
        decay, r = self.pooling_decay, self.pooling_decay_r
        decay_mx = decay * loguniform(rng, r, size=self._pooling_activations.size)
        return decay_mx


def loguniform(rng, radius, size):
    low = np.log(1 / radius)
    high = np.log(radius)
    return np.exp(rng.uniform(low, high, size))
