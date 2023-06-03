#  Copyright (c) 2023 Autonomous Non-Profit Organization "Artificial Intelligence Research
#  Institute" (AIRI); Moscow Institute of Physics and Technology (National Research University).
#  All rights reserved.
#
#  Licensed under the AGPLv3 license. See LICENSE in the project root for license information.
from hima.modules.htm.connections import Connections
from hima.modules.belief.utils import softmax, normalize
from hima.modules.belief.utils import EPS, INT_TYPE, UINT_DTYPE, REAL_DTYPE, REAL64_DTYPE
from hima.modules.htm.spatial_pooler import SPEnsemble

from htm.bindings.sdr import SDR
from htm.bindings.math import Random

import numpy as np
import warnings


class Layer:
    """
        This class represents a layer of the neocortex model.
    """
    def __init__(
            self,
            n_obs_vars: int,
            n_obs_states: int,
            cells_per_column: int,
            n_vars_per_factor: int,
            factors_per_var: int,
            spatial_pooler: SPEnsemble = None,
            n_context_vars: int = 0,
            n_context_states: int = 0,
            n_external_vars: int = 0,
            n_external_states: int = 0,
            external_vars_boost: float = 0,
            unused_vars_boost: float = 0,
            lr: float = 0.01,
            segment_activity_lr: float = 0.001,
            var_score_lr: float = 0.001,
            inverse_temp: float = 1.0,
            initial_factor_value: float = 0,
            cell_activation_threshold: float = EPS,
            max_segments_per_cell: int = 255,
            max_segments: int = 10000,
            developmental_period: int = 10000,
            fraction_of_segments_to_prune: float = 0.5,
            seed: int = None,
    ):
        self._rng = np.random.default_rng(seed)

        if seed:
            self._legacy_rng = Random(seed)
        else:
            self._legacy_rng = Random()

        self.timestep = 1
        self.developmental_period = developmental_period
        self.fraction_of_segments_to_prune = fraction_of_segments_to_prune
        self.n_obs_vars = n_obs_vars
        self.n_hidden_vars = n_obs_vars
        self.n_obs_states = n_obs_states
        self.n_external_vars = n_external_vars
        self.n_external_states = n_external_states
        self.n_context_vars = n_context_vars
        self.n_context_states = n_context_states
        self.external_vars_boost = external_vars_boost
        self.unused_vars_boost = unused_vars_boost

        self.cells_per_column = cells_per_column
        self.n_hidden_states = cells_per_column * n_obs_states

        self.internal_cells = self.n_hidden_vars * self.n_hidden_states
        self.external_input_size = self.n_external_vars * self.n_external_states
        self.context_input_size = self.n_context_vars * self.n_context_states
        self.total_cells = (
                self.internal_cells +
                self.external_input_size +
                self.context_input_size
        )

        self.input_sdr_size = n_obs_vars * n_obs_states
        self.spatial_pooler = spatial_pooler
        self.sp_input = SDR(self.input_sdr_size)
        self.sp_output = SDR(self.spatial_pooler.getNumColumns())

        self.total_segments = max_segments
        self.max_segments_per_cell = max_segments_per_cell

        self.lr = lr
        self.segment_activity_lr = segment_activity_lr
        self.var_score_lr = var_score_lr
        self.factors_per_var = factors_per_var
        self.total_factors = self.n_hidden_vars * self.factors_per_var

        self.inverse_temp = inverse_temp

        self.n_columns = self.n_obs_vars * self.n_obs_states

        # number of variables assigned to a segment
        self.n_vars_per_factor = n_vars_per_factor

        # for now leave it strict
        self.segment_activation_threshold = n_vars_per_factor

        # low probability clipping
        self.cell_activation_threshold = cell_activation_threshold

        self.internal_active_cells = SDR(self.internal_cells)
        self.internal_active_cells.sparse = np.arange(self.n_hidden_vars) * self.n_hidden_states

        self.external_active_cells = SDR(self.external_input_size)
        self.context_active_cells = SDR(self.context_input_size)

        self.internal_forward_messages = np.zeros(
            self.internal_cells,
            dtype=REAL64_DTYPE
        )
        self.internal_forward_messages[self.internal_active_cells.sparse] = 1
        self.external_messages = np.zeros(self.external_input_size)
        self.context_messages = np.zeros(self.context_input_size)

        self.prediction_cells = None
        self.prediction_columns = None

        # cells are numbered in the following order:
        # internal cells | context cells | external cells
        self.connections = Connections(
            numCells=self.total_cells,
            connectedThreshold=0.5
        )

        # each segment corresponds to a factor value
        self.initial_factor_value = initial_factor_value
        self.log_factor_values_per_segment = np.full(
            self.total_segments,
            fill_value=self.initial_factor_value,
            dtype=REAL64_DTYPE
        )

        self.segment_activity = np.ones(
            self.total_segments,
            dtype=REAL64_DTYPE
        )

        self.factor_for_segment = np.full(
            self.total_segments,
            fill_value=-1,
            dtype=INT_TYPE
        )

        # receptive fields for each segment
        self.receptive_fields = np.full(
            (self.total_segments, self.n_vars_per_factor),
            fill_value=-1,
            dtype=INT_TYPE
        )

        # treat factors as segments
        self.factor_connections = Connections(
            numCells=self.total_cells,
            connectedThreshold=0.5
        )

        self.segments_in_use = np.empty(0, dtype=UINT_DTYPE)
        self.factors_in_use = np.empty(0, dtype=UINT_DTYPE)
        self.factors_score = np.empty(0, dtype=REAL_DTYPE)

        self.factor_vars = np.full(
            (self.total_factors, self.n_vars_per_factor),
            fill_value=-1,
            dtype=INT_TYPE
        )

        self.var_score = np.ones(
            self.n_hidden_vars + self.n_context_vars + self.n_external_vars,
            dtype=REAL64_DTYPE
        )

    def reset(self):
        self.internal_forward_messages = np.zeros(
            self.internal_cells,
            dtype=REAL64_DTYPE
        )
        self.internal_forward_messages[self.internal_active_cells.sparse] = 1
        self.external_messages = np.zeros(self.external_input_size)
        self.context_messages = np.zeros(self.context_input_size)

        self.prediction_cells = None
        self.prediction_columns = None

    def propagate_belief(self, messages: np.ndarray):
        """
        Calculate messages for internal cells based on messages from all cells.
            messages: should be an array of size total_cells
        """
        # filter dendrites that have low activation likelihood
        active_cells = SDR(self.total_cells)

        active_cells.sparse = np.flatnonzero(
            messages >= self.cell_activation_threshold
        )

        num_connected_segment = self.connections.computeActivity(
            active_cells,
            False
        )

        active_segments = np.flatnonzero(
            num_connected_segment >= self.segment_activation_threshold
        )
        cells_for_active_segments = self.connections.mapSegmentsToCells(active_segments)

        log_next_messages = np.full(
            self.internal_cells,
            fill_value=-np.inf,
            dtype=REAL_DTYPE
        )

        # excitation activity
        if len(active_segments) > 0:
            factors_for_active_segments = self.factor_for_segment[active_segments]
            log_factor_value = self.log_factor_values_per_segment[active_segments]

            likelihood = messages[self.receptive_fields[active_segments]]
            log_likelihood = np.sum(np.log(likelihood), axis=-1)

            log_excitation_per_segment = log_likelihood + log_factor_value

            # uniquely encode pairs (factor, cell) for each segment
            cell_factor_id_per_segment = (
                    factors_for_active_segments * self.internal_cells
                    + cells_for_active_segments
            )

            # group segments by factors
            sorting_inxs = np.argsort(cell_factor_id_per_segment)
            cells_for_active_segments = cells_for_active_segments[sorting_inxs]
            cell_factor_id_per_segment = cell_factor_id_per_segment[sorting_inxs]
            log_excitation_per_segment = log_excitation_per_segment[sorting_inxs]

            cell_factor_id_excitation, reduce_inxs = np.unique(
                cell_factor_id_per_segment, return_index=True
            )

            # approximate log sum with max
            log_excitation_per_factor = np.maximum.reduceat(log_excitation_per_segment, reduce_inxs)

            # group segments by cells
            cells_for_factors = cells_for_active_segments[reduce_inxs]

            sort_inxs = np.argsort(cells_for_factors)
            cells_for_factors = cells_for_factors[sort_inxs]
            log_excitation_per_factor = log_excitation_per_factor[sort_inxs]

            cells_with_factors, reduce_inxs = np.unique(cells_for_factors, return_index=True)

            log_prediction_for_cells_with_factors = np.add.reduceat(
                log_excitation_per_factor, indices=reduce_inxs
            )

            log_next_messages[cells_with_factors] = log_prediction_for_cells_with_factors

        log_next_messages = log_next_messages.reshape((self.n_hidden_vars, self.n_hidden_states))

        # shift log value for stability
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', category=RuntimeWarning)

            means = log_next_messages.mean(
                axis=-1,
                where=~np.isinf(log_next_messages)
            ).reshape((-1, 1))

        means[np.isnan(means)] = 0

        log_next_messages -= means

        log_next_messages = self.inverse_temp * log_next_messages

        next_messages = normalize(np.exp(log_next_messages))

        next_messages = next_messages.flatten()

        assert ~np.any(np.isnan(next_messages))

        self.internal_forward_messages = next_messages

    def set_external_messages(self, messages=None):
        # update external cells
        if messages is not None:
            self.external_messages = messages
        elif self.external_input_size != 0:
            self.external_messages = normalize(
                np.zeros(self.external_input_size).reshape((self.n_external_vars, -1))
            ).flatten()

    def set_context_messages(self, messages=None):
        # update external cells
        if messages is not None:
            self.context_messages = messages
        elif self.context_input_size != 0:
            self.context_messages = normalize(
                np.zeros(self.context_input_size).reshape((self.n_context_vars, -1))
            ).flatten()

    def predict(self):
        # step 1: predict cells based on context
        # block all messages except context messages
        # think about it as thalamus orchestration of the neocortex
        messages = np.zeros(self.total_cells)
        messages[
            self.internal_cells:
            self.internal_cells + self.context_input_size
        ] = self.context_messages

        self.propagate_belief(messages)

        # step 2: update predictions based on internal and external connections
        # block context messages
        messages = np.zeros(self.total_cells)

        messages[: self.internal_cells] = self.internal_forward_messages

        messages[
            self.internal_cells + self.context_input_size:
            -1
        ] = self.external_messages

        self.propagate_belief(messages)

        self.prediction_cells = self.internal_forward_messages.copy()

        self.prediction_columns = self.prediction_cells.reshape(
            (self.n_columns, self.cells_per_column)
        ).sum(axis=-1)

    def observe(
            self,
            observation: np.ndarray,
            learn: bool = True
    ):
        """
            observation: pattern in sparse representation
        """
        # encode observation
        if self.spatial_pooler is not None:
            self.spatial_pooler.compute(self.sp_input, learn, self.sp_output)
            observation = self.sp_output.sparse

        # update messages
        cells = self._get_cells_for_observation(observation)
        obs_factor = np.zeros_like(self.internal_forward_messages)
        obs_factor[cells] = 1
        self.internal_forward_messages *= obs_factor

        with warnings.catch_warnings():
            warnings.filterwarnings('error')
            self.internal_forward_messages = normalize(
                self.internal_forward_messages.reshape((self.n_hidden_vars, -1)),
                obs_factor.reshape((self.n_hidden_vars, -1))
            ).flatten()

        # update connections
        if learn and (len(self.internal_active_cells.sparse) > 0):
            # sample cells from messages (1-step Monte-Carlo learning)
            self.internal_active_cells.sparse = self._sample_cells(
                self.internal_forward_messages.reshape((self.n_hidden_vars, -1))
            )
            self.context_active_cells.sparse = self._sample_cells(
                self.context_messages.reshape((self.n_context_vars, -1))
            )
            self.external_active_cells.sparse = self._sample_cells(
                self.external_messages.reshape((self.n_external_vars, -1))
            )

            # learn context segments
            self._learn(
                np.concatenate(
                    [
                        self.internal_cells + self.context_active_cells.sparse,
                        (
                                self.internal_cells +
                                self.context_input_size +
                                self.external_active_cells.sparse
                         )
                    ]
                ),
                self.internal_active_cells.sparse,
                prune_segments=(self.timestep % self.developmental_period) == 0
            )

            # learn internal segments
            self._learn(
                self.internal_active_cells.sparse,
                self.internal_active_cells.sparse
            )

        self.timestep += 1

    def _learn(self, active_cells, next_active_cells, prune_segments=False):
        (
            segments_to_reinforce,
            segments_to_punish,
            cells_to_grow_new_segments
        ) = self._calculate_learning_segments(
            active_cells,
            next_active_cells
        )

        new_segments = self._grow_new_segments(
            cells_to_grow_new_segments,
            active_cells
        )

        self.segments_in_use = np.append(
            self.segments_in_use,
            new_segments[np.isin(new_segments, self.segments_in_use, invert=True)]
        )

        self._update_factors(
            np.concatenate(
                [
                    segments_to_reinforce,
                    new_segments
                ]
            ),
            segments_to_punish,
            prune=prune_segments
        )

    def _calculate_learning_segments(self, active_cells, next_active_cells):
        # determine which segments are learning and growing
        active_cells_sdr = SDR(self.internal_cells)
        active_cells_sdr.sparse = active_cells

        num_connected = self.connections.computeActivity(
            active_cells_sdr,
            False
        )

        active_segments = np.flatnonzero(num_connected >= self.segment_activation_threshold)

        cells_for_active_segments = self.connections.mapSegmentsToCells(active_segments)

        mask = np.isin(cells_for_active_segments, next_active_cells)
        segments_to_learn = active_segments[mask]
        segments_to_punish = active_segments[~mask]

        cells_to_grow_new_segments = next_active_cells[
            np.isin(next_active_cells, cells_for_active_segments, invert=True)
        ]

        return (
            segments_to_learn.astype(UINT_DTYPE),
            segments_to_punish.astype(UINT_DTYPE),
            cells_to_grow_new_segments.astype(UINT_DTYPE)
        )

    def _update_factors(self, segments_to_reinforce, segments_to_punish, prune=False):
        w = self.log_factor_values_per_segment[segments_to_reinforce]
        self.log_factor_values_per_segment[
            segments_to_reinforce
        ] += np.log1p(self.lr * (np.exp(-w) - 1))

        self.log_factor_values_per_segment[
            segments_to_punish
        ] += np.log1p(-self.lr)

        active_segments = np.concatenate([segments_to_reinforce, segments_to_punish])
        non_active_segments = self.segments_in_use[
            np.isin(self.segments_in_use, active_segments, invert=True)
        ]

        self.segment_activity[active_segments] += self.segment_activity_lr * (
                1 - self.segment_activity[active_segments]
        )
        self.segment_activity[non_active_segments] -= self.segment_activity_lr * self.segment_activity[
            non_active_segments
        ]

        vars_for_correct_segments = np.unique(
            self.receptive_fields[segments_to_reinforce].flatten() // self.n_hidden_states
        )

        vars_for_incorrect_segments = np.unique(
            self.receptive_fields[segments_to_punish].flatten() // self.n_hidden_states
        )

        self.var_score[vars_for_correct_segments] += self.var_score_lr * (
                1 - self.var_score[vars_for_correct_segments]
        )

        self.var_score[vars_for_incorrect_segments] -= self.var_score_lr * self.var_score[
            vars_for_incorrect_segments
        ]

        if prune:
            n_segments_to_prune = int(
                self.fraction_of_segments_to_prune * len(self.segments_in_use)
            )
            self._prune_segments(n_segments_to_prune)

    def _prune_segments(self, n_segments):
        log_value = self.log_factor_values_per_segment[self.segments_in_use]
        activity = self.segment_activity[self.segments_in_use]

        score = (
                np.exp(log_value) * activity
        )

        segments_to_prune = self.segments_in_use[
            np.argpartition(score, n_segments)[:n_segments]
        ]

        filter_destroyed_segments = np.isin(
            self.segments_in_use, segments_to_prune, invert=True
        )
        self.segments_in_use = self.segments_in_use[filter_destroyed_segments]

        for segment in segments_to_prune:
            self.connections.destroySegment(segment)

        return segments_to_prune

    def _get_cells_for_observation(self, obs_states):
        vars_for_obs_states = obs_states // self.n_obs_states
        all_vars = np.arange(self.n_obs_vars)
        vars_without_states = all_vars[np.isin(all_vars, vars_for_obs_states, invert=True)]

        cells_for_empty_vars = self._get_cells_in_vars(vars_without_states)

        cells_in_columns = (
                (
                    obs_states * self.cells_per_column
                ).reshape((-1, 1)) +
                np.arange(self.cells_per_column, dtype=UINT_DTYPE)
            ).flatten()

        return np.concatenate([cells_for_empty_vars, cells_in_columns])

    def _get_cells_in_vars(self, variables):
        local_vars_mask = variables < self.n_hidden_vars

        cells_in_local_vars = (
                (variables[local_vars_mask] * self.n_hidden_states).reshape((-1, 1)) +
                np.arange(self.n_hidden_states, dtype=UINT_DTYPE)
        ).flatten()

        cells_in_ext_vars = (
                ((variables[~local_vars_mask] - self.n_hidden_vars) *
                 self.n_external_states).reshape((-1, 1)) +
                np.arange(self.n_external_states, dtype=UINT_DTYPE)
        ).flatten() + self.internal_cells

        return np.concatenate([cells_in_local_vars, cells_in_ext_vars])

    def _filter_cells_by_vars(self, cells, variables):
        cells_in_vars = self._get_cells_in_vars(variables)

        mask = np.isin(cells, cells_in_vars)

        return cells[mask]

    def _sample_cells(self, messages, shift=0):
        """
            messages.shape = (n_vars, n_states)
        """
        n_vars, n_states = messages.shape

        next_states = self._sample_categorical_variables(
            messages
        )
        # transform states to cell ids
        next_cells = next_states + np.arange(
            0,
            n_states*n_vars,
            n_states
        )
        next_cells += shift

        return next_cells.astype(UINT_DTYPE)

    def _sample_categorical_variables(self, probs):
        assert np.allclose(probs.sum(axis=-1), 1)

        gammas = self._rng.uniform(size=probs.shape[0]).reshape((-1, 1))

        dist = np.cumsum(probs, axis=-1)

        ubounds = dist
        lbounds = np.zeros_like(dist)
        lbounds[:, 1:] = dist[:, :-1]

        cond = (gammas >= lbounds) & (gammas < ubounds)

        states = np.zeros_like(probs) + np.arange(probs.shape[1])

        samples = states[cond]

        return samples

    def _grow_new_segments(
            self,
            new_segment_cells,
            growth_candidates,
    ):
        # free space for new segments
        n_segments_after_growing = len(self.segments_in_use) + len(new_segment_cells)
        if n_segments_after_growing > self.total_segments:
            n_segments_to_prune = n_segments_after_growing - self.total_segments
            self._prune_segments(n_segments_to_prune)

        # sum factor values for every factor
        if len(self.segments_in_use) > 0:
            factor_for_segment = self.factor_for_segment[self.segments_in_use]
            log_factor_values = self.log_factor_values_per_segment[self.segments_in_use]
            segment_activation_freq = self.segment_activity[self.segments_in_use]

            sort_ind = np.argsort(factor_for_segment)
            factors_sorted = factor_for_segment[sort_ind]
            segments_sorted_values = log_factor_values[sort_ind]
            segments_sorted_freq = segment_activation_freq[sort_ind]

            factors_with_segments, split_ind, counts = np.unique(
                factors_sorted,
                return_index=True,
                return_counts=True
            )

            score = np.exp(segments_sorted_values) * segments_sorted_freq
            factor_score = np.add.reduceat(score, split_ind) / counts

            # destroy factors without segments
            mask = np.isin(self.factors_in_use, factors_with_segments, invert=True)
            factors_without_segments = self.factors_in_use[mask]

            for factor in factors_without_segments:
                self.factor_connections.destroySegment(factor)
                self.factor_vars[factor] = np.full(self.n_vars_per_factor, fill_value=-1)

            self.factors_in_use = factors_with_segments.copy()
        else:
            factors_with_segments = np.empty(0)
            factor_score = np.empty(0)

        self.factor_score = factor_score.copy()

        new_segments = list()

        # each cell corresponds to one variable
        for cell in new_segment_cells:
            n_segments = self.connections.numSegments(cell)

            # this condition is usually loose,
            # so it's just a placeholder for extreme cases
            if n_segments >= self.max_segments_per_cell:
                continue

            # get factors for cell
            var = cell // self.n_hidden_states
            cell_factors = np.array(
                self.factor_connections.segmentsForCell(var)
            )

            score = np.zeros(self.factors_per_var)
            factors = np.full(self.factors_per_var, fill_value=-1)

            if len(cell_factors) > 0:
                mask = np.isin(factors_with_segments, cell_factors)

                score[:len(cell_factors)] = factor_score[mask]
                factors[:len(cell_factors)] = factors_with_segments[mask]

            factor_id = self._rng.choice(
                factors,
                size=1,
                p=softmax(score)
            )

            if factor_id != -1:
                variables = self.factor_vars[factor_id]
            else:
                # select cells for a new factor
                h_vars = np.arange(self.n_hidden_vars + self.n_external_vars)
                var_score = self.var_score.copy()

                used_vars, counts = np.unique(
                    self.factor_vars[self.factors_in_use].flatten(),
                    return_counts=True
                )

                var_score[used_vars] *= np.exp(-self.unused_vars_boost * counts)
                var_score[h_vars >= self.n_hidden_vars] += self.external_vars_boost

                # sample size can't be smaller than number of variables
                sample_size = min(self.n_vars_per_factor, len(h_vars))

                if sample_size == 0:
                    return np.empty(0, dtype=UINT_DTYPE)

                variables = self._rng.choice(
                    h_vars,
                    size=sample_size,
                    p=softmax(var_score),
                    replace=False
                )

                factor_id = self.factor_connections.createSegment(
                    var,
                    maxSegmentsPerCell=self.factors_per_var
                )

                self.factor_connections.growSynapses(
                    factor_id,
                    variables,
                    0.6,
                    self._legacy_rng,
                    maxNew=self.n_vars_per_factor
                )

                self.factor_vars[factor_id] = variables
                self.factors_in_use = np.append(self.factors_in_use, factor_id)

            candidates = self._filter_cells_by_vars(growth_candidates, variables)

            # don't create a segment that will never activate
            if len(candidates) < self.segment_activation_threshold:
                continue

            new_segment = self.connections.createSegment(cell, self.max_segments_per_cell)

            self.connections.growSynapses(
                new_segment,
                candidates,
                0.6,
                self._legacy_rng,
                maxNew=self.n_vars_per_factor
            )

            self.factor_for_segment[new_segment] = factor_id
            self.log_factor_values_per_segment[new_segment] = self.initial_factor_value
            self.receptive_fields[new_segment] = candidates

            new_segments.append(new_segment)

        return np.array(new_segments, dtype=UINT_DTYPE)
