#  Copyright (c) 2022 Autonomous Non-Profit Organization "Artificial Intelligence Research
#  Institute" (AIRI); Moscow Institute of Physics and Technology (National Research University).
#  All rights reserved.
#
#  Licensed under the AGPLv3 license. See LICENSE in the project root for license information.
from hima.modules.htm.connections import Connections
from htm.bindings.sdr import SDR
from htm.bindings.math import Random

import numpy as np
from scipy.stats import entropy

EPS = 1e-24
INT_TYPE = "int32"
UINT_DTYPE = "uint32"
REAL_DTYPE = "float32"
REAL64_DTYPE = "float64"
_TIE_BREAKER_FACTOR = 1e-24


def softmax(x, beta=1.0):
    e_x = np.exp(beta * (x - x.mean()))
    return e_x / e_x.sum()


def normalize(x):
    norm = x.sum(axis=-1)
    mask = norm == 0
    x[mask] = 1
    norm[mask] = x.shape[-1]
    return x / norm.reshape((-1, 1))


class DCHMM:
    def __init__(
            self,
            n_obs_vars: int,
            n_obs_states: int,
            shape: tuple[int, int],
            cells_per_column: int,
            use_off_states: bool,
            n_vars_per_factor: int,
            factors_per_var: int,
            factor_activation_threshold: int,
            initial_factor_value: float = 0,
            initial_alpha_value: float = 0,
            lr: float = 0.01,
            lr_off: float = 0.01,
            beta: float = 0.0,
            gamma: float = 0.1,
            punishment: float = 0.0,
            punishment_off: float = 0.0,
            cell_activation_threshold: float = EPS,
            max_segments_per_cell: int = 255,
            max_segments_for_off_state: int = 1000,
            segment_prune_threshold: float = 0.001,
            allow_synapses_to_off_states: bool = False,
            seed: int = None,
    ):
        self._rng = np.random.default_rng(seed)

        if seed:
            self._legacy_rng = Random(seed)
        else:
            self._legacy_rng = Random()

        self.n_obs_vars = n_obs_vars
        self.n_hidden_vars = n_obs_vars
        self.shape = shape

        self.n_obs_states = n_obs_states

        self.use_off_states = use_off_states

        self.n_hidden_states = cells_per_column * n_obs_states + int(self.use_off_states)
        self.total_cells = self.n_hidden_vars * self.n_hidden_states

        if self.use_off_states:
            self.n_off_states = 1
            self.off_states = (np.arange(self.n_hidden_vars) + 1) * self.n_hidden_states - 1
        else:
            self.n_off_states = 0
            self.off_states = np.empty(0, dtype=UINT_DTYPE)

        self.filter_off_states_mask = np.ones(self.total_cells, dtype=bool)
        self.filter_off_states_mask[self.off_states] = False

        self.input_sdr_size = n_obs_vars * n_obs_states
        self.cells_per_column = cells_per_column
        self.max_segments_per_cell = max_segments_per_cell
        self.max_segments_for_off_state = max_segments_for_off_state

        self.total_segments = (
                (self.n_hidden_states - self.n_off_states) * self.max_segments_per_cell +
                self.n_off_states * self.max_segments_for_off_state
        ) * self.n_hidden_vars

        self.factors_per_var = factors_per_var
        self.total_factors = self.n_hidden_vars * self.factors_per_var

        self.n_columns = self.n_obs_vars * self.n_obs_states

        # number of variables assigned to a segment
        self.n_vars_per_factor = n_vars_per_factor

        self.segment_prune_threshold = segment_prune_threshold

        self.allow_synapses_to_off_states = allow_synapses_to_off_states

        # for now leave it strict
        self.segment_activation_threshold = n_vars_per_factor

        self.lr = lr
        self.beta = beta
        self.gamma = gamma
        self.punishment = punishment

        self.lr_off = lr_off
        self.punishment_off = punishment_off

        # low probability clipping
        self.cell_activation_threshold = cell_activation_threshold

        self.active_cells = SDR(self.total_cells)
        # set all variables to the reset state
        self.active_cells.sparse = self.off_states

        self.forward_messages = np.zeros(
            self.total_cells,
            dtype=REAL64_DTYPE
        )
        self.forward_messages[self.active_cells.sparse] = 1

        self.next_forward_messages = None
        self.prediction = None

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

        self.factor_for_segment = np.zeros(
            self.total_segments,
            dtype=UINT_DTYPE
        )

        self.receptive_fields = np.zeros(
            (self.total_segments, self.n_vars_per_factor),
            dtype=UINT_DTYPE
        )

        # treat factors as segments
        self.factor_connections = Connections(
            numCells=self.n_hidden_vars,
            connectedThreshold=0.5
        )

        self.factor_activation_threshold = factor_activation_threshold

        self.factor_vars = np.zeros(
            (self.total_factors, self.n_vars_per_factor),
            dtype=UINT_DTYPE
        )

        self.factors_for_var = np.zeros(
            (self.n_hidden_vars, self.factors_per_var),
            dtype=UINT_DTYPE
        )

        self.initial_alpha_value = initial_alpha_value
        self.alpha = np.full(
            self.n_hidden_vars,
            fill_value=self.initial_alpha_value,
            dtype=REAL64_DTYPE
        )

    def reset(self):
        self.active_cells.sparse = self.off_states

        self.forward_messages = np.zeros(
            self.total_cells,
            dtype=REAL64_DTYPE
        )

        self.forward_messages[self.active_cells.sparse] = 1
        self.next_forward_messages = None
        self.prediction = None

    def predict_cells(self):
        # filter dendrites that have low activation likelihood
        active_cells = SDR(self.total_cells)
        active_cells.sparse = np.flatnonzero(
            self.forward_messages >= self.cell_activation_threshold
        )

        if not self.allow_synapses_to_off_states:
            active_cells.sparse = active_cells.sparse[
                np.isin(active_cells.sparse, self.off_states, invert=True)
            ]

        num_connected_segment = self.connections.computeActivity(
            active_cells,
            False
        )

        active_segments = np.flatnonzero(num_connected_segment >= self.segment_activation_threshold)
        cells_for_active_segments = self.connections.mapSegmentsToCells(active_segments)

        active_factors = np.unique(self.factor_for_segment[active_segments])

        if len(active_factors) > 0:
            # base activity level
            message_norm = self.forward_messages.reshape(
                (self.n_hidden_vars, self.n_hidden_states)
            ).sum(axis=-1)

            base = message_norm[self.factor_vars[active_factors]]
            # base per factor
            log_base = np.sum(np.log(base), axis=-1)
            vars_for_factors = self.factor_connections.mapSegmentsToCells(
                active_factors
            )
            cells_for_factor_vars = self._get_cells_in_vars(vars_for_factors)
            log_base_for_cells = np.repeat(log_base, self.n_hidden_states)
            factors_for_cells = np.repeat(active_factors, self.n_hidden_states)

            # uniquely encode pairs (factor, cell)
            cell_factor_id_per_cell = (
                factors_for_cells * self.total_cells + cells_for_factor_vars
            )

            # deviation activity
            if len(active_segments) > 0:
                factors_for_active_segments = self.factor_for_segment[active_segments]
                shifted_factor_value = np.expm1(
                    self.log_factor_values_per_segment[active_segments]
                )

                likelihood = self.forward_messages[self.receptive_fields[active_segments]]
                log_likelihood = np.sum(np.log(likelihood), axis=-1)

                # advantage per segment
                log_advantage = log_likelihood + np.log(shifted_factor_value)

                # uniquely encode pairs (factor, cell) for each segment
                cell_factor_id_per_segment = (
                        factors_for_active_segments * self.total_cells
                        + cells_for_active_segments
                )

                # group segments by factors
                sorting_inxs = np.argsort(cell_factor_id_per_segment)
                cell_factor_id_per_segment = cell_factor_id_per_segment[sorting_inxs]
                log_advantage = log_advantage[sorting_inxs]

                cell_factor_id_deviation, reduce_inxs = np.unique(
                    cell_factor_id_per_segment, return_index=True
                )

                # deviation per factor and cell
                # approximate log sum with max
                log_deviation = np.maximum.reduceat(log_advantage, reduce_inxs)

                deviation_mask = np.isin(cell_factor_id_per_cell, cell_factor_id_deviation)
                log_base_for_cells[deviation_mask] = np.logaddexp(
                    log_base_for_cells[deviation_mask], log_deviation
                )

            sort_inxs = np.argsort(cells_for_factor_vars)
            log_base_for_cells = log_base_for_cells[sort_inxs]
            cells_for_factor_vars = cells_for_factor_vars[sort_inxs]

            cells_with_factors, reduce_inxs = np.unique(cells_for_factor_vars, return_index=True)

            log_prediction_for_cells_with_factors = np.add.reduceat(
                log_base_for_cells, indices=reduce_inxs
            )

            log_prediction = np.zeros(self.total_cells)

            log_prediction[cells_with_factors] = log_prediction_for_cells_with_factors
        else:
            log_prediction = np.zeros(self.total_cells)

        log_prediction = log_prediction.reshape((self.n_hidden_vars, self.n_hidden_states))

        # rescale
        log_prediction -= log_prediction.min(axis=-1).reshape((-1, 1))

        prediction = normalize(np.exp(log_prediction))

        # boost off states
        if self.use_off_states:
            prediction_entropy = entropy(
                prediction[:, :-1],
                axis=-1
            )

            prediction[:, -1] *= (1 + self.alpha * np.exp(prediction_entropy))

            prediction = normalize(prediction)

        prediction = prediction.flatten()

        assert ~np.any(np.isnan(prediction))

        self.next_forward_messages = prediction

        self.prediction = prediction.copy()

    def predict_columns(self):
        assert self.prediction is not None

        prediction = self.prediction[self.filter_off_states_mask]
        prediction = prediction.reshape((self.n_columns, self.cells_per_column))
        return prediction.sum(axis=-1)

    def observe(self, observation: np.ndarray, learn: bool = True):
        assert self.next_forward_messages is not None

        cells = self._get_cells_for_observation(observation)
        obs_factor = np.zeros_like(self.forward_messages)
        obs_factor[cells] = 1

        self.next_forward_messages *= obs_factor
        self.next_forward_messages = normalize(
            self.next_forward_messages.reshape((self.n_hidden_vars, -1))
        ).flatten()

        if learn:
            next_active_cells = self._sample_cells(
                cells
            )

            (
                segments_to_reinforce,
                segments_to_punish,
                cells_to_grow_new_segments
            ) = self._calculate_learning_segments(
                self.active_cells.sparse,
                next_active_cells
            )

            # adapt off states
            if self.use_off_states:
                off_states_to_reinforce = np.isin(self.off_states, next_active_cells)
                p = self.forward_messages[self.off_states[off_states_to_reinforce]]
                self.alpha[off_states_to_reinforce] += (1 - p) * self.lr_off
                self.alpha[~off_states_to_reinforce] -= self.punishment_off
                np.clip(self.alpha, a_min=0, a_max=None, out=self.alpha)

            new_segments = self._grow_new_segments(
                cells_to_grow_new_segments,
                self.active_cells.sparse
            )

            segments_to_prune = self._update_factors(
                np.concatenate(
                    [
                        segments_to_reinforce,
                        new_segments
                    ]
                ),
                segments_to_punish
            )

            for segment in segments_to_prune:
                self.connections.destroySegment(segment)

            self.active_cells.sparse = next_active_cells

        self.forward_messages = self.next_forward_messages

    def _calculate_learning_segments(self, prev_active_cells, next_active_cells):
        # determine which segments are learning and growing
        active_cells = SDR(self.total_cells)
        active_cells.sparse = prev_active_cells

        num_connected = self.connections.computeActivity(
            active_cells,
            False
        )

        active_segments = np.flatnonzero(num_connected >= self.segment_activation_threshold)

        cells_for_active_segments = self.connections.mapSegmentsToCells(active_segments)

        mask = np.isin(cells_for_active_segments, next_active_cells)
        segments_to_learn = active_segments[mask]
        segments_to_punish = active_segments[~mask]

        cells_to_grow_new_segments = next_active_cells[
            ~np.isin(next_active_cells, cells_for_active_segments)
        ]

        if self.max_segments_for_off_state == 0:
            cells_to_grow_new_segments = cells_to_grow_new_segments[
                np.isin(cells_to_grow_new_segments, self.off_states, invert=True)
            ]

        return (
            segments_to_learn.astype(UINT_DTYPE),
            segments_to_punish.astype(UINT_DTYPE),
            cells_to_grow_new_segments.astype(UINT_DTYPE)
        )

    def _update_factors(self, segments_to_reinforce, segments_to_punish):
        cells_for_segments_reinforce = self.connections.mapSegmentsToCells(segments_to_reinforce)
        cells_for_segments_punish = self.connections.mapSegmentsToCells(segments_to_punish)

        w = self.log_factor_values_per_segment[segments_to_reinforce]
        self.log_factor_values_per_segment[
            segments_to_reinforce
        ] += self.lr * (
                1 - self.prediction[cells_for_segments_reinforce]
        ) * np.exp(-self.gamma*w)

        self.log_factor_values_per_segment[
            segments_to_punish
        ] -= self.punishment * self.prediction[cells_for_segments_punish]

        self.log_factor_values_per_segment[segments_to_punish] = (
            np.clip(
                self.log_factor_values_per_segment[segments_to_punish],
                a_min=0.0,
                a_max=None
            )
        )

        segments = np.concatenate(
                [
                    segments_to_punish,
                    segments_to_reinforce
                ]
            )

        w = self.log_factor_values_per_segment[segments]

        segments_to_prune = segments[np.abs(w) < self.segment_prune_threshold]

        return segments_to_prune

    def _get_cells_for_observation(self, obs_states):
        vars_for_obs_states = obs_states // self.n_obs_states
        cells_in_columns = (
                (
                        obs_states * self.cells_per_column +
                        self.n_off_states * vars_for_obs_states
                ).reshape((-1, 1)) +
                np.arange(self.cells_per_column, dtype=UINT_DTYPE)
            ).flatten()

        if self.use_off_states:

            vars_without_states = ~np.isin(np.arange(self.n_obs_vars), vars_for_obs_states)

            empty_states = self.off_states[vars_without_states]

            cells = np.concatenate([empty_states, cells_in_columns])
        else:
            cells = cells_in_columns

        return cells

    def _get_cells_in_vars(self, variables):
        cells_in_vars = (
                (variables * self.n_hidden_states).reshape((-1, 1)) +
                np.arange(self.n_hidden_states, dtype=UINT_DTYPE)
        ).flatten()

        return cells_in_vars

    def _filter_cells_by_vars(self, cells, variables):
        cells_in_vars = self._get_cells_in_vars(variables)

        mask = np.isin(cells, cells_in_vars)

        return cells[mask]

    def _sample_cells(self, cells_for_obs):
        # sample predicted distribution
        next_states = self._sample_categorical_variables(
            self.prediction.reshape((self.n_hidden_vars, self.n_hidden_states))
        )
        # transform states to cell ids
        next_cells = next_states + np.arange(
            0,
            self.n_hidden_states*self.n_hidden_vars,
            self.n_hidden_states
        )

        wrong_predictions = ~np.isin(next_cells, cells_for_obs)
        wrong_predicted_vars = (
                next_cells[wrong_predictions] // self.n_hidden_states
        ).astype(UINT_DTYPE)

        # resample cells for wrong predictions
        new_forward_message = self.next_forward_messages.reshape(
            (self.n_hidden_vars, self.n_hidden_states)
        )[wrong_predicted_vars]

        new_forward_message /= new_forward_message.sum(axis=-1).reshape(-1, 1)

        next_states2 = self._sample_categorical_variables(
            new_forward_message
        )
        # transform states to cell ids
        next_cells2 = (
                next_states2 + wrong_predicted_vars * self.n_hidden_states
        )
        # replace wrong predicted cells with resampled
        next_cells[wrong_predictions] = next_cells2

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
        # determine factor activity
        active_vars = SDR(self.n_hidden_vars)

        active_cells = growth_candidates[
            np.isin(
                growth_candidates, self.off_states, invert=True
            )
        ]

        active_vars.sparse = active_cells // self.n_hidden_states
        num_active_vars_per_factor = self.factor_connections.computeActivity(
            active_vars, False
        )

        new_segments = list()

        for cell in new_segment_cells:
            # get factors for cell
            var = cell // self.n_hidden_states
            cell_factors = np.array(
                self.factor_connections.segmentsForCell(var)
            )

            # sample factors by their activity
            # TODO score factors by their predictive ability, remove factor if it cause errors a lot
            if len(cell_factors) > 0:
                num_active = num_active_vars_per_factor[cell_factors]
                active_factors_mask = num_active >= self.factor_activation_threshold
                active_factors = cell_factors[active_factors_mask]
                num_active = num_active[active_factors_mask]
            else:
                active_factors = []

            if len(active_factors) > 0:
                factor_id = self._rng.choice(active_factors, size=1, p=softmax(num_active))
                variables = self.factor_vars[factor_id]
            else:
                # select cells for a new factor
                # TODO check factor intersections
                h_vars = np.arange(self.n_hidden_vars)
                rows = h_vars // self.shape[1]
                cols = h_vars % self.shape[1]

                row = var // self.shape[1]
                col = var % self.shape[1]
                distance = np.abs(rows - row) + np.abs(cols - col)

                # beta doesn't have effect if synapses to off states are not allowed
                score = - distance + self.beta * active_vars.dense

                if not self.allow_synapses_to_off_states:
                    h_vars = h_vars[active_vars.sparse]
                    score = score[active_vars.sparse]

                # sample size can't be smaller than number of variables
                sample_size = min(self.n_vars_per_factor, len(h_vars))

                if sample_size == 0:
                    return np.empty(0, dtype=UINT_DTYPE)

                variables = self._rng.choice(
                    h_vars,
                    size=sample_size,
                    p=softmax(score),
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

            candidates = self._filter_cells_by_vars(growth_candidates, variables)

            if self.filter_off_states_mask[cell]:
                max_segments = self.max_segments_per_cell
            else:
                max_segments = self.max_segments_for_off_state

            new_segment = self.connections.createSegment(cell, max_segments)
            self.factor_for_segment[new_segment] = factor_id

            new_segments.append(new_segment)

            self.connections.growSynapses(
                new_segment,
                candidates,
                0.6,
                self._legacy_rng,
                maxNew=self.n_vars_per_factor
            )

            self.receptive_fields[new_segment] = candidates

        return np.array(new_segments, dtype=UINT_DTYPE)
