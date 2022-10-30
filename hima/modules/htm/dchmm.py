#  Copyright (c) 2022 Autonomous Non-Profit Organization "Artificial Intelligence Research
#  Institute" (AIRI); Moscow Institute of Physics and Technology (National Research University).
#  All rights reserved.
#
#  Licensed under the AGPLv3 license. See LICENSE in the project root for license information.
from hima.modules.htm.connections import Connections
from htm.bindings.sdr import SDR
from htm.bindings.math import Random
import numpy as np

EPS = 1e-24
INT_TYPE = "int32"
UINT_DTYPE = "uint32"
REAL_DTYPE = "float32"
REAL64_DTYPE = "float64"
_TIE_BREAKER_FACTOR = 1e-24


class DCHMM:
    def __init__(
            self,
            n_obs_vars: int,
            n_obs_states: int,
            cells_per_column: int,
            n_vars_per_factor: int,
            initial_log_factor_value: float = 0,
            lr: float = 0.01,
            gamma: float = 1.0,
            regularization: float = 0.01,
            punishment: float = 0.0,
            cell_activation_threshold: float = EPS,
            max_segments_per_cell: int = 255,
            segment_prune_threshold: float = 0.001,
            loop_sequence: bool = False,
            seed: int = None,
    ):
        self._rng = np.random.default_rng(seed)

        if seed:
            self._legacy_rng = Random(seed)
        else:
            self._legacy_rng = Random()

        self.n_obs_vars = n_obs_vars
        self.n_hidden_vars = n_obs_vars

        self.n_obs_states = n_obs_states

        # plus reset state
        self.n_hidden_states = cells_per_column*n_obs_states + 1

        self.loop_sequence = loop_sequence

        if self.loop_sequence:
            self.n_spec_states = 1
            self.reset_states = (np.arange(self.n_hidden_vars) + 1) * self.n_hidden_states - 1
            self.terminal_states = np.empty(0, dtype=UINT_DTYPE)
        else:
            self.n_spec_states = 2
            self.n_hidden_states += 1  # plus terminal state
            self.reset_states = (np.arange(self.n_hidden_vars) + 1) * self.n_hidden_states - 2
            self.terminal_states = (np.arange(self.n_hidden_vars) + 1) * self.n_hidden_states - 1

        self.input_sdr_size = n_obs_vars * n_obs_states
        self.cells_per_column = cells_per_column
        self.max_segments_per_cell = max_segments_per_cell
        self.total_cells = self.n_hidden_vars * self.n_hidden_states
        self.total_segments = self.total_cells * self.max_segments_per_cell
        self.n_columns = self.n_obs_vars * self.n_obs_states

        self.filter_reset_states_mask = np.ones(self.total_cells, dtype=bool)
        self.filter_reset_states_mask[self.reset_states] = False

        self.filter_terminal_states_mask = np.ones(self.total_cells, dtype=bool)
        self.filter_terminal_states_mask[self.terminal_states] = False

        # number of variables assigned to a segment
        self.n_vars_per_factor = n_vars_per_factor

        # this makes things much easier
        assert (self.n_hidden_vars % self.n_vars_per_factor) == 0

        self.max_factors = self.n_hidden_vars // self.n_vars_per_factor

        self.segment_prune_threshold = segment_prune_threshold

        # for now leave it strict
        self.segment_activation_threshold = n_vars_per_factor

        self.lr = lr
        self.gamma = gamma
        self.regularization = regularization
        self.punishment = punishment

        # low probability clipping
        self.cell_activation_threshold = cell_activation_threshold

        self.active_cells = SDR(self.total_cells)
        # set all variables to the reset state
        self.active_cells.sparse = self.reset_states

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
        self.initial_log_factor_value = initial_log_factor_value
        self.log_factor_values_per_segment = np.full(
            self.total_segments,
            fill_value=self.initial_log_factor_value,
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

        # to each factor corresponds unique combination of observation variables
        # for now just randomly distribute variables among factors
        # TODO make an adaptive partition
        self.factor_vars = np.arange(
                self.n_hidden_vars,
                dtype=UINT_DTYPE
            )
        self._rng.shuffle(
            self.factor_vars
        )
        self.factor_vars = self.factor_vars.reshape(
            (self.max_factors, n_vars_per_factor)
        )

    def reset(self):
        self.active_cells.sparse = self.reset_states

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

        num_connected = self.connections.computeActivity(
            active_cells,
            False
        )

        active_segments = np.flatnonzero(num_connected >= self.segment_activation_threshold)
        cells_for_active_segments = self.connections.mapSegmentsToCells(active_segments)

        # base activity level
        message_norm = self.forward_messages.reshape(
            (self.n_hidden_vars, self.n_hidden_states)
        ).sum(axis=-1)
        # group by factors, it is the same for all cells
        base = message_norm[self.factor_vars]
        base = np.prod(base, axis=-1)

        prediction = np.tile(base, (self.total_cells, 1))

        # deviation activity
        if len(active_segments) > 0:
            # group segments by cells
            sorting_inxs = np.argsort(cells_for_active_segments)

            segments_in_use = active_segments[sorting_inxs]
            cell_for_segment = cells_for_active_segments[sorting_inxs]

            factor_for_segment = self.factor_for_segment[segments_in_use]
            shifted_factor_value = np.exp(
                self.log_factor_values_per_segment[segments_in_use]
            ) - 1

            likelihood = self.forward_messages[self.receptive_fields[segments_in_use]]
            likelihood = np.prod(likelihood, axis=-1)
            likelihood *= shifted_factor_value

            factor_for_segment_spec = cell_for_segment * self.max_factors + factor_for_segment

            # group segments by factors
            sorting_inxs = np.argsort(factor_for_segment_spec)
            factor_for_segment_spec = factor_for_segment_spec[sorting_inxs]
            likelihood = likelihood[sorting_inxs]

            factors_spec, split_inxs = np.unique(factor_for_segment_spec, return_index=True)

            deviation = np.add.reduceat(likelihood, split_inxs)

            prediction = prediction.flatten()
            prediction[factors_spec] += deviation
            prediction = prediction.reshape((self.total_cells, -1))

        prediction = prediction.prod(axis=-1)

        if not self.loop_sequence:
            # prevent reset states prediction
            prediction[self.reset_states] = 0

        prediction = prediction.reshape((self.n_hidden_vars, self.n_hidden_states))
        prediction /= prediction.sum(axis=-1).reshape((-1, 1))
        prediction = prediction.flatten()
        self.next_forward_messages = prediction.copy()

        prediction = prediction.reshape((self.n_hidden_vars, self.n_hidden_states))
        prediction /= prediction.sum(axis=-1).reshape((-1, 1))
        self.prediction = prediction.flatten()

    def predict_columns(self):
        assert self.prediction is not None

        filter_special_states = self.filter_reset_states_mask & self.filter_terminal_states_mask
        prediction = self.prediction[filter_special_states]
        prediction = prediction.reshape((self.n_columns, self.cells_per_column))
        return prediction.sum(axis=-1)

    def observe(self, observation: np.ndarray, learn: bool = True):
        assert self.next_forward_messages is not None

        cells = self._get_cells_for_observation(observation)
        obs_factor = np.zeros_like(self.forward_messages)
        obs_factor[cells] = 1

        self.next_forward_messages *= obs_factor

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

        return (
            segments_to_learn.astype(UINT_DTYPE),
            segments_to_punish.astype(UINT_DTYPE),
            cells_to_grow_new_segments.astype(UINT_DTYPE)
        )

    def _update_factors(self, segments_to_reinforce, segments_to_punish):
        w = self.log_factor_values_per_segment[segments_to_reinforce]
        self.log_factor_values_per_segment[
            segments_to_reinforce
        ] += self.lr * (np.exp(-self.gamma*w) - self.regularization * w)

        self.log_factor_values_per_segment[
            segments_to_punish
        ] -= self.punishment

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
                        self.n_spec_states * vars_for_obs_states
                ).reshape((-1, 1)) +
                np.arange(self.cells_per_column, dtype=UINT_DTYPE)
            ).flatten()

        vars_without_states = ~np.isin(np.arange(self.n_obs_vars), vars_for_obs_states)

        if self.loop_sequence:
            empty_states = self.reset_states[vars_without_states]
        else:
            empty_states = self.terminal_states[vars_without_states]

        cells = np.concatenate([empty_states, cells_in_columns])
        return cells

    def _filter_cells_by_vars(self, cells, variables):
        cells_in_vars = (
                    (variables * self.n_hidden_states).reshape((-1, 1)) +
                    np.arange(self.n_hidden_states, dtype=UINT_DTYPE)
                ).flatten()

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
        wrong_predicted_vars = np.flatnonzero(
            wrong_predictions
        )

        # resample cells for wrong predictions
        new_forward_message = self.next_forward_messages.reshape(
            (self.n_hidden_vars, self.n_hidden_states)
        )[wrong_predictions]

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
        new_segments = list()
        for cell in new_segment_cells:
            factor_id = self._rng.integers(self.max_factors)
            variables = self.factor_vars[factor_id]
            candidates = self._filter_cells_by_vars(growth_candidates, variables)

            new_segment = self.connections.createSegment(cell, self.max_segments_per_cell)
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
