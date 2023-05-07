#  Copyright (c) 2022 Autonomous Non-Profit Organization "Artificial Intelligence Research
#  Institute" (AIRI); Moscow Institute of Physics and Technology (National Research University).
#  All rights reserved.
#
#  Licensed under the AGPLv3 license. See LICENSE in the project root for license information.
from hima.modules.htm.connections import Connections
from htm.bindings.sdr import SDR
from htm.bindings.math import Random

import numpy as np
import pygraphviz as pgv
import warnings
import seaborn as sns
import matplotlib.pyplot as plt

EPS = 1e-24
INT_TYPE = "int64"
UINT_DTYPE = "uint32"
REAL_DTYPE = "float32"
REAL64_DTYPE = "float64"
_TIE_BREAKER_FACTOR = 1e-24


def softmax(x, beta=1.0):
    e_x = np.exp(beta * (x - x.mean()))
    return e_x / e_x.sum()


def normalize(x, default_values=None):
    norm = x.sum(axis=-1)
    mask = norm == 0

    if default_values is None:
        default_values = np.ones_like(x)

    x[mask] = default_values[mask]
    norm[mask] = x[mask].sum(axis=-1)
    return x / norm.reshape((-1, 1))


class DCHMM:
    def __init__(
            self,
            n_obs_vars: int,
            n_obs_states: int,
            cells_per_column: int,
            n_vars_per_factor: int,
            factors_per_var: int,
            lr: float = 1.0,
            factor_boost_scale: float = 10,
            factor_boost_decay: float = 0.01,
            predicted_cell_boost_factor: float = 1.0,
            prediction_inverse_temp: float = 1.0,
            initial_factor_value: float = 0,
            cell_activation_threshold: float = EPS,
            max_segments_per_cell: int = 255,
            segment_prune_threshold: float = 0.001,
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

        self.n_hidden_states = cells_per_column * n_obs_states
        self.total_cells = self.n_hidden_vars * self.n_hidden_states

        self.input_sdr_size = n_obs_vars * n_obs_states
        self.cells_per_column = cells_per_column
        self.max_segments_per_cell = max_segments_per_cell

        self.total_segments = (
                self.n_hidden_states * self.max_segments_per_cell
        ) * self.n_hidden_vars

        self.lr = lr
        self.factors_per_var = factors_per_var
        self.factor_boost_scale = factor_boost_scale
        self.factor_boost_decay = factor_boost_decay
        self.total_factors = self.n_hidden_vars * self.factors_per_var

        self.prediction_inverse_temp = prediction_inverse_temp
        self.predicted_cell_boost_factor = predicted_cell_boost_factor

        self.n_columns = self.n_obs_vars * self.n_obs_states

        # number of variables assigned to a segment
        self.n_vars_per_factor = n_vars_per_factor

        self.segment_prune_threshold = segment_prune_threshold

        # for now leave it strict
        self.segment_activation_threshold = n_vars_per_factor

        # low probability clipping
        self.cell_activation_threshold = cell_activation_threshold

        self.active_cells = SDR(self.total_cells)
        self.active_cells.sparse = np.arange(self.n_hidden_vars) * self.n_hidden_states

        self.predicted_cells = SDR(self.total_cells)

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
            numCells=self.n_hidden_vars,
            connectedThreshold=0.5
        )

        self.segments_in_use = np.empty(0, dtype=UINT_DTYPE)
        self.factors_in_use = np.empty(0, dtype=UINT_DTYPE)
        self.factors_boost = np.empty(0, dtype=REAL_DTYPE)
        self.factors_score = np.empty(0, dtype=REAL_DTYPE)

        self.factor_vars = np.full(
            (self.total_factors, self.n_vars_per_factor),
            fill_value=-1,
            dtype=INT_TYPE
        )

        self.factors_for_var = np.full(
            (self.n_hidden_vars, self.factors_per_var),
            fill_value=-1,
            dtype=INT_TYPE
        )

    def reset(self):
        self.active_cells.sparse = np.arange(self.n_hidden_vars) * self.n_hidden_states

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

        num_connected_segment = self.connections.computeActivity(
            active_cells,
            False
        )

        active_segments = np.flatnonzero(
            num_connected_segment >= self.segment_activation_threshold
        )
        cells_for_active_segments = self.connections.mapSegmentsToCells(active_segments)
        self.predicted_cells.sparse = np.unique(cells_for_active_segments)

        log_prediction = np.full(
            self.total_cells,
            fill_value=-np.inf,
            dtype=REAL_DTYPE
        )

        # deviation activity
        if len(active_segments) > 0:
            factors_in_use = np.unique(self.factor_for_segment[active_segments])
            vars_for_factors = self.factor_connections.mapSegmentsToCells(factors_in_use)
            cells_for_factor_vars = self._get_cells_in_vars(vars_for_factors)

            factors_for_active_segments = self.factor_for_segment[active_segments]
            log_factor_value = self.log_factor_values_per_segment[active_segments]

            likelihood = self.forward_messages[self.receptive_fields[active_segments]]
            log_likelihood = np.sum(np.log(likelihood), axis=-1)

            log_excitation_per_segment = log_likelihood + log_factor_value

            # uniquely encode pairs (factor, cell) for each segment
            cell_factor_id_per_segment = (
                    factors_for_active_segments * self.total_cells
                    + cells_for_active_segments
            )

            # group segments by factors
            sorting_inxs = np.argsort(cell_factor_id_per_segment)
            cell_factor_id_per_segment = cell_factor_id_per_segment[sorting_inxs]
            log_excitation_per_segment = log_excitation_per_segment[sorting_inxs]

            cell_factor_id_excitation, reduce_inxs = np.unique(
                cell_factor_id_per_segment, return_index=True
            )

            # approximate log sum with max
            log_excitation_per_factor = np.maximum.reduceat(log_excitation_per_segment, reduce_inxs)

            sort_inxs = np.argsort(cells_for_factor_vars)
            log_excitation = log_excitation_per_factor[sort_inxs]
            cells_for_factor_vars = cells_for_factor_vars[sort_inxs]

            cells_with_factors, reduce_inxs = np.unique(cells_for_factor_vars, return_index=True)

            log_prediction_for_cells_with_factors = np.add.reduceat(
                log_excitation, indices=reduce_inxs
            )

            log_prediction[cells_with_factors] = log_prediction_for_cells_with_factors

        log_prediction = log_prediction.reshape((self.n_hidden_vars, self.n_hidden_states))

        # shift log value for stability
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', category=RuntimeWarning)

            means = log_prediction.mean(
                axis=-1,
                where=~np.isinf(log_prediction)
            ).reshape((-1, 1))

        means[np.isnan(means)] = 0

        log_prediction -= means

        log_prediction = self.prediction_inverse_temp * log_prediction

        prediction = normalize(np.exp(log_prediction))

        prediction = prediction.flatten()

        assert ~np.any(np.isnan(prediction))

        self.next_forward_messages = prediction

        self.prediction = prediction.copy()

    def predict_columns(self):
        assert self.prediction is not None

        prediction = self.prediction.reshape((self.n_columns, self.cells_per_column))
        return prediction.sum(axis=-1)

    def observe(self, observation: np.ndarray, learn: bool = True):
        assert self.next_forward_messages is not None

        cells = self._get_cells_for_observation(observation)
        obs_factor = np.zeros_like(self.forward_messages)
        obs_factor[cells] = 1

        self.next_forward_messages *= obs_factor

        with warnings.catch_warnings():
            warnings.filterwarnings('error')
            self.next_forward_messages = normalize(
                self.next_forward_messages.reshape((self.n_hidden_vars, -1)),
                obs_factor.reshape((self.n_hidden_vars, -1))
            ).flatten()

        next_active_cells = self._sample_cells(
            cells
        )

        if learn and (len(self.active_cells.sparse) > 0):
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

            self.segments_in_use = np.append(
                self.segments_in_use,
                new_segments[np.isin(new_segments, self.segments_in_use, invert=True)]
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

            filter_destroyed_segments = np.isin(
                self.segments_in_use, segments_to_prune, invert=True
            )
            self.segments_in_use = self.segments_in_use[filter_destroyed_segments]

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
        ] += np.log1p(self.lr/np.exp(w))

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
        # boost cells with active segments
        prediction = self.prediction.copy()
        prediction[self.predicted_cells.sparse] *= self.predicted_cell_boost_factor
        prediction = normalize(prediction.reshape((self.n_hidden_vars, self.n_hidden_states)))

        # sample predicted distribution
        next_states = self._sample_categorical_variables(
            prediction
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
        # TODO add pruning or rewriting of inefficient factors
        # sum factor values for every factor
        if len(self.segments_in_use) > 0:
            factor_for_segment = self.factor_for_segment[self.segments_in_use]
            log_factor_values = self.log_factor_values_per_segment[self.segments_in_use]

            sort_ind = np.argsort(factor_for_segment)
            factors_sorted = factor_for_segment[sort_ind]
            segments_sorted = log_factor_values[sort_ind]

            factors_with_segments, split_ind, counts = np.unique(
                factors_sorted,
                return_index=True,
                return_counts=True
            )

            factor_score = self.factor_boost_scale * self.factors_boost
            mask = np.isin(self.factors_in_use, factors_with_segments)
            factor_eff = np.add.reduceat(segments_sorted, split_ind) / counts
            factor_score[mask] += factor_eff
        else:
            factor_score = np.empty(0)

        self.factor_score = factor_score

        new_segments = list()

        # each cell corresponds to one variable
        for cell in new_segment_cells:
            # get factors for cell
            var = cell // self.n_hidden_states
            cell_factors = np.array(
                self.factor_connections.segmentsForCell(var)
            )

            score = np.zeros(self.factors_per_var)
            factors = np.full(self.factors_per_var, fill_value=-1)

            if len(cell_factors) > 0:
                score[:len(cell_factors)] = factor_score[cell_factors]
                factors[:len(cell_factors)] = cell_factors

            factor_id = self._rng.choice(
                factors,
                size=1,
                p=softmax(score)
            )

            if factor_id != -1:
                variables = self.factor_vars[factor_id]
                self.factors_boost[self.factors_in_use == factor_id] *= (
                        1 - self.factor_boost_decay
                )
            else:
                # select cells for a new factor
                h_vars = np.arange(self.n_hidden_vars)
                var_score = np.zeros_like(h_vars)

                used_vars, counts = np.unique(
                    self.factor_vars[self.factors_in_use].flatten(),
                    return_counts=True
                )

                var_score[used_vars] = -counts

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
                self.factors_boost = np.append(self.factors_boost, 1)

            candidates = self._filter_cells_by_vars(growth_candidates, variables)
            # TODO it may rewrite segments randomly, but we want to do it wisely
            #   so, check if we reached the limit of segments and delete a segment with
            #   least log factor value
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

    def draw_factor_graph(self, path):
        g = pgv.AGraph(strict=False, directed=False)
        for fid in self.factors_in_use:
            var_next = self.factor_connections.cellForSegment(fid)
            g.add_node(f'f{fid}', shape='box')
            g.add_edge(f'v{var_next}(t+1)', f'f{fid}')
            for var_prev in self.factor_vars[fid]:
                g.add_edge(f'f{fid}', f'v{var_prev}(t)',)
        g.layout(prog='dot')
        g.draw(path)

    def draw_messages(self):
        fig, ax = plt.subplots(2, 1)
        sns.heatmap(
            self.forward_messages.reshape((self.n_hidden_vars, -1)),
            ax=ax[0]
        )
        sns.heatmap(
            self.next_forward_messages.reshape((self.n_hidden_vars, -1)),
            ax=ax[1]
        )
        plt.show()
