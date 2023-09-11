#  Copyright (c) 2022 Autonomous Non-Profit Organization "Artificial Intelligence Research
#  Institute" (AIRI); Moscow Institute of Physics and Technology (National Research University).
#  All rights reserved.
#
#  Licensed under the AGPLv3 license. See LICENSE in the project root for license information.
import numpy as np
from typing import Literal, Optional
from hima.common.utils import softmax
from hmmlearn.hmm import CategoricalHMM
from copy import copy
from hima.modules.baselines.cscg import CHMM

import warnings
from hima.modules.belief.utils import normalize, sample_categorical_variables
from hima.modules.belief.utils import EPS, UINT_DTYPE, REAL64_DTYPE


L_MODE = Literal['bw', 'bw_base', 'bw_iter', 'htm']
INI_MODE = Literal['dirichlet', 'normal', 'uniform']


class CHMMBasic:
    def __init__(
            self,
            n_columns: int,
            cells_per_column: int,
            lr: float = 0.1,
            temp: float = 1.0,
            gamma: float = 0.0,
            punishment: float = 0.0,
            batch_size: int = 1,
            learning_mode: L_MODE = 'htm',
            initialization: INI_MODE = 'uniform',
            sigma: float = 1.0,
            alpha: float = 1.0,
            seed: Optional[int] = None
    ):
        self.n_columns = n_columns
        self.cells_per_column = cells_per_column
        self.n_states = cells_per_column * n_columns
        self.states = np.arange(self.n_states)
        self.lr = lr
        self.temp = temp
        self.gamma = gamma
        self.punishment = punishment
        self.batch_size = batch_size
        self.learning_mode = learning_mode
        self.initialization = initialization
        self.is_first = True
        self.seed = seed

        self._rng = np.random.default_rng(self.seed)

        if self.initialization == 'dirichlet':
            self.transition_probs = self._rng.dirichlet(
                alpha=[alpha]*self.n_states,
                size=self.n_states
            )
            self.state_prior = self._rng.dirichlet(alpha=[alpha]*self.n_states)
        elif self.initialization == 'normal':
            self.log_transition_factors = self._rng.normal(
                scale=sigma,
                size=(self.n_states, self.n_states)
            )
            self.log_state_prior = self._rng.normal(scale=sigma, size=self.n_states)
        elif self.initialization == 'uniform':
            self.log_transition_factors = np.zeros((self.n_states, self.n_states))
            self.log_state_prior = np.zeros(self.n_states)

        if self.initialization != 'dirichlet':
            self.transition_probs = np.vstack(
                [softmax(x, temp=self.temp) for x in self.log_transition_factors]
            )

            self.state_prior = softmax(self.log_state_prior, temp=self.temp)
        else:
            self.log_transition_factors = np.log(self.transition_probs)
            self.log_state_prior = np.log(self.state_prior)

        self.forward_message = None

        self.stats_trans_mat = self.transition_probs
        self.stats_state_prior = self.state_prior

        self.active_state = None
        self.prediction = None
        self.observations = list()
        self.obs_sequences = list()
        self.fm_sequences = list()
        self.forward_messages = list()

        if self.learning_mode == 'bw_base':
            self.model = CategoricalHMM(
                n_features=self.n_columns,
                n_components=self.n_states,
                params='st',
                init_params='',
                random_state=self.seed
            )

            emission_probs = list()
            for i in range(self.n_columns):
                p = np.zeros(self.n_states)
                p[i*self.cells_per_column: (i+1)*self.cells_per_column] = 1
                emission_probs.append(p)

            self.model.startprob_ = self.state_prior
            self.model.transmat_ = self.transition_probs
            self.model.emissionprob_ = np.vstack(emission_probs).T
        else:
            self.model = None

    def observe(self, observation_state: int, learn: bool = True) -> None:
        assert 0 <= observation_state < self.n_columns, "Invalid observation state."
        assert self.prediction is not None, "Run predict_columns() first."

        states_for_obs = self._obs_state_to_hidden(observation_state)

        obs_factor = np.zeros(self.n_states)
        obs_factor[states_for_obs] = 1

        new_forward_message = self.prediction * obs_factor

        if learn:
            if self.learning_mode == 'htm':
                self._htm_like_learning(new_forward_message, states_for_obs)
            elif self.learning_mode == 'bw_iter':
                self._iterative_baum_welch_learning(obs_factor)
            elif (self.learning_mode == 'bw') or (self.learning_mode == 'bw_base'):
                if self.is_first and (len(self.observations) > 0):
                    self.obs_sequences.append(copy(self.observations))
                    self.fm_sequences.append(copy(self.forward_messages))

                    if len(self.obs_sequences) == self.batch_size:
                        if self.learning_mode == 'bw':
                            self._baum_welch_learning()
                        else:
                            self._baum_welch_base_learning()

                        self.obs_sequences.clear()
                        self.fm_sequences.clear()

                    self.observations.clear()
                    self.forward_messages.clear()
            else:
                raise ValueError

        if self.is_first:
            self.is_first = False

        self.forward_message = new_forward_message
        self.forward_messages.append(new_forward_message)
        self.observations.append(observation_state)

    def predict_columns(self):
        if self.is_first:
            self.prediction = self.state_prior
        else:
            self.prediction = np.dot(self.forward_message, self.transition_probs)
        prediction = self.prediction / self.prediction.sum()
        prediction = np.reshape(prediction, (self.n_columns, self.cells_per_column))
        prediction = prediction.sum(axis=-1)
        return prediction

    def reset(self):
        self.forward_message = None
        self.active_state = None
        self.prediction = None
        self.is_first = True

    def _obs_state_to_hidden(self, obs_state):
        return np.arange(
            self.cells_per_column * obs_state,
            self.cells_per_column * (obs_state + 1),
        )

    def _baum_welch_learning(self):
        new_prior_stats = np.zeros_like(self.state_prior)
        new_transition_stats = np.zeros_like(self.transition_probs)

        for observations, forward_messages in zip(self.obs_sequences, self.fm_sequences):
            backward_message = np.ones(self.n_states)
            backward_messages = [backward_message]

            for observation in observations[1:][::-1]:
                states_for_obs = self._obs_state_to_hidden(observation)
                obs_factor = np.zeros(self.n_states)
                obs_factor[states_for_obs] = 1

                backward_message = np.dot(backward_message * obs_factor, self.transition_probs.T)
                backward_messages.append(copy(backward_message))

            # priors
            states_for_obs = self._obs_state_to_hidden(observations[0])
            obs_factor = np.zeros(self.n_states)
            obs_factor[states_for_obs] = 1

            posterior = self.state_prior * obs_factor * backward_messages[-1]
            posterior /= posterior.sum()

            new_prior_stats += posterior

            # transitions
            backward_messages = backward_messages[::-1]

            for i, forward_message in enumerate(forward_messages[:-1]):
                states_for_obs = self._obs_state_to_hidden(observations[i+1])
                obs_factor = np.zeros(self.n_states)
                obs_factor[states_for_obs] = 1

                forward_message = forward_message.reshape((-1, 1))
                backward_message = backward_messages[i+1].reshape((1, -1))
                obs_factor = obs_factor.reshape((1, -1))

                posterior = forward_message * self.transition_probs * obs_factor * backward_message
                posterior /= posterior.sum()
                new_transition_stats += posterior

        # update
        self.stats_trans_mat += self.lr * (new_transition_stats - self.stats_trans_mat)

        self.stats_state_prior += self.lr * (new_prior_stats - self.stats_state_prior)

        self.transition_probs = self.stats_trans_mat / self.stats_trans_mat.sum(axis=1).reshape((-1, 1))
        self.state_prior = self.stats_state_prior / self.stats_state_prior.sum()

    def _baum_welch_base_learning(self):
        self.model.fit(
            np.array([x for y in self.obs_sequences for x in y]).reshape((-1, 1)),
            [len(x) for x in self.obs_sequences]
        )

        new_transition_matrix = self.model.transmat_
        new_priors = self.model.startprob_

        # update
        self.stats_trans_mat += self.lr * (new_transition_matrix - self.stats_trans_mat)

        self.stats_state_prior += self.lr * (new_priors - self.stats_state_prior)

        self.transition_probs = self.stats_trans_mat / self.stats_trans_mat.sum(axis=1).reshape((-1, 1))
        self.state_prior = self.stats_state_prior / self.stats_state_prior.sum()

        self.model.transmat_ = self.transition_probs
        self.model.startprob_ = self.state_prior

    def _iterative_baum_welch_learning(self, obs_factor):
        if self.is_first:
            posterior = self.state_prior * obs_factor
            posterior /= posterior.sum()
            self.stats_state_prior += posterior
            self.state_prior = self.stats_state_prior / self.stats_state_prior.sum()
        else:
            posterior = self.forward_message.reshape((-1, 1)) * self.transition_probs * obs_factor.reshape((1, -1))
            posterior /= posterior.sum()
            self.stats_trans_mat += posterior
            self.transition_probs = self.stats_trans_mat / self.stats_trans_mat.sum(axis=1).reshape((-1, 1))

    def _htm_like_learning(self, new_forward_message, states_for_obs):
        prev_state = self.active_state

        prediction = self.prediction / self.prediction.sum()
        predicted_state = self._rng.choice(self.states, p=prediction)

        wrong_prediction = not np.in1d(predicted_state, states_for_obs)

        if wrong_prediction:
            new_forward_message /= new_forward_message.sum()
            next_state = self._rng.choice(self.states, p=new_forward_message)
        else:
            next_state = predicted_state

        if self.is_first:
            w = self.log_state_prior[next_state]
            self.log_state_prior[next_state] += self.lr * np.exp(-self.gamma*w) * (
                    1 - prediction[next_state]
            )

            if wrong_prediction:
                w = self.log_state_prior[predicted_state]
                self.log_state_prior[predicted_state] -= self.punishment * prediction[
                    predicted_state
                ]

            self.state_prior = softmax(self.log_state_prior, temp=self.temp)
        else:
            w = self.log_transition_factors[prev_state, next_state]

            self.log_transition_factors[prev_state, next_state] += (
                    self.lr * np.exp(-self.gamma*w) * (
                        1 - prediction[next_state]
                    )
            )

            if wrong_prediction:
                w = self.log_transition_factors[prev_state, predicted_state]

                self.log_transition_factors[prev_state, predicted_state] -= (
                        self.punishment * prediction[predicted_state]
                )

            self.transition_probs = np.vstack(
                [softmax(x, temp=self.temp) for x in self.log_transition_factors]
            )

        self.active_state = next_state


class HMM:
    def __init__(
            self,
            n_hidden_states: int,
            n_obs_states: int,
            learn_every: int,
            n_iter: int,
            tol: float,
            seed: Optional[int] = None
    ):
        self.n_obs_states = n_obs_states
        self.learn_every = learn_every
        self.n_hidden_states = n_hidden_states
        self.states = np.arange(self.n_hidden_states)
        self.is_first = True
        self.seed = seed

        self._rng = np.random.default_rng(self.seed)

        self.forward_message = None
        self.episode = 0
        self.observations = list()
        self.obs_sequences = list()

        self.model = CategoricalHMM(
            n_features=self.n_obs_states,
            n_components=self.n_hidden_states,
            n_iter=n_iter,
            tol=tol,
            random_state=self.seed,
        )

        self.state_prior = np.ones(self.n_hidden_states) / self.n_hidden_states
        self.transition_probs = normalize(np.ones((self.n_hidden_states, self.n_hidden_states)))
        self.emission_probs = normalize(np.ones((self.n_hidden_states, self.n_obs_states)))

    def observe(self, observation_state: int, learn: bool = True) -> None:
        assert 0 <= observation_state < self.n_obs_states, "Invalid observation state."
        assert self.forward_message is not None, "Run predict_columns() first."

        self.forward_message *= self.emission_probs[:, observation_state]

        if self.is_first:
            self.is_first = False

            if len(self.observations) > 0:
                self.obs_sequences.append(copy(self.observations))
                self.observations.clear()
                self.episode += 1

            if learn and ((self.episode % self.learn_every) == 0) and (len(self.obs_sequences) > 0):
                self._baum_welch_base_learning()

        self.observations.append(observation_state)

    def predict_columns(self):
        if self.forward_message is None:
            self.forward_message = self.state_prior
        else:
            self.forward_message = np.dot(self.forward_message, self.transition_probs)

        if np.isclose(self.forward_message.sum(), 0):
            self.forward_message = normalize(self.forward_message.reshape((1, -1))).flatten()

        prediction = np.dot(self.forward_message, self.emission_probs)
        prediction = normalize(prediction.reshape((1, -1))).flatten()
        return prediction

    def reset(self):
        self.forward_message = None
        self.is_first = True

    def _baum_welch_base_learning(self):
        self.model.fit(
            np.array([x for y in self.obs_sequences for x in y]).reshape((-1, 1)),
            [len(x) for x in self.obs_sequences]
        )

        self.transition_probs = self.model.transmat_.copy()
        self.state_prior = self.model.startprob_.copy()
        self.emission_probs = self.model.emissionprob_.copy()


class CHMMLayer:
    def __init__(
            self,
            n_obs_states: int,
            cells_per_column: int,
            batch_size: int = 100,
            n_context_states: int = 0,
            n_external_states: int = 0,
            alpha: float = 1.0,
            lr: float = 1.0,
            em_iterations: int = 100,
            seed: int = None,
    ):
        self._rng = np.random.default_rng(seed)

        self.timestep = 1
        self.n_obs_vars = 1
        self.n_hidden_vars = 1
        self.n_obs_states = n_obs_states
        self.n_external_vars = 1
        self.n_external_states = n_external_states
        self.n_context_vars = 1
        self.n_context_states = n_context_states
        self.batch_size = batch_size
        self.em_iterations = em_iterations
        self.lr = lr

        self.cells_per_column = cells_per_column
        self.n_hidden_states = cells_per_column * n_obs_states

        self.internal_cells = self.n_hidden_vars * self.n_hidden_states
        self.external_input_size = self.n_external_vars * self.n_external_states
        self.context_input_size = self.n_context_vars * self.n_context_states

        self.input_sdr_size = n_obs_states

        # trainable parameters
        self.transition_probs = self._rng.dirichlet(
            alpha=[alpha] * self.n_hidden_states * self.n_external_states,
            size=self.n_hidden_states
        ).reshape((self.n_external_states, self.n_hidden_states, self.n_hidden_states))
        self.state_prior = self._rng.dirichlet(alpha=[alpha] * self.n_hidden_states)

        self.transition_stats = np.zeros_like(self.transition_probs)
        self.state_prior_stats = np.zeros_like(self.state_prior)

        self.n_columns = self.n_obs_states

        self.internal_forward_messages = np.zeros(
            self.internal_cells,
            dtype=REAL64_DTYPE
        )
        self.external_messages = np.zeros(
            self.external_input_size,
            dtype=REAL64_DTYPE
        )
        self.context_messages = np.zeros(
            self.context_input_size,
            dtype=REAL64_DTYPE
        )

        self.prediction_cells = None
        self.prediction_columns = None

        self.observations = []
        self.actions = []

        # cells are numbered in the following order:
        # internal cells | context cells | external cells
        self.internal_cells_range = (
            0,
            self.internal_cells
        )
        self.context_cells_range = (
            self.internal_cells_range[1],
            self.internal_cells_range[1] + self.context_input_size
        )
        self.external_cells_range = (
            self.context_cells_range[1],
            self.context_cells_range[1] + self.external_input_size
        )

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

    def make_state_snapshot(self):
        return (
            # mutable attributes:
            self.internal_forward_messages.copy(),
            # immutable attributes:
            self.external_messages,
            self.context_messages,
            self.prediction_cells,
            self.prediction_columns
        )

    def restore_last_snapshot(self, snapshot):
        if snapshot is None:
            return

        (
            self.internal_forward_messages,
            self.external_messages,
            self.context_messages,
            self.prediction_cells,
            self.prediction_columns
        ) = snapshot

        # explicitly copy mutable attributes:
        self.internal_forward_messages = self.internal_forward_messages.copy()

    def reset(self):
        self.internal_forward_messages = np.zeros(
            self.internal_cells,
            dtype=REAL64_DTYPE
        )
        self.external_messages = np.zeros(
            self.external_input_size,
            dtype=REAL64_DTYPE
        )
        self.context_messages = np.zeros(
            self.context_input_size,
            dtype=REAL64_DTYPE
        )

        self.prediction_cells = None
        self.prediction_columns = None

    def predict(self, include_context_connections=True, include_internal_connections=False, **_):
        if self.context_messages.size == 0:
            self.internal_forward_messages = self.state_prior.copy()
        else:
            trans_probs_action = np.sum(
                self.transition_probs * self.external_messages.reshape((-1, 1, 1)),
                axis=0
            )
            self.internal_forward_messages = np.dot(self.context_messages, trans_probs_action)
            self.internal_forward_messages = normalize(
                self.internal_forward_messages.reshape(
                    (self.n_hidden_vars, self.n_hidden_states)
                )
            ).flatten()

        self.prediction_cells = self.internal_forward_messages.copy()
        self.prediction_columns = self.prediction_cells.reshape(
            (self.n_columns, self.cells_per_column)
        ).sum(axis=-1)

    def observe(
            self,
            observation: np.ndarray,
            learn: bool = True
    ):
        # TODO handle missing observations
        if len(observation) == 0:
            return

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

        if learn:
            self.observations.append(observation[0])
            if self.external_messages.size != 0:
                self.actions.append(
                    sample_categorical_variables(
                        self.external_messages.reshape((1, -1)),
                        self._rng
                    )[0]
                )
            else:
                self.actions.append(-1)

            if (self.timestep % self.batch_size) == 0:
                x = np.array(self.observations, dtype=np.int64)
                self.actions.append(-1)
                a = np.array(self.actions[1:], dtype=np.int64)

                chmm = CHMM(
                    np.full(self.n_obs_states, fill_value=self.cells_per_column),
                    x,
                    a,
                    pseudocount=EPS,
                    dtype=REAL64_DTYPE,
                    seed=self._rng.integers(np.iinfo(np.int32).max)
                )
                chmm.T = self.transition_probs
                chmm.Pi_x = self.state_prior
                chmm.learn_em_T_Pi_x(x, a, n_iter=self.em_iterations, term_early=False)

                self.transition_stats += self.lr * (chmm.T.copy() - self.transition_stats)
                self.state_prior_stats += self.lr * (chmm.Pi_x.copy() - self.state_prior_stats)

                self.transition_probs = self.transition_stats / self.transition_stats.sum(
                    axis=2
                )[:, :, None]
                self.state_prior = self.state_prior_stats / self.state_prior_stats.sum()

                self.observations.clear()
                self.actions.clear()

        self.timestep += 1

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
        internal_vars_mask = variables < self.n_hidden_vars
        context_vars_mask = (
                (variables >= self.n_hidden_vars) &
                (variables < (self.n_hidden_vars + self.n_context_vars))
        )
        external_vars_mask = (
                variables >= (self.n_hidden_vars + self.n_context_vars)
        )

        cells_in_internal_vars = (
                (variables[internal_vars_mask] * self.n_hidden_states).reshape((-1, 1)) +
                np.arange(self.n_hidden_states, dtype=UINT_DTYPE)
        ).flatten()

        cells_in_context_vars = (
                ((variables[context_vars_mask] - self.n_hidden_vars) *
                 self.n_context_states).reshape((-1, 1)) +
                np.arange(self.n_context_states, dtype=UINT_DTYPE)
        ).flatten() + self.context_cells_range[0]

        cells_in_external_vars = (
                ((variables[external_vars_mask] - self.n_hidden_vars - self.n_context_vars) *
                 self.n_external_states).reshape((-1, 1)) +
                np.arange(self.n_external_states, dtype=UINT_DTYPE)
        ).flatten() + self.external_cells_range[0]

        return np.concatenate(
            [
                cells_in_internal_vars,
                cells_in_context_vars,
                cells_in_external_vars
            ]
        )
