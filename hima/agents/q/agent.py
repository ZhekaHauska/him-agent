from typing import Optional

import numpy as np
from numpy.random import Generator

from hima.agents.q.eligibility_traces import EligibilityTraces
from hima.agents.q.qvn import QValueNetwork
from hima.agents.q.input_changes_detector import InputChangesDetector
from hima.agents.q.ucb_estimator import UcbEstimator
from hima.common.sdr import SparseSdr
from hima.common.utils import exp_decay, softmax, DecayingValue, isnone


class QAgent:
    n_actions: int
    Q: QValueNetwork
    E_traces: EligibilityTraces
    input_changes_detector: InputChangesDetector

    train: bool
    softmax_temp: DecayingValue
    exploration_eps: DecayingValue
    ucb_estimate: UcbEstimator

    _step: int
    _current_sa_sdr: Optional[SparseSdr]
    _rng: Generator

    def __init__(
            self,
            seed: int,
            qvn: dict,
            n_actions: int,
            n_states: int,
            eligibility_traces: dict = None,
            softmax_temp: DecayingValue = (0., 0.),
            exploration_eps: DecayingValue = (0., 0.),
            ucb_estimate: dict = None,
    ):
        self.n_actions = n_actions
        self.n_states = n_states
        self.total_states = n_actions*n_states
        self.Q = QValueNetwork(self.total_states, seed, **qvn)
        self.E_traces = EligibilityTraces(
            self.total_states,
            **isnone(eligibility_traces, {})
        )
        self.input_changes_detector = InputChangesDetector(self.total_states)

        self.train = True
        self.softmax_temp = softmax_temp
        self.exploration_eps = exploration_eps
        self.ucb_estimate = UcbEstimator(
            self.total_states, **isnone(ucb_estimate, {})
        )

        self._rng = np.random.default_rng(seed)
        self._current_sa_sdr = None
        self._step = 0

    @property
    def name(self):
        return 'q'

    def on_new_episode(self):
        self._step = 0
        self.E_traces.reset()
        if self.train:
            self.Q.decay_learning_factors()
            self.input_changes_detector.reset()
            self.softmax_temp = exp_decay(self.softmax_temp)
            self.exploration_eps = exp_decay(self.exploration_eps)
            if self.ucb_estimate.enabled:
                self.ucb_estimate.decay_learning_factors()

    def act(self, reward: float, state: SparseSdr, first: bool) -> Optional[int]:
        if first and self._step > 0:
            self.on_new_episode()
            return None

        train = self.train
        prev_sa_sdr = self._current_sa_sdr
        input_changed = self.input_changes_detector.changed(state, train)
        s = state
        actions_sa_sdr = self._encode_s_actions(s)

        if train and not first:
            self.E_traces.update(prev_sa_sdr, with_reset=not input_changed)
            self._make_q_learning_step(
                sa=prev_sa_sdr, r=reward, next_actions_sa_sdr=actions_sa_sdr
            )

        action = self._choose_action(actions_sa_sdr)
        chosen_sa_sdr = actions_sa_sdr[action]

        if train and self.ucb_estimate.enabled:
            self.ucb_estimate.update(chosen_sa_sdr)

        self._current_sa_sdr = chosen_sa_sdr
        self._step += 1
        return action

    def _choose_action(self, next_actions_sa_sdr: list[SparseSdr]) -> int:
        if self.softmax_temp[0] > .0:
            # SOFTMAX
            action_values = self.Q.values(next_actions_sa_sdr)
            p = softmax(action_values, temp=self.softmax_temp[0])
            return self._rng.choice(self.n_actions, p=p)

        if self.train and self._should_make_random_action():
            # RND
            return self._rng.integers(self.n_actions)

        if self.train and self.ucb_estimate.enabled:
            # UCB
            action_values = self.Q.values(next_actions_sa_sdr)
            ucb_values = self.ucb_estimate.ucb_terms(next_actions_sa_sdr)
            # noinspection PyTypeChecker
            return np.argmax(action_values + ucb_values)

        # GREEDY
        action_values = self.Q.values(next_actions_sa_sdr)
        greedy_action = np.argmax(action_values)
        # noinspection PyTypeChecker
        return greedy_action

    def _make_q_learning_step(
            self, sa: SparseSdr, r: float, next_actions_sa_sdr: list[SparseSdr]
    ):
        action_values = self.Q.values(next_actions_sa_sdr)
        greedy_action = np.argmax(action_values)
        greedy_sa_sdr = next_actions_sa_sdr[greedy_action]
        self.Q.update(
            sa=sa, reward=r, sa_next=greedy_sa_sdr,
            E_traces=self.E_traces.E
        )

    def _should_make_random_action(self) -> bool:
        if self.exploration_eps[0] < 1e-6:
            # === disabled
            return False
        return self._rng.random() < self.exploration_eps[0]

    def _encode_s_actions(self, s: SparseSdr) -> list[SparseSdr]:
        return [
            [s[0] + self.n_states*action]
            for action in range(self.n_actions)
        ]
