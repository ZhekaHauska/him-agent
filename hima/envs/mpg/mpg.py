#  Copyright (c) 2022 Autonomous Non-Profit Organization "Artificial Intelligence Research
#  Institute" (AIRI); Moscow Institute of Physics and Technology (National Research University).
#  All rights reserved.
#
#  Licensed under the AGPLv3 license. See LICENSE in the project root for license information.

import numpy as np
import pygraphviz as pgv

class MarkovProcessGrammar:
    def __init__(
            self,
            transition_probs,
            transition_letters,
            alphabet,
            initial_state,
            autoreset=False,
            seed=None
    ):
        transition_probs = np.array(transition_probs, dtype=np.float)
        norm = transition_probs.sum(axis=-1).reshape(-1, 1)
        norm_transition_probs = np.divide(
            transition_probs, norm,
            where=(norm != 0.0),
            out=np.zeros_like(transition_probs, dtype=np.float)
        )

        self.terminal_states = np.flatnonzero(norm == 0)

        self.transition_probs = norm_transition_probs
        self.transition_letters = transition_letters
        self.alphabet = alphabet
        self.char_to_num = {x: i for i, x in enumerate(alphabet)}

        self.states = np.arange(transition_probs.shape[0])
        self.initial_state = initial_state

        self.current_state = initial_state
        self.autoreset = autoreset

        self.letter_probs = self.init_letter_probs()

        self.rng = np.random.default_rng(seed)

    def init_letter_probs(self):
        letter_probs = np.zeros((len(self.states), len(self.alphabet)))
        for state in self.states:
            for i, letter in enumerate(self.transition_letters[state]):
                if letter != 0:
                    letter_probs[state, self.char_to_num[letter]] = self.transition_probs[
                        state, i]
        return letter_probs

    def set_current_state(self, state):
        self.current_state = state

    def reset(self):
        self.current_state = self.initial_state

    def next_state(self):
        if np.any(self.current_state == self.terminal_states):
            if self.autoreset:
                self.reset()
            else:
                return None

        transition_dist = self.transition_probs[self.current_state]

        new_state = self.rng.choice(self.states, p=transition_dist)

        letter = self.transition_letters[self.current_state][new_state]

        self.current_state = new_state

        return letter

    def predict_states(self, from_state=None, steps=0):
        if from_state is None:
            from_state = self.current_state
        return np.linalg.matrix_power(self.transition_probs, steps)[from_state]

    def predict_letters(self, from_state=None, steps=0):
        if from_state is None:
            from_state = self.current_state
        states_probs = np.linalg.matrix_power(self.transition_probs, steps)
        return np.dot(states_probs, self.letter_probs)[from_state]


class MultiMarkovProcessGrammar(MarkovProcessGrammar):
    def __init__(
            self,
            policy_transition_probs,
            policy_transition_letters,
            alphabet,
            initial_state,
            initial_policy,
            autoreset=False,
            seed=None):

        super(MultiMarkovProcessGrammar, self).__init__(
            policy_transition_probs[initial_policy],
            policy_transition_letters[initial_policy],
            alphabet,
            initial_state,
            autoreset,
            seed
        )

        self.initial_policy = initial_policy
        self.current_policy = initial_policy

        transition_probs = np.array(policy_transition_probs, dtype=np.float)
        norm = transition_probs.sum(axis=-1)[:, :, None]
        norm_transition_probs = np.divide(
            transition_probs, norm,
            where=(norm != 0),
            out=np.zeros_like(transition_probs)
        )
        self.policy_transition_probs = norm_transition_probs
        self.policy_transition_letters = policy_transition_letters

    def set_policy(self, policy_id):
        self.current_policy = policy_id
        self.transition_probs = self.policy_transition_probs[policy_id]
        self.transition_letters = self.policy_transition_letters[policy_id]
        self.letter_probs = self.init_letter_probs()

    def reset(self):
        super(MultiMarkovProcessGrammar, self).reset()
        self.current_policy = self.initial_policy
        self.set_policy(self.initial_policy)


def draw_mpg(path, transition_probs, transition_letters):
    g = pgv.AGraph(strict=False, directed=True)
    for i in range(transition_probs.shape[0]):
        for j in range(transition_probs.shape[1]):
            prob = transition_probs[i][j]
            letter = transition_letters[i][j]
            if prob > 0:
                g.add_edge(i, j, label=f'{letter}:{round(prob, 2)}')

    g.layout(prog='dot')
    g.draw(path)

