#  Copyright (c) 2022 Autonomous Non-Profit Organization "Artificial Intelligence Research
#  Institute" (AIRI); Moscow Institute of Physics and Technology (National Research University).
#  All rights reserved.
#
#  Licensed under the AGPLv3 license. See LICENSE in the project root for license information.

import numpy as np
from htm.bindings.sdr import SDR
from htm.bindings.algorithms import TemporalMemory
from itertools import product
import json
from hima.common.sdr import SparseSdr


class Memory:
    """
    Memory saves state representations. It compares new state with stored ones. If the new is
    included more than 'threshold' in some of stored states, than it replaces the old one. In
    the other case new state replace the least using state. Also every 'del_step' time step
    using statistics of one random state is set to 0.

    Parameters
    ----------
    seed : int
        The seed for random generator.
    size : int
        The number of bits represented state. Should be fixed.
    threshold : float
        The threshold is used to determine the corresponding stored state.
    memory_size : int
        The maximum amount of stored states.
    del_step : int
        How often one of the states is deleted.

    Attributes
    ----------
    size : int
        It stores 'size' parameter.
    states : np.array
        The array stores state representations. Shape: ('memory_size', 'size').
    visits : np.array
        The array determines the frequency of the corresponding state. Shape: ('memory_size')
    threshold : float
        It stores 'threshold' parameter.
    del_step : int
        It stores 'del_step' parameter.
    time : int
        The number of recordings to memory.
    """

    def __init__(self, seed: int, size: int, threshold: float, memory_size: int, del_step: int):
        self._rng = np.random.default_rng(seed)
        self.states = np.zeros((memory_size, size), dtype=int)
        self.visits = np.zeros(memory_size, dtype=int)
        self.threshold = threshold
        self.size = size
        self.time = 0
        self.del_step = del_step

    @property
    def stored_states(self):
        return len(np.flatnonzero(self.visits))

    def add(self, state: SparseSdr):
        """ Add a new state representation to memory.

        Parameters
        ----------
        state: SparseSdr
            Representation of new state should be stored.
        """
        self.time += 1
        if self.time % self.del_step == 0:
            probs = 1 - self.visits.astype(float)/np.max(self.visits)
            probs = probs / np.sum(probs)
            del_ind = self._rng.choice(len(probs), p=probs)
            self.visits[del_ind] = 0

        stored_inds = np.flatnonzero(self.visits)
        max_inclusion = 0
        ins = -1
        for ind in stored_inds:
            inclusion = len(np.intersect1d(state, self.states[ind])) / self.size
            if inclusion > self.threshold and inclusion > max_inclusion:
                max_inclusion = inclusion
                ins = ind
                if inclusion == 1:
                    break
        if ins < 0:
            ins = np.argmin(self.visits)
            self.visits[ins] = 0
        self.states[ins] = state
        self.visits[ins] += 1

    def get_n_components(self, superposition: SparseSdr) -> int:
        """Calculates the number of separate states in given superposition.

        Parameters
        ----------
        superposition: SparseSdr
            Superposition representation of a set of states.

        Returns
        -------
        int:
            The number of states in the superposition.
        """
        stored_inds = np.flatnonzero(self.visits)
        n_states = 0
        for ind in stored_inds:
            inclusion = len(np.intersect1d(superposition, self.states[ind])) / self.size
            if inclusion > self.threshold:
                n_states += 1
        return n_states


class Empowerment:
    """
    The Empowerment contains Temporal Memory algorithm as the model of an environment to evaluate
    a deterministic form of empowerment. It is possible to improve evaluation by using simple form
    of pattern memory: Memory, that stores states and helps to separate a superposition of states.

    Parameters
    ----------
    seed : int
        The seed for random generator.
    encode_size : int
        The size of SDR representations which is taken by model.
    tm_config : dict
        It contains all parameters for initialisation of the TemporalMemory without the
        'columnDimensions'. 'columnDimensions' is defined inside Empowerment.
    sparsity : float
        The sparsity of SDR representations which are used in the TemporalMemory algorithm.
    memory : bool, optional
        Will be used the Memory for storing states or not. By default is False.
    similarity_threshold : float, optional
        The threshold for distinguishing states. It is used then memory is True. By default: 0.6.
    evaluate : bool, optional
        This flag defines the necessity of storing some statistics to evaluate the learning process.
        By default is True.
    filename : str, optional
        Defines the path to the file for saved values of empowerment. By default: None.
    memory_size : int, optional
        Defines the memory size, then Memory is used. By default: 100.
    memory_clean_step : int, optional
        Defines how often Memory is cleaned. By default: 100.

    Attributes
    ----------
    filename : str or None
        Stores the same parameter.
    evaluate : bool
        Stores the same parameter.
    anomalies : list[float]
        Stores the anomaly values of TM for each time step after learning.
        Only then 'evaluate' is True.
    IoU : list[float]
        Stores the Intersection over Union values of TM predictions and real ones for each
        time step after learning. Only then 'evaluate' is True.
    sparsity : float
        Stores the same parameter.
    tm : TemporalMemory
        Contains the TemporalMemory object.
    size : int
        Stores the 'encode_size' parameter.
    memory : Memory
        Contains the Memory object if 'memory' parameter is True, else None.
    empowerment_data : dict[tuple[int, int], float]
        Contains empowerment values reading from 'filename', only if 'filename' is defined.
    """

    def __init__(
            self, seed: int, encode_size: int, tm_config: dict, sparsity: float,
            memory: bool = False, similarity_threshold: float = 0.6, evaluate: bool = True,
            filename: str = None, memory_size: int = 100, memory_clean_step: int = 100):
        self.filename = filename
        if self.filename is None:
            self.evaluate = evaluate
            if evaluate:
                self.anomalies = []
                self.IoU = []
            self.sdr_0 = SDR(encode_size)
            self.sdr_1 = SDR(encode_size)
            self.sparsity = sparsity

            self.tm = TemporalMemory(
                columnDimensions=[encode_size], seed=seed, **tm_config,
            )
            self.size = self.tm.getColumnDimensions()[0]

            if memory:
                self.memory = Memory(
                    seed, int(self.size*self.sparsity), similarity_threshold,
                    memory_size, memory_clean_step
                )
            else:
                self.memory = None
        else:
            with open(self.filename) as json_file:
                self.empowerment_data = json.load(json_file)

    def eval_from_file(self, position):
        return self.empowerment_data[str(position[0])][str(position[1])]

    def eval_state(
            self, state: SparseSdr, horizon: int
    ) -> float:
        """This function evaluates empowerment for given state.

        Parameters
        ----------
        state : SparseSdr
            The SDR representation (sparse) of the state.
        horizon : int
            The horizon of evaluating for given state. The good value is 3.

        Returns
        -------
        float
            The empowerment value (always > 0).
        """
        superposition = state.copy()
        for _ in range(horizon):
            superposition = self.predict(superposition)

        num_predicted_states = len(superposition) / (self.sparsity * self.size)
        if self.memory:
            num_predicted_states = self.memory.get_n_components(superposition)
        if num_predicted_states < 1:
            # the case of no predictions
            num_predicted_states = 1
        return np.log2(num_predicted_states)

    def predict(self, state: SparseSdr) -> SparseSdr:
        """This function predicts next states for given state.

        Parameters
        ----------
        state : SparseSdr
            The SDR representation (sparse) of the state.

        Returns
        -------
        SparseSdr
            The superposition of predicted states.
        """
        self.sdr_0.sparse = state
        self.tm.reset()
        self.tm.compute(self.sdr_0, learn=False)
        self.tm.activateDendrites(learn=False)
        predictive_cells = self.tm.getPredictiveCells().sparse
        predicted_columns = [self.tm.columnForCell(i) for i in predictive_cells]
        prediction = np.unique(predicted_columns)
        return prediction

    def learn(self, state_0: SparseSdr, state_1: SparseSdr):
        """This function realizes learning of TM.

        Parameters
        ----------
        state_0 : SparseSdr
            The SDR representation of the state (sparse form).
        state_1 : SparseSdr
            The SDR representation of the next state (sparse form).
        """
        self.sdr_0.sparse = state_0
        self.sdr_1.sparse = state_1
        self.tm.reset()

        if self.memory is not None:
            self.memory.add(state_0)
            self.memory.add(state_1)
        self.tm.compute(self.sdr_0, learn=True)

        if self.evaluate:
            self.tm.activateDendrites(learn=False)
            predictiveCells = self.tm.getPredictiveCells().sparse
            predictedColumnIndices = np.unique([self.tm.columnForCell(i) for i in predictiveCells])

        self.tm.compute(self.sdr_1, learn=True)
        if self.evaluate:
            intersection = np.intersect1d(self.sdr_1.sparse, predictedColumnIndices)
            union = np.union1d(self.sdr_1.sparse, predictedColumnIndices)
            self.IoU.append(len(intersection) / len(union))
            self.anomalies.append(self.tm.anomaly)
        self.tm.reset()


def real_empowerment(env, position, horizon):
    data = np.zeros(env.env.shape)

    for actions in product(range(4), repeat=horizon):
        env.env.agent.position = position
        for a in actions:
            env.act(a)

        data[env.env.agent.position] += 1
    return np.sum(-data / data.sum() * np.log(data / data.sum(), where=data != 0), where=data != 0), data

