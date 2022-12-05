#  Copyright (c) 2022 Autonomous Non-Profit Organization "Artificial Intelligence Research
#  Institute" (AIRI); Moscow Institute of Physics and Technology (National Research University).
#  All rights reserved.
#
#  Licensed under the AGPLv3 license. See LICENSE in the project root for license information.

from typing import Any, Optional, Union

import numpy as np

from hima.common.sdr import SparseSdr
from hima.common.sds import Sds
from hima.experiments.temporal_pooling._depr.metrics import (
    standardize_sample_distribution,
    sequence_similarity_elementwise, distribution_similarity, DISTR_SIM_PMF,
    DISTR_SIM_KL
)


SdrSequence = Optional[list[SparseSdr]]
SeqHistogram = Optional[np.ndarray]


class SimilarityMatrix:
    tag: str

    _raw_mx: Union[np.ndarray, np.ma.MaskedArray]
    _mx: Optional[np.ndarray]

    unbias_func: str
    online: bool

    def __init__(
            self, tag: str, raw_mx: np.ma.MaskedArray, normalization: str, online: bool = False
    ):
        self.tag = tag
        self.online = online
        self.unbias_func = normalization
        self._raw_mx = raw_mx
        self._mx = None

    def on_new_sequence(self, sequence_id: int):
        ...

    def on_step(self, **kwargs):
        ...

    def final_metrics(self) -> dict[str, Any]:
        tag = self.tag
        if self._mx is None:
            self._mx = standardize_sample_distribution(self._raw_mx, normalization=self.unbias_func)

        return {
            f'raw_sim_mx_{tag}': self._raw_mx,
            f'sim_mx_{tag}': self._mx,
        }


class OfflineElementwiseSimilarityMatrix(SimilarityMatrix):
    def __init__(
            self, sequences: list[list],
            normalization: str, discount: float = None, symmetrical: bool = False
    ):
        n = len(sequences)
        diagonal_mask = np.identity(n, dtype=bool)
        sm = np.empty((n, n))

        for i in range(n):
            for j in range(n):
                sm[i, j] = sequence_similarity_elementwise(
                    sequences[i], sequences[j], discount=discount, symmetrical=symmetrical
                )
        raw_mx = np.ma.array(sm, mask=diagonal_mask)

        super(OfflineElementwiseSimilarityMatrix, self).__init__(
            tag='el', raw_mx=raw_mx, normalization=normalization
        )


class OfflinePmfSimilarityMatrix(SimilarityMatrix):
    def __init__(
            self, pmfs: list[np.ndarray],
            normalization: str, algorithm: str, symmetrical: bool = False, sds: Sds = None
    ):
        n = len(pmfs)
        diagonal_mask = np.identity(n, dtype=bool)
        sm = np.empty((n, n))

        for i in range(n):
            for j in range(n):
                sm[i, j] = distribution_similarity(
                    pmfs[i], pmfs[j], algorithm=algorithm, sds=sds, symmetrical=symmetrical
                )
        raw_mx = np.ma.array(sm, mask=diagonal_mask)

        if algorithm == DISTR_SIM_PMF:
            tag = 'pmf'
        elif algorithm == DISTR_SIM_KL:
            tag = '1_nkl'
        else:
            raise KeyError(f'Algorithm {algorithm} is not supported')

        super(OfflinePmfSimilarityMatrix, self).__init__(
            tag=tag, raw_mx=raw_mx, normalization=normalization
        )


class OnlineSimilarityMatrix(SimilarityMatrix):
    symmetrical: Optional[bool]
    discount: Optional[float]

    current_seq_id: Optional[int]
    current_seq: SdrSequence

    def __init__(
            self, tag: str, n_sequences: int, normalization: str, discount: float, symmetrical: bool
    ):
        self.symmetrical = symmetrical
        self.discount = discount

        self.current_seq_id = -1
        self.current_seq = None

        raw_mx = np.empty((n_sequences, n_sequences))
        mask = np.ones_like(raw_mx, dtype=bool)
        raw_mx = np.ma.array(raw_mx, mask=mask)

        super(OnlineSimilarityMatrix, self).__init__(
            tag=tag, raw_mx=raw_mx, normalization=normalization, online=True
        )

    def on_new_sequence(self, sequence_id: int = -1):
        if self.current_seq:
            self._handle_finished_sequence()

        self.current_seq = []
        self.current_seq_id = sequence_id

    def on_step(self, sdr: SparseSdr):
        self.current_seq.append(sdr)

    def _handle_finished_sequence(self):
        ...

    def final_metrics(self) -> dict[str, Any]:
        self.on_new_sequence()

        self._mx = standardize_sample_distribution(self._raw_mx, normalization=self.unbias_func)
        return super(OnlineSimilarityMatrix, self).final_metrics()


class OnlineElementwiseSimilarityMatrix(OnlineSimilarityMatrix):
    sequences: list[SdrSequence]

    def __init__(
            self, n_sequences: int, normalization: str, discount: float, symmetrical: bool
    ):
        super(OnlineElementwiseSimilarityMatrix, self).__init__(
            tag='prfx_el', n_sequences=n_sequences, normalization=normalization,
            discount=discount, symmetrical=symmetrical
        )
        self.sequences = [None] * n_sequences

    def _handle_finished_sequence(self):
        n_seqs = len(self.sequences)
        n_seq_elements = len(self.current_seq)

        self.current_seq = [set(sdr) for sdr in self.current_seq]
        similarity_sum = np.zeros(n_seqs)

        for step in range(n_seq_elements):
            prefix_size = step + 1
            for j in range(n_seqs):
                if self.sequences[j] is None:
                    continue

                similarity_sum[j] += sequence_similarity_elementwise(
                    s1=self.current_seq[:prefix_size],
                    s2=self.sequences[j][:prefix_size],
                    discount=self.discount, symmetrical=self.symmetrical
                )

        # store mean similarity
        mean_similarity = similarity_sum / n_seq_elements
        for j in range(n_seqs):
            if self.sequences[j] is None:
                continue
            self._raw_mx[self.current_seq_id][j] = mean_similarity[j]

        # store sequence; MUST BE stored after similarity calculations to prevent rewriting prev seq
        self.sequences[self.current_seq_id] = self.current_seq


class OnlinePmfSimilarityMatrix(OnlineSimilarityMatrix):
    sds: Sds
    algorithm: str
    pmfs: list[SeqHistogram]

    def __init__(
            self, n_sequences: int, sds: Sds,
            normalization: str, discount: float, symmetrical: bool, algorithm: str
    ):
        self.sds = sds
        self.pmfs = [None] * n_sequences

        self.algorithm = algorithm
        if algorithm == DISTR_SIM_PMF:
            tag = 'pmf'
        elif algorithm == DISTR_SIM_KL:
            tag = '1_nkl'
        else:
            raise KeyError(f'Algorithm {algorithm} is not supported')

        super(OnlinePmfSimilarityMatrix, self).__init__(
            tag=tag, n_sequences=n_sequences, normalization=normalization,
            discount=discount, symmetrical=symmetrical
        )

    def _handle_finished_sequence(self):
        n_pmfs = len(self.pmfs)
        n_seq_elements = len(self.current_seq)

        similarity_sum = np.zeros(n_pmfs)
        prefix_histogram = np.zeros(self.sds.size)
        prefix_histogram_sum = 0

        # calculate similarity for every prefix pmf against all stored pmfs
        prefix_pmf = 1
        for sdr in self.current_seq:
            prefix_histogram_sum = self._update_pmf(
                sdr, histogram=prefix_histogram, histogram_sum=prefix_histogram_sum
            )
            prefix_pmf = prefix_histogram / prefix_histogram_sum

            for j in range(n_pmfs):
                if self.pmfs[j] is None:
                    continue
                similarity_sum[j] += distribution_similarity(
                    p=prefix_pmf, q=self.pmfs[j],
                    algorithm=self.algorithm, sds=self.sds, symmetrical=self.symmetrical
                )

        # store mean similarity for entire `current_seq_id` row
        mean_similarity = similarity_sum / n_seq_elements
        for j in range(n_pmfs):
            if self.pmfs[j] is None:
                continue
            self._raw_mx[self.current_seq_id][j] = mean_similarity[j]

        # store pmf; MUST BE stored after similarity calculations to prevent rewriting prev pmf
        self.pmfs[self.current_seq_id] = prefix_pmf

    def _update_pmf(self, sdr: SparseSdr, histogram: SeqHistogram, histogram_sum: float):
        if self.discount < 1.:
            # discount preserving `hist / step = 1 * active_size`
            histogram *= self.discount
            histogram_sum *= self.discount

        # add new sdr to pmf
        histogram[sdr] += 1
        return histogram_sum + 1
