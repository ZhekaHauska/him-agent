#  Copyright (c) 2022 Autonomous Non-Profit Organization "Artificial Intelligence Research
#  Institute" (AIRI); Moscow Institute of Physics and Technology (National Research University).
#  All rights reserved.
#
#  Licensed under the AGPLv3 license. See LICENSE in the project root for license information.

from typing import Union

import numpy as np

from hima.common.sdr import SparseSdr, DenseSdr
from hima.common.sds import Sds
from hima.common.utils import safe_divide, isnone

SEQ_SIM_ELEMENTWISE = 'elementwise'
DISTR_SIM_PMF = 'pmf_pointwise'
DISTR_SIM_KL = 'kl-divergence'

MEAN_STD_NORMALIZATION = 'mean-std'
MIN_MINMAX_NORMALIZATION = 'min-minmax'
NO_NORMALIZATION = 'no'


# ==================== Sdr [sequence] similarity ====================
def dense_similarity(x1: DenseSdr, x2: DenseSdr, symmetrical: bool = False) -> float:
    overlap = np.count_nonzero(x1 == x2)
    if symmetrical:
        union_size = np.count_nonzero(np.logical_or(x1, x2))
        return safe_divide(overlap, union_size)

    return safe_divide(overlap, np.count_nonzero(x2))


def sdr_similarity(x1: set, x2: set, symmetrical: bool = False) -> float:
    assert isinstance(x1, set) and isinstance(x2, set)
    overlap = len(x1 & x2)
    if symmetrical:
        return safe_divide(overlap, len(x1 | x2))
    return safe_divide(overlap, len(x2))


def tuple_similarity(
        t1: tuple[SparseSdr, ...], t2: tuple[SparseSdr, ...], symmetrical=False
) -> float:
    # noinspection PyTypeChecker
    return np.prod([
        sdr_similarity(t1[i], t2[i], symmetrical=symmetrical)
        for i in range(len(t1))
    ])


def sequence_similarity(
        s1: list, s2: list,
        algorithm: str, discount: float = None, symmetrical: bool = False,
        sds: Sds = None
) -> float:
    if algorithm == 'elementwise':
        # reflects strictly ordered similarity
        return sequence_similarity_elementwise(s1, s2, discount=discount, symmetrical=symmetrical)
    elif algorithm.startswith('union'):
        # reflects unordered (=set) similarity
        algorithm = algorithm[6:]
        return sequence_similarity_as_union(
            s1, s2, sds=sds, algorithm=algorithm, symmetrical=symmetrical
        )
    elif algorithm.startswith('prefix'):
        # reflects balance between the other two
        algorithm = algorithm[7:]
        return sequence_similarity_by_prefixes(
            s1, s2, sds=sds, algorithm=algorithm, discount=discount, symmetrical=symmetrical
        )
    else:
        raise KeyError(f'Invalid algorithm: {algorithm}')


def distribution_similarity(
        p: np.ndarray, q: np.ndarray, algorithm: str, sds: Sds = None, symmetrical: bool = False
) -> float:
    if algorithm == 'kl-divergence':
        # We take 1 - KL to make it similarity metric. NB: normalized KL div for SDS can be > 1
        return 1 - kl_divergence(p, q, sds, symmetrical=symmetrical)
    elif algorithm == 'pmf_pointwise':
        return point_pmf_similarity(p, q, sds=sds)
    elif algorithm == 'wasserstein':
        return -wasserstein_distance(p, q, sds=sds)
    else:
        raise KeyError(f'Invalid algorithm: {algorithm}')


def similarity_matrix(
        a: Union[list[set[int]], list[np.ndarray], list[list]],
        algorithm: str = None, discount: float = None, symmetrical: bool = False,
        sds: Sds = None
) -> np.ndarray:
    n = len(a)
    diagonal_mask = np.identity(n, dtype=bool)
    sm = np.empty((n, n))

    if isinstance(a[0], set):
        # SDR representations
        regime = 0
    elif isinstance(a[0], np.ndarray):
        # representations distribution (PMF)
        regime = 1
    elif isinstance(a[0], list):
        # sequences
        regime = 2
    else:
        raise KeyError(f'List of {type(a[0])} is not supported')

    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            x, y = a[i], a[j]

            if regime == 0:
                sim = sdr_similarity(x, y, symmetrical=symmetrical)
            elif regime == 1:
                # noinspection PyTypeChecker
                sim = distribution_similarity(
                    x, y, algorithm=algorithm, sds=sds, symmetrical=symmetrical
                )
            else:
                # noinspection PyTypeChecker
                sim = sequence_similarity(
                    x, y, algorithm=algorithm, discount=discount, symmetrical=symmetrical, sds=sds
                )
            sm[i, j] = sim
    return np.ma.array(sm, mask=diagonal_mask)


# FIXME:
# def rank_matrix(sim_matrix: np.ndarray) -> np.ndarray:
#     n = sim_matrix.shape[0]
#     rm = np.ma.argsort(sim_matrix, axis=None)
#
#
# def normalised_kendall_tau_distance(ranking1, ranking2):
#     """Compute the Kendall tau distance."""
#     n = len(ranking1)
#     i, j = np.meshgrid(np.arange(n), np.arange(n))
#     a = np.ma.argsort(ranking1)
#     b = np.argsort(ranking2)
#     n_disordered = np.logical_or(
#         np.logical_and(a[i] < a[j], b[i] > b[j]),
#         np.logical_and(a[i] > a[j], b[i] < b[j])
#     ).sum()
#     return n_disordered / (n * (n - 1))


def sequence_similarity_elementwise(
        s1: list, s2: list, discount: float = None, symmetrical: bool = False
) -> float:
    n = len(s1)
    assert n == len(s2)
    if not n:
        # arguable: empty sequences are equal
        return 1.

    sim_func = tuple_similarity if isinstance(s1[0], tuple) else sdr_similarity
    sims = np.array([
        sim_func(s1[i], s2[i], symmetrical=symmetrical)
        for i in range(n)
    ])
    return discounted_mean(sims, gamma=discount)


def sequence_similarity_as_union(
        s1: list, s2: list, sds: Sds, algorithm: str, symmetrical: bool = False
) -> float:
    n = len(s1)
    assert n == len(s2)
    if not n:
        # arguable: empty sequences are equal
        return 1.

    p = aggregate_pmf(s1, sds)
    q = aggregate_pmf(s2, sds)

    return distribution_similarity(p, q, algorithm=algorithm, sds=sds, symmetrical=symmetrical)


def sequence_similarity_by_prefixes(
        seq1: list, seq2: list, sds: Sds, algorithm: str,
        discount: float = None, symmetrical=False,
) -> float:
    n = len(seq1)
    assert n == len(seq2)
    if not n:
        # arguable: empty sequences are equal
        return 1.

    if algorithm == 'elementwise':
        sims = [
            sequence_similarity_elementwise(
                seq1[:i+1], seq2[:i+1], discount=discount, symmetrical=symmetrical
            )
            for i in range(n)
        ]
    else:   # distribution (with specified `algorithm`)
        sims = []
        histogram1, histogram2 = np.zeros(sds.size), np.zeros(sds.size)
        for i in range(n):
            s1, s2 = seq1[i], seq2[i]
            if isinstance(s1, set):
                s1, s2 = list(s1), list(s2)
            if i == 0:
                histogram1[s1] = 1
                histogram2[s2] = 1
            else:
                histogram1 *= discount
                histogram2 *= discount
                histogram1[s1] += 1 - discount
                histogram2[s2] += 1 - discount

            sims.append(
                distribution_similarity(
                    histogram1, histogram2, algorithm=algorithm, sds=sds, symmetrical=symmetrical
                )
            )

    # noinspection PyTypeChecker
    return np.mean(sims)


# ==================== Distributions or cluster distribution similarity ====================
def aggregate_pmf(seq: list, sds: Sds, decay: float = 1.0) -> np.ndarray:
    """Return empirical probability-mass-like function for a sequence."""
    is_tuple = isinstance(seq[0], tuple)
    histogram = np.zeros(sds.size)
    cnt = 0
    for s in seq:
        if is_tuple:
            s = s[0]
        if isinstance(s, set):
            s = list(s)
        if decay < 1.:
            histogram *= decay
            cnt *= decay

        histogram[s] += 1.
        cnt += 1.
    return histogram / cnt


def representation_from_pmf(pmf: np.ndarray, sds: Sds) -> SparseSdr:
    representative_sdr = np.argpartition(pmf, -sds.active_size)[-sds.active_size:]
    representative_sdr.sort()
    return representative_sdr


def _correct_information_metric_for_sds(metric: float, sds: Sds = None) -> float:
    # if SDS params are passed, we treat each distribution as cluster distribution
    # and normalize it
    if sds is not None:
        # as probability mass functions do not sum to 1, but to `active_size`
        metric /= sds.active_size
        # normalize relative to max possible value, i.e. uniform bucket encoding
        # NB: it's equivalent to changing the logarithm base of the main equation
        metric /= -np.log(sds.sparsity)
    return metric


def kl_divergence(
        p: np.ndarray, q: np.ndarray, sds: Sds = None,
        symmetrical=False
) -> float:
    if symmetrical:
        return (kl_divergence(p, q, sds) + kl_divergence(q, p, sds)) / 2

    # noinspection PyTypeChecker
    kl_div: float = np.dot(p, np.ma.log(p) - np.ma.log(q))
    kl_div = _correct_information_metric_for_sds(kl_div, sds)
    # we take abs as for SDS KL-div actually can be < 0! But I think it's ok to consider abs value
    kl_div = abs(kl_div)
    return kl_div


def cross_entropy(d1: np.ndarray, d2: np.ndarray, sds: Sds = None) -> float:
    ce = -np.dot(d1, np.ma.log(d2))
    ce = _correct_information_metric_for_sds(ce, sds)
    return ce


def entropy(x: np.ndarray, sds: Sds = None) -> float:
    return cross_entropy(x, x, sds)


def wasserstein_distance(p: np.ndarray, q: np.ndarray, sds: Sds = None) -> float:
    d = np.zeros_like(p)
    for i in range(1, d.shape[0]):
        d[i] = p[i-1] + d[i-1] - q[i-1]

    # normalize by "domain" size
    res = np.sum(d) / d.shape[0]
    if sds is not None:
        # normalize by the distribution mass
        res /= sds.active_size

    # same as for KL-div
    res = abs(res)
    # noinspection PyTypeChecker
    return res


def point_pmf_similarity(p: np.ndarray, q: np.ndarray, sds: Sds = None) -> float:
    # -> [0, active_size]
    similarity = np.fmin(p, q).sum()
    if sds is not None:
        # -> [0, 1]
        similarity /= sds.active_size
    return similarity


# ==================== Errors ====================
def standardize_sample_distribution(x: np.ndarray, normalization: str) -> np.ndarray:
    if normalization == MEAN_STD_NORMALIZATION:
        unbiased_x = x - np.mean(x)
        return safe_divide(unbiased_x, np.std(x))
    elif normalization == MIN_MINMAX_NORMALIZATION:
        unbiased_x = x - np.min(x)
        return safe_divide(unbiased_x, np.max(x) - np.min(x))
    elif isnone(normalization, 'no') == NO_NORMALIZATION:
        return x
    else:
        raise KeyError(f'Normalization {normalization} is not supported')


def mean_absolute_error(x: np.ndarray, y: np.ndarray) -> float:
    # noinspection PyTypeChecker
    return np.mean(np.abs(x - y))


# ==================== Loss ====================
def simple_additive_loss(smae, pmf_coverage):
    return smae + 0.1 * (1 - pmf_coverage) / pmf_coverage


def multiplicative_loss(smae, pmf_coverage):
    # Google it: y = 0.25 * ((1 - x) / (1.4 * x))^0.5 + 0.75
    # == 1 around 0.41 pmf coverage — it's a target value
    pmf_weight = (1 - pmf_coverage) / (1.4 * pmf_coverage)
    # smooth with sqrt and shift it up
    pmf_weight = 0.25 * (pmf_weight ** 0.55) + 0.75

    # == 1 at smae = 0.08. At ~0.2 SMAE we get almost garbage
    smae_weight = (smae / 0.06)**1.5

    return pmf_weight * smae_weight


# ================== Utility ====================
def discounted_mean(arr: np.ndarray, gamma: float = None) -> float:
    if isinstance(arr, list):
        arr = np.array(arr)

    norm = arr.shape[0]
    if gamma is not None and gamma < 1.:
        # [g, g^2, g^3, ..., g^N] === g * [1, g, g^2,..., g^N-1]
        weights = np.cumprod(np.repeat(gamma, norm))
        # gamma * geometric sum
        norm = gamma * (1 - weights[-1]) / (1 - gamma)

        # not normalized weighted sum
        arr *= weights[::-1]

    return np.sum(arr) / norm
