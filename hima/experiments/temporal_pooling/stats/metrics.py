#  Copyright (c) 2022 Autonomous Non-Profit Organization "Artificial Intelligence Research
#  Institute" (AIRI); Moscow Institute of Physics and Technology (National Research University).
#  All rights reserved.
#
#  Licensed under the AGPLv3 license. See LICENSE in the project root for license information.
from typing import Any

import numpy as np

from hima.common.sdr import SparseSdr, DenseSdr, SetSdr
from hima.common.sdrr import AnySparseSdr, RateSdr
from hima.common.sds import Sds
from hima.common.utils import safe_divide, isnone

TMetrics = dict[str, Any]

SdrSequence = list[AnySparseSdr]
SetSdrSequence = list[SetSdr]
SeqHistogram = np.ndarray

SEQ_SIM_ELEMENTWISE = 'elementwise'
DISTR_SIM_PMF = 'pmf_pointwise'
DISTR_SIM_KL = 'kl-divergence'

MEAN_STD_NORMALIZATION = 'mean-std'
MIN_MINMAX_NORMALIZATION = 'min-minmax'
NO_NORMALIZATION = 'no'


# TODO: CLEAN UP, SIMPLIFY, AND REWRITE

def sdr_similarity(
        x1: AnySparseSdr, x2: AnySparseSdr, symmetrical: bool = False,
        sds: Sds = None, dense_cache: DenseSdr = None
) -> float:
    """
    Compute similarity between two SDRs (both Sdr or RateSdr).
    This is the most abstract function that will induce a suited implementation itself.

    It is useful for a single-time computation. If you need to compute many pairwise
    similarities, use sequential variants from the next group of methods. Otherwise,
    it will be much less efficient.
    """
    if isinstance(x1, set):
        assert isinstance(x2, set)
        return _sdr_similarity_for_sets(x1, x2, symmetrical=symmetrical)

    if dense_cache is None:
        dense_cache = np.zeros(sds.size)

    sim_func = _sdrr_similarity if isinstance(x1, RateSdr) else _sdr_similarity
    return sim_func(x1, x2, dense_cache=dense_cache, symmetrical=symmetrical)


# ==================== SDR similarity ====================
# HOW TO USE: for a single-time it's convenient to use the most abstract function.
#   It will induce a suited implementation itself.
#
#   NB: there are different implementations that are suited for different SDR storage
#       representations having different computational optimizations.
#   NB: if you need to compute many pairwise similarities, use sequential variants from the next
#       group of methods.

def _sdr_similarity_for_sets(x1: SetSdr, x2: SetSdr, symmetrical: bool = False) -> float:
    """Optimized for SDRs represented with sets."""
    overlap = len(x1 & x2)

    # sim is a fraction of their union or x2. For the former, len(x1 | x2) = x1 + x2 - overlap
    norm = len(x1) + len(x2) - overlap if symmetrical else len(x2)
    return safe_divide(overlap, norm)


def _sdr_similarity(
        x1: SparseSdr, x2: SparseSdr, dense_cache: DenseSdr, symmetrical: bool = False
) -> float:
    """
    Optimized for SDRs represented with arrays. For fast computations, it utilizes
    a zeroed-out dense SDR array (will be cleared after using before returning the result).
    """
    dense_cache[x2] = 1
    overlap = dense_cache[x1].sum()
    # clear it
    dense_cache[x2] = 0

    # sim is a fraction of their union or x2. For the former, len(x1 | x2) = x1 + x2 - overlap
    norm = len(x1) + len(x2) - overlap if symmetrical else len(x2)
    return safe_divide(overlap, norm)


# ==================== SDRR similarity ====================
# HOW TO USE: for a single-time it's convenient to use the most abstract function.
#   It will induce a suited implementation itself.
#
#   NB: there are different implementations that are suited for different Rate SDR storage
#       representations having different computational optimizations.
#   NB: if you need to compute many pairwise similarities, use sequential variants from the next
#       group of methods.

def _sdrr_similarity(
        x1: RateSdr, x2: RateSdr, dense_cache: DenseSdr, symmetrical: bool = False
) -> float:
    """
    Optimized for SDRs represented with arrays. For fast computations, it utilizes
    a zeroed-out dense SDR array (will be cleared after using before returning the result).

    NB: Both x1 and x2 are expected to have their sdr bits to have non-zero rates.
        Otherwise, supp(...) calculation will be incorrect.
    """
    dense_cache[x2.sdr] = x2.values
    dense_cache[x1.sdr] -= x1.values

    # similarity is 1 minus an average L1 distance from x1 to x2 over supp(x2)
    raw_distance = np.sum(np.abs(dense_cache[x2.sdr]))
    norm = len(x2.sdr)

    # clear cache
    dense_cache[x2.sdr] = 0

    if symmetrical:
        # similarity is 1 minus an average L1 distance from x1 to x2 over supp(x1 | x2)
        # so, we also need to add (x1 \ x2) part
        raw_distance += np.sum(np.abs(dense_cache[x1.sdr]))
        norm += np.count_nonzero(dense_cache[x1.sdr])

    # finish clearing cache
    dense_cache[x1.sdr] = 0

    return 1 - safe_divide(raw_distance, norm)


# ==================== Sdr [sequence] similarity ====================
def sequence_similarity(
        s1: SdrSequence, s2: SdrSequence,
        algorithm: str, discount: float = None, symmetrical: bool = False,
        sds: Sds = None
) -> float:
    if algorithm == 'elementwise':
        # reflects strictly ordered similarity
        return sequence_similarity_elementwise(s1, s2, discount=discount, symmetrical=symmetrical)
    elif algorithm.startswith('union'):
        # reflects unordered (=set) similarity
        # "union.xxx" -> "xxx", where "xxx" — from `distribution_similarity`
        algorithm = algorithm[6:]
        return sequence_similarity_as_union(
            s1, s2, sds=sds, algorithm=algorithm, symmetrical=symmetrical
        )
    elif algorithm.startswith('prefix'):
        # reflects balance between the other two
        # "prefix.xxx" -> "xxx", where "xxx" — from `distribution_similarity`
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
        # We take |1 - KL| to make it similarity metric. NB: normalized KL div for SDS can be > 1
        return np.abs(1 - kl_divergence(p, q, sds, symmetrical=symmetrical))
    elif algorithm == 'pmf_pointwise':
        return point_pmf_similarity(p, q, sds=sds)
    elif algorithm == 'wasserstein':
        return -wasserstein_distance(p, q, sds=sds)
    else:
        raise KeyError(f'Invalid algorithm: {algorithm}')


def similarity_matrix(
        a: list[AnySparseSdr],
        algorithm: str = None, discount: float = None, symmetrical: bool = False,
        sds: Sds = None
) -> np.ndarray:
    n = len(a)
    diagonal_mask = np.identity(n, dtype=bool)
    sm = np.empty((n, n))

    if isinstance(a[0], set) or isinstance(a[0], RateSdr):
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
                sim = _sdr_similarity_for_sets(x, y, symmetrical=symmetrical)
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


def sequence_similarity_elementwise(
        s1: SdrSequence, s2: SdrSequence,
        discount: float = None, symmetrical: bool = False,
        sds: Sds = None, dense_cache: np.ndarray = None
) -> float:
    n = len(s1)
    assert n == len(s2)
    if not n:
        # arguable: empty sequences are equal
        return 1.

    if isinstance(s1[0], set):
        sims = np.array([
            _sdr_similarity_for_sets(s1[i], s2[i], symmetrical=symmetrical)
            for i in range(n)
        ])
    else:
        if dense_cache is None:
            dense_cache = np.zeros(sds.size)
        sim_func = _sdrr_similarity if isinstance(s1[0], RateSdr) else _sdr_similarity

        sims = np.array([
            sim_func(s1[i], s2[i], symmetrical=symmetrical, dense_cache=dense_cache)
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
        seq1: list[AnySparseSdr], seq2: list[AnySparseSdr],
        sds: Sds, algorithm: str,
        discount: float = None, symmetrical=False,
) -> float:
    """
    Compute similarity between two SDR sequences using their prefixes.
    It is averaged-by-time online version of sequence similarity, as it
    computes similarity at each timestep (hence using prefixes) and then averages them.

    It supports different algorithms of similarity computation, e.g.:
        - elementwise SDR similarity
        - similarity based on aggregate distributions (see `distribution_similarity`)
    """
    n = len(seq1)
    assert n == len(seq2)
    if n == 0:
        # arguable: empty sequences are equal
        return 1.

    if algorithm == 'elementwise':
        sims = [
            sequence_similarity_elementwise(
                seq1[:i+1], seq2[:i+1], discount=discount, symmetrical=symmetrical
            )
            for i in range(n)
        ]
    else:
        # distribution specified by `algorithm`

        # TODO: rewrite it using incremental histogram update
        # TODO: support RateSDR
        sims = []
        histogram1, histogram2 = np.zeros(sds.size), np.zeros(sds.size)
        for t in range(n):
            s1, s2 = seq1[t], seq2[t]
            if isinstance(s1, set):
                s1, s2 = list(s1), list(s2)
            if t == 0:
                # init histograms at t=0
                histogram1[s1] = 1
                histogram2[s2] = 1
            else:
                # each timestep we discount previous histograms and add new elements
                histogram1 *= discount
                histogram2 *= discount
                histogram1[s1] += 1 - discount
                histogram2[s2] += 1 - discount

            sims.append(
                distribution_similarity(
                    histogram1, histogram2, algorithm=algorithm, sds=sds, symmetrical=symmetrical
                )
            )
    return np.mean(sims)


# ==================== Distributions or cluster distribution similarity ====================
def aggregate_pmf(seq: list[AnySparseSdr], sds: Sds, decay: float = 1.0) -> np.ndarray:
    """
    Return empirical probability-mass-like function for a sequence.
    Decay is used to weight the contribution of past elements in the sequence.
    """
    histogram = np.zeros(sds.size)
    if not seq:
        return histogram

    s = seq[0]
    is_rate_sdr = isinstance(s, RateSdr)
    s = s.sdr if is_rate_sdr else s
    is_set = isinstance(s, set)
    is_decaying = decay < 1.

    # TODO: extract incremental histogram update into a separate function

    cnt = 0
    val = 1.

    for s in seq:
        if is_rate_sdr:
            # extract rate SDR
            val = s.values
            s = s.sdr
        if is_set:
            # convert to list so that we can use it as slicing indices
            s = list(s)
        if is_decaying:
            histogram *= decay
            cnt *= decay

        histogram[s] += val
        cnt += 1.
    return histogram / cnt


def representation_from_pmf(pmf: np.ndarray, sds: Sds) -> SparseSdr:
    """
    Return an SDR representative from a probability-mass-like function, i.e.
    `active_size` the most probable elements.
    """
    representative_sdr = np.argpartition(pmf, -sds.active_size)[-sds.active_size:]
    # indices are ordered by probability, reorder them by index
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
