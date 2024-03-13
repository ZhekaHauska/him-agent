#  Copyright (c) 2022 Autonomous Non-Profit Organization "Artificial Intelligence Research
#  Institute" (AIRI); Moscow Institute of Physics and Technology (National Research University).
#  All rights reserved.
#
#  Licensed under the AGPLv3 license. See LICENSE in the project root for license information.
from typing import Any

import numba
import numpy as np
import numpy.typing as npt

from hima.common.sdr import SparseSdr, SetSdr
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
        sds: Sds = None, dense_cache: npt.NDArray[float] = None
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

    is_rate_sdr1 = isinstance(x1, RateSdr)
    is_rate_sdr2 = isinstance(x2, RateSdr)
    if is_rate_sdr1 or is_rate_sdr2:
        if not is_rate_sdr1:
            x1 = RateSdr(x1, np.repeat(1., len(x1)))
        if not is_rate_sdr2:
            x2 = RateSdr(x2, np.repeat(1., len(x2)))
        sim_func = _sdrr_similarity
    else:
        sim_func = _sdr_similarity

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

    # NB: if x1 — prediction, x2 — positives, then for non-symmetrical case:
    #   sim(x1, x2) = recall = 1 - miss_rate;
    #   sim(x2, x1) = precision = 1 - imprecision

    # sim is a fraction of their union or x2. For the former, len(x1 | x2) = x1 + x2 - overlap
    norm = len(x1) + len(x2) - overlap if symmetrical else len(x2)
    return safe_divide(overlap, norm)


@numba.jit(nopython=True, cache=True)
def _sdr_similarity(
        x1: SparseSdr, x2: SparseSdr, dense_cache: npt.NDArray[float], symmetrical: bool = False
) -> float:
    """
    Optimized for SDRs represented with arrays. For fast computations, it utilizes
    a zeroed-out dense SDR array (will be cleared after using before returning the result).
    """
    # SDR similarity is implemented as a special case of Rate SDR similarity for easier
    # results comparison with different output modes.

    # NB: sym(pred, gt) — recall; sym(gt, pred) — precision.

    dense_cache[x2] = 1
    overlap = dense_cache[x1].sum()
    # clear it
    dense_cache[x2] = 0

    sim = _normalize_distance(overlap, len(x2))
    if symmetrical:
        sim_ = _normalize_distance(overlap, len(x1))
        sim = (sim + sim_) / 2

    return sim


# ==================== SDRR similarity ====================
# HOW TO USE: for a single-time it's convenient to use the most abstract function.
#   It will induce a suited implementation itself.
#
#   NB: there are different implementations that are suited for different Rate SDR storage
#       representations having different computational optimizations.
#   NB: if you need to compute many pairwise similarities, use sequential variants from the next
#       group of methods.

def _sdrr_similarity(
        x1: RateSdr, x2: RateSdr, dense_cache: npt.NDArray[float], symmetrical: bool = False
) -> float:
    """
    Optimized for SDRs represented with arrays. For fast computations, it utilizes
    a zeroed-out dense SDR array (will be cleared after using before returning the result).
    """
    return _sdrr_similarity_numba(
        x1.sdr, x1.values, x2.sdr, x2.values,
        dense_cache=dense_cache, symmetrical=symmetrical
    )


@numba.jit(nopython=True, cache=True)
def _sdrr_similarity_numba(
        x1_sdr: npt.NDArray[int], x1_values: npt.NDArray[float],
        x2_sdr: npt.NDArray[int], x2_values: npt.NDArray[float],
        dense_cache: npt.NDArray[float], symmetrical: bool = False
) -> float:
    """
    Optimized for SDRs represented with arrays. For fast computations, it utilizes
    a zeroed-out dense SDR array (will be cleared after using before returning the result).
    """
    if len(x1_sdr) == 0 or len(x2_sdr) == 0:
        # we define that empty SDRs aren't similar to anything, even to each other
        return 0.

    # NB: sym(pred, gt) — recall; sym(gt, pred) — precision.

    dense_cache[x2_sdr] = x2_values
    norm = np.sum(x2_values)
    if np.isclose(norm, 0.):
        return 0.

    if symmetrical:
        norm_ = np.sum(x1_values)
        if np.isclose(norm_, 0.):
            return 0.

    dense_cache[x1_sdr] -= x1_values

    # we define non-symmetrical distance as: d(x1, x2) = |x1 - x2| / |x2|
    dense_cache[x2_sdr] = np.abs(dense_cache[x2_sdr])
    raw_distance = np.sum(dense_cache[x2_sdr])

    # I clip all intermediate results to [0, 1] to avoid incorrect results for cases
    # like [1, 1, 1] vs [<<1, <<1, <<1] —> which would get one of non-symmetrical
    # distances >> 1 and the total distance > 1 => similarity = 0, which is wrong as
    # for this case similarity should be 0 < sim << 1.
    # E.g. [1, 1, 1] vs [0.1, 0.1, 0.1] should give us a similarity ~= 0.1. It gives us
    # a symmetrical similarity = 0.1 / 2 with intermediate clipping, which is a good enough.
    # NB: this problem is relevant only for rate SDRs with significantly different masses.
    distance = _normalize_distance(raw_distance, norm)

    if symmetrical:
        # symmetrical distance is an average of two non-symmetrical distances
        dense_cache[x1_sdr] = np.abs(dense_cache[x1_sdr])
        raw_distance = np.sum(dense_cache[x1_sdr])

        # noinspection PyUnboundLocalVariable
        distance_ = _normalize_distance(raw_distance, norm_)
        distance = (distance + distance_) / 2

    # clear cache
    dense_cache[x2_sdr] = 0
    dense_cache[x1_sdr] = 0

    result = 1 - distance
    return result


@numba.jit(nopython=True, cache=True)
def _normalize_distance(dist: float, norm: float) -> float:
    """Safely divide distance by norm and clip it to [0, 1]. Auxiliary function for sim funcs."""
    if np.isclose(norm, 0.):
        return 0.

    dist /= norm
    if dist < 0.:
        dist = 0.
    elif dist > 1.:
        dist = 1.
    return dist


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
        p: npt.NDArray[float], q: npt.NDArray[float],
        algorithm: str, sds: Sds = None, symmetrical: bool = False
) -> float:
    if algorithm == 'kl-divergence':
        # We take |1 - KL| to make it similarity metric. NB: normalized KL div for SDS can be > 1
        return np.abs(1 - kl_divergence(p, q, sds=sds, symmetrical=symmetrical))
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
) -> npt.NDArray[float]:
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
        sds: Sds = None, dense_cache: npt.NDArray[float] = None
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


def representation_from_pmf(pmf: npt.NDArray[float], sds: Sds) -> SparseSdr:
    """
    Return an SDR representative from a probability-mass-like function, i.e.
    `active_size` the most probable elements.
    """
    representative_sdr = np.argpartition(pmf, -sds.active_size)[-sds.active_size:]
    # indices are ordered by probability, reorder them by index
    representative_sdr.sort()
    return representative_sdr


def _correct_information_metric_for_sds(
        metric: float, p: npt.NDArray[float], sds: Sds = None
) -> float:
    # if SDS params are passed, we treat each distribution as cluster distribution
    # and normalize it.
    # There are two different cases:
    #   - for binary SDR pmf the sum of it equals to `sds.active_size`
    #   - for Rate SDRs pmf the sum of it equals to the average rate mass (= sum of rates)

    # we distinguish these cases by checking if `sds` is passed
    if sds is not None:
        # for SDR pmf does not sum to 1, but to `active_size`
        metric /= sds.active_size
        # normalize relative to max possible value, i.e. uniform bucket encoding
        # NB: it's equivalent to changing the logarithm base of the main equation
        metric /= -np.log(sds.sparsity)
    else:
        # for Rate SDRs pmf does not sum to 1, but to the average rate mass
        # (= sum of rates in rate SDRs)
        avg_rate_mass = p.sum()
        if avg_rate_mass != 0:
            metric /= avg_rate_mass
            # normalize relative to max possible value, i.e. uniform pmf
            metric /= -np.log(avg_rate_mass / p.size)
    return metric


def kl_divergence(
        p: npt.NDArray[float], q: npt.NDArray[float],
        normalize=True, sds: Sds = None, symmetrical=False
) -> float:
    if symmetrical:
        return (kl_divergence(p, q, normalize, sds) + kl_divergence(q, p, normalize, sds)) / 2

    # noinspection PyTypeChecker
    kl_div: float = np.dot(p, np.ma.log(p) - np.ma.log(q))
    if normalize:
        kl_div = _correct_information_metric_for_sds(kl_div, p=p, sds=sds)
    # we take abs as for SDS KL-div actually can be < 0! But I think it's ok to consider abs value
    kl_div = abs(kl_div)
    return kl_div


def cross_entropy(
        p: npt.NDArray[float], q: npt.NDArray[float], normalize=True, sds: Sds = None
) -> float:
    ce = -np.dot(p, np.ma.log(q))
    if normalize:
        ce = _correct_information_metric_for_sds(ce, p=p, sds=sds)
    return ce


def entropy(x: np.ndarray, sds: Sds = None, normalize=True) -> float:
    return cross_entropy(x, x, normalize=normalize, sds=sds)


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
