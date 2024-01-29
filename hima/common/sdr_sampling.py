#  Copyright (c) 2023 Autonomous Non-Profit Organization "Artificial Intelligence Research
#  Institute" (AIRI); Moscow Institute of Physics and Technology (National Research University).
#  All rights reserved.
#
#  Licensed under the AGPLv3 license. See LICENSE in the project root for license information.
import numpy as np
from numpy.random import Generator

from hima.common.sdr import SparseSdr
from hima.common.sdrr import RateSdr
from hima.common.sds import Sds


# ==================== SDR generation ====================
def sample_sdr(rng: Generator, sds: Sds):
    return rng.choice(sds.size, sds.active_size, replace=False)


def sample_noisy_sdr(rng: Generator, sds: Sds, sdr: SparseSdr, frac: float):
    """Sample noisy SDR from the given SDR with the given fraction of noise."""
    set_sdr = set(sdr)
    active_size = len(sdr)

    n_to_remove = max(1, round(frac * sds.active_size))
    target_interim_len = max(sds.active_size, active_size) - n_to_remove
    n_will_remove = max(0, active_size - target_interim_len)

    if n_will_remove > 0:
        # NB: permutation is significantly faster than choice for short arrays
        to_remove = rng.permutation(sdr)[:n_will_remove]
        set_sdr -= set(to_remove)

    n_will_add = sds.active_size - len(set_sdr)
    if n_will_add > 0:
        to_add = rng.integers(0, sds.size, n_will_add)
        set_sdr |= set(to_add)

        # in case of collisions, add the rest
        while len(set_sdr) < sds.active_size:
            x = rng.integers(0, sds.size)
            set_sdr.add(x)

    result = np.array(list(set_sdr))
    result.sort()
    return result


def sample_melded_sdr(rng: Generator, sds: Sds, sdr: SparseSdr, other: SparseSdr, frac: float):
    """Randomly meld two SDRs together, preserving the given fraction of the first one."""
    set_sdr = set(sdr)
    set_other = set(other)

    intersect = set_sdr & set_other
    n_intersect = len(intersect)
    n_to_fill = max(sds.active_size - n_intersect, 0)

    n_frac = round(frac * n_to_fill)
    n_from_left_uniques = min(n_to_fill - n_frac, len(sdr) - n_intersect)
    n_from_right_uniques = min(n_frac, len(other) - n_intersect)

    left_uniques = list(set_sdr - intersect)
    right_uniques = list(set_other - intersect)

    # NB: in-place shuffle is significantly faster than choice for short arrays
    rng.shuffle(left_uniques)
    rng.shuffle(right_uniques)

    result = np.concatenate((
        list(intersect),
        left_uniques[:n_from_left_uniques],
        right_uniques[:n_from_right_uniques]
    ))
    result.sort()
    return result


# ==================== Rate SDR generation ====================
def sample_rates(rng: Generator, size: int, temp: float = 0.3):
    """Sample rates array in [0, 1] with the given size and the temperature (noise scale)."""
    x = rng.normal(0, temp, size)
    np.abs(x, out=x)
    x[x > 1] = 1
    x[:] = 1 - x
    return x


def sample_rate_sdr(rng: Generator, sdr: SparseSdr, temp: float = 0.3, scale: float = 1.):
    """Sample RateSdr from the given SDR with the given temperature (noise scale)."""
    return RateSdr(
        sdr=sdr,
        values=sample_rates(rng, len(sdr), temp=temp) * scale
    )


def sample_noisy_rates_rate_sdr(rng: Generator, rate_sdr: RateSdr, frac: float, temp: float = 0.3):
    """
    Sample noisy RateSdr from the given RateSdr with the given fraction of rates noise.
    The noise applied only to the rates, while the original SDR is kept intact.
    """
    other_rates = sample_rates(rng, size=len(rate_sdr.sdr), temp=temp)
    # NB: meld rates with the given fraction of the original rates
    new_rates = (1. - frac) * rate_sdr.values + frac * other_rates
    return RateSdr(rate_sdr.sdr, values=new_rates)


def sample_noisy_sdr_rate_sdr(rng: Generator, sds: Sds, rate_sdr: RateSdr, frac: float):
    """
    Sample noisy rate SDR from the given rates SDR with the given fraction of sdr noise.

    The noise applied only to the SDR, while the original rates are kept intact.
    However, the new active bits are assigned new sampled rates.
    """
    n = len(rate_sdr.sdr)
    n_to_change = max(1, round(frac * n))

    to_change_sdr = np.array(list(
        set(rng.integers(0, sds.size, n_to_change + 4)) - set(rate_sdr.sdr)
    ))
    n_to_change = min(n_to_change, len(to_change_sdr))

    to_change_sdr = to_change_sdr[:n_to_change]
    to_change_values = sample_rates(rng, n_to_change)
    idx_to_change = rng.permutation(n)[:n_to_change]

    sdr = rate_sdr.sdr.copy()
    values = rate_sdr.values.copy()

    sdr[idx_to_change] = to_change_sdr
    values[idx_to_change] = to_change_values

    indices = np.argsort(sdr)
    sdr = sdr[indices]
    values = values[indices]

    return RateSdr(sdr=sdr, values=values)
