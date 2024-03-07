#  Copyright (c) 2022 Autonomous Non-Profit Organization "Artificial Intelligence Research
#  Institute" (AIRI); Moscow Institute of Physics and Technology (National Research University).
#  All rights reserved.
#
#  Licensed under the AGPLv3 license. See LICENSE in the project root for license information.
from __future__ import annotations

from typing import cast

import numpy as np
import numpy.typing as npt

from hima.common.sdr import SparseSdr
from hima.common.sdrr import AnySparseSdr, RateSdr, OutputMode
from hima.common.sds import Sds, TSdsShortNotation
from hima.common.utils import isnone
from hima.common.sdr_sampling import sample_sdr, sample_rate_sdr

INT_TYPE = "int64"
UINT_DTYPE = "uint32"


# TODO:
#   1. Support caching for bucket encoder


class IntBucketEncoder:
    """
    Encodes integer values from the range [0, `n_values`) as SDR with `output_sdr_size` total bits.
    SDR bit space is divided into `n_values` possibly overlapping buckets of `bucket_size` bits
        where i-th bucket starts from i * `buckets_step` index.
    Each value is encoded with corresponding bucket of active bits (see example).
    This is a sparse encoder, so the output is in sparse SDR format.

    Non-overlapping example, i.e. when `buckets_step` == `bucket_size` = 4, for `n_values` = 3:
        0000 1111 0000
    """

    ALL = -1

    n_values: int
    output_sds: Sds

    _bucket_size: int
    _buckets_step: int

    def __init__(self, n_values: int, bucket_size: int, buckets_step: int = None):
        """
        Initializes encoder.

        :param n_values: int defines a range [0, n_values) of values that can be encoded
        :param bucket_size: number of bits in a bucket used to encode single value
        :param buckets_step: number of bits between beginnings of consecutive buckets
        """
        self._bucket_size = bucket_size
        self._buckets_step = isnone(buckets_step, bucket_size)
        self.n_values = n_values

        max_value = self.n_values - 1
        sds_size = self._bucket_starting_pos(max_value) + self._bucket_size
        self.output_sds = Sds(
            size=sds_size,
            active_size=self._bucket_size
        )

    @property
    def output_sdr_size(self) -> int:
        return self.output_sds.size

    @property
    def n_active_bits(self) -> int:
        return self.output_sds.active_size

    def encode(self, x: int) -> SparseSdr:
        """Encodes value x to sparse SDR format using bucket-based non-overlapping encoding."""
        assert x is None or x == self.ALL or 0 <= x < self.n_values, \
            f'Value must be in [0, {self.n_values}] or {self.ALL} or None; got {x}'

        if x is None:
            return np.array([], dtype=int)
        if x == self.ALL:
            return np.arange(self.output_sdr_size, dtype=int)

        left = self._bucket_starting_pos(x)
        right = left + self._bucket_size
        return np.arange(left, right, dtype=int)

    def _bucket_starting_pos(self, i):
        return i * self._buckets_step


class IntRandomEncoder:
    """
    Encodes integer values from range [0, `n_values`) as SDR with `total_bits` bits.
    Each value x is encoded with `total_bits` * `sparsity` random bits.
    Any two encoded values may overlap. Encoding scheme is initialized once and then remains fixed.
    """

    output_sds: Sds

    encoding_map: np.array

    def __init__(
            self, n_values: int, seed: int,
            sds: Sds | TSdsShortNotation = None,
            space_compression: float = None,
            active_size: int = None,
            output_mode: str = 'binary'
    ):
        """
        Initializes encoder that maps each categorical value to a fixed random SDR.

        :param n_values: defines a range [0, n_values) of values that can be encoded
        :param sds: SDR space
        :param space_compression: compression factor for SDR space relatively to a bucket encoder
        :param seed: random seed for the random encoding scheme generation
        """

        if sds is None:
            sds_size = int(n_values * active_size * space_compression)
            sds = (sds_size, active_size)

        self.output_sds = Sds.make(sds)
        self.output_mode = OutputMode[output_mode.upper()]

        self.encoding_map = self._make_encoding_map(
            seed=seed, n_values=n_values, sds=self.output_sds, output_mode=self.output_mode
        )

    @property
    def output_sdr_size(self) -> int:
        return self.output_sds.size

    @property
    def n_active_bits(self) -> int:
        return self.output_sds.active_size

    @property
    def n_values(self) -> int:
        return self.encoding_map.shape[0]

    def encode(
            self, x: int | list[int] | np.ndarray
    ) -> AnySparseSdr | list[AnySparseSdr]:
        """
        Encodes value x to sparse SDR format using random overlapping encoding.
        It is vectorized, so an array-like x is accepted too.
        """
        if self.output_mode == OutputMode.BINARY:
            return self.encoding_map[x]

        # Rate SDR
        if isinstance(x, (list, np.ndarray)):
            return [self.encoding_map[i] for i in x]
        return self.encoding_map[x]

    @staticmethod
    def _make_encoding_map(
            seed: int, n_values, sds, output_mode: OutputMode
    ) -> np.ndarray:
        rng = np.random.default_rng(seed=seed)
        encoding_map = np.array([sample_sdr(rng, sds) for _ in range(n_values)], dtype=int)
        if output_mode == OutputMode.RATE:
            encoding_map = [
                sample_rate_sdr(rng, sdr, scale=1.0)
                for sdr in encoding_map
            ]
            avg_encoding_mass = np.mean([
                np.sum(sdr.values) / len(sdr.sdr)
                for sdr in encoding_map
            ])
            print(f'Average encoding mass: {avg_encoding_mass:.2f}')
        return encoding_map


class SdrConcatenator:
    """Concatenates sparse SDRs."""
    output_sds: Sds

    _shifts: list[int] | npt.NDArray

    def __init__(self, sdr_spaces: list[Sds] | list[int]):
        if len(sdr_spaces) > 0 and isinstance(sdr_spaces[0], int):
            # sdr_spaces is a list of SDR sizes ==> sparsity is unknown, so we set
            # it to 1.0 as a placeholder
            # noinspection PyTypeChecker
            sdr_spaces = [Sds(size=size, sparsity=1.0) for size in sdr_spaces]

        cumulative_sizes = np.cumsum([sds.size for sds in sdr_spaces])
        total_size = cast(int, cumulative_sizes[-1])
        total_active_size = sum([sds.active_size for sds in sdr_spaces])

        # NB: note that zero shift at the beginning is omitted
        self._shifts = cumulative_sizes[:-1]
        self.output_sds = Sds(size=total_size, active_size=total_active_size)

    def concatenate(self, *sparse_sdrs: AnySparseSdr) -> AnySparseSdr:
        """Concatenates `sparse_sdrs` fixing their relative indexes."""
        if len(sparse_sdrs) == 0:
            return []

        is_rate_sdr = [isinstance(sdr, RateSdr) for sdr in sparse_sdrs]
        is_any_rate_sdr = any(is_rate_sdr)

        sizes = [
            len(sdr.sdr) if isinstance(sdr, RateSdr) else len(sdr)
            for sdr in sparse_sdrs
        ]
        total_size = sum(sizes)
        result = np.empty(total_size, dtype=int)

        # to speed things up do not apply zero shift to the first sdr
        first = sparse_sdrs[0].sdr if is_rate_sdr[0] else sparse_sdrs[0]
        l, r = 0, sizes[0]
        result[l:r] = first

        # apply corresponding shifts to the rest inputs
        for i in range(1, len(sparse_sdrs)):
            sdr = sparse_sdrs[i]
            if is_rate_sdr[i]:
                sdr = sdr.sdr
            sz = sizes[i]
            l = r
            r = r + sz
            result[l:r] = sdr + self._shifts[i - 1]

        if is_any_rate_sdr:
            values = np.concatenate([
                sparse_sdrs[i].values if is_rate_sdr[i] else np.repeat(1.0, sizes[i])
                for i in range(len(sparse_sdrs))
            ])
            return RateSdr(result, values=values)
        return result

    @property
    def output_sdr_size(self):
        return self.output_sds.size


class RangeDynamicEncoder:
    def __init__(self,
                 min_value,
                 max_value,
                 min_delta,
                 n_active_bits,
                 cyclic: bool,
                 max_delta=None,
                 min_speed=None,
                 max_speed=None,
                 use_speed_modulation: bool = False,
                 seed=None):
        self.min_value = min_value
        self.max_value = max_value
        self.min_speed = min_speed
        self.max_speed = max_speed
        self.min_delta = min_delta
        self.max_delta = max_delta
        self.use_speed_modulation = use_speed_modulation

        if self.use_speed_modulation:
            if (min_speed is None) or (max_speed is None) or (max_delta is None):
                raise ValueError
            else:
                assert min_speed < max_speed
                assert min_delta < max_delta
        assert min_value < max_value

        self.n_active_bits = n_active_bits
        self.cyclic = cyclic

        self.min_diameter = self.n_active_bits

        self.rng = np.random.default_rng(seed)

        assert self.min_delta > 0
        self.output_sdr_size = self.n_active_bits * int(
            round((self.max_value - self.min_value) / self.min_delta))
        assert self.output_sdr_size > 0
        if self.use_speed_modulation:
            self.max_diameter = self.output_sdr_size // int(
                round((self.max_value - self.min_value) / self.max_delta))
        else:
            self.max_diameter = self.min_diameter

        self.sample_order = self.rng.random(size=self.output_sdr_size)

    def encode(self, value, speed=None):
        # print(f"joint pos {value}")
        assert self.min_value <= value <= self.max_value
        if self.use_speed_modulation:
            if speed is None:
                raise ValueError
            else:
                speed = min(max(self.min_speed, speed), self.max_speed)
                norm_speed = (speed - self.min_speed) / (self.max_speed - self.min_speed)
        else:
            norm_speed = 0

        diameter = int(round(
            self.min_diameter + norm_speed * (self.max_diameter - self.min_diameter)
        ))

        norm_value = (value - self.min_value) / (self.max_value - self.min_value)

        center = int(round(norm_value * self.output_sdr_size))

        l_radius = (diameter - 1) // 2
        r_radius = diameter // 2

        if not self.cyclic:
            if (center - l_radius) <= 0:
                start = 0
                end = diameter - 1
            elif (center + r_radius) >= self.output_sdr_size:
                start = self.output_sdr_size - diameter
                end = self.output_sdr_size - 1
            else:
                start = center - l_radius
                end = center + r_radius

            potential = np.arange(start, end + 1)
        else:
            if (center - l_radius) < 0:
                start = self.output_sdr_size + center - l_radius
                end = center + r_radius

                potential = np.concatenate([np.arange(start, self.output_sdr_size),
                                            np.arange(0, end + 1)])
            elif (center + r_radius) >= self.output_sdr_size:
                start = center - l_radius
                end = center + r_radius - self.output_sdr_size

                potential = np.concatenate([np.arange(start, self.output_sdr_size),
                                            np.arange(0, end + 1)])
            else:
                start = center - l_radius
                end = center + r_radius
                potential = np.arange(start, end + 1)

        active_arg = np.argpartition(self.sample_order[potential],
                                     kth=-self.n_active_bits)[-self.n_active_bits:]
        active = potential[active_arg]

        return active


class VectorDynamicEncoder:
    def __init__(self, size, encoder: RangeDynamicEncoder):
        self.size = size
        self.encoder = encoder
        self.output_sdr_size = size * encoder.output_sdr_size

    def encode(self, value_vector, speed_vector):
        assert len(value_vector) == len(speed_vector)
        outputs = list()
        shift = 0
        for i in range(len(value_vector)):
            sparse = self.encoder.encode(value_vector[i], speed_vector[i])
            outputs.append(sparse + shift)
            shift += self.encoder.output_sdr_size

        return np.concatenate(outputs)


def _test():
    encoder = RangeDynamicEncoder(0, 1, 0.3, 10, True, seed=5)
    for x in np.linspace(0, 1, 11):
        code = encoder.encode(x)
        dense = np.zeros(encoder.output_sdr_size, dtype=int)
        dense[code] = 1
        print(f"{round(x, 2)}: {dense}")


if __name__ == '__main__':
    _test()
