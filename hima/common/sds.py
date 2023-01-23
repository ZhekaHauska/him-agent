#  Copyright (c) 2022 Autonomous Non-Profit Organization "Artificial Intelligence Research
#  Institute" (AIRI); Moscow Institute of Physics and Technology (National Research University).
#  All rights reserved.
#
#  Licensed under the AGPLv3 license. See LICENSE in the project root for license information.
from __future__ import annotations

from typing import Union

import numpy as np

from hima.common.config.values import is_resolved_value

TSdsShortNotation = Union[
    # tuple[shape|size, active_size|sparsity]
    tuple[
        Union[tuple, int],
        Union[int, float]
    ],

    # tuple[sparsity, active_size]
    tuple[float, int]
]


# TODO: rework parsing and probably split building from the dataclass itself

class Sds:
    """
    Sparse Distributed Space (SDS) parameters.

    Short notation helps to correctly define SDS with the minimal number of params. In each case
    we want to specify only a sufficient subset of all params and just let the others be
    inducted.

    Here's all supported notations (note that they all have distinguishable types):
        a) (100, 0.02) — the total size and sparsity
        b) (100, 10) — the total size and active SDR size
        c) ((20, 20), 0.02) — shape and sparsity
        d) ((20, 20), 10) — shape and active SDR size
        e) (0.02, 10) — sparsity and active SDR size

    The same goes for the keyword-only __init__ arguments — you only need to specify
    the sufficient subset of them.
    """

    shape: tuple[int, ...]
    size: int
    sparsity: float
    active_size: int

    def __init__(
            self,
            # short notation is the only positional argument — default way to define SDS via config
            short_notation: TSdsShortNotation = None,
            *,
            shape: tuple[int, ...] = None,
            size: int = None,
            sparsity: float = None,
            active_size: int = None,
    ):
        if short_notation is not None:
            # ignore keyword-only params
            shape, size, sparsity, active_size = self.parse_short_notation(*short_notation)

        self.shape, self.size, self.sparsity, self.active_size = self.resolve_sds_components(
            shape=shape, size=size, sparsity=sparsity, active_size=active_size
        )

    @staticmethod
    def parse_short_notation(first, second):
        shape, size, sparsity, active_size = None, None, None, None

        if isinstance(first, tuple) or isinstance(first, int):
            # (shape/size, sparsity/active_size)

            if isinstance(first, tuple):
                # shape
                shape = first
            else:
                # size
                size = first

            if isinstance(second, float):
                # sparsity
                sparsity = second
            elif isinstance(second, int):
                # active size
                active_size = second
            else:
                raise TypeError(
                    f'Parsed as (shape/size, sparsity/active_size). TSecond: {type(second)}'
                )
        elif isinstance(first, float):
            # (sparsity, active_size)
            if not isinstance(second, int):
                raise TypeError(
                    f'Parsed as (sparsity, active_size). TSecond is not int: {type(second)}'
                )
            sparsity = first
            active_size = second
        else:
            raise TypeError(f'TFirst is not tuple/int/float: {type(first)}')

        return shape, size, sparsity, active_size

    def __str__(self):
        return f'({self.shape}, {self.size}, {self.active_size}, {self.sparsity})'

    @staticmethod
    def resolve_sds_components(
            shape: tuple[int, ...] = None,
            size: int = None,
            sparsity: float = None,
            active_size: int = None
    ):
        if shape is None and size is None:
            # defined: sparsity & active size
            #   resolve size and shape
            size = round(active_size / sparsity)
            shape = (size, )
        else:
            # defined: shape | size + sparsity | active size
            #   1) resolve size; 2) resolve sparsity and active size

            if size is None:
                size = np.prod(shape)
            else:
                shape = (size,)

            if active_size is None:
                active_size = int(size * sparsity)
            else:
                sparsity = active_size / size

        return shape, size, sparsity, active_size

    @staticmethod
    def as_sds(sds: Sds | TSdsShortNotation) -> Sds:
        if not is_resolved_value(sds):
            # allow keeping unresolved values as is, because there's nothing you can do with it RN
            return sds

        if sds is None:
            # allow empty sds
            print("WARNING: allow empty SDS?")
            return Sds(size=0, sparsity=0., active_size=0)
        if isinstance(sds, Sds):
            return sds
        return Sds(short_notation=sds)
