#  Copyright (c) 2022 Autonomous Non-Profit Organization "Artificial Intelligence Research
#  Institute" (AIRI); Moscow Institute of Physics and Technology (National Research University).
#  All rights reserved.
#
#  Licensed under the AGPLv3 license. See LICENSE in the project root for license information.
from __future__ import annotations

from typing import Union

import numpy as np

TSdsShortNotation = Union[
    # tuple[shape|size, active_size|sparsity]
    tuple[
        Union[tuple, int],
        Union[int, float]
    ],

    # tuple[sparsity, active_size]
    tuple[float, int]
]


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
            try:
                shape, size, sparsity, active_size = self.parse_short_notation(*short_notation)
            except:
                print(short_notation)
                raise

        self.shape, self.size, self.sparsity, self.active_size = self.induce_all_components(
            shape=shape, size=size, sparsity=sparsity, active_size=active_size
        )

    def __eq__(self, other):
        assert isinstance(other, Sds)
        return self.size == other.size and self.active_size == other.active_size

    def __repr__(self):
        return str(self)

    def __str__(self):
        return f'({self.shape}, {self.size}, {self.active_size}, {self.sparsity:.4f})'

    def with_active_size(self, active_size: int) -> Sds:
        """Produce another SDS with the specified active size."""
        return Sds(shape=self.shape, active_size=active_size)

    @staticmethod
    def parse_short_notation(first, second):
        shape, size, sparsity, active_size = None, None, None, None
        tuple_type = (tuple, list)

        if isinstance(first, tuple_type) or isinstance(first, int):
            # (shape/size, sparsity/active_size)

            if isinstance(first, tuple_type):
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

    @staticmethod
    def induce_all_components(
            shape: tuple[int, ...] = None,
            size: int = None,
            sparsity: float = None,
            active_size: int = None
    ):
        """
        Resolve all SDS components from the given subset of them.

        As all components are interdependent, it is convenient to define SDS by specifying
        only a necessary subset of them and let the others be induced.
        """
        if shape is None and size is None:
            # defined: sparsity & active size
            #   resolve size and shape
            size = round(active_size / sparsity)
            shape = (size, )
        else:
            # defined: shape | size + sparsity | active size
            #   1) resolve size; 2) resolve sparsity and active size

            if size is None:
                shape = tuple(shape)
                size = np.prod(shape)
            else:
                shape = (size,)

            if active_size is None:
                active_size = int(size * sparsity)
            else:
                sparsity = active_size / size

        return shape, size, sparsity, active_size

    @staticmethod
    def make(sds: Sds | TSdsShortNotation) -> Sds:
        if isinstance(sds, Sds):
            return sds

        if isinstance(sds, dict):
            # full key-value notation aka TConfig
            return Sds(**sds)

        # otherwise, a short notation is expected, which is a two-element sequence-like object
        return Sds(short_notation=sds)
