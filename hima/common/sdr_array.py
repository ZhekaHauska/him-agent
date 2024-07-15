#  Copyright (c) 2023 Autonomous Non-Profit Organization "Artificial Intelligence Research
#  Institute" (AIRI); Moscow Institute of Physics and Technology (National Research University).
#  All rights reserved.
#
#  Licensed under the AGPLv3 license. See LICENSE in the project root for license information.

from __future__ import annotations

from numpy import typing as npt

from hima.common.sdr import RateSdr, AnySparseSdr


class SdrArray:
    sparse: list[RateSdr] | None
    dense: npt.NDArray[float] | None

    sdr_size: int

    def __init__(self, sparse=None, dense=None, sdr_size: int = None):
        self.sparse = sparse
        self.dense = dense

        if sdr_size is not None:
            self.sdr_size = sdr_size
        else:
            self.sdr_size = dense.shape[1]

    def __len__(self):
        return len(self.sparse) if self.sparse is not None else self.dense.shape[0]

    def get_sdr(self, ind: int, binary: bool = False) -> AnySparseSdr:
        return self.sparse[ind].sdr if binary else self.sparse[ind]

    def create_modified(self, fn):
        sparse, dense = None, None

        if self.sparse is not None:
            sparse = [RateSdr(sdr=sdr.sdr, values=fn(sdr.values)) for sdr in self.sparse]
        if self.dense is not None:
            dense = fn(self.dense)

        return SdrArray(sparse=sparse, dense=dense, sdr_size=self.sdr_size)


def fill_dense(dst: npt.NDArray[float], srs: SdrArray, ixs):
    for i, ix in enumerate(ixs):
        rate_sdr = srs.sparse[ix]
        dst[i, rate_sdr.sdr] = rate_sdr.values
