#  Copyright (c) 2022 Autonomous Non-Profit Organization "Artificial Intelligence Research
#  Institute" (AIRI); Moscow Institute of Physics and Technology (National Research University).
#  All rights reserved.
#
#  Licensed under the AGPLv3 license. See LICENSE in the project root for license information.

from __future__ import annotations

import numpy as np

from hima.common.config.values import get_unresolved_value
from hima.common.sdr_sampling import (
    sample_noisy_sdr_rate_sdr, sample_noisy_rates_rate_sdr,
    sample_noisy_sdr
)
from hima.common.sdrr import RateSdr
from hima.common.sds import Sds
from hima.experiments.temporal_pooling.graph.node import Node, Stretchable
from hima.experiments.temporal_pooling.graph.stream import Stream, SdrStream


class Pipe(Node):
    """Pipe connects two blocks' streams. Thus, both streams operate in the same SDS."""

    src: Stream
    dst: Stream

    def __init__(self, src: Stream, dst: Stream):
        assert src.is_sdr == dst.is_sdr
        self.src = src
        self.dst = dst

    def expand(self):
        yield self

    def forward(self) -> None:
        self.dst.set(self.src.get())

    def __repr__(self) -> str:
        return f'{self.src} -> {self.dst}'


class SdrPipe(Stretchable, Pipe):
    """Pipe connects two blocks' streams. Thus, both streams operate in the same SDS."""

    src: SdrStream
    dst: SdrStream

    def __init__(self, src: SdrStream, dst: SdrStream, sds: Sds = get_unresolved_value()):
        super().__init__(src, dst)

        self.src.set_sds(sds)
        self.src.exchange_sds(self.dst)

    def fit_dimensions(self) -> bool:
        """
        Align dimensions of the streams connected via this pipe.
        Returns True if the streams' dimensions are resolved and correctly aligned,
        and False otherwise.
        """
        self.src.exchange_sds(self.dst)
        return self.src.valid_sds

    def forward(self) -> None:
        self.dst.set(self.src.get())

    def __repr__(self) -> str:
        pipe_str = super().__repr__()
        return f'{pipe_str} | {self.src.sds}'


class NoisySdrPipe(SdrPipe):
    """Pipe connects two blocks' streams. Thus, both streams operate in the same SDS."""

    src: SdrStream
    dst: SdrStream

    def __init__(
            self, seed: int,
            src: SdrStream, dst: SdrStream, sds: Sds = get_unresolved_value(),
            noise_frac: float = 0.12
    ):
        super().__init__(src, dst)

        self.noise_frac = noise_frac
        self.rng = np.random.default_rng(seed)

    def fit_dimensions(self) -> bool:
        """
        Align dimensions of the streams connected via this pipe.
        Returns True if the streams' dimensions are resolved and correctly aligned,
        and False otherwise.
        """
        self.src.exchange_sds(self.dst)
        return self.src.valid_sds

    def forward(self) -> None:
        sdr = self.src.get()
        if isinstance(sdr, RateSdr):
            sdr = sample_noisy_sdr_rate_sdr(self.rng, self.src.sds, sdr, self.noise_frac)
            sdr = sample_noisy_rates_rate_sdr(self.rng, sdr, self.noise_frac)
        else:
            sdr = sample_noisy_sdr(self.rng, self.src.sds, sdr, self.noise_frac)

        self.dst.set(sdr)

    def __repr__(self) -> str:
        pipe_str = f'{self.src} ~> {self.noise_frac} ~> {self.dst}'
        return f'{pipe_str} | {self.src.sds}'
