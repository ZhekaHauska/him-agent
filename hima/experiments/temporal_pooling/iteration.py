#  Copyright (c) 2022 Autonomous Non-Profit Organization "Artificial Intelligence Research
#  Institute" (AIRI); Moscow Institute of Physics and Technology (National Research University).
#  All rights reserved.
#
#  Licensed under the AGPLv3 license. See LICENSE in the project root for license information.

from __future__ import annotations

import math

from hima.common.utils import isnone, clip


class IterationConfig:
    epochs: int

    total_sequences: int
    resample_frequency: int | None
    resample_with_replacement: bool
    sequences: int
    sequence_repeats: int

    elements: int
    element_repeats: int

    def __init__(
            self,
            epochs: int,
            sequences: int | tuple[int, int],
            elements: int | tuple[int, int],
            total_sequences: int = None,
            resample_frequency: int | None = None,
            resample_with_replacement: bool = False,
    ):
        self.epochs = epochs
        self.sequences, self.sequence_repeats = self._split_repeats(sequences)
        self.elements, self.element_repeats = self._split_repeats(elements)

        self.resample_frequency = clip(isnone(resample_frequency, epochs), low=1, high=epochs)
        self.resample_with_replacement = resample_with_replacement
        self.total_sequences = self.get_total_sequences(total_sequences)

    @staticmethod
    def _split_repeats(
            with_or_without_repeats: int | tuple[int, int], repeats: int = 1
    ) -> tuple[int, int]:
        if isinstance(with_or_without_repeats, int):
            return with_or_without_repeats, repeats
        return with_or_without_repeats

    def get_total_sequences(self, total_sequences: int | None):
        if not self.resample_with_replacement:
            n_stages = math.ceil(self.epochs / self.resample_frequency)
            return self.sequences * n_stages

        assert isinstance(total_sequences, int)
        return total_sequences
