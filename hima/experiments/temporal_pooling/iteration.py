#  Copyright (c) 2022 Autonomous Non-Profit Organization "Artificial Intelligence Research
#  Institute" (AIRI); Moscow Institute of Physics and Technology (National Research University).
#  All rights reserved.
#
#  Licensed under the AGPLv3 license. See LICENSE in the project root for license information.

from __future__ import annotations


class IterationConfig:
    epochs: int

    sequences: int
    sequence_repeats: int

    elements: int
    element_repeats: int

    def __init__(
            self,
            epochs: int,
            sequences: int | tuple[int, int],
            elements: int | tuple[int, int]
    ):
        self.epochs = epochs
        self.sequences, self.sequence_repeats = self._split_repeats(sequences)
        self.elements, self.element_repeats = self._split_repeats(elements)

    @staticmethod
    def _split_repeats(
            with_or_without_repeats: int | tuple[int, int], repeats: int = 1
    ) -> tuple[int, int]:
        if isinstance(with_or_without_repeats, int):
            return with_or_without_repeats, repeats
        return with_or_without_repeats
