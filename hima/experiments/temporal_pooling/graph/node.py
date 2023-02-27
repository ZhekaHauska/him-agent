#  Copyright (c) 2022 Autonomous Non-Profit Organization "Artificial Intelligence Research
#  Institute" (AIRI); Moscow Institute of Physics and Technology (National Research University).
#  All rights reserved.
#
#  Licensed under the AGPLv3 license. See LICENSE in the project root for license information.

from abc import ABC, abstractmethod
from typing import Iterator


ListIndentFirst = '  * '
ListIndentRest = '    '


class Node(ABC):
    @abstractmethod
    def expand(self) -> Iterator['Node']:
        raise NotImplementedError()

    @abstractmethod
    def align_dimensions(self) -> bool:
        raise NotImplementedError()

    @abstractmethod
    def forward(self) -> None:
        raise NotImplementedError()

    @abstractmethod
    def __repr__(self) -> str:
        raise NotImplementedError()
