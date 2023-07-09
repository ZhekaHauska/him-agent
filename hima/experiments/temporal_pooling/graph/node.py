#  Copyright (c) 2022 Autonomous Non-Profit Organization "Artificial Intelligence Research
#  Institute" (AIRI); Moscow Institute of Physics and Technology (National Research University).
#  All rights reserved.
#
#  Licensed under the AGPLv3 license. See LICENSE in the project root for license information.

from abc import ABC, abstractmethod


class Node(ABC):
    @abstractmethod
    def forward(self) -> None:
        raise NotImplementedError()

    @abstractmethod
    def __repr__(self) -> str:
        raise NotImplementedError()


class Stretchable(ABC):
    @abstractmethod
    def fit_dimensions(self) -> bool:
        raise NotImplementedError()


class Stateful(ABC):
    @abstractmethod
    def reset(self):
        raise NotImplementedError()
