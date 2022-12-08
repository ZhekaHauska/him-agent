#  Copyright (c) 2022 Autonomous Non-Profit Organization "Artificial Intelligence Research
#  Institute" (AIRI); Moscow Institute of Physics and Technology (National Research University).
#  All rights reserved.
#
#  Licensed under the AGPLv3 license. See LICENSE in the project root for license information.

class RunProgress:
    epoch: int
    step: int

    def __init__(self):
        self.epoch = -1
        self.step = -1

    def next_epoch(self):
        self.epoch += 1

    def next_step(self):
        self.step += 1
