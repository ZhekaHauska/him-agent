#  Copyright (c) 2022 Autonomous Non-Profit Organization "Artificial Intelligence Research
#  Institute" (AIRI); Moscow Institute of Physics and Technology (National Research University).
#  All rights reserved.
#
#  Licensed under the AGPLv3 license. See LICENSE in the project root for license information.

from __future__ import annotations

from hima.experiments.temporal_pooling.test_on_obs_layered import resolve_epoch_runs


class RunSetup:
    n_sequences: int
    steps_per_sequence: int | None
    sequence_repeats: int
    epochs: int
    log_repeat_schedule: int
    log_epoch_schedule: int

    def __init__(
            self, n_sequences: int, steps_per_sequence: int | None,
            sequence_repeats: int, epochs: int, total_repeats: int,
            log_repeat_schedule: int = 1, log_epoch_schedule: int = 1
    ):
        self.n_sequences = n_sequences
        self.steps_per_sequence = steps_per_sequence
        self.sequence_repeats, self.epochs = resolve_epoch_runs(
            sequence_repeats, epochs, total_repeats
        )
        self.log_repeat_schedule = log_repeat_schedule
        self.log_epoch_schedule = log_epoch_schedule
