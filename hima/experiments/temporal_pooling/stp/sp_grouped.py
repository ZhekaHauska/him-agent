#  Copyright (c) 2022 Autonomous Non-Profit Organization "Artificial Intelligence Research
#  Institute" (AIRI); Moscow Institute of Physics and Technology (National Research University).
#  All rights reserved.
#
#  Licensed under the AGPLv3 license. See LICENSE in the project root for license information.
import numpy as np

from hima.common.float_sdr import FloatSparseSdr
from hima.common.sds import Sds
from hima.experiments.temporal_pooling.stp.sp import SpatialPooler, SpOutputMode


class SpatialPoolerGrouped(SpatialPooler):
    group_size: int
    group_shifts: np.ndarray

    cross_group_inhibition_scale: float

    def __init__(
            self, *,
            feedforward_sds: Sds,
            # newborn / mature
            initial_max_rf_sparsity: float, target_max_rf_sparsity: float,
            initial_rf_to_input_ratio: float, target_rf_to_input_ratio: float,
            output_sds: Sds,
            learning_rate: float,
            newborn_pruning_cycle: float, newborn_pruning_stages: int,
            prune_grow_cycle: float,
            boosting_k: float, seed: int,
            adapt_to_ff_sparsity: bool = True,
            newborn_pruning_mode: str = 'powerlaw',
            cross_group_inhibition_scale: float = 0.1
    ):
        super().__init__(
            feedforward_sds=feedforward_sds, initial_max_rf_sparsity=initial_max_rf_sparsity,
            target_max_rf_sparsity=target_max_rf_sparsity,
            initial_rf_to_input_ratio=initial_rf_to_input_ratio,
            target_rf_to_input_ratio=target_rf_to_input_ratio, output_sds=output_sds,
            learning_rate=learning_rate, newborn_pruning_cycle=newborn_pruning_cycle,
            newborn_pruning_stages=newborn_pruning_stages, prune_grow_cycle=prune_grow_cycle,
            boosting_k=boosting_k, seed=seed, adapt_to_ff_sparsity=adapt_to_ff_sparsity,
            newborn_pruning_mode=newborn_pruning_mode
        )
        assert self.output_size % self.n_groups == 0, f'Non-divisible groups {self.output_sds}'
        self.group_size = self.output_sds.size // self.n_groups
        self.group_shifts = np.arange(self.n_groups) * self.group_size
        self.cross_group_inhibition_scale = cross_group_inhibition_scale
        self.sub_winners = []

    def select_winners(self, learn=False):
        overlaps_grouped = self.potentials.reshape(self.n_groups, -1)

        if learn:
            # find sub winners too
            winners_grouped = np.argpartition(overlaps_grouped, -2, axis=-1)[:, -2:]
            winners = winners_grouped[:, -1].flatten() + self.group_shifts
            sub_winners = winners_grouped[:, -2].flatten() + self.group_shifts

            # keep only sub_winners stronger than others
            weakest_overlap = self.potentials[winners].min()
            self.sub_winners = sub_winners[self.potentials[sub_winners] > weakest_overlap]
        else:
            # regular grouped winners
            winners_grouped = np.argpartition(overlaps_grouped, -1, axis=-1)[:, -1:]
            winners = winners_grouped.flatten() + self.group_shifts
            self.sub_winners = []

        self.winners = winners[self.potentials[winners] > 0]

    def reinforce_winners(self, learn: bool):
        if not learn:
            return

        self.stdp(self.winners, self.potentials[self.winners])
        self.stdp(
            self.sub_winners, self.potentials[self.sub_winners],
            modulation=-self.cross_group_inhibition_scale
        )

    @property
    def n_groups(self):
        return self.output_sds.active_size
