#  Copyright (c) 2022 Autonomous Non-Profit Organization "Artificial Intelligence Research
#  Institute" (AIRI); Moscow Institute of Physics and Technology (National Research University).
#  All rights reserved.
#
#  Licensed under the AGPLv3 license. See LICENSE in the project root for license information.

from typing import Any

import numpy as np
from htm.bindings.sdr import SDR

from hima.common.config_utils import extracted_type, resolve_init_params, resolve_absolute_quantity
from hima.common.sdr import SparseSdr
from hima.common.sds import Sds
from hima.experiments.temporal_pooling.blocks.base_block_stats import BlockStats
from hima.experiments.temporal_pooling.sdr_seq_stats import SdrSequenceStats


class TemporalPoolerBlockStats(BlockStats):
    seq_stats: SdrSequenceStats

    def __init__(self, output_sds: Sds):
        super(TemporalPoolerBlockStats, self).__init__(output_sds)
        self.seq_stats = SdrSequenceStats(self.output_sds)

    def update(self, current_output_sdr: SparseSdr):
        self.seq_stats.update(current_output_sdr)

    def step_metrics(self) -> dict[str, Any]:
        return self.seq_stats.step_metrics()

    def final_metrics(self) -> dict[str, Any]:
        return self.seq_stats.final_metrics()


class TemporalPoolerBlock:
    id: int
    name: str
    feedforward_sds: Sds
    output_sds: Sds

    output_sdr: SparseSdr
    tp: Any
    stats: TemporalPoolerBlockStats

    _input_active_cells: SDR
    _input_predicted_cells: SDR

    def __init__(self, feedforward_sds: Sds, output_sds: Sds, tp: Any):
        self.feedforward_sds = feedforward_sds
        self.output_sds = output_sds
        self.tp = tp
        self.stats = TemporalPoolerBlockStats(self.output_sds)

        self.output_sdr = []
        self._input_active_cells = SDR(self.feedforward_sds.size)
        self._input_predicted_cells = SDR(self.feedforward_sds.size)

    @property
    def tag(self) -> str:
        return f'{self.id}_tp'

    def reset(self):
        self.tp.reset()
        self.output_sdr = []

    def reset_stats(self, stats: TemporalPoolerBlockStats = None):
        if stats is None:
            self.stats = TemporalPoolerBlockStats(self.output_sds)
        else:
            self.stats = stats

    def compute(self, active_input: SparseSdr, predicted_input: SparseSdr, learn: bool):
        self._input_active_cells.sparse = active_input.copy()
        self._input_predicted_cells.sparse = predicted_input.copy()

        output_sdr: SDR = self.tp.compute(
            self._input_active_cells, self._input_predicted_cells, learn
        )
        self.output_sdr = np.array(output_sdr.sparse, copy=True)

        self.stats.update(self.output_sdr)
        return self.output_sdr


def resolve_tp(tp_config, feedforward_sds: Sds, output_sds: Sds, seed: int):
    tp_config, tp_type = extracted_type(tp_config)

    if tp_type == 'UnionTp':
        from hima.modules.htm.spatial_pooler import UnionTemporalPooler
        tp_config = resolve_init_params(
            tp_config,
            inputDimensions=feedforward_sds.shape, localAreaDensity=output_sds.sparsity,
            columnDimensions=output_sds.shape, maxUnionActivity=output_sds.sparsity,
            potentialRadius=feedforward_sds.size, seed=seed
        )
        tp = UnionTemporalPooler(**tp_config)

    elif tp_type == 'AblationUtp':
        from hima.experiments.temporal_pooling.ablation_utp import AblationUtp
        tp_config = resolve_init_params(
            tp_config,
            inputDimensions=feedforward_sds.shape, localAreaDensity=output_sds.sparsity,
            columnDimensions=output_sds.shape, maxUnionActivity=output_sds.sparsity,
            potentialRadius=feedforward_sds.size, seed=seed
        )
        tp = AblationUtp(**tp_config)

    elif tp_type == 'CustomUtp':
        from hima.experiments.temporal_pooling.custom_utp import CustomUtp
        tp_config = resolve_init_params(
            tp_config,
            inputDimensions=feedforward_sds.shape,
            columnDimensions=output_sds.shape, union_sdr_sparsity=output_sds.sparsity,
            seed=seed
        )
        tp = CustomUtp(**tp_config)

    elif tp_type == 'SandwichTp':
        from hima.experiments.temporal_pooling.sandwich_tp import SandwichTp
        tp_config = resolve_init_params(tp_config, seed=seed)

        # hacky hack to set pooling restriction propagated to upper SP
        if 'max_intermediate_used' in tp_config and tp_config['max_intermediate_used'] is not None:
            # FIXME: due to the wandb bug https://github.com/wandb/client/issues/3555 I have to
            # explicitly use only the float (i.e. relative) version that counts in active sizes
            tp_config['max_intermediate_used'] = resolve_absolute_quantity(
                float(tp_config['max_intermediate_used']),
                feedforward_sds.active_size if tp_config['only_upper'] else output_sds.active_size
            )

        if not tp_config['only_upper']:
            tp_config['lower_sp_conf'] = resolve_init_params(
                tp_config['lower_sp_conf'],
                inputDimensions=feedforward_sds.shape,
                columnDimensions=output_sds.shape, localAreaDensity=output_sds.sparsity,
                potentialRadius=feedforward_sds.size
            )
            tp_config['upper_sp_conf'] = resolve_init_params(
                tp_config['upper_sp_conf'],
                inputDimensions=output_sds.shape,
                columnDimensions=output_sds.shape, localAreaDensity=output_sds.sparsity,
                potentialRadius=output_sds.size
            )
        else:
            tp_config['upper_sp_conf'] = resolve_init_params(
                tp_config['upper_sp_conf'],
                inputDimensions=feedforward_sds.shape,
                columnDimensions=output_sds.shape, localAreaDensity=output_sds.sparsity,
                potentialRadius=output_sds.size
            )

        tp = SandwichTp(**tp_config)

    else:
        raise KeyError(f'Temporal Pooler type "{tp_type}" is not supported')

    tp_block = TemporalPoolerBlock(feedforward_sds=feedforward_sds, output_sds=output_sds, tp=tp)
    return tp_block
