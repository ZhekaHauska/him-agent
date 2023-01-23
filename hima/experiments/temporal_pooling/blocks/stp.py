#  Copyright (c) 2022 Autonomous Non-Profit Organization "Artificial Intelligence Research
#  Institute" (AIRI); Moscow Institute of Physics and Technology (National Research University).
#  All rights reserved.
#
#  Licensed under the AGPLv3 license. See LICENSE in the project root for license information.

from typing import Any

import numpy as np
from htm.bindings.sdr import SDR

from hima.common.config.values import resolve_init_params
from hima.common.config.base import resolve_absolute_quantity, extracted
from hima.common.sdr import SparseSdr
from hima.common.sds import Sds
from hima.experiments.temporal_pooling.blocks.graph import Block


class SpatiotemporalPoolerBlock(Block):
    family = "spatiotemporal_pooler"

    FEEDFORWARD = 'feedforward'
    OUTPUT = 'output'
    supported_streams = {FEEDFORWARD, OUTPUT}

    stp: Any
    _active_input: SDR
    _active_output: SDR

    def __init__(self, id: int, name: str, **stp_config):
        super(SpatiotemporalPoolerBlock, self).__init__(id, name)

        stp_config, ff_sds, output_sds = extracted(stp_config, 'ff_sds', 'output_sds')

        self.register_stream(self.FEEDFORWARD).resolve_sds(ff_sds)
        self.register_stream(self.OUTPUT).resolve_sds(output_sds)

        self._stp_config = stp_config

    def build(self, **kwargs):
        stp_config = self._stp_config
        ff_sds = self.streams[self.FEEDFORWARD].sds
        output_sds = self.streams[self.OUTPUT].sds

        self.stp = resolve_stp(self._stp_config, feedforward_sds=ff_sds, output_sds=output_sds)

        self._active_input = SDR(ff_sds.size)
        self._active_output = SDR(output_sds.size)

    def reset(self):
        self.stp.reset()
        super(SpatiotemporalPoolerBlock, self).reset()

    def compute(self, data: dict[str, SparseSdr], **kwargs):
        self._compute(**data, **kwargs)

    def _compute(
            self, feedforward: SparseSdr, learn: bool, predicted_feedforward: SparseSdr = None
    ):
        self._active_input.sparse = feedforward.copy()
        assert predicted_feedforward is None, 'Not implemented'

        output_sdr: SDR = self.stp.compute(
            self._active_input, self._active_input, learn
        )
        self.streams[self.OUTPUT].sdr = np.array(output_sdr.sparse, copy=True)


def resolve_stp(stp_config, feedforward_sds: Sds, output_sds: Sds):
    stp_config, stp_type = extracted_type(stp_config)

    if stp_type == 'UnionTp':
        from hima.modules.htm.spatial_pooler import UnionTemporalPooler
        stp_config = resolve_init_params(
            stp_config,
            inputDimensions=feedforward_sds.shape, localAreaDensity=output_sds.sparsity,
            columnDimensions=output_sds.shape, maxUnionActivity=output_sds.sparsity,
            potentialRadius=feedforward_sds.size
        )
        stp = UnionTemporalPooler(**stp_config)

    elif stp_type == 'AblationUtp':
        from hima.experiments.temporal_pooling._depr.ablation_utp import AblationUtp
        stp_config = resolve_init_params(
            stp_config,
            inputDimensions=feedforward_sds.shape, localAreaDensity=output_sds.sparsity,
            columnDimensions=output_sds.shape, maxUnionActivity=output_sds.sparsity,
            potentialRadius=feedforward_sds.size
        )
        stp = AblationUtp(**stp_config)

    elif stp_type == 'CustomUtp':
        from hima.experiments.temporal_pooling._depr.custom_utp import CustomUtp
        stp_config = resolve_init_params(
            stp_config,
            inputDimensions=feedforward_sds.shape,
            columnDimensions=output_sds.shape, union_sdr_sparsity=output_sds.sparsity
        )
        stp = CustomUtp(**stp_config)

    elif stp_type == 'SandwichTp':
        from hima.experiments.temporal_pooling._depr.sandwich_tp import SandwichTp

        # hacky hack to set pooling restriction propagated to upper SP
        if 'max_intermediate_used' in stp_config and stp_config['max_intermediate_used'] is not None:
            # FIXME: due to the wandb bug https://github.com/wandb/client/issues/3555 I have to
            # explicitly use only the float (i.e. relative) version that counts in active sizes
            stp_config['max_intermediate_used'] = resolve_absolute_quantity(
                float(stp_config['max_intermediate_used']),
                feedforward_sds.active_size if stp_config['only_upper'] else output_sds.active_size
            )

        if not stp_config['only_upper']:
            stp_config['lower_sp_conf'] = resolve_init_params(
                stp_config['lower_sp_conf'],
                inputDimensions=feedforward_sds.shape,
                columnDimensions=output_sds.shape, localAreaDensity=output_sds.sparsity,
                potentialRadius=feedforward_sds.size
            )
            stp_config['upper_sp_conf'] = resolve_init_params(
                stp_config['upper_sp_conf'],
                inputDimensions=output_sds.shape,
                columnDimensions=output_sds.shape, localAreaDensity=output_sds.sparsity,
                potentialRadius=output_sds.size
            )
        else:
            stp_config['upper_sp_conf'] = resolve_init_params(
                stp_config['upper_sp_conf'],
                inputDimensions=feedforward_sds.shape,
                columnDimensions=output_sds.shape, localAreaDensity=output_sds.sparsity,
                potentialRadius=output_sds.size
            )

        stp = SandwichTp(**stp_config)

    else:
        raise KeyError(f'Temporal Pooler type "{stp_type}" is not supported')

    return stp
