#  Copyright (c) 2023 Autonomous Non-Profit Organization "Artificial Intelligence Research
#  Institute" (AIRI); Moscow Institute of Physics and Technology (National Research University).
#  All rights reserved.
#
#  Licensed under the AGPLv3 license. See LICENSE in the project root for license information.
import numpy as np
from hima.modules.belief.utils import normalize


def make_decoder(encoder, decoder, decoder_conf):
    if decoder == 'naive':
        from hima.experiments.temporal_pooling.stp.sp_decoder import SpatialPoolerDecoder
        return SpatialPoolerDecoder(encoder)
    elif decoder == 'learned':
        from hima.experiments.temporal_pooling.stp.sp_decoder import SpatialPoolerLearnedDecoder
        return SpatialPoolerLearnedDecoder(encoder, **decoder_conf)
    else:
        raise ValueError(f'Decoder {decoder} is not supported')


def print_digest(metrics: dict):
    ep_len = int(metrics['main_metrics/steps'])
    digest = f'{ep_len:2d}:'

    if 'main_metrics/reward' in metrics:
        ep_return = metrics['main_metrics/reward']
        digest += f' R = {ep_return:5.2f}'
    if 'sr/td_error' in metrics:
        td_error = metrics['sr/td_error']
        digest += f' | TD = {td_error:12.8f}'
    if 'layer/surprise_hidden' in metrics:
        surprise = metrics['layer/surprise_hidden']
        digest += f' | Srp = {surprise:.7f}'
    if 'layer/loss' in metrics:
        loss = metrics['layer/loss']
        digest += f'| Loss = {loss:.7f}'
    print(digest)

