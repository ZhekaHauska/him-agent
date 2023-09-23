#  Copyright (c) 2023 Autonomous Non-Profit Organization "Artificial Intelligence Research
#  Institute" (AIRI); Moscow Institute of Physics and Technology (National Research University).
#  All rights reserved.
#
#  Licensed under the AGPLv3 license. See LICENSE in the project root for license information.


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
    ep_return = metrics['main_metrics/reward']
    td_error = metrics['agent/td_error']
    surprise = metrics['layer/surprise_hidden']
    stats = f'{ep_len:2d}: R = {ep_return:5.2f} | TD = {td_error:12.8f} | Srp = {surprise:.7f}'

    if 'layer/loss' in metrics:
        loss = metrics['layer/loss']
        stats += f'| Loss = {loss:.7f}'

    print(stats)
