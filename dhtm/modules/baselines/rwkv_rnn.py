#  Copyright (c) 2023 Autonomous Non-Profit Organization "Artificial Intelligence Research
#  Institute" (AIRI); Moscow Institute of Physics and Technology (National Research University).
#  All rights reserved.
#
#  Licensed under the AGPLv3 license. See LICENSE in the project root for license information.

import numpy as np
import torch
from torch import nn


class RwkvChannelMix(nn.Module):
    def __init__(self, n_hidden, dim_ffn=-1):
        super().__init__()

        self.n_hidden = n_hidden
        self.ln = nn.LayerNorm(self.n_hidden)

        if dim_ffn <= 0:
            dim_ffn = self.n_hidden * 4

        with torch.no_grad():  # fancy init of time_mix
            # NB: (1,1, ..,) shape is probably for batch processing?
            # ddd = torch.ones(1, 1, self.n_hidden)
            # ddd[0, 0] = torch.arange(0, self.n_hidden) / self.n_hidden
            ddd = torch.arange(0, self.n_hidden) / self.n_hidden

            ratio_1_to_almost0 = 0.5
            self.time_mix_k = nn.Parameter(torch.pow(ddd, ratio_1_to_almost0))
            self.time_mix_r = nn.Parameter(torch.pow(ddd, ratio_1_to_almost0))

        self.key = nn.Linear(self.n_hidden, dim_ffn, bias=False)
        self.receptance = nn.Linear(self.n_hidden, self.n_hidden, bias=False)
        self.value = nn.Linear(dim_ffn, self.n_hidden, bias=False)

    # @torch.jit.script_method
    def forward(self, x, state):
        # print(f'{x.shape=}')
        xk = x * self.time_mix_k + state * (1 - self.time_mix_k)
        xr = x * self.time_mix_r + state * (1 - self.time_mix_r)
        next_state = x
        # print(f'{xk.shape=}')

        r = torch.sigmoid(self.receptance(xr))
        # print(f'{r.shape=}')
        # print('R:', r)

        k = torch.square(torch.relu(self.key(xk)))  # square relu, primer paper
        # print(f'{k.shape=}')

        # print(
        #     f'XK: {torch.max(torch.abs(xk)).item():.3f}',
        #     f'XR: {torch.max(torch.abs(xr)).item():.3f}',
        #     f'K: {torch.max(torch.abs(k)).item():.3f}'
        # )

        return r * self.value(k), next_state


class RwkvTimeMix(nn.Module):
    def __init__(self, n_hidden, dim_att=-1):
        super().__init__()

        self.n_hidden = n_hidden
        self.ln = nn.LayerNorm(self.n_hidden)

        if dim_att <= 0:
            dim_att = self.n_hidden

        with torch.no_grad():  # fancy init
            # NB: (1,1, ..,) shape is probably for batch processing?
            # ddd = torch.ones(1, 1, self.n_hidden)
            # ddd[0, 0] = torch.arange(0, self.n_hidden) / self.n_hidden
            ddd = torch.arange(0, self.n_hidden) / self.n_hidden

            # fancy time_decay
            ratio_0_to_1 = 0.5  # 0 to 1
            ratio_1_to_almost0 = 0.5  # 1 to ~0
            _x = torch.arange(dim_att) / (dim_att - 1)
            _pow = 0.7 + 1.3 * ratio_0_to_1
            decay_speed = -5 + 8 * _x ** _pow

            self.time_decay = nn.Parameter(decay_speed)

            # fancy time_first
            zigzag = torch.tensor([(i + 1) % 3 - 1 for i in range(dim_att)]) * 0.5
            self.time_first = nn.Parameter(torch.ones(dim_att) * np.log(0.3) + zigzag)

            # fancy time_mix
            self.time_mix_k = nn.Parameter(torch.pow(ddd, ratio_1_to_almost0))
            self.time_mix_v = nn.Parameter(torch.pow(ddd, ratio_1_to_almost0) + 0.3 * ratio_0_to_1)
            self.time_mix_r = nn.Parameter(torch.pow(ddd, 0.5 * ratio_1_to_almost0))

        self.key = nn.Linear(self.n_hidden, dim_att, bias=False)
        self.value = nn.Linear(self.n_hidden, dim_att, bias=False)
        self.receptance = nn.Linear(self.n_hidden, dim_att, bias=False)
        self.output = nn.Linear(dim_att, self.n_hidden, bias=False)

    # @torch.jit.script_method
    def forward(self, x, state):
        # print(f'{x.shape=}')

        alpha, aa, bb, pp = state

        xk = x * self.time_mix_k + alpha * (1 - self.time_mix_k)
        xv = x * self.time_mix_v + alpha * (1 - self.time_mix_v)
        xr = x * self.time_mix_r + alpha * (1 - self.time_mix_r)
        next_alpha = x
        # print(f'{xk.shape=}')

        r = torch.sigmoid(self.receptance(xr))
        k = self.key(xk)
        v = self.value(xv)
        # print(
        #     f'R: {torch.max(torch.abs(r)).item():.3f}',
        #     f'K: {torch.max(torch.abs(r)).item():.3f}',
        #     f'V: {torch.max(torch.abs(r)).item():.3f}',
        # )
        # print(f'{r.shape=}', f'{k.shape=}', f'{v.shape=}')

        ww = self.time_first + k
        qq = torch.maximum(pp, ww)
        e1 = torch.exp(pp - qq)
        e2 = torch.exp(ww - qq)
        a = e1 * aa + e2 * v
        b = e1 * bb + e2
        wkv = a / b
        # print(f'{wkv.shape=}')

        ww = pp + self.time_decay
        qq = torch.maximum(ww, k)
        e1 = torch.exp(ww - qq)
        e2 = torch.exp(k - qq)

        # print(
        #     f'e1: {torch.max(torch.abs(e1)).item():.3f}',
        #     f'e2: {torch.max(torch.abs(e2)).item():.3f}',
        #     f'bb: {torch.max(torch.abs(bb)).item():.3f}',
        # )

        next_aa = e1 * aa + e2 * v
        next_bb = e1 * bb + e2
        next_pp = qq
        # print(f'{next_aa.shape=}')
        # print(f'{next_bb.shape=}')
        # print(f'{next_pp.shape=}')
        # print(
        #     f'NextAA: {torch.max(torch.abs(next_aa)).item():.3f}',
        #     f'NextBB: {torch.max(torch.abs(next_bb)).item():.3f}',
        #     f'NextPP: {torch.max(torch.abs(next_pp)).item():.3f}',
        # )

        next_state = (next_alpha, next_aa, next_bb, next_pp)
        return self.output(r * wkv), next_state


class RwkvCell(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.dropout = nn.Dropout(p=0.01)

        self.tm_layer_norm = nn.LayerNorm(self.hidden_size)
        self.time_mix = RwkvTimeMix(hidden_size, dim_att=hidden_size)

        self.cm_layer_norm = nn.LayerNorm(self.hidden_size)
        self.chan_mix = RwkvChannelMix(hidden_size, dim_ffn=hidden_size)

    def get_initial_state(self):
        state = torch.zeros(5, self.hidden_size)
        # set `pp` param to -inf, as it defines the minimum threshold in max(pp, ww) oper
        state[4] = -1e30  # -infinity
        return state

    def forward(self, x, state):
        # print(f'Input:', x)
        # print(f'State:', state)

        # extract state: (output_state, hidden_state)
        _, state = state

        # NB: state: channel state (top) then time state (bottom)
        chan_state, time_state = state[0], state[1:]

        time_x, next_time_state = self.time_mix(self.tm_layer_norm(x), time_state)
        # residual connection + dropout
        x = x + time_x
        # x = self.dropout(x + self.dropout(time_x))
        # print(f'TMix:', x)

        chan_x, next_chan_state = self.chan_mix(self.cm_layer_norm(x), chan_state)
        # residual connection + dropout
        x = x + chan_x
        # x = self.dropout(x + self.dropout(chan_x))
        # print(f'TCha:', x)

        # NB: state: channel state (top) then time state (bottom)
        state = torch.stack((next_chan_state, ) + next_time_state)
        # print(f'OutState:', state)
        # print('====================================================')
        # print()
        return x, state
