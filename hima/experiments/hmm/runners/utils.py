#  Copyright (c) 2022 Autonomous Non-Profit Organization "Artificial Intelligence Research
#  Institute" (AIRI); Moscow Institute of Physics and Technology (National Research University).
#  All rights reserved.
#
#  Licensed under the AGPLv3 license. See LICENSE in the project root for license information.
import numpy as np


def get_surprise(probs, obs, mode='bernoulli', normalize=True):
    """
    Calculate the surprise -log(p(o)), where o is observation

    'probs': distribution parameters

    'obs': indexes of variables in state 1

    'mode': bernoulli | categorical

        bernoulli
            'probs' are parameters of Bernoulli distributed vector

        categorical
            'probs' are parameters of Categorical distributed vector

    'normalize': bool
    """
    is_coincide = np.isin(
        np.arange(len(probs)), obs
    )

    surprise = - np.sum(
        np.log(
            np.clip(probs[is_coincide], 1e-7, 1)
        )
    )

    if mode == 'bernoulli':
        surprise += - np.sum(
            np.log(
                np.clip(1 - probs[~is_coincide], 1e-7, 1)
            )
        )
        if normalize:
            surprise /= len(probs)
    elif mode == 'categorical':
        if normalize:
            surprise /= len(obs)
    else:
        raise ValueError(f'There is no such mode "{mode}"')

    return surprise
