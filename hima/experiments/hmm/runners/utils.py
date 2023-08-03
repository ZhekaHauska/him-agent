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


def get_surprise_2(probs, obs, mode='bernoulli', normalize=True):
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
    def clip(p):
        return np.clip(p, 1e-7, 1.)

    surprise = -np.sum(np.log(clip(probs[obs])))
    if mode == 'bernoulli':
        not_in_obs_mask = np.ones_like(probs, dtype=bool)
        not_in_obs_mask[obs] = False

        surprise += -np.sum(np.log(clip(1. - probs[not_in_obs_mask])))
        if normalize:
            surprise /= len(probs)
    elif mode == 'categorical':
        if normalize:
            surprise /= len(obs)
    else:
        raise ValueError(f'There is no such mode "{mode}"')

    return surprise
