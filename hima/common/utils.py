#  Copyright (c) 2022 Autonomous Non-Profit Organization "Artificial Intelligence Research
#  Institute" (AIRI); Moscow Institute of Physics and Technology (National Research University).
#  All rights reserved.
#
#  Licensed under the AGPLv3 license. See LICENSE in the project root for license information.
from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt

DecayingValue = tuple[float, float]
Coord2d = tuple[int, int]


def isnone(x, default):
    """Return x if it's not None, or default value instead."""
    return x if x is not None else default


def ensure_list(arr: Any | list[Any] | None) -> list[Any] | None:
    """Wrap single value to list or return list as it is."""
    if arr is not None and not isinstance(arr, list):
        arr = [arr]
    return arr


def safe_ith(arr: list | None, ind: int, default: Any = None) -> Any | None:
    """Perform safe index access. If array is None, returns default."""
    if arr is not None:
        return arr[ind]
    return default


def exp_sum(ema, decay, val):
    """Return updated exponential moving average (EMA) with the added new value."""
    return ema * decay + val


def lin_sum(x, lr, y):
    """Return linear sum."""
    return x + lr * (y - x)


def update_slice_exp_sum(s, ind, decay, val):
    """Update EMA only for specified slice."""
    s[ind] *= decay
    s[ind] += val


def update_slice_lin_sum(s, ind, lr, val):
    """Update slice value estimate with specified learning rate."""
    s[ind] = (1 - lr) * s[ind] + lr * val


def update_exp_trace(traces, tr, decay, val=1., with_reset=False):
    """Update an exponential trace."""
    traces *= decay
    if with_reset:
        traces[tr] = val
    else:
        traces[tr] += val


def exp_decay(value: DecayingValue) -> DecayingValue:
    """Apply decay to specified DecayingValue."""
    x, decay = value
    return x * decay, decay


def multiply_decaying_value(value: DecayingValue, alpha: float) -> DecayingValue:
    """Return new tuple with the first value multiplied by the specified factor."""
    x, decay = value
    return x * alpha, decay


def softmax(
        x: npt.NDArray[float], *, temp: float = None, beta: float = None, axis: int = -1
) -> npt.NDArray[float]:
    """
    Compute softmax values for a vector `x` with a given temperature or inverse temperature.
    The softmax operation is applied over the last axis by default, or over the specified axis.
    """
    beta = isnone(beta, 1.0)
    temp = isnone(temp, 1 / beta)
    temp = clip(temp, 1e-5, 1e+4)

    e_x = np.exp((x - np.max(x, axis=axis, keepdims=True)) / temp)
    return e_x / np.sum(e_x, axis=axis, keepdims=True)


def symlog(x: npt.NDArray[float]) -> npt.NDArray[float]:
    """Compute symlog values for a vector `x`. It's an inverse operation for symexp."""
    return np.sign(x) * np.log(np.abs(x) + 1)


def symexp(x: npt.NDArray[float]) -> npt.NDArray[float]:
    """Compute symexp values for a vector `x`. It's an inverse operation for symlog."""
    return np.sign(x) * (np.exp(np.abs(x)) - 1.0)


def clip(x: Any, low=None, high=None) -> Any:
    """Clip the value with the provided thresholds. NB: doesn't support vectorization."""

    # both x < None and x > None are False, so consider them as safeguards
    if x < low:
        x = low
    elif x > high:
        x = high
    return x


def safe_divide(x, y: int | float):
    """
    Return x / y or just x itself if y == 0 preventing NaNs.
    Warning: it may not work as you might expect for floats, use it only when you need exact match!
    """
    return x / y if y != 0 else x


def prepend_dict_keys(d: dict[str, Any], prefix, separator='/'):
    """Add specified prefix to all the dict keys."""
    return {
        f'{prefix}{separator}{k}': d[k]
        for k in d
    }


def to_gray_img(
        img: npt.NDArray, like: tuple[int, int] | npt.NDArray = None
) -> npt.NDArray[np.uint8]:
    img = img * 255
    if like is not None:
        if isinstance(like, np.ndarray):
            shape = like.shape
        else:
            shape = like
        img = img.reshape(shape)

    return img.astype(np.uint8)


def standardize(value, mean, std):
    return (value - mean) / std
