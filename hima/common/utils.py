#  Copyright (c) 2022 Autonomous Non-Profit Organization "Artificial Intelligence Research
#  Institute" (AIRI); Moscow Institute of Physics and Technology (National Research University).
#  All rights reserved.
#
#  Licensed under the AGPLv3 license. See LICENSE in the project root for license information.

from functools import wraps
from timeit import default_timer as timer
from typing import Any, Union, Optional

import numpy as np


DecayingValue = tuple[float, float]
Coord2d = tuple[int, int]


def isnone(x, default):
    """Return x if it's not None, or default value instead."""
    return x if x is not None else default


def ensure_list(arr: Optional[Union[Any, list[Any]]]) -> Optional[list[Any]]:
    """Wraps single value to list or return list as it is."""
    if arr is not None and not isinstance(arr, list):
        arr = [arr]
    return arr


def safe_ith(arr: Optional[list], ind: int, default: Any = None) -> Optional[Any]:
    """Performs safe index access. If array is None, returns default."""
    if arr is not None:
        return arr[ind]
    return default


def wrap(obj, *wrappers):
    """Sequentially wraps object."""
    for wrapper in wrappers:
        obj = wrapper(obj)
    return obj


def timed(f):
    """Wraps function with the timer that returns tuple: result, elapsed_time."""
    @wraps(f)
    def _wrap(*args, **kw):
        start = timer()
        result = f(*args, **kw)
        elapsed = timer() - start
        return result, elapsed
    return _wrap


def exp_sum(ema, decay, val):
    """Returns new exponential moving average (EMA) adding next value."""
    return ema * decay + val


def lin_sum(x, lr, y):
    """Returns linear sum."""
    return x + lr * (y - x)


def update_slice_exp_sum(s, ind, decay, val):
    """Updates EMA for specified slice."""
    s[ind] *= decay
    s[ind] += val


def update_slice_lin_sum(s, ind, lr, val):
    """Updates slice value estimate with specified learning rate."""
    s[ind] = (1 - lr) * s[ind] + lr * val


def update_exp_trace(traces, tr, decay, val=1., with_reset=False):
    """Updates exponential trace."""
    traces *= decay
    if with_reset:
        traces[tr] = val
    else:
        traces[tr] += val


def exp_decay(value: DecayingValue) -> DecayingValue:
    """Applies decay to specified DecayingValue."""
    x, decay = value
    return x * decay, decay


def multiply_decaying_value(value: DecayingValue, alpha: float) -> DecayingValue:
    """Returns new tuple with the first value multiplied by specified factor."""
    x, decay = value
    return x * alpha, decay


def softmax(x: np.ndarray, temp=1.) -> np.ndarray:
    """Computes softmax values for a vector `x` with a given temperature."""
    temp = clip(temp, 1e-5, 1e+3)
    e_x = np.exp((x - np.max(x, axis=-1)) / temp)
    return e_x / e_x.sum(axis=-1)


def clip(x: Any, low=None, high=None) -> Any:
    """Clips `x` with the provided thresholds."""

    # both x < x and x > x are False, so consider them as safeguards
    if x < isnone(low, x):
        x = low
    elif x > isnone(high, x):
        x = high
    return x


def safe_divide(x, y, default=0.):
    """
    Allows specifying a default value that will be returned in divide-by-zero case.
    Warning: it may not work as you might expect for floats!
    """
    if y == 0:
        return default
    return x / y
