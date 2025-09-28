#  Copyright (c) 2022 Autonomous Non-Profit Organization "Artificial Intelligence Research
#  Institute" (AIRI); Moscow Institute of Physics and Technology (National Research University).
#  All rights reserved.
#
#  Licensed under the AGPLv3 license. See LICENSE in the project root for license information.
from __future__ import annotations

from typing import Union, Any
from ast import literal_eval

from epm.common.config.base import TKeyPathValue
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


def normalize(x, default_values=None, return_zeroed_variables_count=False):
    if len(x.shape) == 1:
        x = x[None]

    norm_x = x.copy()
    norm = x.sum(axis=-1)
    mask = norm == 0

    if default_values is None:
        default_values = np.ones_like(x)

    norm_x[mask] = default_values[mask]
    norm[mask] = norm_x[mask].sum(axis=-1)
    if return_zeroed_variables_count:
        return norm_x / norm.reshape((-1, 1)), np.sum(mask)
    else:
        return norm_x / norm.reshape((-1, 1))

# ========================= Binary SDR ===============================

# SDR representation optimized for set operations. It is segregated to
# clarify, when a function work with this exact representation.
SetSdr = set[int]

# General sparse form SDR. In most cases, ndarray or list is expected.
SparseSdr = Union[list[int], npt.NDArray[int], SetSdr]

# Dense SDR form. Could be a list too, but in general it's ndarray.
DenseSdr = npt.NDArray[Union[int, float]]

def sparse_to_dense(
        sdr: SparseSdr,
        size: int | tuple | DenseSdr = None,
        shape: int | tuple | DenseSdr = None,
        dtype=float,
        like: DenseSdr = None
) -> DenseSdr:
    """
    Converts SDR from sparse representation to dense.

    Size, shape and dtype define resulting dense vector params.
    The size should be at least inducible (from shape or like).
    The shape default is 1-D, dtype: float.

    Like param is a shorthand, when you have an array with all three params set correctly.
    Like param overwrites all others!
    """

    if like is not None:
        shape, size, dtype = like.shape, like.size, like.dtype
    else:
        if isinstance(size, np.ndarray):
            size = size.size
        if isinstance(shape, np.ndarray):
            shape = shape.shape

        # -1 for reshape means flatten.
        # It is also invalid size, which we need here for the unset shape case.
        shape = isnone(shape, -1)
        size = isnone(size, np.prod(shape))

    dense_vector = np.zeros(size, dtype=dtype)
    if len(sdr) > 0:
        dense_vector[sdr] = 1
    return dense_vector.reshape(shape)


def dense_to_sparse(dense_vector: DenseSdr) -> SparseSdr:
    return np.flatnonzero(dense_vector)


def parse_arg_list(args: list[str]) -> list[TKeyPathValue]:
    """Parse a list of command line arguments to the list of key-value pairs."""
    return list(map(parse_arg, args))


def parse_arg(arg: str | tuple[str, Any]) -> TKeyPathValue:
    """Parse a single command line argument to the key-value pair."""
    try:
        if isinstance(arg, str):
            # raw arg string: "key=value"

            # "--key=value" --> ["--key", "value"]
            key_path, value = arg.split('=', maxsplit=1)

            # "--key" --> "key"
            key_path = key_path.removeprefix('--')

            # parse value represented as str
            value = parse_str(value)
        else:
            # tuple ("key", value) from wandb config of the sweep single run
            # we assume that the passed value is already correctly parsed
            key_path, value = arg
    except:
        print(arg)
        raise

    # parse key tokens as they can represent array indices
    # NB: skip empty key tokens (see [1] in the end of the file for an explanation)
    key_path = [
        parse_str(key_token)
        for key_token in key_path.split('.')
        if key_token
    ]

    return key_path, value


def parse_str(s: str) -> Any:
    """Parse string value to the most appropriate type."""
    # noinspection PyShadowingNames
    def boolify(s):
        if s in ['True', 'true']:
            return True
        if s in ['False', 'false']:
            return False
        raise ValueError('Not a boolean value!')

    # NB: try/except is widely accepted pythonic way to parse things
    assert isinstance(s, str)

    # NB: order of casters is important (from most specific to most general)
    for caster in (boolify, int, float, literal_eval):
        try:
            return caster(s)
        except (ValueError, SyntaxError):
            pass
    return s


# [1]: Using sweeps we have a problem with config logging. All parameters provided to
# a run from the sweep via run args are logged to wandb automatically. At the same time,
# when we also log our compiled config dictionary, its content is flattened such that
# each param key is represented as `path.to.nested.dict.key`. Note that we declare
# params in a sweep config the same way. Therefore, each sweep run will have such params
# visibly duplicated in wandb and there's no correct way to distinguish them
# (although, wandb itself does it)! Also, only sweep runs will have params duplicated.
# Simple runs don't have the duplicate entry because they don't have sweep param args.
#
# Problem: when you want to filter or group by a param in wandb interface,
# you cannot be sure which of the duplicated entries to select, while they're different
# — the only entry that is presented in all runs [either sweep or simple] is the entry
# from our config, not from a sweep.
#
# Solution: That's why we introduced a trick - you are allowed to specify sweep param
# with insignificant additional dots (e.g. `path..to...key.`) to de-duplicate entries.
# We ignore these dots [or empty path elements introduced by them after split-by-dots]
# while parsing the nested key path.