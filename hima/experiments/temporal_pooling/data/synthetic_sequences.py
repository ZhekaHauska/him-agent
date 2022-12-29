#  Copyright (c) 2022 Autonomous Non-Profit Organization "Artificial Intelligence Research
#  Institute" (AIRI); Moscow Institute of Physics and Technology (National Research University).
#  All rights reserved.
#
#  Licensed under the AGPLv3 license. See LICENSE in the project root for license information.
import numpy as np

from hima.common.utils import clip


def generate_synthetic_sequences(
        n_sequences: int, sequence_length: int, n_values: int, seed: int,
        sequence_similarity: float,
        sequence_similarity_std: float = 0.
) -> np.ndarray:
    rng = np.random.default_rng(seed)

    base_sequence = rng.integers(0, high=n_values, size=(1, sequence_length))
    sequences = base_sequence.repeat(n_sequences, axis=0)

    # to-change indices
    for i in range(n_sequences - 1):
        if sequence_similarity_std < 1e-5:
            sim = sequence_similarity
        else:
            sim = rng.normal(sequence_similarity, scale=sequence_similarity_std)
            sim = clip(sim, 0, 1)

        n_values_to_change = int(sequence_length * (1 - sim))
        if n_values_to_change == 0:
            continue
        indices = rng.choice(sequence_length, n_values_to_change, replace=False)

        # re-sample values from reduced value space (note n_values-1 below)
        new_values = rng.integers(0, n_values - 1, n_values_to_change)
        old_values = sequences[0][indices]

        # that's how we exclude origin value: |0|1|2| -> |0|.|2|3| — value 1 is excluded
        mask = new_values >= old_values
        new_values[mask] += 1

        # replace origin values for specified positions with new values
        sequences[i+1, indices] = new_values

    return sequences


def generate_synthetic_single_element_sequences(
        n_sequences: int, n_values: int, seed: int
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(0, high=n_values, size=(n_sequences, 1))
