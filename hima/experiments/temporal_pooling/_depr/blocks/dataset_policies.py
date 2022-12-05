#  Copyright (c) 2022 Autonomous Non-Profit Organization "Artificial Intelligence Research
#  Institute" (AIRI); Moscow Institute of Physics and Technology (National Research University).
#  All rights reserved.
#
#  Licensed under the AGPLv3 license. See LICENSE in the project root for license information.
from typing import Iterator, Any

import numpy as np
from numpy.random import Generator

from hima.common.sdr import SparseSdr
from hima.common.sds import Sds
from hima.experiments.temporal_pooling._depr.blocks.base_block_stats import BlockStats
from hima.experiments.temporal_pooling._depr.blocks.dataset_resolver import resolve_encoder
from hima.experiments.temporal_pooling._depr.blocks.dataset_synth_sequences import \
    generate_synthetic_sequences
from hima.experiments.temporal_pooling._depr.sdr_seq_cross_stats import OfflineElementwiseSimilarityMatrix
from hima.experiments.temporal_pooling._depr.stats_config import StatsMetricsConfig


class Policy:
    """Sequence of (action, state)."""
    id: int
    _policy: list[tuple[SparseSdr, SparseSdr]]

    def __init__(self, id_: int, policy):
        self.id = id_
        self._policy = policy

    def __iter__(self) -> Iterator[tuple[SparseSdr, SparseSdr]]:
        return iter(self._policy)

    def __len__(self):
        return len(self._policy)


class SyntheticPoliciesDatasetBlockStats(BlockStats):
    n_policies: int
    actions_sds: Sds
    policies: list[list[set[int]]]
    cross_stats: OfflineElementwiseSimilarityMatrix

    def __init__(self, policies: list[Policy], actions_sds: Sds, stats_config: StatsMetricsConfig):
        super(SyntheticPoliciesDatasetBlockStats, self).__init__(output_sds=actions_sds)

        self.n_policies = len(policies)
        self.policies = [
            [set(a) for a, s in p]
            for p in policies
        ]
        self.actions_sds = actions_sds
        self.cross_stats = OfflineElementwiseSimilarityMatrix(
            sequences=self.policies,
            normalization=stats_config.normalization,
            discount=stats_config.prefix_similarity_discount,
            symmetrical=False
        )

    def final_metrics(self) -> dict[str, Any]:
        return self.cross_stats.final_metrics()


class SyntheticPoliciesDatasetBlock:
    id: int
    name: str
    n_states: int
    n_actions: int

    context_sds: Sds
    output_sds: Sds

    stats: SyntheticPoliciesDatasetBlockStats

    _policies: list[Policy]
    _rng: Generator

    def __init__(
            self, n_states: int, states_sds: Sds, n_actions: int, actions_sds: Sds,
            policies: list[Policy], seed: int, stats_config: StatsMetricsConfig
    ):
        self.n_states = n_states
        self.n_actions = n_actions
        self.context_sds = states_sds
        self.output_sds = actions_sds
        self._policies = policies

        self.stats = SyntheticPoliciesDatasetBlockStats(self._policies, self.output_sds, stats_config)
        self._rng = np.random.default_rng(seed)

    @property
    def tag(self) -> str:
        return f'{self.id}_in'

    def __iter__(self) -> Iterator[Policy]:
        return iter(self._policies)

    def reset_stats(self):
        ...


class SyntheticPoliciesGenerator:
    n_states: int
    n_actions: int

    policy_similarity: float
    policy_similarity_std: float

    seed: int
    _rng: Generator

    def __init__(
            self, config: dict,
            n_states: int, n_actions: int, active_size: int,
            state_encoder: str, action_encoder: str,
            policy_similarity: float,
            seed: int,
            policy_similarity_std: float = 0.
    ):
        self.n_states = n_states
        self.n_actions = n_actions
        self.state_encoder = resolve_encoder(
            config, state_encoder, 'encoders',
            n_values=self.n_states,
            active_size=active_size,
            seed=seed
        )
        self.states_sds = self.state_encoder.output_sds

        self.action_encoder = resolve_encoder(
            config, action_encoder, 'encoders',
            n_values=self.n_actions,
            active_size=active_size,
            seed=seed
        )
        self.actions_sds = self.action_encoder.output_sds

        self.policy_similarity = policy_similarity
        self.policy_similarity_std = policy_similarity_std

        self.seed = seed
        self._rng = np.random.default_rng(seed)

    def generate_policies(
            self, n_policies, stats_config: StatsMetricsConfig
    ) -> SyntheticPoliciesDatasetBlock:
        policies = generate_synthetic_sequences(
            n_sequences=n_policies,
            sequence_length=self.n_states,
            n_values=self.n_actions,
            seed=self.seed,
            sequence_similarity=self.policy_similarity,
            sequence_similarity_std=self.policy_similarity_std,
        )

        states_encoding = [self.state_encoder.encode(s) for s in range(self.n_states)]
        action_encoding = [self.action_encoder.encode(a) for a in range(self.n_actions)]

        encoded_policies = []
        for i_policy in range(n_policies):
            policy = []
            for state in range(self.n_states):
                action = policies[i_policy, state]
                s = states_encoding[state]
                a = action_encoding[action]
                policy.append((a, s))

            encoded_policies.append(Policy(id_=i_policy, policy=policy))

        return SyntheticPoliciesDatasetBlock(
            n_states=self.n_states, states_sds=self.states_sds,
            n_actions=self.n_actions, actions_sds=self.actions_sds,
            policies=encoded_policies, seed=self.seed, stats_config=stats_config
        )
