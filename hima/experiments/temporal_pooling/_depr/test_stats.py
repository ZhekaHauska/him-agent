#  Copyright (c) 2022 Autonomous Non-Profit Organization "Artificial Intelligence Research
#  Institute" (AIRI); Moscow Institute of Physics and Technology (National Research University).
#  All rights reserved.
#
#  Licensed under the AGPLv3 license. See LICENSE in the project root for license information.

from typing import Optional, Any

import numpy as np
import seaborn as sns
import wandb
from matplotlib import pyplot as plt
from wandb.sdk.wandb_run import Run

from hima.common.sdr import SparseSdr
from hima.experiments.temporal_pooling._depr.blocks.base_block_stats import BlockStats
from hima.experiments.temporal_pooling._depr.metrics import (
    multiplicative_loss, DISTR_SIM_PMF
)
from hima.experiments.temporal_pooling._depr.sdr_seq_cross_stats import (
    OnlinePmfSimilarityMatrix, OfflinePmfSimilarityMatrix, SimilarityMatrix
)
from hima.experiments.temporal_pooling._depr.stats_config import StatsMetricsConfig
from hima.experiments.temporal_pooling._depr.utils import rename_dict_keys


class RunProgress:
    epoch: int
    step: int

    def __init__(self):
        self.epoch = -1
        self.step = -1

    def next_epoch(self):
        self.epoch += 1

    def next_step(self):
        self.step += 1


class ExperimentStats:
    n_sequences: int
    progress: RunProgress
    logger: Optional[Run]
    blocks: dict[str, Any]
    stats_config: StatsMetricsConfig

    sequence_ids_order: list[int]
    logging_temporally_disabled: int
    sequences_block_stats: dict[int, dict[str, BlockStats]]
    sequences_block_cross_stats: dict[str, dict[str, SimilarityMatrix]]

    debug: bool

    def __init__(
            self, n_sequences: int, progress: RunProgress, logger: Optional[Run],
            blocks: dict[str, Any], stats_config: StatsMetricsConfig, debug: bool
    ):
        self.n_sequences = n_sequences
        self.progress = progress
        self.logger = logger
        self.blocks = blocks
        self.stats_config = stats_config
        self.debug = debug

        self.sequences_block_stats = {}
        self.sequence_ids_order = []
        self.logging_temporally_disabled = True

        self.sequences_block_cross_stats = {}
        self._init_cross_stats()

    def on_new_epoch(self):
        self.sequences_block_stats.clear()
        self.sequence_ids_order.clear()

    def on_new_sequence(self, sequence_id: int, logging_scheduled: bool):
        self.logging_temporally_disabled = not logging_scheduled
        self.notify_cross_stats_new_sequence(sequence_id)

        if sequence_id == self.current_sequence_id:
            return

        self.sequence_ids_order.append(sequence_id)

        self.sequences_block_stats[sequence_id] = {}
        for block_name in self.blocks:
            block = self.blocks[block_name]
            block.reset_stats()
            self.current_block_stats[block.name] = block.stats

    @property
    def current_sequence_id(self):
        return self.sequence_ids_order[-1] if self.sequence_ids_order else None

    @property
    def previous_sequence_id(self):
        return self.sequence_ids_order[-2] if len(self.sequence_ids_order) >= 2 else None

    @property
    def current_block_stats(self):
        return self.sequences_block_stats[self.current_sequence_id]

    def on_block_step(self, block, block_output_sdr: SparseSdr):
        if not self.logger and not self.debug:
            return
        if self.logging_temporally_disabled:
            return

        self.update_block_cross_stats(block, block_output_sdr)

    def on_step(self):
        if self.logger is None and not self.debug:
            return
        if self.logging_temporally_disabled:
            return

        metrics = {
            'epoch': self.progress.epoch
        }
        for block_name in self.current_block_stats:
            block = self.blocks[block_name]
            block_stats = self.current_block_stats[block_name]
            block_metrics = block_stats.step_metrics()
            block_metrics = rename_dict_keys(block_metrics, add_prefix=f'{block.tag}/')
            metrics |= block_metrics

        if self.logger:
            self.logger.log(metrics, step=self.progress.step)

    def on_finish(self, logging_scheduled: bool):
        if not self.logger and not self.debug:
            return
        if not logging_scheduled:
            return

        metrics = {}
        diff_metrics = []
        optimized_metrics = []
        for block_name in self.current_block_stats:
            block = self.blocks[block_name]
            if block_name.startswith('generator'):
                self.summarize_input(block, metrics, diff_metrics)
            elif block_name.startswith('spatial_pooler'):
                self.summarize_sp(block, metrics)
            elif block_name.startswith('temporal_memory'):
                ...
            elif block.name.startswith('temporal_pooler'):
                self.summarize_tp(block, metrics, diff_metrics, optimized_metrics)
            else:
                raise KeyError(f'Block {block.name} is not supported')

        metrics |= self.summarize_similarity_errors(diff_metrics, optimized_metrics)

        if self.logger:
            self.logger.log(metrics, step=self.progress.step)

    def summarize_input(self, block, metrics: dict, diff_metrics: list):
        block_metrics = self.current_block_stats[block.name].final_metrics()

        block_diff_metrics = {
            'raw_sim_mx': block_metrics['raw_sim_mx_el'],
            'sim_mx': block_metrics['sim_mx_el']
        }
        diff_metrics.append((block.tag, block_diff_metrics))

        block_metrics = rename_dict_keys(block_metrics, add_prefix=f'{block.tag}/epoch/')
        self.transform_sim_mx_to_plots(block_metrics)

        metrics |= block_metrics

    def summarize_sp(self, block, metrics: dict):
        block_metrics = self._collect_block_final_stats(block)

        # noinspection PyUnresolvedReferences
        pmfs = [
            self.sequences_block_stats[seq_id][block.name].seq_stats.aggregate_pmf()
            for seq_id in range(self.n_sequences)
        ]
        # offline pmf similarity matrices sim_mx
        offline_pmf_similarity = OfflinePmfSimilarityMatrix(
            pmfs, sds=block.output_sds,
            normalization=self.stats_config.normalization,
            algorithm=DISTR_SIM_PMF, symmetrical=self.stats_config.symmetrical_similarity
        )
        block_metrics |= offline_pmf_similarity.final_metrics()

        # online pmf similarity matrices
        for block_online_similarity_matrix in self.sequences_block_cross_stats[block.name].values():
            block_metrics |= block_online_similarity_matrix.final_metrics()

        block_metrics = rename_dict_keys(block_metrics, add_prefix=f'{block.tag}/epoch/')
        self.transform_sim_mx_to_plots(block_metrics)

        metrics |= block_metrics

    def summarize_tp(self, block, metrics: dict, diff_metrics: list, optimized_metrics: list):
        block_metrics = self._collect_block_final_stats(block)
        optimized_metrics.append(block_metrics['mean_pmf_coverage'])

        # noinspection PyUnresolvedReferences
        pmfs = [
            self.sequences_block_stats[seq_id][block.name].seq_stats.aggregate_pmf()
            for seq_id in range(self.n_sequences)
        ]
        # offline pmf similarity matrices sim_mx
        offline_pmf_similarity_matrix = OfflinePmfSimilarityMatrix(
            pmfs, sds=block.output_sds,
            normalization=self.stats_config.normalization,
            algorithm=DISTR_SIM_PMF, symmetrical=self.stats_config.symmetrical_similarity
        )
        offline_pmf_similarity = offline_pmf_similarity_matrix.final_metrics()

        block_metrics |= offline_pmf_similarity
        block_diff_metrics = {
            'raw_sim_mx': offline_pmf_similarity['raw_sim_mx_pmf'],
            'sim_mx': offline_pmf_similarity['sim_mx_pmf']
        }
        diff_metrics.append((block.tag, block_diff_metrics))

        # online pmf similarity matrices
        for block_online_similarity_matrix in self.sequences_block_cross_stats[block.name].values():
            block_metrics |= block_online_similarity_matrix.final_metrics()

        # track first TP pmf coverage
        if block.tag[0] in {'2', 3}:
            pmf_coverage = block_metrics['mean_pmf_coverage']
            key = f'{block.tag}_mean_pmf_coverage'
            metrics[key] = pmf_coverage

        block_metrics = rename_dict_keys(block_metrics, add_prefix=f'{block.tag}/epoch/')
        self.transform_sim_mx_to_plots(block_metrics)
        metrics |= block_metrics

    def summarize_similarity_errors(self, diff_metrics, optimized_metrics):
        input_tag, input_sims = diff_metrics[0]
        metrics = {
            sim_key: {input_tag: input_sims[sim_key]}
            for sim_key in input_sims
        }

        loss_components = []
        i = 0
        for block_tag, block_sim_metrics in diff_metrics[1:]:
            for metric_key in block_sim_metrics:
                sim_mx = block_sim_metrics[metric_key]

                metrics[metric_key][block_tag] = sim_mx
                abs_err_mx = np.ma.abs(sim_mx - input_sims[metric_key])
                metrics[metric_key][f'{block_tag}_abs_err'] = abs_err_mx

                mae = abs_err_mx.mean()
                if metric_key.startswith('raw_'):
                    metrics[f'{block_tag}/epoch/similarity_mae'] = mae
                else:
                    metrics[f'{block_tag}/epoch/similarity_smae'] = mae

            if block_tag.endswith('_tp'):
                key = 'mae' if self.stats_config.loss_on_mae else 'smae'
                mae = metrics[f'{block_tag}/epoch/similarity_{key}']

                # duplicate target metric to the general wandb panel
                metrics[f'{block_tag}_similarity_{key}'] = mae

                # store to compute loss later
                pmf_coverage = optimized_metrics[i]
                loss_components.append((mae, pmf_coverage))
                i += 1

        result = {}
        for metric_key in metrics.keys():
            metric = metrics[metric_key]
            if isinstance(metric, dict):
                # dict of similarity matrices
                result[f'diff/{metric_key}'] = plot_heatmaps_row(**metric)
            else:
                result[metric_key] = metric

        result['loss'] = compute_loss(loss_components, self.stats_config.loss_layer_discount)
        return result

    def _collect_block_final_stats(self, block) -> dict[str, Any]:
        result = {}
        # collect/reorder from (seq_id, block, metric) -> (block, metric, seq_id)
        for seq_id in range(self.n_sequences):
            block_stat = self.sequences_block_stats[seq_id][block.name]
            final_metrics = block_stat.final_metrics()
            for metric_key in final_metrics:
                if metric_key not in result:
                    result[metric_key] = [None]*self.n_sequences
                result[metric_key][seq_id] = final_metrics[metric_key]

        for metric_key in result:
            if isinstance(result[metric_key][0], np.ndarray):
                result[metric_key] = np.vstack(result[metric_key])
            else:
                result[metric_key] = np.mean(result[metric_key])
        return result

    def notify_cross_stats_new_sequence(self, sequence_id: int):
        for block_name in self.blocks:
            current_block_cross_stats = self.sequences_block_cross_stats[block_name]
            for block_online_similarity_matrix in current_block_cross_stats.values():
                block_online_similarity_matrix.on_new_sequence(sequence_id=sequence_id)

    def update_block_cross_stats(self, block, block_output_sdr: SparseSdr):
        current_block_cross_stats = self.sequences_block_cross_stats[block.name]
        for block_online_similarity_matrix in current_block_cross_stats.values():
            block_online_similarity_matrix.on_step(sdr=block_output_sdr)

    def _init_cross_stats(self):
        self.sequences_block_cross_stats = {}

        for block_name in self.blocks:
            block = self.blocks[block_name]

            if block_name.startswith('generator'):
                self.sequences_block_cross_stats[block.name] = {}
            elif block_name.startswith('spatial_pooler'):
                self.sequences_block_cross_stats[block.name] = {}
                self.sequences_block_cross_stats[block.name] = {
                    # 'online_el': OnlineElementwiseSimilarityMatrix(
                    #     n_sequences=self.n_sequences,
                    #     unbias_func=self.stats_config.normalization_unbias,
                    #     discount=self.stats_config.prefix_similarity_discount,
                    #     symmetrical=self.stats_config.symmetrical_similarity
                    # ),
                    'online_pmf': OnlinePmfSimilarityMatrix(
                        n_sequences=self.n_sequences,
                        sds=block.output_sds,
                        normalization=self.stats_config.normalization,
                        discount=self.stats_config.prefix_similarity_discount,
                        symmetrical=self.stats_config.symmetrical_similarity,
                        algorithm=DISTR_SIM_PMF
                    ),
                    # FIXME: kl-div shows peculiar similarity matrix, investigate later
                    # 'online_kl': OnlinePmfSimilarityMatrix(
                    #     n_sequences=self.n_sequences,
                    #     sds=block.output_sds,
                    #     unbias_func=self.stats_config.normalization_unbias,
                    #     discount=self.stats_config.prefix_similarity_discount,
                    #     symmetrical=self.stats_config.symmetrical_similarity,
                    #     algorithm=DISTR_SIM_KL
                    # )
                }
            elif block_name.startswith('temporal_memory'):
                self.sequences_block_cross_stats[block.name] = {}
            elif block.name.startswith('temporal_pooler'):
                self.sequences_block_cross_stats[block.name] = {
                    # 'online_el': OnlineElementwiseSimilarityMatrix(
                    #     n_sequences=self.n_sequences,
                    #     unbias_func=self.stats_config.normalization_unbias,
                    #     discount=self.stats_config.prefix_similarity_discount,
                    #     symmetrical=self.stats_config.symmetrical_similarity
                    # ),
                    'online_pmf': OnlinePmfSimilarityMatrix(
                        n_sequences=self.n_sequences,
                        sds=block.output_sds,
                        normalization=self.stats_config.normalization,
                        discount=self.stats_config.prefix_similarity_discount,
                        symmetrical=self.stats_config.symmetrical_similarity,
                        algorithm=DISTR_SIM_PMF
                    ),
                    # FIXME: kl-div shows peculiar similarity matrix, investigate later
                    # 'online_kl': OnlinePmfSimilarityMatrix(
                    #     n_sequences=self.n_sequences,
                    #     sds=block.output_sds,
                    #     unbias_func=self.stats_config.normalization_unbias,
                    #     discount=self.stats_config.prefix_similarity_discount,
                    #     symmetrical=self.stats_config.symmetrical_similarity,
                    #     algorithm=DISTR_SIM_KL
                    # )
                }
            else:
                raise KeyError(f'Block {block.name} is not supported')

    @staticmethod
    def transform_sim_mx_to_plots(metrics):
        for metric_key in metrics:
            metric_value = metrics[metric_key]
            if isinstance(metric_value, np.ndarray) and metric_value.ndim == 2:
                metrics[metric_key] = plot_single_heatmap(metric_value)


def compute_loss(components, layer_discount) -> float:
    gamma = 1
    loss = 0
    for mae, pmf_coverage in components:
        loss += gamma * multiplicative_loss(mae, pmf_coverage)
        gamma *= layer_discount

    return loss


HEATMAP_SIDE_SIZE = 7


def plot_single_heatmap(repr_matrix):
    fig, ax = plt.subplots(1, 1, figsize=(HEATMAP_SIDE_SIZE+1, HEATMAP_SIDE_SIZE-1))
    plot_heatmap(repr_matrix, ax)

    fig.tight_layout()
    img = wandb.Image(fig)
    plt.close(fig)
    return img


def plot_heatmaps_row(**sim_matrices):
    n = len(sim_matrices)
    fig, axes = plt.subplots(
        nrows=1, ncols=n, sharey='all',
        figsize=(HEATMAP_SIDE_SIZE * n + 1, HEATMAP_SIDE_SIZE - 1)
    )

    for ax, (name, sim_matrix) in zip(axes, sim_matrices.items()):
        plot_heatmap(sim_matrix, ax)
        ax.set_title(name, size=10)

    fig.tight_layout()
    img = wandb.Image(fig)
    plt.close(fig)
    return img


def plot_heatmap(heatmap: np.ndarray, ax):
    v_min, v_max = calculate_heatmap_value_boundaries(heatmap)
    if isinstance(heatmap, np.ma.MaskedArray):
        sns.heatmap(
            heatmap, mask=heatmap.mask,
            vmin=v_min, vmax=v_max, cmap='plasma', ax=ax, annot=True, annot_kws={"size": 6}
        )
    else:
        sns.heatmap(heatmap, vmin=v_min, vmax=v_max, cmap='plasma', ax=ax, annot=True)


def calculate_heatmap_value_boundaries(arr: np.ndarray) -> tuple[float, float]:
    v_min, v_max = np.min(arr), np.max(arr)
    if -1 <= v_min < 0:
        v_min = -1
    elif v_min >= 0:
        v_min = 0

    if v_max < 0:
        v_max = 0
    elif v_max < 1:
        v_max = 1
    return v_min, v_max
