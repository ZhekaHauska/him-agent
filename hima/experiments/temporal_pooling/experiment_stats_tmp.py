#  Copyright (c) 2022 Autonomous Non-Profit Organization "Artificial Intelligence Research
#  Institute" (AIRI); Moscow Institute of Physics and Technology (National Research University).
#  All rights reserved.
#
#  Licensed under the AGPLv3 license. See LICENSE in the project root for license information.
from __future__ import annotations

from typing import Optional, TYPE_CHECKING

import numpy as np

from hima.common.config.base import TConfig
from hima.experiments.temporal_pooling.blocks.general_feedback_tm import \
    GeneralFeedbackTemporalMemoryBlock
from hima.experiments.temporal_pooling.graph.block import Block
from hima.experiments.temporal_pooling.graph.stream import Stream
from hima.experiments.temporal_pooling.run_progress import RunProgress
from hima.experiments.temporal_pooling.stats.config import StatsMetricsConfig
from hima.experiments.temporal_pooling.stats.metrics import (
    multiplicative_loss
)
from hima.experiments.temporal_pooling.stats.recall_tracker import AnomalyTracker
from hima.experiments.temporal_pooling.stats.stream_tracker import StreamTracker, rename_dict_keys

if TYPE_CHECKING:
    from wandb.sdk.wandb_run import Run


class ExperimentStats:
    TSequenceId = int
    TStreamName = str
    TMetricsName = str

    logger: Optional[Run]
    n_sequences: int
    progress: RunProgress
    stats_config: StatsMetricsConfig

    diff_stats: TConfig
    loss_items: tuple[str, str] | None
    # charts: list[str]

    stream_trackers: dict[TStreamName, StreamTracker]
    current_sequence_id: int

    logging_temporally_disabled: bool

    def __init__(
            self, *, n_sequences: int, progress: RunProgress, logger,
            blocks: dict[str, Block], track_streams: TConfig,
            stats_config: StatsMetricsConfig,
            diff_stats: TConfig,
            loss: list[str] = None,
    ):
        self.n_sequences = n_sequences
        self.progress = progress
        self.logger = logger
        self.stats_config = stats_config
        self.logging_temporally_disabled = True
        self.current_sequence_id = -1
        self.diff_stats = diff_stats
        self.loss_items = (loss[0], loss[1]) if loss else []

        if self.logger:
            import wandb
            from matplotlib import pyplot as plt
            import seaborn as sns

        self.stream_trackers = self._make_stream_trackers(
            track_streams=track_streams, blocks=blocks,
            stats_config=stats_config, n_sequences=n_sequences
        )
        self.tms = [
            (blocks[block_name], AnomalyTracker())
            for block_name in blocks
            if blocks[block_name].family in 'temporal_memory'
        ]

    @staticmethod
    def _make_stream_trackers(
            track_streams: TConfig, blocks: dict[str, Block],
            stats_config: StatsMetricsConfig, n_sequences: int
    ) -> dict[TStreamName, StreamTracker]:
        trackers = {}
        for stream_name in track_streams:
            stream = parse_stream_name(stream_name, blocks)
            if not stream:
                continue
            stream_trackers_list = track_streams[stream_name]
            tracker = StreamTracker(
                stream=stream, trackers=stream_trackers_list,
                config=stats_config, n_sequences=n_sequences
            )
            trackers[tracker.name] = tracker
        return trackers

    def define_metrics(self):
        if not self.logger:
            return

        self.logger.define_metric('epoch')
        self.logger.define_metric('loss', step_metric='epoch')
        for name in self.stream_trackers:
            tracker = self.stream_trackers[name]
            self.logger.define_metric(f'{tracker.name}/epoch/*', step_metric='epoch')

    def on_epoch_started(self):
        for name in self.stream_trackers:
            self.stream_trackers[name].on_epoch_started()

    def on_sequence_started(self, sequence_id: int, logging_scheduled: bool):
        self.logging_temporally_disabled = not logging_scheduled
        if sequence_id == self.current_sequence_id:
            return

        self.current_sequence_id = sequence_id
        for name in self.stream_trackers:
            self.stream_trackers[name].on_sequence_started(sequence_id)

    def on_sequence_finished(self):
        for name in self.stream_trackers:
            self.stream_trackers[name].on_sequence_finished()

    def on_step(self):
        if self.logger is None:
            return
        if self.logging_temporally_disabled:
            return

        for name in self.stream_trackers:
            tracker = self.stream_trackers[name]
            tracker.on_step(tracker.stream.sdr)

        for block, anomaly_tracker in self.tms:
            anomaly_tracker.on_step(block.tm.anomaly[-1])

        metrics = {
            'epoch': self.progress.epoch
        }
        for name in self.stream_trackers:
            metrics |= self.stream_trackers[name].step_metrics()
        for block, anomaly_tracker in self.tms:
            res = anomaly_tracker.step_metrics()
            metrics |= rename_dict_keys(res, add_prefix=f'{block.name}/')

        self.logger.log(metrics, step=self.progress.step)

    def on_epoch_finished(self, logging_scheduled: bool):
        if not self.logger:
            return
        if not logging_scheduled:
            return

        for name in self.stream_trackers:
            self.stream_trackers[name].on_epoch_finished()

        metrics = {}
        for name in self.stream_trackers:
            metrics |= self.stream_trackers[name].aggregate_metrics()

        for name in self.diff_stats:
            self.append_sim_mae(diff_tag=name, tags=self.diff_stats[name], metrics=metrics)

        self.transform_sim_mx_to_plots(metrics)

        loss_items = [metrics[key] for key in self.loss_items if key in metrics]
        if loss_items:
            loss_components = [(loss_items[0], loss_items[1])]
            metrics['loss'] = compute_loss(loss_components, self.stats_config.loss_layer_discount)

        self.logger.log(metrics, step=self.progress.step)

    @staticmethod
    def append_sim_mae(diff_tag, tags: list[str], metrics: dict):
        i = 0
        while tags[i] not in metrics:
            i += 1
        baseline_tag = tags[i]
        baseline_sim_mx = metrics[baseline_tag]
        diff_dict = {baseline_tag: baseline_sim_mx}

        for tag in tags[i+1:]:
            if tag not in metrics:
                continue
            sim_mx = metrics[tag]
            abs_err_mx = np.ma.abs(sim_mx - baseline_sim_mx)
            diff_dict[tag] = sim_mx
            diff_dict[f'{tag}_abs_err'] = abs_err_mx

            mae = abs_err_mx.mean()
            metrics[f'{tag}_mae'] = mae

        metrics[f'diff/{diff_tag}'] = diff_dict

    @staticmethod
    def transform_sim_mx_to_plots(metrics):
        for metric_key in metrics:
            metric_value = metrics[metric_key]
            if isinstance(metric_value, np.ndarray) and metric_value.ndim == 2:
                metrics[metric_key] = plot_single_heatmap(metric_value)
            if isinstance(metric_value, dict):
                metrics[metric_key] = plot_heatmaps_row(**metric_value)


def parse_stream_name(stream_name: str, blocks: dict[str, Block]) -> Optional[Stream]:
    block_name, stream_name = stream_name.split('.')
    if block_name not in blocks:
        # skip unused blocks
        return None
    return blocks[block_name].streams[stream_name]


def compute_loss(components, layer_discount) -> float:
    gamma = 1
    loss = 0
    for mae, pmf_coverage in components:
        loss += gamma * multiplicative_loss(mae, pmf_coverage)
        gamma *= layer_discount

    return loss


HEATMAP_SIDE_SIZE = 7


def plot_single_heatmap(repr_matrix):
    import wandb
    from matplotlib import pyplot as plt

    fig, ax = plt.subplots(1, 1, figsize=(HEATMAP_SIDE_SIZE+1, HEATMAP_SIDE_SIZE-1))
    plot_heatmap(repr_matrix, ax)

    fig.tight_layout()
    img = wandb.Image(fig)
    plt.close(fig)
    return img


def plot_heatmaps_row(**sim_matrices):
    import wandb
    from matplotlib import pyplot as plt

    n = len(sim_matrices)
    fig, axes = plt.subplots(
        nrows=1, ncols=n, sharey='all',
        figsize=(HEATMAP_SIDE_SIZE * n + 1, HEATMAP_SIDE_SIZE - 1)
    )

    axes = axes.flat if n > 1 else [axes]
    for ax, (name, sim_matrix) in zip(axes, sim_matrices.items()):
        plot_heatmap(sim_matrix, ax)
        ax.set_title(name, size=10)

    fig.tight_layout()
    img = wandb.Image(fig)
    plt.close(fig)
    return img


def plot_heatmap(heatmap: np.ndarray, ax):
    import seaborn as sns

    v_min, v_max = calculate_heatmap_value_boundaries(heatmap)

    h, w = heatmap.shape
    annotate = h * w <= 200

    heatmap_params = dict(
        vmin=v_min, vmax=v_max, cmap='plasma', ax=ax, annot=annotate, annot_kws={'size': 6}
    )
    if isinstance(heatmap, np.ma.MaskedArray):
        heatmap_params['mask'] = heatmap.mask

    sns.heatmap(heatmap, **heatmap_params)


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
