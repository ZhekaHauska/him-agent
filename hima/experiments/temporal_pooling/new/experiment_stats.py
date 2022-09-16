#  Copyright (c) 2022 Autonomous Non-Profit Organization "Artificial Intelligence Research
#  Institute" (AIRI); Moscow Institute of Physics and Technology (National Research University).
#  All rights reserved.
#
#  Licensed under the AGPLv3 license. See LICENSE in the project root for license information.

from typing import Optional

import numpy as np
import seaborn as sns
import wandb
from matplotlib import pyplot as plt
from wandb.sdk.wandb_run import Run

from hima.common.config_utils import TConfig
from hima.experiments.temporal_pooling.new.blocks.graph import Block, Stream
from hima.experiments.temporal_pooling.new.metrics import (
    multiplicative_loss
)
from hima.experiments.temporal_pooling.new.stats.config import StatsMetricsConfig
from hima.experiments.temporal_pooling.new.stats.stream_tracker import StreamTracker


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
    TSequenceId = int
    TStreamName = str
    TMetricsName = str

    n_sequences: int
    progress: RunProgress
    logger: Optional[Run]
    stats_config: StatsMetricsConfig
    diff_stats: TConfig

    stream_trackers: dict[TStreamName, StreamTracker]
    current_sequence_id: int

    debug: bool
    logging_temporally_disabled: bool

    def __init__(
            self, *, n_sequences: int, progress: RunProgress, logger: Optional[Run],
            blocks: dict[str, Block], track_streams: TConfig,
            stats_config: StatsMetricsConfig, debug: bool,
            diff_stats: TConfig,
    ):
        self.n_sequences = n_sequences
        self.progress = progress
        self.logger = logger
        self.stats_config = stats_config
        self.debug = debug
        self.logging_temporally_disabled = True
        self.current_sequence_id = -1
        self.diff_stats = diff_stats

        self.stream_trackers = self._make_stream_trackers(
            track_streams=track_streams, blocks=blocks,
            stats_config=stats_config, n_sequences=n_sequences
        )

    @staticmethod
    def _make_stream_trackers(
            track_streams: TConfig, blocks: dict[str, Block],
            stats_config: StatsMetricsConfig, n_sequences: int
    ) -> dict[TStreamName, StreamTracker]:
        trackers = {}
        for stream_name in track_streams:
            stream = parse_stream_name(stream_name, blocks)
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
        if self.logger is None and not self.debug:
            return
        if self.logging_temporally_disabled:
            return

        for name in self.stream_trackers:
            tracker = self.stream_trackers[name]
            tracker.on_step(tracker.stream.sdr)

        metrics = {
            'epoch': self.progress.epoch
        }
        for name in self.stream_trackers:
            metrics |= self.stream_trackers[name].step_metrics()

        if self.logger:
            self.logger.log(metrics, step=self.progress.step)

    def on_epoch_finished(self, logging_scheduled: bool):
        if not self.logger and not self.debug:
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
        if self.logger:
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

    axes = axes.flat if n > 1 else [axes]
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
