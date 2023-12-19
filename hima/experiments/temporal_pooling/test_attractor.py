#  Copyright (c) 2022 Autonomous Non-Profit Organization "Artificial Intelligence Research
#  Institute" (AIRI); Moscow Institute of Physics and Technology (National Research University).
#  All rights reserved.
#
#  Licensed under the AGPLv3 license. See LICENSE in the project root for license information.
from __future__ import annotations

from pathlib import Path

import numpy as np

from hima.common.config.base import TConfig
from hima.common.config.global_config import GlobalConfig
from hima.common.sdrr import RateSdr
from hima.common.lazy_imports import lazy_import
from hima.common.run.wandb import get_logger
from hima.common.sdr import SparseSdr
from hima.common.timer import timer, print_with_timestamp
from hima.common.utils import isnone, prepend_dict_keys
from hima.experiments.temporal_pooling.data.mnist import MnistDataset
from hima.experiments.temporal_pooling.resolvers.type_resolver import StpLazyTypeResolver
from hima.experiments.temporal_pooling.stats.metrics import sdr_similarity
from hima.experiments.temporal_pooling.utils import resolve_random_seed

wandb = lazy_import('wandb')
sns = lazy_import('seaborn')
pd = lazy_import('pandas')


class TrainConfig:
    n_epochs: int
    n_steps: int

    def __init__(self, n_epochs: int, n_steps: int):
        self.n_epochs = n_epochs
        self.n_steps = n_steps


class TestConfig:
    items_per_class: int

    def __init__(self, items_per_class: int):
        self.items_per_class = items_per_class


class AttractionConfig:
    n_steps: int
    learn_in_attraction: bool

    def __init__(self, n_steps: int, learn_in_attraction: bool):
        self.n_steps = n_steps
        self.learn_in_attraction = learn_in_attraction


# This is an attractor experiment on MNIST dataset.
class SpAttractorExperiment:
    training: TrainConfig
    attraction: AttractionConfig
    testing: TestConfig

    data: MnistDataset
    binary: bool

    def __init__(
            self, config: TConfig, config_path: Path,
            log: bool, seed: int, binary: bool,
            train: TConfig, test: TConfig, attraction: TConfig,
            encoder: TConfig, attractor: TConfig,
            project: str = None,
            wandb_init: TConfig = None,
            **_
    ):
        self.init_time = timer()
        self.print_with_timestamp('==> Init')

        self.config = GlobalConfig(
            config=config, config_path=config_path, type_resolver=StpLazyTypeResolver()
        )
        self.logger = self.config.resolve_object(
            isnone(wandb_init, {}),
            object_type_or_factory=get_logger,
            config=config, log=log, project=project
        )
        self.seed = resolve_random_seed(seed)
        self.rng = np.random.default_rng(self.seed)

        self.data = MnistDataset()
        self.binary = binary

        input_sds = self.data.output_sds
        self.encoder = self.config.resolve_object(encoder, feedforward_sds=input_sds)
        if self.encoder is not None:
            print(f'Encoder: {self.encoder.feedforward_sds} -> {self.encoder.output_sds}')
            input_sds = self.encoder.output_sds

        self.attractor = self.config.resolve_object(attractor, feedforward_sds=input_sds)
        print(f'Attractor: {self.attractor.feedforward_sds} -> {self.attractor.output_sds}')

        self.training = self.config.resolve_object(train, object_type_or_factory=TrainConfig)
        self.attraction = self.config.resolve_object(
            attraction, object_type_or_factory=AttractionConfig
        )
        self.testing = self.config.resolve_object(test, object_type_or_factory=TestConfig)

    def run(self):
        self.print_with_timestamp('==> Run')
        # self.define_metrics()

        self.print_with_timestamp(f'Epoch {0}')
        self.log_results(0)

        for epoch in range(1, self.training.n_epochs + 1):
            self.print_with_timestamp(f'Epoch {epoch}')
            self.train_epoch()
            self.log_results(epoch)

    def train_epoch(self):
        sample_indices = self.rng.choice(self.data.n_images, size=self.training.n_steps)
        for step in range(self.training.n_steps):
            sample_ind = sample_indices[step]
            # noinspection PyTypeChecker
            sample = self.data.get_sdr(sample_ind, binary=self.binary)
            self.process_sample(sample, learn=True)


    def test_epoch(self):
        trajectories = []
        for digit_class in range(self.data.n_classes):
            subset = self.data.classes[digit_class]
            sample_indices = self.rng.choice(subset, size=self.testing.items_per_class)

            intra_cls_trajectories = [
                self.process_sample(
                    self.data.get_sdr(sample_ind, binary=self.binary),
                    learn=False
                )
                for sample_ind in sample_indices
            ]
            trajectories.append(intra_cls_trajectories)

        return self.analyse_trajectories(trajectories)

    def process_sample(
            self, sample: SparseSdr | RateSdr, learn: bool
    ) -> list[SparseSdr]:
        sdrs = [sample]
        if self.encoder is not None:
            sdrs.append(
                self.encoder.compute(sdrs[-1], learn=learn)
            )

        for attractor_steps in range(self.attraction.n_steps):
            _learn = learn and (attractor_steps == 0 or self.attraction.learn_in_attraction)
            sdrs.append(
                self.attractor.compute(sdrs[-1], learn=_learn)
            )
        return [
            set(sdr.sdr) if isinstance(sdr, RateSdr) else set(sdr)
            for sdr in sdrs
        ]

    def analyse_trajectories(self, trajectories):
        n_cls = len(trajectories)
        n_cls_samples = len(trajectories[0])
        n_total_samples = n_cls * n_cls_samples
        n_attractor_steps = len(trajectories[0][0])

        sim_mx = np.zeros((n_attractor_steps, n_cls, n_cls))
        sim_mx_counts = np.zeros((n_cls, n_cls))

        for i in range(n_total_samples):
            i_cls, i_sample = divmod(i, n_cls_samples)
            i_trajectory = trajectories[i_cls][i_sample]

            for j in range(i, n_total_samples):
                j_cls, j_sample = divmod(j, n_cls_samples)
                j_trajectory = trajectories[j_cls][j_sample]

                for step in range(n_attractor_steps):
                    sim = sdr_similarity(i_trajectory[step], j_trajectory[step])
                    sim_mx[step, i_cls, j_cls] += sim
                    sim_mx[step, j_cls, i_cls] += sim
                    sim_mx_counts[i_cls, j_cls] += 1
                    sim_mx_counts[j_cls, i_cls] += 1

        sim_mx /= np.expand_dims(sim_mx_counts, 0)
        # divide each row in each matrix by its diagonal element
        rel_sim_mx = sim_mx / np.diagonal(sim_mx, axis1=1, axis2=2)[:, :, None]

        return sim_mx, rel_sim_mx

    def log_results(self, epoch: int):
        sim_mx, rel_sim_mx = self.test_epoch()

        if self.logger is None:
            with np.printoptions(precision=2, suppress=True):
                print(sim_mx)
            return

        import matplotlib.pyplot as plt

        main_metrics = dict(attractor_entropy=self.attractor.output_entropy)
        if self.encoder is not None:
            main_metrics |= dict(encoder_entropy=self.encoder.output_entropy)

        main_metrics = personalize_metrics(main_metrics, 'main')

        step_metrics = {}
        rel_step_metrics = {}
        for step in range(self.attraction.n_steps):
            step_metrics[f'{step=}'] = sim_mx[step].mean()
            rel_step_metrics[f'rel_{step=}'] = rel_sim_mx[step].mean()

        step_metrics = personalize_metrics(step_metrics, 'step')
        rel_step_metrics = personalize_metrics(rel_step_metrics, 'rel_step')

        sim_df = pd.DataFrame(
            sim_mx.mean(-1), columns=np.arange(sim_mx.shape[1], dtype=int)
        )
        rel_sim_df = pd.DataFrame(
            rel_sim_mx.mean(-1), columns=np.arange(rel_sim_mx.shape[1], dtype=int)
        )
        convergence_metrics = dict(
            sim=wandb.Image(sns.lineplot(data=sim_df))
        )
        plt.close('all')
        convergence_metrics |= dict(
            rel_sim=wandb.Image(sns.lineplot(data=rel_sim_df))
        )
        plt.close('all')

        # fig, axs = plt.subplots(ncols=4, sharey='row', figsize=(16, 4))
        # axs[0].set_title('raw_sim')
        # axs[1].set_title('1-step')
        # axs[2].set_title(f'{self.testing.items_per_class // 2}-step')
        # axs[3].set_title(f'{self.testing.items_per_class}-step')
        # sns.heatmap(sim_matrices[0], ax=axs[0], cmap='viridis')
        # sns.heatmap(sim_matrices[1], ax=axs[1], cmap='viridis')
        # sns.heatmap(
        #     sim_matrices[self.testing.items_per_class // 2], ax=axs[2], cmap='viridis'
        # )
        # sns.heatmap(sim_matrices[-1], ax=axs[3], cmap='viridis')
        #
        # convergence_metrics |= dict(
        #     similarity_per_class=wandb.Image(fig)
        # )
        # plt.close('all')

        convergence_metrics = personalize_metrics(convergence_metrics, 'convergence')

        self.logger.log(
            main_metrics | convergence_metrics | step_metrics | rel_step_metrics,
            step=epoch
        )

    def print_with_timestamp(self, *args, cond: bool = True):
        if not cond:
            return
        print_with_timestamp(self.init_time, *args)

    def define_metrics(self):
        if self.logger is None:
            return

        self.logger.define_metric(
            name='main_metrics/relative_similarity',
            step_metric='epoch'
        )
        self.logger.define_metric(
            name='convergence/io_hist',
            step_metric='epoch'
        )

        for cls in range(10):
            self.logger.define_metric(
                name=f'relative_similarity/class {cls}',
                step_metric='epoch'
            )


def personalize_metrics(metrics: dict, prefix: str):
    return prepend_dict_keys(metrics, prefix, separator='/')
