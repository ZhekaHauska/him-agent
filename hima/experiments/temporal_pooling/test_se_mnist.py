#  Copyright (c) 2022 Autonomous Non-Profit Organization "Artificial Intelligence Research
#  Institute" (AIRI); Moscow Institute of Physics and Technology (National Research University).
#  All rights reserved.
#
#  Licensed under the AGPLv3 license. See LICENSE in the project root for license information.
from __future__ import annotations

from pathlib import Path

import numpy as np
import numpy.typing as npt

from hima.common.config.base import TConfig
from hima.common.config.global_config import GlobalConfig
from hima.common.lazy_imports import lazy_import
from hima.common.run.wandb import get_logger
from hima.common.sdrr import OutputMode, split_sdr_values, AnySparseSdr
from hima.common.sds import Sds
from hima.common.timer import timer, print_with_timestamp
from hima.common.utils import isnone, prepend_dict_keys
from hima.experiments.temporal_pooling.data.mnist_ext import MnistDataset
from hima.experiments.temporal_pooling.resolvers.type_resolver import StpLazyTypeResolver
from hima.experiments.temporal_pooling.stats.sdr_tracker import SdrTracker
from hima.experiments.temporal_pooling.stp.mlp_classifier_torch import MlpClassifier
from hima.experiments.temporal_pooling.stp.sp_utils import (
    make_repeating_counter,
    RepeatingCountdown, tick
)
from hima.experiments.temporal_pooling.utils import resolve_random_seed

wandb = lazy_import('wandb')
sns = lazy_import('seaborn')
pd = lazy_import('pandas')


class TrainConfig:
    n_epochs: int
    n_steps: int | None

    def __init__(self, n_epochs: int, n_steps: int | None = None):
        self.n_epochs = n_epochs
        self.n_steps = n_steps


class TestConfig:
    eval_countdown: RepeatingCountdown
    n_epochs: int

    def __init__(self, eval_schedule: int, n_epochs: int):
        self.eval_countdown = make_repeating_counter(eval_schedule)
        self.n_epochs = n_epochs

    def tick(self):
        now, self.eval_countdown = tick(self.eval_countdown)
        return now


class EpochStats:
    def __init__(self):
        pass


class SpatialEncoderExperiment:
    training: TrainConfig
    testing: TestConfig

    input_sds: Sds
    output_sds: Sds

    stats: EpochStats

    def __init__(
            self, config: TConfig, config_path: Path,
            log: bool, seed: int, output_sds: Sds,
            train: TConfig, test: TConfig,
            setup: TConfig, classifier: TConfig,
            sdr_tracker: TConfig,
            project: str = None,
            wandb_init: TConfig = None,
            **_
    ):
        self.init_time = timer()
        self.print_with_timestamp('==> Init')

        self.config = GlobalConfig(
            config=config, config_path=config_path, type_resolver=StpLazyTypeResolver()
        )
        self.log = log
        self.logger = self.config.resolve_object(
            isnone(wandb_init, {}),
            object_type_or_factory=get_logger,
            config=config, log=log, project=project
        )
        self.seed = resolve_random_seed(seed)
        self.rng = np.random.default_rng(self.seed)

        setup = self.config.config_resolver.resolve(setup, config_type=dict)
        encoder, input_mode = self._get_setup(**setup)
        self.input_mode = OutputMode[input_mode.upper()]
        self.is_binary = self.input_mode == OutputMode.BINARY

        self.data = MnistDataset(binary=self.is_binary)

        self.input_sds = self.data.output_sds
        self.output_sds = Sds.make(output_sds)

        self.encoder = self.config.resolve_object(
            encoder, feedforward_sds=self.input_sds, output_sds=self.output_sds
        )
        self.n_classes = self.data.n_classes
        self.classifier: MlpClassifier = self.config.resolve_object(
            classifier, object_type_or_factory=MlpClassifier,
            feedforward_sds=self.output_sds, n_classes=self.n_classes
        )

        print(f'Encoder: {self.encoder.feedforward_sds} -> {self.encoder.output_sds}')
        print(f'Classif: {self.classifier.feedforward_sds} -> {self.classifier.n_classes}')

        self.training = self.config.resolve_object(train, object_type_or_factory=TrainConfig)
        self.testing = self.config.resolve_object(test, object_type_or_factory=TestConfig)

        self.sdr_tracker: SdrTracker = self.config.resolve_object(sdr_tracker, sds=self.output_sds)

        self.classify_loss = []

        self.i_train_epoch = 0

    def run(self):
        self.print_with_timestamp('==> Run')

        self.i_train_epoch = 0
        self.test_epoch()

        while self.i_train_epoch < self.training.n_epochs:
            self.i_train_epoch += 1
            self.print_with_timestamp(f'Epoch {self.i_train_epoch}')
            self.train_epoch()

            if self.testing.tick() or self.i_train_epoch == 1:
                self.test_epoch()
            # self.log_progress(epoch)

    def train_epoch(self):
        n_steps = self.training.n_steps

        sample_indices: npt.NDArray[int]
        if n_steps is not None:
            sample_indices = self.rng.choice(self.data.n_images, size=n_steps)
        else:
            sample_indices = self.rng.permutation(self.data.n_images)
            n_steps = self.data.n_images

        self.stats = EpochStats()
        for step in range(1, n_steps + 1):
            # noinspection PyTypeChecker
            sample_ind: int = sample_indices[step - 1]
            self.process_sample(sample_ind, learn=True)

    def test_epoch(self):
        print(f'==> Test after {self.i_train_epoch}')
        encoded_sdrs = [
            self.process_sample(i, learn=False)
            for i in range(self.data.n_images)
        ]
        for i in range(self.data.n_images):
            self.sdr_tracker.on_sdr_updated(encoded_sdrs[i], ignore=False)

        metrics = {}
        metrics |= self.sdr_tracker.on_sequence_finished(None, ignore=False)

        metrics |= self.eval_with_ensemble(encoded_sdrs)

        metrics = personalize_metrics(metrics, 'eval')
        self.log_progress(metrics)
        print('<== Test')

    def eval_with_ensemble(self, encoded_sdrs: list[AnySparseSdr]):
        classifier = self.train_ensemble_classifier(encoded_sdrs)
        accuracy = 0.
        for i in range(self.data.n_images):
            target_cls = self.data.target[i]
            encoded_sdr = encoded_sdrs[i]

            sdr, rates = split_sdr_values(encoded_sdr)
            prediction = np.mean(classifier[sdr] * rates[:, None], axis=0)
            accuracy += prediction.argmax() == target_cls

        accuracy /= self.data.n_images

        print(f'Accuracy: {accuracy:.3%}')
        return {
            'ens_accuracy': accuracy,
        }

    def train_ensemble_classifier(self, encoded_sdrs: list[AnySparseSdr]):
        classifier = np.zeros((self.output_sds.size, self.n_classes))
        cls_counter = np.zeros(self.n_classes)

        for i in range(self.data.n_images):
            target_cls = self.data.target[i]
            encoded_sdr = encoded_sdrs[i]

            sdr, rates = split_sdr_values(encoded_sdr)
            classifier[sdr, target_cls] += rates
            cls_counter[target_cls] += 1

        neuron_norm = classifier.sum(axis=1)
        mask = neuron_norm > 0
        classifier[mask] /= classifier.sum(axis=1)[mask, None]
        return classifier

    def process_sample(self, obs_ind: int, learn: bool):
        obs_sdr = self.data.get_sdr(obs_ind)
        encoded_sdr = self.encoder.compute(obs_sdr, learn=learn)
        return encoded_sdr

    def log_progress(self, metrics: dict):
        if self.logger is None:
            return

        self.logger.log(metrics, step=self.i_train_epoch)


    @staticmethod
    def _get_setup(encoder: TConfig, input_mode: str):
        return encoder, input_mode

    def print_with_timestamp(self, *args, cond: bool = True):
        if not cond:
            return
        print_with_timestamp(self.init_time, *args)


def personalize_metrics(metrics: dict, prefix: str):
    return prepend_dict_keys(metrics, prefix, separator='/')
