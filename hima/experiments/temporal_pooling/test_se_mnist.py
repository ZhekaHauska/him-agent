#  Copyright (c) 2022 Autonomous Non-Profit Organization "Artificial Intelligence Research
#  Institute" (AIRI); Moscow Institute of Physics and Technology (National Research University).
#  All rights reserved.
#
#  Licensed under the AGPLv3 license. See LICENSE in the project root for license information.
from __future__ import annotations

from pathlib import Path

import numpy as np
from tqdm import tqdm, trange

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
    batch_size: int
    online: bool

    def __init__(self, n_epochs: int, batch_size: int, online: bool):
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.online = online


class TestConfig:
    eval_first: int
    eval_countdown: RepeatingCountdown
    n_epochs: int

    def __init__(self, eval_first: int, eval_schedule: int, n_epochs: int):
        self.eval_first = eval_first
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

        self.data = MnistDataset(seed=seed, binary=self.is_binary)

        self.input_sds = self.data.output_sds
        self.output_sds = Sds.make(output_sds)

        if encoder is not None:
            self.encoder = self.config.resolve_object(
                encoder, feedforward_sds=self.input_sds, output_sds=self.output_sds
            )

            self.sdr_tracker: SdrTracker = self.config.resolve_object(
                sdr_tracker, sds=self.output_sds
            )
            print(f'Encoder: {self.encoder.feedforward_sds} -> {self.encoder.output_sds}')
        else:
            classifier['hidden_layer'] = self.output_sds.size
            self.encoder = None

        self.n_classes = self.data.n_classes
        self.classifier: TConfig = classifier
        self.full_time_ann_classifier = self.make_ann_classifier()

        self.training = self.config.resolve_object(train, object_type_or_factory=TrainConfig)
        self.testing = self.config.resolve_object(test, object_type_or_factory=TestConfig)
        self.i_train_epoch = 0

    def run(self):
        self.print_with_timestamp('==> Run')
        if self.encoder is None:
            self.run_ann()
        else:
            self.run_se_ann()

    def run_se_ann(self):
        self.i_train_epoch = 0
        if self.logger is not None:
            self.test_epoch_se_ann()

        while self.i_train_epoch < self.training.n_epochs:
            self.i_train_epoch += 1
            self.print_with_timestamp(f'Epoch {self.i_train_epoch}')
            self.train_epoch_se()
            self.test_epoch_se_ann()

    def train_epoch_se(self):
        for ix in tqdm(self.rng.permutation(self.data.train.n_images)):
            sdr = self.data.train.get_sdr(ix)
            self.encoder.compute(sdr, learn=True)

    def test_epoch_se_ann(self):
        def encode_dataset(data):
            encoded_sdrs = []
            for sample_ix in range(data.n_images):
                obs_sdr = data.get_sdr(sample_ix)
                enc_sdr = self.encoder.compute(obs_sdr, learn=False)
                self.sdr_tracker.on_sdr_updated(enc_sdr, ignore=False)
                encoded_sdrs.append(enc_sdr)
            return encoded_sdrs

        if not self.should_test():
            return

        print(f'==> Test after {self.i_train_epoch}')
        train_sdrs, test_sdrs = encode_dataset(self.data.train), encode_dataset(self.data.test)

        metrics = {}
        metrics |= self.sdr_tracker.on_sequence_finished(None, ignore=False)

        if not self.training.online:
            ensemble_classifier = self.train_ensemble_classifier(train_sdrs)
            metrics |= self.evaluate_ensemble_classifier(ensemble_classifier, test_sdrs)

            # ==> train and test epoch-specific ANN classifier
            epoch_ann_classifier = self.make_ann_classifier()
            for _ in trange(self.testing.n_epochs):
                self.train_epoch_ann_classifier(epoch_ann_classifier, train_sdrs)
            metrics |= self.evaluate_ann_classifier(epoch_ann_classifier, test_sdrs)

        # ==> train and test full-time ANN classifier
        self.train_epoch_ann_classifier(self.full_time_ann_classifier, train_sdrs)
        online_metrics = self.evaluate_ann_classifier(self.full_time_ann_classifier, test_sdrs)
        if not self.training.online:
            metrics |= personalize_metrics(online_metrics, 'online')

        metrics = personalize_metrics(metrics, 'eval')
        self.log_progress(metrics)
        print('<== Test')

    def run_ann(self):
        def get_data(data):
            return [data.get_sdr(i) for i in range(data.n_images)]

        train_sdrs, test_sdrs = get_data(self.data.train), get_data(self.data.test)

        self.i_train_epoch = 0
        while self.i_train_epoch < self.training.n_epochs:
            self.i_train_epoch += 1
            self.print_with_timestamp(f'Epoch {self.i_train_epoch}')
            self.train_epoch_ann_classifier(self.full_time_ann_classifier, train_sdrs)
            self.test_epoch_ann(test_sdrs)

    def test_epoch_ann(self, test_sdrs: list[AnySparseSdr]):
        if not self.should_test():
            return
        metrics = self.evaluate_ann_classifier(self.full_time_ann_classifier, test_sdrs)
        metrics = personalize_metrics(metrics, 'eval')
        self.log_progress(metrics)

    def evaluate_ann_classifier(self, classifier, test_sdrs: list[AnySparseSdr]):
        accuracy = 0.

        data = self.data.test
        batched_indices = np.array_split(
            np.arange(data.n_images), data.n_images // self.training.batch_size
        )

        for batch_ix in batched_indices:
            batch = np.zeros((len(batch_ix), classifier.feedforward_sds.size))
            for i, encoded_sdr_ix in enumerate(batch_ix):
                encoded_sdr_ix: int
                sdr, rates = split_sdr_values(test_sdrs[encoded_sdr_ix])
                batch[i, sdr] = rates

            target_cls = data.targets[batch_ix]
            prediction = classifier.predict(batch)
            accuracy += np.count_nonzero(np.argmax(prediction, axis=-1) == target_cls)

        accuracy /= data.n_images
        loss = np.mean(classifier.losses)

        print(f'MLP Accuracy: {accuracy:.3%} | Loss: {loss:.3f}')
        return {
            'mlp_accuracy': accuracy,
            'mlp_loss': loss,
        }

    def train_epoch_ann_classifier(self, classifier, train_sdrs: list[AnySparseSdr]):
        data = self.data.train
        batched_indices = np.array_split(
            self.rng.permutation(data.n_images), data.n_images // self.training.batch_size
        )

        for batch_ix in batched_indices:
            batch = np.zeros((len(batch_ix), classifier.feedforward_sds.size))
            for i, encoded_sdr_ix in enumerate(batch_ix):
                encoded_sdr_ix: int
                sdr, rates = split_sdr_values(train_sdrs[encoded_sdr_ix])
                batch[i, sdr] = rates

            target_cls = data.targets[batch_ix]
            classifier.learn(batch, target_cls)

    def make_ann_classifier(self) -> MlpClassifier:
        feedforward_sds = self.input_sds if self.encoder is None else self.output_sds
        return self.config.resolve_object(
            self.classifier, object_type_or_factory=MlpClassifier,
            feedforward_sds=feedforward_sds, n_classes=self.n_classes,

        )

    def evaluate_ensemble_classifier(self, classifier, test_sdrs: list[AnySparseSdr]):
        accuracy = 0.
        data = self.data.test
        for i in range(data.n_images):
            target_cls = data.targets[i]
            sdr, rates = split_sdr_values(test_sdrs[i])

            prediction = np.mean(classifier[sdr] * rates[:, None], axis=0)
            accuracy += prediction.argmax() == target_cls

        accuracy /= data.n_images

        print(f'Ensemble Accuracy: {accuracy:.3%}')
        return {
            'ens_accuracy': accuracy,
        }

    def train_ensemble_classifier(self, train_enc_sdrs: list[AnySparseSdr]):
        classifier = np.zeros((self.output_sds.size, self.n_classes))
        data = self.data.train
        for i in range(data.n_images):
            target_cls = data.targets[i]
            encoded_sdr = train_enc_sdrs[i]

            sdr, rates = split_sdr_values(encoded_sdr)
            classifier[sdr, target_cls] += rates

        neuron_norm = classifier.sum(axis=1)
        mask = neuron_norm > 0
        classifier[mask] /= classifier.sum(axis=1)[mask, None]
        return classifier

    def process_sample(self, data, obs_ind: int, learn: bool):
        obs_sdr = data.get_sdr(obs_ind)
        encoded_sdr = self.encoder.compute(obs_sdr, learn=learn)
        return encoded_sdr

    def log_progress(self, metrics: dict):
        if self.logger is None:
            return

        self.logger.log(metrics, step=self.i_train_epoch)

    @staticmethod
    def _get_setup(input_mode: str, encoder: TConfig = None):
        return encoder, input_mode

    def should_test(self):
        return (
            self.testing.tick() or
            self.i_train_epoch <= self.testing.eval_first or
            self.i_train_epoch == self.training.n_epochs
        )

    def print_with_timestamp(self, *args, cond: bool = True):
        if not cond:
            return
        print_with_timestamp(self.init_time, *args)


def personalize_metrics(metrics: dict, prefix: str):
    return prepend_dict_keys(metrics, prefix, separator='/')
