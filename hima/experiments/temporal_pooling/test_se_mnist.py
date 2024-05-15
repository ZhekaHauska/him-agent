#  Copyright (c) 2022 Autonomous Non-Profit Organization "Artificial Intelligence Research
#  Institute" (AIRI); Moscow Institute of Physics and Technology (National Research University).
#  All rights reserved.
#
#  Licensed under the AGPLv3 license. See LICENSE in the project root for license information.
from __future__ import annotations

from itertools import zip_longest
from pathlib import Path

import numpy as np
from tqdm import tqdm, trange

from hima.common.config.base import TConfig
from hima.common.config.global_config import GlobalConfig
from hima.common.lazy_imports import lazy_import
from hima.common.run.wandb import get_logger
from hima.common.sdrr import OutputMode, split_sdr_values
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
    n_online_epochs: int

    def __init__(self, n_epochs: int, batch_size: int, n_online_epochs: int):
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.n_online_epochs = n_online_epochs


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

        self.training = self.config.resolve_object(train, object_type_or_factory=TrainConfig)
        self.testing = self.config.resolve_object(test, object_type_or_factory=TestConfig)

        if encoder is not None:
            # spatial encoding layer + 1-layer linear ANN classifier
            self.encoder = self.config.resolve_object(
                encoder, feedforward_sds=self.input_sds, output_sds=self.output_sds
            )

            self.sdr_tracker: SdrTracker = self.config.resolve_object(
                sdr_tracker, sds=self.output_sds
            )
            self.online_loss_metric_key = f'online_loss_{self.training.n_online_epochs}'
            print(f'Encoder: {self.encoder.feedforward_sds} -> {self.encoder.output_sds}')
        else:
            # 2-layer ANN classifier
            # NB: set hidden layer to make the classifier 2-layer
            classifier['hidden_layer'] = self.output_sds.size
            self.encoder = None

        self.n_classes = self.data.n_classes
        self.classifier: TConfig = classifier
        self.online_ann_classifier = self.make_ann_classifier()
        self.i_train_epoch = 0
        self.metrics = {}

    def run(self):
        self.print_with_timestamp('==> Run')
        if self.encoder is None:
            self.run_ann()
        else:
            self.run_se_ann()

    def run_se_ann(self):
        """
        Train linear ANN classifier over a spatial encoding layer.
        There are three modes operating simultaneously:
        - we train SE in an epoch-based regime
        - [on testing schedule] we train ANN classifier for N epochs over
            a frozen K-epoch pretrained SE (K-N mode) and then test it
        - [for the first M epochs] we train ANN classifier batch-ONLINE mode
            without testing to report only the training losses (ONLINE mode).
            It is used to compare representation stability of SE and SP.
        """
        train_sdrs, test_sdrs = get_data(self.data.train), get_data(self.data.test)

        self.i_train_epoch = 0
        while self.i_train_epoch < self.testing.n_epochs:
            self.i_train_epoch += 1
            self.print_with_timestamp(f'Epoch {self.i_train_epoch}')
            if self.i_train_epoch <= self.training.n_online_epochs:
                # beside training SE, train ANN classifier in batch-ONLINE mode
                epoch_online_losses = self.train_epoch_se_ann_online(
                    train_sdrs, self.online_ann_classifier, self.data.train.targets
                )

                # append online losses for the epoch to the total list
                step_metrics = self.metrics.setdefault('steps', {})
                accumulated_online_losses = step_metrics.setdefault(self.online_loss_metric_key, [])
                accumulated_online_losses.extend(epoch_online_losses)
            elif self.i_train_epoch <= self.training.n_epochs:
                # just regular SE training
                self.train_epoch_se(train_sdrs)

            # [on testing schedule] train and test ANN classifier in K-N mode
            self.test_epoch_se_ann_kn_mode(train_sdrs, test_sdrs)

        self.log_progress(self.metrics)

    def train_epoch_se(self, train_sdrs):
        for ix in tqdm(self.rng.permutation(len(train_sdrs))):
            self.encoder.compute(train_sdrs[ix], learn=True)

    def train_epoch_se_ann_online(self, train_sdrs, classifier, targets):
        """Train spatial encoder. Train also ANN classifier in batch-ONLINE mode."""
        order = self.rng.permutation(len(train_sdrs))
        batched_indices = split_to_batches(order, self.training.batch_size)
        losses = []
        for batch_ix in batched_indices:
            batch = np.zeros((len(batch_ix), classifier.feedforward_sds.size))
            fill_batch(batch, train_sdrs, batch_ix, self.encoder, learn=True)

            target_cls = targets[batch_ix]
            classifier.learn(batch, target_cls)
            losses.append(classifier.losses[-1])

        return losses

    def test_epoch_se_ann_kn_mode(self, train_sdrs, test_sdrs):
        def encode_dataset(data):
            encoded_sdrs = []
            for obs_sdr in data:
                enc_sdr = self.encoder.compute(obs_sdr, learn=False)
                self.sdr_tracker.on_sdr_updated(enc_sdr, ignore=False)
                encoded_sdrs.append(enc_sdr)
            return encoded_sdrs

        if not self.should_test():
            return

        print(f'==> Test after {self.i_train_epoch}')

        entropy = self.sdr_tracker.on_sequence_finished(None, ignore=False)['H']

        # ==> train and test epoch-specific ANN classifier
        kn_ann_classifier = self.make_ann_classifier()
        train_sdrs = encode_dataset(train_sdrs)
        first_epoch_kn_losses = None
        final_epoch_kn_losses = None
        for _ in trange(self.testing.n_epochs):
            kn_epoch_losses = self.train_epoch_ann_classifier(
                kn_ann_classifier, train_sdrs, self.data.train.targets
            )
            # NB: stores only the first epoch losses and remains unchanged further on
            first_epoch_kn_losses = isnone(first_epoch_kn_losses, kn_epoch_losses)
            # NB: is overwritten every epoch => stores the last epoch losses after the loop
            final_epoch_kn_losses = kn_epoch_losses

        final_epoch_kn_loss = np.mean(final_epoch_kn_losses)

        test_sdrs = encode_dataset(test_sdrs)
        accuracy = self.evaluate_ann_classifier(
            kn_ann_classifier, test_sdrs, self.data.test.targets
        )

        # add metrics
        epoch_metrics = self.metrics.setdefault('epochs', {})
        epoch_metrics[self.i_train_epoch] = {
            'se_entropy': entropy,
            'kn_loss': final_epoch_kn_loss,
            'kn_accuracy': accuracy,
        }

        if self.i_train_epoch == 1:
            step_metrics = self.metrics.setdefault('steps', {})
            step_metrics['1-1_loss'] = first_epoch_kn_losses
        print('<== Test')

    def run_ann(self):
        """
        Train 2-layer ANN classifier for N epochs with Batch SGD. Every epoch, the train
        dataset is split into batches, and the classifier is updated with each batch.
        We also collect all losses and provide it to the logger.

        Testing schedule determines when to evaluate the classifier on the test dataset.
        """
        train_sdrs, test_sdrs = get_data(self.data.train), get_data(self.data.test)

        self.i_train_epoch = 0
        while self.i_train_epoch < self.testing.n_epochs:
            self.i_train_epoch += 1
            self.print_with_timestamp(f'Epoch {self.i_train_epoch}')
            # NB: it is `nn_` instead of `kn` as both first and second layers trained for N epochs,
            # i.e. K-N mode for 2-layer ANN is N-N mode.
            nn_epoch_losses = self.train_epoch_ann_classifier(
                self.online_ann_classifier, train_sdrs, self.data.train.targets
            )
            self.test_epoch_ann(
                self.online_ann_classifier, test_sdrs, self.data.test.targets, nn_epoch_losses
            )

        self.log_progress(self.metrics)

    def test_epoch_ann(self, classifier, test_sdrs, targets, nn_epoch_losses):
        if not self.should_test():
            return

        nn_epoch_loss = np.mean(nn_epoch_losses)
        accuracy = self.evaluate_ann_classifier(classifier, test_sdrs, targets)
        print(f'MLP Accuracy: {accuracy:.3%} | Loss: {nn_epoch_loss:.3f}')

        epoch_metrics = self.metrics.setdefault('epochs', {})
        epoch_metrics[self.i_train_epoch] = {
            'kn_loss': nn_epoch_loss,
            'kn_accuracy': accuracy,
        }
        if self.i_train_epoch == 1:
            step_metrics = self.metrics.setdefault('steps', {})
            step_metrics['1-1_loss'] = nn_epoch_losses

    def evaluate_ann_classifier(self, classifier, test_sdrs, targets):
        batched_indices = split_to_batches(len(test_sdrs), self.training.batch_size)

        accuracy = 0.0
        for batch_ix in batched_indices:
            batch = np.zeros((len(batch_ix), classifier.feedforward_sds.size))
            fill_batch(batch, test_sdrs, batch_ix)

            target_cls = targets[batch_ix]
            prediction = classifier.predict(batch)
            accuracy += np.count_nonzero(np.argmax(prediction, axis=-1) == target_cls)

        accuracy /= len(test_sdrs)
        return accuracy

    def train_epoch_ann_classifier(self, classifier, train_sdrs, targets):
        order = self.rng.permutation(len(train_sdrs))
        batched_indices = split_to_batches(order, self.training.batch_size)

        losses = []
        for batch_ix in batched_indices:
            batch = np.zeros((len(batch_ix), classifier.feedforward_sds.size))
            fill_batch(batch, train_sdrs, batch_ix)

            target_cls = targets[batch_ix]
            classifier.learn(batch, target_cls)
            losses.append(classifier.losses[-1])

        return losses

    def make_ann_classifier(self) -> MlpClassifier:
        feedforward_sds = self.input_sds if self.encoder is None else self.output_sds
        return self.config.resolve_object(
            self.classifier, object_type_or_factory=MlpClassifier,
            feedforward_sds=feedforward_sds, n_classes=self.n_classes,

        )

    def log_progress(self, metrics: dict):
        if self.logger is None:
            return

        # NB: I log all collected metrics for the entire run in the end, simulating the correct
        # steps order, such that all step metrics (losses) have correct step number, and all
        # epoch summary metrics are logged in the first step.

        self.logger.define_metric("epoch")
        self.logger.define_metric("se_entropy", step_metric="epoch")
        self.logger.define_metric("kn_loss", step_metric="epoch")
        self.logger.define_metric("kn_accuracy", step_metric="epoch")

        # {metric: [batch stats], ...}
        step_metrics = metrics.pop('steps', {})
        # {epoch: {metric: value, ...}, ...}
        epoch_metrics = metrics.pop('epochs', {})

        # first, log step (=batch) stats
        step_metrics_names, step_metrics_arrays = zip(*step_metrics.items())
        step_metrics_names = list(step_metrics_names)
        for step_items in zip_longest(*step_metrics_arrays, fillvalue=None):
            self.logger.log({
                step_metrics_names[i_item]: item
                for i_item, item in enumerate(step_items)
                if item is not None
            })

        # second, log epoch stats
        for i_epoch, epoch in epoch_metrics.items():
            self.logger.log({'epoch': i_epoch, **epoch})

        # third, log epoch stats as summary
        self.logger.log({
            f'{key}/epoch_{i_epoch}': value
            for i_epoch, epoch in epoch_metrics.items()
            for key, value in epoch.items()
        })

    @staticmethod
    def _get_setup(input_mode: str, encoder: TConfig = None):
        return encoder, input_mode

    def should_test(self):
        return (
            self.i_train_epoch <= self.testing.eval_first or
            self.i_train_epoch == self.testing.n_epochs
        )

    def print_with_timestamp(self, *args, cond: bool = True):
        if not cond:
            return
        print_with_timestamp(self.init_time, *args)


def personalize_metrics(metrics: dict, prefix: str):
    return prepend_dict_keys(metrics, prefix, separator='/')


def get_data(data):
    return [data.get_sdr(i) for i in range(data.n_images)]


def split_to_batches(ds_size_or_order, batch_size):
    if isinstance(ds_size_or_order, int):
        ds_order = np.arange(ds_size_or_order)
        ds_size = ds_size_or_order
    else:
        ds_order = ds_size_or_order
        ds_size = len(ds_order)
    return np.array_split(ds_order, ds_size // batch_size)


def fill_batch(batch, ds, batch_ix, encoder=None, learn=False):
    if encoder is None:
        for i, sdr_ix in enumerate(batch_ix):
            sdr, rates = split_sdr_values(ds[sdr_ix])
            batch[i, sdr] = rates
    else:
        for i, sdr_ix in enumerate(batch_ix):
            sdr, rates = split_sdr_values(
                encoder.compute(ds[sdr_ix], learn=learn)
            )
            batch[i, sdr] = rates
