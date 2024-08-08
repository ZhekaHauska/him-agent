#  Copyright (c) 2022 Autonomous Non-Profit Organization "Artificial Intelligence Research
#  Institute" (AIRI); Moscow Institute of Physics and Technology (National Research University).
#  All rights reserved.
#
#  Licensed under the AGPLv3 license. See LICENSE in the project root for license information.
from __future__ import annotations

from functools import partial
from pathlib import Path

import numpy as np
from tqdm import tqdm, trange

from hima.common.config.base import TConfig
from hima.common.config.global_config import GlobalConfig
from hima.common.lazy_imports import lazy_import
from hima.common.run.wandb import get_logger
from hima.common.sdr import OutputMode, wrap_as_rate_sdr
from hima.common.sdr_array import SdrArray
from hima.common.sds import Sds
from hima.common.timer import timer, print_with_timestamp
from hima.common.utils import isnone, prepend_dict_keys
from hima.experiments.temporal_pooling.resolvers.type_resolver import StpLazyTypeResolver
from hima.experiments.temporal_pooling.stats.sdr_tracker import SdrTracker
from hima.experiments.temporal_pooling.stp.mlp_torch import MlpClassifier
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

    def __init__(self, n_epochs: int, batch_size: int):
        self.n_epochs = n_epochs
        self.batch_size = batch_size


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


class SpatialEncoderOfflineExperiment:
    training: TrainConfig
    testing: TestConfig

    dataset_sds: Sds
    encoding_sds: Sds

    stats: EpochStats

    def __init__(
            self, config: TConfig, config_path: Path,
            log: bool, seed: int, train: TConfig, test: TConfig,
            setup: TConfig, classifier: TConfig, data: str,
            sdr_tracker: TConfig, debug: bool,
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
        (
            encoder, encoding_sds, input_mode, req_sdr_tracker,
            classifier_symexp_logits, ds_norm
        ) = self._get_setup(**setup)
        self.input_mode = OutputMode[input_mode.upper()]
        self.is_binary = self.input_mode == OutputMode.BINARY
        self.classifier_symexp_logits = classifier_symexp_logits

        if data in ['mnist', 'cifar']:
            from hima.experiments.temporal_pooling.data.mnist_ext import MnistDataset
            self.data = MnistDataset(seed=seed, binary=self.is_binary, ds=data, debug=debug)
            self.classification = True
        elif data == 'dvs':
            from hima.experiments.temporal_pooling.data.dvs_ext import DvsDataset
            ds_filepath = Path('~/data/outdoors_walking').expanduser()
            self.data = DvsDataset(seed=seed, filepath=ds_filepath, binary=self.is_binary)
            self.classification = False
        else:
            raise ValueError(f'Data type {data} is unsupported')

        self.dataset_sds = self.data.sds
        self.encoding_sds = Sds.make(encoding_sds)

        self.training = self.config.resolve_object(train, object_type_or_factory=TrainConfig)
        self.testing = self.config.resolve_object(test, object_type_or_factory=TestConfig)

        if encoder is not None:
            # spatial encoding layer + 1-layer linear ANN classifier
            self.encoder = self.config.resolve_object(
                encoder, feedforward_sds=self.dataset_sds, output_sds=self.encoding_sds
            )

            self.sdr_tracker = None
            if req_sdr_tracker:
                self.sdr_tracker: SdrTracker = self.config.resolve_object(
                    sdr_tracker, sds=self.encoding_sds
                )
            print(f'Encoder: {self.encoder.feedforward_sds} -> {self.encoder.output_sds}')

            normalizer = partial(
                normalize_ds, norm=ds_norm,
                p=getattr(self.encoder, 'lebesgue_p', None)
            )
            self.data.train.normalize(normalizer)
            self.data.test.normalize(normalizer)
        else:
            self.encoder = None

        self.n_classes = self.data.n_classes
        self.classifier: TConfig = classifier
        self.persistent_ann_classifier = self.make_ann_classifier()
        self.i_train_epoch = 0
        self.set_metrics()

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
        n_epochs = self.training.n_epochs
        self.i_train_epoch = 0
        while self.i_train_epoch < n_epochs:
            self.i_train_epoch += 1
            self.print_with_timestamp(f'Epoch {self.i_train_epoch}')

            self.train_epoch_se(self.data.train)

            # [on testing schedule] train and test ANN classifier in K-N mode
            self.test_epoch_se_ann_kn_mode(self.data.train, self.data.test)

    def train_epoch_se(self, data):
        n_samples = len(data)
        order = self.rng.permutation(n_samples)
        self.encode_array(data.sdrs, order=order, learn=True, track=False)
        self.print_encoding_speed('train')

    def test_epoch_se_ann_kn_mode(self, train_data, test_data):
        if not self.should_test():
            return

        print(f'==> Test after {self.i_train_epoch}')

        train_speed = self.get_encoding_speed()
        entropy = None
        if self.sdr_tracker is not None:
            entropy = self.sdr_tracker.on_sequence_finished(None, ignore=False)['H']

        # ==> train and test epoch-specific ANN classifier
        kn_ann_classifier = self.make_ann_classifier()
        track_sdrs = self.sdr_tracker is not None
        n_train_samples = len(train_data)
        train_order = np.arange(n_train_samples)
        encoded_train_sdrs = self.encode_array(
            train_data.sdrs, order=train_order, learn=False, track=track_sdrs
        )
        eval_speed = self.get_encoding_speed()

        first_epoch_kn_losses = None
        final_epoch_kn_losses = None
        for _ in trange(self.testing.n_epochs):
            kn_epoch_losses = self.train_epoch_ann_classifier(
                kn_ann_classifier, encoded_train_sdrs, train_data.targets
            )
            # NB: stores only the first epoch losses and remains unchanged further on
            first_epoch_kn_losses = isnone(first_epoch_kn_losses, kn_epoch_losses)
            # NB: is overwritten every epoch => stores the last epoch losses after the loop
            final_epoch_kn_losses = kn_epoch_losses

        final_epoch_kn_loss = np.mean(final_epoch_kn_losses)

        n_test_samples = len(test_data)
        test_order = np.arange(n_test_samples)
        encoded_test_sdrs = self.encode_array(
            test_data.sdrs, order=test_order, learn=False, track=track_sdrs
        )
        accuracy = self.evaluate_ann_classifier(
            kn_ann_classifier, encoded_test_sdrs, self.data.test.targets
        )
        self.print_decoder_quality(accuracy, final_epoch_kn_loss)

        # add metrics
        epoch_metrics = {
            'kn_loss': final_epoch_kn_loss,
            'kn_accuracy': accuracy,
        }
        if train_speed is not None and eval_speed is not None:
            epoch_metrics |= {
                'train_speed_kcps': train_speed,
                'eval_speed_kcps': eval_speed,
            }
        if entropy is not None:
            epoch_metrics['se_entropy'] = entropy

        self.log_progress(epoch_metrics)
        self.print_encoding_speed('eval')
        print('<== Test')

    def run_ann(self):
        """
        Train 2-layer ANN classifier for N epochs with Batch SGD. Every epoch, the train
        dataset is split into batches, and the classifier is updated with each batch.
        We also collect all losses and provide it to the logger.

        Testing schedule determines when to evaluate the classifier on the test dataset.
        """
        classifier = self.persistent_ann_classifier
        train_sdrs, train_targets = self.data.train.sdrs, self.data.train.targets
        test_data = self.data.test

        self.i_train_epoch = 0
        while self.i_train_epoch < self.testing.n_epochs:
            self.i_train_epoch += 1
            self.print_with_timestamp(f'Epoch {self.i_train_epoch}')
            # NB: it is `nn_` instead of `kn` as both first and second layers trained for N epochs,
            # i.e. K-N mode for 2-layer ANN is N-N mode.
            nn_epoch_losses = self.train_epoch_ann_classifier(classifier, train_sdrs, train_targets)
            self.test_epoch_ann(classifier, test_data, nn_epoch_losses)

    def train_epoch_ann_classifier(self, classifier, sdrs, targets):
        n_samples = len(sdrs)
        order = self.rng.permutation(n_samples)
        batched_indices = split_to_batches(order, self.training.batch_size)

        losses = []
        for batch_ixs in batched_indices:
            batch = sdrs.get_batch_dense(batch_ixs)
            target_cls = targets[batch_ixs]
            classifier.learn(batch, target_cls)
            losses.append(classifier.losses[-1])

        return losses

    def test_epoch_ann(self, classifier, data, nn_epoch_losses):
        if not self.should_test():
            return

        nn_epoch_loss = np.mean(nn_epoch_losses)
        accuracy = self.evaluate_ann_classifier(classifier, data.sdrs, data.targets)
        self.print_decoder_quality(accuracy, nn_epoch_loss)

        epoch_metrics = {
            'kn_loss': nn_epoch_loss,
            'kn_accuracy': accuracy,
        }
        self.log_progress(epoch_metrics)

    def evaluate_ann_classifier(self, classifier, sdrs, targets):
        n_samples = len(sdrs)
        order = np.arange(n_samples)
        batched_indices = split_to_batches(order, self.training.batch_size)

        sum_accuracy = 0.0
        for batch_ixs in batched_indices:
            batch = sdrs.get_batch_dense(batch_ixs)
            target_cls = targets[batch_ixs]

            prediction = classifier.predict(batch)
            sum_accuracy += self.get_accuracy(prediction, target_cls)

        return sum_accuracy / n_samples

    def encode_array(self, sdrs: SdrArray, *, order, learn=False, track=False):
        assert self.encoder is not None, 'Encoder is not defined'
        encoded_sdrs = []

        if getattr(self.encoder, 'compute_batch', False):
            # I expect that batch computing is defined for dense SDRs, as only for this kind
            # of encoding batch computing is reasonable.
            batched_indices = split_to_batches(order, self.training.batch_size)
            for batch_ixs in tqdm(batched_indices):
                batch_sdrs = sdrs.create_slice(batch_ixs)
                encoded_batch: SdrArray = self.encoder.compute_batch(batch_sdrs, learn=learn)
                if track:
                    self.sdr_tracker.on_sdr_batch_updated(encoded_batch, ignore=False)
                encoded_sdrs.extend(encoded_batch.sparse)
        else:
            # for single SDR encoding, compute expects a sparse SDR.
            for ix in tqdm(order):
                obs_sdr = sdrs.get_sdr(ix, binary=self.is_binary)
                enc_sdr = self.encoder.compute(obs_sdr, learn=learn)
                enc_sdr = wrap_as_rate_sdr(enc_sdr)
                if track:
                    self.sdr_tracker.on_sdr_updated(enc_sdr, ignore=False)
                encoded_sdrs.append(enc_sdr)

        return SdrArray(sparse=encoded_sdrs, sdr_size=self.encoding_sds.size)

    def log_progress(self, metrics: dict):
        if self.logger is None:
            return

        metrics['epoch'] = self.i_train_epoch
        self.logger.log(metrics)

    def get_accuracy(self, predictions, targets):
        if self.classification:
            # number of correct predictions in a batch
            return np.count_nonzero(np.argmax(predictions, axis=-1) == targets)

        # MSE over each prediction coordinates => sum MSE over a batch
        return np.mean((predictions - targets) ** 2, axis=-1).sum()

    def make_ann_classifier(self) -> MlpClassifier:
        if self.encoder is not None:
            layers = [self.encoding_sds.size, self.n_classes]
        else:
            layers = [self.dataset_sds.size, self.encoding_sds.size, self.n_classes]

        return self.config.resolve_object(
            self.classifier, object_type_or_factory=MlpClassifier,
            layers=layers, classification=self.classification,
            symexp_logits=self.classifier_symexp_logits
        )

    def print_decoder_quality(self, accuracy, nn_epoch_loss):
        if self.classification:
            print(f'MLP Accuracy: {accuracy:.3%} | Loss: {nn_epoch_loss:.3f}')
        else:
            print(f'MLP MSE: {accuracy:.3} | Loss: {nn_epoch_loss:.3f}')

    @staticmethod
    def _get_setup(
            input_mode: str, encoding_sds, encoder: TConfig = None, sdr_tracker: bool = True,
            classifier_symexp_logits: bool = False, ds_norm: str = None
    ):
        return encoder, encoding_sds, input_mode, sdr_tracker, classifier_symexp_logits, ds_norm

    def should_test(self):
        if self.i_train_epoch <= self.training.n_epochs:
            eval_scheduled = self.testing.tick()
            if eval_scheduled or self.i_train_epoch <= self.testing.eval_first:
                return True

        with_encoder = self.encoder is not None
        if with_encoder:
            # last training epoch (for encoder)
            return self.i_train_epoch == self.training.n_epochs

        # last training epoch (for ANN)
        return self.i_train_epoch == self.testing.n_epochs

    def print_with_timestamp(self, *args, cond: bool = True):
        if not cond:
            return
        print_with_timestamp(self.init_time, *args)

    def get_encoding_speed(self):
        if not hasattr(self.encoder, 'computation_speed'):
            return None

        return round(1.0 / self.encoder.computation_speed.get() / 1000.0, 2)

    def print_encoding_speed(self, kind):
        speed = self.get_encoding_speed()
        if speed is None:
            return
        print(f'{kind} speed_kcps: {speed:.2f}')

    def set_metrics(self):
        if self.logger is None:
            return

        self.logger.define_metric("epoch")
        self.logger.define_metric("se_entropy", step_metric="epoch")
        self.logger.define_metric("kn_loss", step_metric="epoch")
        self.logger.define_metric("kn_accuracy", step_metric="epoch")


def personalize_metrics(metrics: dict, prefix: str):
    return prepend_dict_keys(metrics, prefix, separator='/')


def split_to_batches(order, batch_size):
    n_samples = len(order)
    return np.array_split(order, n_samples // batch_size)


def normalize_ds(ds, norm, p=None):
    if norm is None:
        return ds
    if norm == 'l1':
        p = 1
    elif norm == 'l2':
        p = 2
    elif norm == 'lp':
        assert p is not None, 'p must be provided for lp norm'
    else:
        raise ValueError(f'Unknown normalization type: {norm}')

    r_p = np.linalg.norm(ds, ord=p, axis=-1)
    if np.ndim(r_p) > 0:
        r_p = r_p[:, np.newaxis]
    return ds / r_p
