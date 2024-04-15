#  Copyright (c) 2022 Autonomous Non-Profit Organization "Artificial Intelligence Research
#  Institute" (AIRI); Moscow Institute of Physics and Technology (National Research University).
#  All rights reserved.
#
#  Licensed under the AGPLv3 license. See LICENSE in the project root for license information.
from __future__ import annotations

from pathlib import Path

import numpy as np
import numpy.typing as npt
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

    def __init__(self, n_epochs: int):
        self.n_epochs = n_epochs


class TestConfig:
    eval_countdown: RepeatingCountdown
    n_epochs: int
    batch_size: int

    def __init__(self, eval_schedule: int, n_epochs: int, batch_size: int):
        self.eval_countdown = make_repeating_counter(eval_schedule)
        self.n_epochs = n_epochs
        self.batch_size = batch_size

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

        self.encoder = self.config.resolve_object(
            encoder, feedforward_sds=self.input_sds, output_sds=self.output_sds
        )
        self.n_classes = self.data.n_classes
        self.classifier: TConfig = classifier

        print(f'Encoder: {self.encoder.feedforward_sds} -> {self.encoder.output_sds}')

        self.training = self.config.resolve_object(train, object_type_or_factory=TrainConfig)
        self.testing = self.config.resolve_object(test, object_type_or_factory=TestConfig)

        self.sdr_tracker: SdrTracker = self.config.resolve_object(sdr_tracker, sds=self.output_sds)
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

    def train_epoch(self):
        data = self.data.train
        for ix in tqdm(self.rng.permutation(data.n_images)):
            sdr = data.get_sdr(ix)
            self.encoder.compute(sdr, learn=True)

    def test_epoch(self):
        def encode_dataset(data):
            encoded_sdrs = []
            for sample_ix in range(data.n_images):
                obs_sdr = data.get_sdr(sample_ix)
                enc_sdr = self.encoder.compute(obs_sdr, learn=False)
                self.sdr_tracker.on_sdr_updated(enc_sdr, ignore=False)
                encoded_sdrs.append(enc_sdr)
            return encoded_sdrs

        print(f'==> Test after {self.i_train_epoch}')

        train_encoded_sdrs = encode_dataset(self.data.train)
        test_encoded_sdrs = encode_dataset(self.data.test)

        metrics = {}
        metrics |= self.sdr_tracker.on_sequence_finished(None, ignore=False)
        metrics |= self.eval_with_ensemble(train_encoded_sdrs, test_encoded_sdrs)
        metrics |= self.eval_with_mlp(train_encoded_sdrs, test_encoded_sdrs)

        metrics = personalize_metrics(metrics, 'eval')
        self.log_progress(metrics)
        print('<== Test')

    def eval_with_ensemble(self, train_sdrs: list[AnySparseSdr], test_sdrs: list[AnySparseSdr]):
        print('Evaluating with ensemble')
        classifier = self.train_ensemble_classifier(train_sdrs)
        accuracy = 0.

        data = self.data.test
        for i in range(data.n_images):
            target_cls = data.targets[i]
            encoded_sdr = test_sdrs[i]

            sdr, rates = split_sdr_values(encoded_sdr)
            prediction = np.mean(classifier[sdr] * rates[:, None], axis=0)
            accuracy += prediction.argmax() == target_cls

        accuracy /= data.n_images

        print(f'Accuracy: {accuracy:.3%}')
        return {
            'ens_accuracy': accuracy,
        }

    def eval_with_mlp(self, train_sdrs: list[AnySparseSdr], test_sdrs: list[AnySparseSdr]):
        print('Evaluating with MLP')
        classifier = self.train_mlp_classifier(train_sdrs)
        accuracy = 0.
        dense_sdr = np.zeros(self.output_sds.size)

        data = self.data.test
        for i in range(data.n_images):
            target_cls = data.targets[i]
            sdr, rates = split_sdr_values(test_sdrs[i])
            dense_sdr[sdr] = rates

            prediction = classifier.predict(dense_sdr)
            accuracy += prediction.argmax() == target_cls
            dense_sdr[sdr] = 0.

        accuracy /= data.n_images

        loss = np.mean(classifier.losses)

        print(f'Accuracy: {accuracy:.3%} | Loss: {loss:.3f}')
        return {
            'mlp_accuracy': accuracy,
            'mlp_loss': loss,
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

    def train_mlp_classifier(self, train_enc_sdrs: list[AnySparseSdr]):
        classifier: MlpClassifier = self.config.resolve_object(
            self.classifier, object_type_or_factory=MlpClassifier,
            feedforward_sds=self.output_sds, n_classes=self.n_classes
        )

        data = self.data.train
        for _ in trange(self.testing.n_epochs):
            indices = self.rng.permutation(data.n_images)
            batched_indices = np.array_split(indices, len(indices) // self.testing.batch_size)

            for batch_ix in batched_indices:
                batch = np.zeros((len(batch_ix), self.output_sds.size))
                for i, encoded_sdr_ix in enumerate(batch_ix):
                    encoded_sdr_ix: int
                    sdr, rates = split_sdr_values(train_enc_sdrs[encoded_sdr_ix])
                    batch[i, sdr] = rates

                target_cls = data.targets[batch_ix]
                classifier.learn(batch, target_cls)

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
    def _get_setup(encoder: TConfig, input_mode: str):
        return encoder, input_mode

    def print_with_timestamp(self, *args, cond: bool = True):
        if not cond:
            return
        print_with_timestamp(self.init_time, *args)


def personalize_metrics(metrics: dict, prefix: str):
    return prepend_dict_keys(metrics, prefix, separator='/')
