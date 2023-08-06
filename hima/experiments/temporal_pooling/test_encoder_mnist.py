#  Copyright (c) 2022 Autonomous Non-Profit Organization "Artificial Intelligence Research
#  Institute" (AIRI); Moscow Institute of Physics and Technology (National Research University).
#  All rights reserved.
#
#  Licensed under the AGPLv3 license. See LICENSE in the project root for license information.
import math
from pathlib import Path

import numpy as np

from hima.common.config.base import TConfig
from hima.common.config.global_config import GlobalConfig
from hima.common.lazy_imports import lazy_import
from hima.common.run.wandb import get_logger
from hima.common.sdr import SparseSdr, sparse_to_dense
from hima.common.sds import Sds
from hima.common.timer import timer, print_with_timestamp
from hima.common.utils import isnone, safe_divide, prepend_dict_keys
from hima.envs.mnist import MNISTEnv
from hima.experiments.hmm.runners.utils import get_surprise_2, get_surprise
from hima.experiments.temporal_pooling.data.mnist import MnistDataset
from hima.experiments.temporal_pooling.resolvers.type_resolver import StpLazyTypeResolver
from hima.experiments.temporal_pooling.stp.sp_decoder import SpatialPoolerDecoder
from hima.experiments.temporal_pooling.utils import resolve_random_seed, Scheduler

wandb = lazy_import('wandb')
sns = lazy_import('seaborn')
pd = lazy_import('pandas')


class TrainConfig:
    n_epochs: int
    n_steps: int

    def __init__(self, n_epochs: int, n_steps: int):
        self.n_epochs = n_epochs
        self.n_steps = n_steps


class EpochStats:
    states: list[SparseSdr]
    decode_errors: list[float]
    decode_surprise: list[float]

    def __init__(self):
        self.states = []
        self.decode_errors = []
        self.decode_surprise = []


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


class SpEncoderExperiment:
    training: TrainConfig
    attraction: AttractionConfig
    testing: TestConfig

    input_sds: Sds
    output_sds: Sds

    stats: EpochStats

    def __init__(
            self, config: TConfig, config_path: Path,
            log: bool, seed: int,
            train: TConfig, test: TConfig,
            encoder: TConfig, decoder_noise: float,
            project: str = None,
            wandb_init: TConfig = None,
            plot_sample: bool = False,
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

        self.data = MnistDataset()

        self.input_sds = self.data.output_sds
        self.encoder = self.config.resolve_object(encoder, feedforward_sds=self.input_sds)
        self.decoder = SpatialPoolerDecoder(self.encoder)
        self.decoder_noise = decoder_noise

        print(f'Encoder: {self.encoder.feedforward_sds} -> {self.encoder.output_sds}')
        self.output_sds = self.encoder.output_sds

        self.training = self.config.resolve_object(train, object_type_or_factory=TrainConfig)
        self.testing = self.config.resolve_object(test, object_type_or_factory=TestConfig)
        self.stats = None

        self.plot_sample = plot_sample

    def run(self):
        self.print_with_timestamp('==> Run')
        # self.define_metrics()

        for epoch in range(1, self.training.n_epochs + 1):
            self.print_with_timestamp(f'Epoch {epoch}')
            self.train_epoch()
            self.log_progress(epoch)

        # self.log_final_results()

    def train_epoch(self):
        sample_indices = self.rng.choice(self.data.n_images, size=self.training.n_steps)

        self.stats = EpochStats()
        for step in range(1, self.training.n_steps + 1):
            sample_ind = sample_indices[step - 1]
            self.process_sample(sample_ind, learn=True)

    def test_epoch(self):
        ...

    def noisy(self, x):
        noisy = x + np.abs(self.rng.normal(scale=self.decoder_noise, size=x.shape))
        s = noisy.sum()
        return safe_divide(noisy, s)

    def process_sample(self, obs_ind: int, learn: bool):
        obs = self.data.sdrs[obs_ind]
        dense_obs = self.data.dense_sdrs[obs_ind]

        state = self.encoder.compute(obs, learn=learn)
        dense_state = sparse_to_dense(state, self.output_sds.size, dtype=float)

        state_probs = self.noisy(dense_state)

        decoded_obs = self.decoder.decode(state_probs)
        error = np.abs(dense_obs - decoded_obs).mean()
        surprise = get_surprise_2(decoded_obs, obs)

        # self.stats.states.append(state)
        self.stats.decode_errors.append(error)
        self.stats.decode_surprise.append(surprise)

    def plot_sample_diff(self):
        obs_ind = self.rng.choice(self.data.n_images)
        obs = self.data.sdrs[obs_ind]
        dense_obs = self.data.dense_sdrs[obs_ind].astype(float).reshape(self.data.image_shape)

        state = self.encoder.compute(obs, learn=False)
        dense_state = sparse_to_dense(state, self.output_sds.size, dtype=float)

        state_probs = self.noisy(dense_state)

        decoded_obs = self.decoder.decode(state_probs).reshape(self.data.image_shape)
        error = np.abs(dense_obs - decoded_obs).mean()

        if self.plot_sample:
            w = self.encoder.weights[state[0]].reshape(-1, 1)
            w = np.repeat(w, 10, axis=1)

            from hima.common.plot_utils import plot_grid_images
            plot_grid_images(
                images=[dense_obs, decoded_obs, w],
                titles=['Orig', 'Decoded', '[0].w'],
                show=True,
                with_value_text_flags=[True, True, True],
                cols_per_row=3
            )
        return np.hstack([dense_obs, decoded_obs])

    def log_progress(self, epoch: int):
        if self.logger is None:
            return

        sample_prediction = self.plot_sample_diff()

        avg_sparsity = self.encoder.ff_avg_active_size / self.encoder.feedforward_sds.size
        mae = np.array(self.stats.decode_errors).mean()
        main_metrics = dict(
            mae=mae,
            nmae=mae / avg_sparsity,
            surprise=np.array(self.stats.decode_surprise).mean(),
            entropy=self.encoder.output_entropy(),
        )
        main_metrics = personalize_metrics(main_metrics, 'main')

        print(main_metrics)
        if isinstance(self.log, bool):
            images = dict(
                sample_prediction=wandb.Image(sample_prediction)
            )
            personalize_metrics(images, 'img')
            main_metrics |= images

        self.logger.log(
            main_metrics,
            step=epoch
        )

    def log_final_results(self):
        if self.logger is None:
            return

        import matplotlib.pyplot as plt

        episode = self.training.n_epochs

        trajectories, targets = self.generate_trajectories()
        similarities, relative_similarity, class_counts, sim_matrices = self.analyse_trajectories(
            trajectories=trajectories, targets=targets
        )

        similarities = np.array(similarities)
        in_sim = similarities[:, 0]

        start_images = [x[0] for x in trajectories]
        dense_start_images = np.zeros(
            (len(start_images), self.encoder.feedforward_sds.size),
            dtype='float32'
        )
        for im_id, x in enumerate(start_images):
            dense_start_images[im_id, x] = 1

        trajectories = np.array(
            [x[1:] for x in trajectories]
        )

        for j in range(relative_similarity.shape[0]):
            out_sim = similarities[:, j]
            hist, x, y = np.histogram2d(in_sim, out_sim)
            x, y = np.meshgrid(x, y)

            self.logger.log(
                {
                    'main_metrics/relative_similarity': relative_similarity[j].mean(),
                    'convergence/io_hist': wandb.Image(
                        plt.pcolormesh(x, y, hist.T)
                    )
                },
                step=episode
            )

            for cls in range(10):
                self.logger.log(
                    {
                        f'relative_similarity/class {cls}': relative_similarity[j, cls]
                    },
                    step=episode
                )

            if not self.so_map.enabled:
                continue

            if j > 0:
                patterns = trajectories[:, j - 1]
                pattern_size = self.encoder.output_sds.size
                n_patterns = patterns.shape[0]
                dense_patterns = np.zeros((n_patterns, pattern_size), dtype='float32')
                for k, p in enumerate(patterns):
                    dense_patterns[k, p] = 1
            else:
                pattern_size = self.encoder.feedforward_sds.size
                dense_patterns = dense_start_images

            dim = int(np.sqrt(self.so_map.size))
            som = minisom.MiniSom(
                dim, dim,
                pattern_size,
                sigma=self.so_map.sigma,
                learning_rate=self.so_map.learning_rate,
                random_seed=self.seed
            )
            som.pca_weights_init(dense_patterns)
            som.train(dense_patterns, self.so_map.iterations)

            activation_map = np.zeros((dim, dim, 10))
            fig = plt.figure(figsize=(8, 8))
            plt.imshow(som.distance_map(), cmap='Greys', alpha=0.5)
            plt.colorbar()

            for p, cls in zip(dense_patterns, targets):
                activation_map[:, :, cls] += som.activate(p)

                cell = som.winner(p)
                plt.text(
                    cell[0],
                    cell[1],
                    str(cls),
                    color=plt.cm.rainbow(cls/10),
                    alpha=0.1,
                    fontdict={'weight': 'bold', 'size': 16}
                )
            # plt.axis([0, som.get_weights().shape[0], 0, som.get_weights().shape[1]])

            self.logger.log({
                'som/clusters': wandb.Image(fig),
                'iteration': j
            }, step=episode)
            plt.close('all')

            # normalize activation map
            activation_map /= dense_patterns.shape[0]
            activation_map /= activation_map.sum(axis=-1).reshape((dim, dim, 1))
            # generate colormap
            colors = [plt.cm.rainbow(c/10)[:-1] for c in range(10)]
            color_map = (np.dot(activation_map.reshape((-1, 10)), colors) * 255)
            color_map = color_map.reshape((dim, dim, 3))

            for cls in range(10):
                self.logger.log(
                    {
                        f'som/activation {cls}': wandb.Image(
                            sns.heatmap(activation_map[:, :, cls], cmap='viridis')
                        )
                    },
                    step=episode
                )
                plt.close('all')

            self.logger.log(
                {
                    'som/soft_clusters': wandb.Image(
                        plt.imshow(color_map.astype('uint8'))
                    )
                },
                step=episode
            )
            plt.close('all')

    def generate_trajectories(self):
        targets = []
        trajectories = []

        for _ in range(self.testing.n_trajectories):
            trajectory = []

            image, cls = self.env.obs(return_class=True)
            self.env.step()

            pattern = self.preprocess(image)
            trajectory.append(pattern)

            for attr_step in range(self.testing.items_per_class):
                if self.encoder is not None and attr_step == 0:
                    pattern = self.encoder.compute(pattern, learn=False)
                else:
                    pattern = self.attractor.compute(
                        pattern, self.testing.learn_attractor_in_loop
                    )

                trajectory.append(pattern)

            trajectories.append(trajectory)
            targets.append(cls)
        return trajectories, targets

    def analyse_trajectories(self, trajectories, targets):
        similarities = list()
        sim_matrices = np.zeros((self.testing.items_per_class + 1, 10, 10))
        class_counts = np.zeros((10, 10))

        # generate non-repetitive trajectory pairs
        pair1 = np.repeat(
            np.arange(len(trajectories) - self.testing.pairs_per_trajectory),
            self.testing.pairs_per_trajectory
        )
        pair2 = np.tile(
            np.arange(self.testing.pairs_per_trajectory) + 1,
            len(trajectories) - self.testing.pairs_per_trajectory
        ) + pair1

        for p1, p2 in zip(pair1, pair2):
            similarity = list()
            cls1 = targets[p1]
            cls2 = targets[p2]
            class_counts[cls1, cls2] += 1
            class_counts[cls2, cls1] += 1

            for att_step, x in enumerate(zip(trajectories[p1], trajectories[p2])):
                sim = self.similarity(x[0], x[1])

                sim_matrices[att_step, cls1, cls2] += sim
                sim_matrices[att_step, cls2, cls1] += sim

                similarity.append(sim)

            similarities.append(similarity)

        sim_matrices /= class_counts
        # divide each row in each matrix by its diagonal element
        relative_similarity = (
            sim_matrices / np.diagonal(sim_matrices, axis1=1, axis2=2)[:, :, None]
        ).mean(axis=-1)

        return similarities, relative_similarity, class_counts, sim_matrices

    def attract(self, steps, pattern, learn=False):
        trajectory = list()

        if self.encoder is not None:
            pattern = self.encoder.compute(pattern, learn=False)

        trajectory.append(pattern)
        for step in range(steps):
            pattern = self.attractor.compute(pattern, learn)
            trajectory.append(pattern)

        return trajectory

    @staticmethod
    def similarity(x1, x2):
        return safe_divide(np.count_nonzero(np.isin(x1, x2)), x2.size)

    def print_with_timestamp(self, *args, cond: bool = True):
        if not cond:
            return
        print_with_timestamp(self.init_time, *args)

    def define_metrics(self):
        if self.logger is None:
            return

        self.logger.define_metric(
            name='main_metrics/relative_similarity',
            step_metric='iteration'
        )
        self.logger.define_metric(
            name='convergence/io_hist',
            step_metric='iteration'
        )
        self.logger.define_metric(
            name='som/clusters',
            step_metric='iteration'
        )

        for cls in range(10):
            self.logger.define_metric(
                name=f'relative_similarity/class {cls}',
                step_metric='iteration'
            )


def personalize_metrics(metrics: dict, prefix: str):
    return prepend_dict_keys(metrics, prefix, separator='/')
