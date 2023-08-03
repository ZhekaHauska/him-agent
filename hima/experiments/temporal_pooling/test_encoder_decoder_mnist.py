#  Copyright (c) 2022 Autonomous Non-Profit Organization "Artificial Intelligence Research
#  Institute" (AIRI); Moscow Institute of Physics and Technology (National Research University).
#  All rights reserved.
#
#  Licensed under the AGPLv3 license. See LICENSE in the project root for license information.
from pathlib import Path

import numpy as np

from hima.common.config.base import TConfig
from hima.common.config.global_config import GlobalConfig
from hima.common.lazy_imports import lazy_import
from hima.common.run.wandb import get_logger
from hima.common.sdr import SparseSdr
from hima.common.sds import Sds
from hima.common.timer import timer, print_with_timestamp
from hima.common.utils import isnone, safe_divide, prepend_dict_keys
from hima.envs.mnist import MNISTEnv
from hima.experiments.temporal_pooling.data.mnist import MnistDataset
from hima.experiments.temporal_pooling.resolvers.type_resolver import StpLazyTypeResolver
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


class SpEncoderDecoderExperiment:
    training: TrainConfig
    attraction: AttractionConfig
    testing: TestConfig

    def __init__(
            self, config: TConfig, config_path: Path,
            log: bool, seed: int,
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

        for epoch in range(1, self.training.n_epochs + 1):
            self.print_with_timestamp(f'Epoch {epoch}')
            self.train_epoch()

        # self.log_final_results()

    def train_epoch(self):
        sample_indices = self.rng.choice(self.data.n_images, size=self.training.n_steps)
        for step in range(1, self.training.n_steps + 1):
            sample_ind = sample_indices[step - 1]

            sample = self.data.sdrs[sample_ind]
            self.process_sample(sample, learn=True)

    def test_epoch(self):
        ...

    def process_sample(self, sample: SparseSdr, learn: bool) -> list[SparseSdr]:
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
        return sdrs

    def log_progress(self, episode: int, attractor_entropy, encoder_entropy):
        if self.logger is None:
            return

        import matplotlib.pyplot as plt

        main_metrics = dict(attractor_entropy=np.array(attractor_entropy).mean())
        if self.encoder is not None:
            main_metrics |= dict(encoder_entropy=np.array(encoder_entropy).mean())

        main_metrics = personalize_metrics(main_metrics, 'main')

        trajectories, targets = self.generate_trajectories()
        similarities, relative_similarity, class_counts, sim_matrices = self.analyse_trajectories(
            trajectories=trajectories, targets=targets
        )

        step_metrics = {}
        for step in range(self.testing.items_per_class):
            step_metrics[f'{step=}'] = relative_similarity[step].mean()
        step_metrics = personalize_metrics(step_metrics, 'step')

        relative_similarity_df = pd.DataFrame(
            relative_similarity,
            columns=np.arange(relative_similarity.shape[1])
        )
        convergence_metrics = dict(
            relative_similarity=wandb.Image(sns.lineplot(data=relative_similarity_df))
        )
        plt.close('all')

        convergence_metrics |= dict(
            class_pair_counts=wandb.Image(sns.heatmap(class_counts))
        )
        plt.close('all')

        fig, axs = plt.subplots(ncols=4, sharey='row', figsize=(16, 4))
        axs[0].set_title('raw_sim')
        axs[1].set_title('1-step')
        axs[2].set_title(f'{self.testing.items_per_class // 2}-step')
        axs[3].set_title(f'{self.testing.items_per_class}-step')
        sns.heatmap(sim_matrices[0], ax=axs[0], cmap='viridis')
        sns.heatmap(sim_matrices[1], ax=axs[1], cmap='viridis')
        sns.heatmap(
            sim_matrices[self.testing.items_per_class // 2], ax=axs[2], cmap='viridis'
        )
        sns.heatmap(sim_matrices[-1], ax=axs[3], cmap='viridis')

        convergence_metrics |= dict(
            similarity_per_class=wandb.Image(fig)
        )
        plt.close('all')

        convergence_metrics = personalize_metrics(convergence_metrics, 'convergence')

        self.logger.log(
            main_metrics | convergence_metrics | step_metrics, step=episode
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
