#  Copyright (c) 2022 Autonomous Non-Profit Organization "Artificial Intelligence Research
#  Institute" (AIRI); Moscow Institute of Physics and Technology (National Research University).
#  All rights reserved.
#
#  Licensed under the AGPLv3 license. See LICENSE in the project root for license information.

import numpy as np
import wandb
from matplotlib import pyplot as plt
from minisom import MiniSom

from hima.common.sds import Sds
from hima.envs.rbits import RandomBits
from hima.experiments.temporal_pooling.stp.sp import SpatialPooler


class SpAttractorRandBitsExperiment:
    def __init__(self, logger, conf):
        self.seed = conf['run']['seed']
        self.logger = logger

        self.env = RandomBits(
            Sds(conf['env']['sds']),
            conf['env']['similarity_range'],
            seed=self.seed
        )

        input_sds = self.env.sds

        if conf.get('encoder') is not None:
            conf['encoder']['seed'] = self.seed
            conf['encoder']['feedforward_sds'] = input_sds
            conf['encoder']['output_sds'] = Sds(conf['encoder']['output_sds'])
            self.encoder = SpatialPooler(**conf['encoder'])

            conf['attractor']['adapt_to_ff_sparsity'] = False
            attractor_sds = self.encoder.output_sds
        else:
            conf['attractor']['adapt_to_ff_sparsity'] = True
            attractor_sds = input_sds
            self.encoder = None

        conf['attractor']['feedforward_sds'] = attractor_sds
        conf['attractor']['output_sds'] = Sds(conf['attractor']['output_sds'])
        conf['attractor']['seed'] = self.seed
        self.attractor = SpatialPooler(**conf['attractor'])

        self.max_steps = conf['run']['max_steps']
        self.n_episodes = conf['run']['n_episodes']
        self.log_update_rate = conf['run'].get('update_rate')
        self.n_trajectories = conf['run'].get('n_trajectories', 0)
        self.attractor_steps = conf['run'].get('attractor_steps', 0)
        self.learn_attractor = conf['run'].get('learn_attractor_in_loop', False)
        self.pairs_per_trajectory = conf['run'].get('pairs_per_trajectory', 1)

        self.som_iterations = conf['run'].get('som_iterations', 100)
        self.som_learning_rate = conf['run'].get('som_learning_rate', 0.5)
        self.som_sigma = conf['run'].get('som_sigma', 1.0)
        self.som_size = conf['run'].get('som_size', 100)

        if self.logger is not None:
            self.logger.define_metric(
                name='convergence/io_hist',
                step_metric='iteration'
            )
            self.logger.define_metric(
                name='som/clusters',
                step_metric='iteration'
            )

    def run(self):
        for i in range(self.n_episodes):
            steps = 0
            att_entropy = []
            enc_entropy = []

            self.env.reset()

            while True:
                obs = self.env.obs()
                self.env.act(obs, None)
                self.env.step()

                if self.encoder is not None:
                    obs = self.encoder.compute(obs, learn=True)
                    enc_entropy.append(self.encoder.output_entropy())

                self.attractor.compute(obs, learn=True)
                att_entropy.append(self.attractor.output_entropy())

                steps += 1
                if steps >= self.max_steps:
                    break

            if self.logger is not None:
                self.logger.log(
                    {'main_metrics/attractor_entropy': np.array(att_entropy).mean()},
                    step=i
                )
                if self.encoder is not None:
                    self.logger.log(
                        {'main_metrics/encoder_entropy': np.array(enc_entropy).mean()},
                        step=i
                    )

                if (self.log_update_rate is not None) and (i % self.log_update_rate == 0):
                    trajectories = list()

                    for _ in range(self.n_trajectories):
                        trajectory = list()

                        pattern = self.env.obs()
                        self.env.act(pattern, None)
                        self.env.step()

                        trajectory.append(pattern)

                        for _ in range(self.attractor_steps):
                            if (self.encoder is not None) and (_ == 0):
                                pattern = self.encoder.compute(pattern, learn=False)
                            else:
                                pattern = self.attractor.compute(pattern, self.learn_attractor)

                            trajectory.append(pattern)

                        trajectories.append(trajectory)

        if self.logger is not None:
            similarities = list()

            # generate non-repetitive trajectory pairs
            pair1 = np.repeat(
                np.arange(len(trajectories) - self.pairs_per_trajectory),
                self.pairs_per_trajectory
            )
            pair2 = (
                    np.tile(
                        np.arange(self.pairs_per_trajectory) + 1,
                        len(trajectories) - self.pairs_per_trajectory
                    ) + pair1
            )
            for p1, p2 in zip(pair1, pair2):
                similarity = list()

                for att_step, x in enumerate(zip(trajectories[p1], trajectories[p2])):
                    sim = self.similarity(x[0], x[1])

                    similarity.append(sim)

                similarities.append(similarity)

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

            for j in range(self.attractor_steps+1):
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

                dim = int(np.sqrt(self.som_size))
                som = MiniSom(
                    dim, dim,
                    pattern_size,
                    sigma=self.som_sigma,
                    learning_rate=self.som_learning_rate,
                    random_seed=self.seed
                )
                som.pca_weights_init(dense_patterns)
                som.train(dense_patterns, self.som_iterations)

                fig = plt.figure(figsize=(8, 8))
                plt.imshow(som.distance_map(), cmap='Greys', alpha=0.5)
                plt.colorbar()

                self.logger.log(
                    {
                        'som/clusters': wandb.Image(fig),
                        'iteration': j
                    },
                    step=i
                )
                plt.close('all')

                out_sim = similarities[:, j]
                hist, x, y = np.histogram2d(in_sim, out_sim)
                x, y = np.meshgrid(x, y)

                self.logger.log(
                    {
                        'convergence/io_hist': wandb.Image(
                            plt.pcolormesh(x, y, hist.T)
                        )
                    },
                    step=i
                )

                i += 1

    def attract(self, steps, pattern, learn=False):
        trajectory = list()

        if self.encoder is not None:
            pattern = self.encoder.compute(pattern, learn=False)

        trajectory.append(pattern)
        for step in range(steps):
            pattern = self.attractor.compute(pattern, learn)
            trajectory.append(pattern)

        return trajectory

    def similarity(self, x1, x2):
        return np.count_nonzero(np.isin(x1, x2)) / x2.size
