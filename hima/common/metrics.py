#  Copyright (c) 2023 Autonomous Non-Profit Organization "Artificial Intelligence Research
#  Institute" (AIRI); Moscow Institute of Physics and Technology (National Research University).
#  All rights reserved.
#
#  Licensed under the AGPLv3 license. See LICENSE in the project root for license information.

import os
import numpy as np

from hima.common.lazy_imports import lazy_import

wandb = lazy_import('wandb')
sns = lazy_import('seaborn')
imageio = lazy_import('imageio')


class ScalarMetrics:
    def __init__(self, metrics, logger):
        self.logger = logger
        self.metrics = {metric: [] for metric in metrics.keys()}
        self.agg_func = {metric: func for metric, func in metrics.items()}

    def update(self, metric_values):
        for key, value in metric_values.items():
            self.metrics[key].append(value)

    def summarize(self):
        return {
            key: self.agg_func[key](values)
            for key, values in self.metrics.items()
            if len(values) > 0
        }

    def reset(self):
        self.metrics = {metric: [] for metric in self.metrics.keys()}

    def log(self, step):
        self.logger.log(self.summarize(), step=step)
        self.reset()


class HeatmapMetrics:
    def __init__(self, metrics, logger):
        self.logger = logger
        self.metrics = {metric: [] for metric in metrics.keys()}
        self.agg_func = {metric: func for metric, func in metrics.items()}

    def update(self, metric_values):
        for key, value in metric_values.items():
            self.metrics[key].append(value)

    def summarize(self):
        return {
            key: self.agg_func[key](values, axis=0)
            for key, values in self.metrics.items()
            if len(values) > 0
        }

    def reset(self):
        self.metrics = {metric: [] for metric in self.metrics.keys()}

    def log(self, step):
        from matplotlib import pyplot as plt
        average_metrics = self.summarize()

        log_dict = {}
        for key, value in average_metrics.items():
            plt.figure()
            log_dict[key] = wandb.Image(sns.heatmap(value))

        self.logger.log(log_dict, step=step)
        plt.close('all')

        self.reset()


class ImageMetrics:
    def __init__(self, metrics, logger, log_fps, log_dir='/tmp'):
        self.metrics = {metric: [] for metric in metrics}
        self.logger = logger
        self.log_fps = log_fps
        self.log_dir = log_dir

    def update(self, metric_values):
        for key, value in metric_values.items():
            self.metrics[key].append(value)

    def log(self, step):
        for metric, values in self.metrics.items():
            if len(values) > 1:
                gif_path = os.path.join(
                    self.log_dir,
                    f'{self.logger.name}_{metric.split("/")[-1]}_{step}.gif'
                )
                # use new v3 API
                imageio.v3.imwrite(
                    # mode 'L': gray 8-bit ints; duration = 1000 / fps; loop == 0: infinitely
                    gif_path, values, mode='L', duration=1000/self.log_fps, loop=0
                )
                self.logger.log({metric: wandb.Video(gif_path)}, step=step)
            elif len(values) == 1:
                self.logger.log({metric: wandb.Image(values[0])}, step=step)

        self.metrics = {metric: [] for metric in self.metrics.keys()}


class SRStackSurprise:
    def __init__(self, name, logger, srs_size, history_length=5, normalize=True):
        self.name = name
        self.logger = logger
        self.srs_size = srs_size
        self.history_length = history_length + 1
        self.normalize = normalize
        self.srs = np.ones((self.history_length, srs_size))
        self.timestep = 0
        self.ages = np.arange(self.history_length)[::-1]
        self.surprises = np.zeros(self.history_length)

    def update(self, sr, events):
        self.srs[self.timestep % self.history_length] = sr
        self.ages += 1
        self.ages %= self.history_length
        self.timestep += 1

        surprises = self.get_surprise(events)
        self.surprises[self.ages] += surprises

    def log(self, step):
        self.logger.log(
            {
                f'{self.name}_{i}':
                self.surprises[i]/(self.timestep - i)
                for i in range(self.history_length)
                if (self.timestep - i) > 0
            },
            step=step
        )
        self.reset()

    def reset(self):
        self.srs = np.ones((self.history_length, self.srs_size))
        self.timestep = 0
        self.ages = np.arange(self.history_length)[::-1]
        self.surprises = np.zeros(self.history_length)

    def get_surprise(self, events):
        if len(events) > 0:
            surprise = - np.sum(
                np.log(
                    np.clip(self.srs[:, events], 1e-7, 1)
                ),
                axis=-1
            )

            if self.normalize:
                surprise /= len(events)
        else:
            surprise = 0

        return surprise


class PredictionsStackSurprise:
    def __init__(self, name, logger, prediction_steps=5, normalize=True, mode='categorical'):
        self.name = name
        self.logger = logger
        self.prediction_steps = prediction_steps
        self.normalize = normalize
        self.mode = mode
        self.predictions = []
        self.surprises = [list() for _ in range(self.prediction_steps)]

    def update(self, predictions, events):
        self.predictions.append(predictions)

        # remove empty lists
        predictions = [x for x in self.predictions if len(x) > 0]

        pred_horizon = [self.prediction_steps - len(x) for x in predictions]
        current_predictions = [x.pop(0) for x in predictions]

        if len(events) > 0:
            for p, s in zip(current_predictions, pred_horizon):
                surp = get_surprise(
                    p.flatten(),
                    events,
                    mode=self.mode,
                    normalize=self.normalize
                )
                self.surprises[s].append(surp)

    def log(self, step):
        self.logger.log(
            {
                f'{self.name}_step_{s+1}': np.mean(x) for s, x in enumerate(self.surprises)
                if len(x) > 0
            },
            step=step
        )
        self.reset()

    def reset(self):
        self.predictions = []
        self.surprises = [list() for t in range(self.prediction_steps)]


def get_surprise(probs, obs, mode='bernoulli', normalize=True):
    """
    Calculate the surprise -log(p(o)), where o is observation

    'probs': distribution parameters

    'obs': indexes of variables in state 1

    'mode': bernoulli | categorical

        bernoulli
            'probs' are parameters of Bernoulli distributed vector

        categorical
            'probs' are parameters of Categorical distributed vector

    'normalize': bool
    """
    is_coincide = np.isin(
        np.arange(len(probs)), obs
    )

    surprise = - np.sum(
        np.log(
            np.clip(probs[is_coincide], 1e-7, 1)
        )
    )

    if mode == 'bernoulli':
        surprise += - np.sum(
            np.log(
                np.clip(1 - probs[~is_coincide], 1e-7, 1)
            )
        )
        if normalize:
            surprise /= len(probs)
    elif mode == 'categorical':
        if normalize:
            surprise /= len(obs)
    else:
        raise ValueError(f'There is no such mode "{mode}"')

    return surprise


def get_surprise_2(probs, obs, mode='bernoulli', normalize=True):
    """
    Calculate the surprise -log(p(o)), where o is observation

    'probs': distribution parameters
    'obs': indexes of variables in state 1
    'mode': bernoulli | categorical
        bernoulli
            'probs' are parameters of Bernoulli distributed vector
        categorical
            'probs' are parameters of Categorical distributed vector
    'normalize': bool
    """
    def clip(p):
        return np.clip(p, 1e-7, 1.)

    surprise = -np.sum(np.log(clip(probs[obs])))
    if mode == 'bernoulli':
        not_in_obs_mask = np.ones_like(probs, dtype=bool)
        not_in_obs_mask[obs] = False

        surprise += -np.sum(np.log(clip(1. - probs[not_in_obs_mask])))
        if normalize:
            surprise /= len(probs)
    elif mode == 'categorical':
        if normalize:
            surprise /= len(obs)
    else:
        raise ValueError(f'There is no such mode "{mode}"')

    return surprise
