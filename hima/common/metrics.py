#  Copyright (c) 2023 Autonomous Non-Profit Organization "Artificial Intelligence Research
#  Institute" (AIRI); Moscow Institute of Physics and Technology (National Research University).
#  All rights reserved.
#
#  Licensed under the AGPLv3 license. See LICENSE in the project root for license information.

import os
import numpy as np

from hima.agents.episodic_control.agent import ECAgent, euclidian_sim, cosine_similarity
from hima.common.lazy_imports import lazy_import
from typing import Dict, Literal, Optional

from hima.common.sdr import sparse_to_dense
from hima.modules.belief.utils import normalize
from scipy.special import rel_entr
import matplotlib.pyplot as plt

wandb = lazy_import('wandb')
sns = lazy_import('seaborn')
imageio = lazy_import('imageio')
minisom = lazy_import('minisom')


class BaseMetric:
    def __init__(self, logger, runner,
                 update_step, log_step, update_period, log_period):
        self.logger = logger
        self.runner = runner
        self.update_step = update_step
        self.log_step = log_step
        self.update_period = update_period
        self.log_period = log_period

        self.last_update_step = None
        self.last_log_step = None

    def step(self):
        update_step = self.get_attr(self.update_step)
        log_step = self.get_attr(self.log_step)

        if (self.last_update_step is None) or (self.last_update_step != update_step):
            if (update_step % self.update_period) == 0:
                self.update()

        if (self.last_log_step is None) or (self.last_log_step != log_step):
            if (log_step % self.log_period) == 0:
                self.log(log_step)

        self.last_update_step = update_step
        self.last_log_step = log_step

    def update(self):
        raise NotImplementedError

    def log(self, step):
        raise NotImplementedError

    def get_attr(self, attr):
        obj = self.runner
        for a in attr.split('.'):
            obj = getattr(obj, a)
        return obj


class MetricsRack:
    metrics: Dict[str, BaseMetric]

    def __init__(self, logger, runner, **kwargs):
        self.metrics = dict()

        for name, params in kwargs.items():
            cls = params['class']
            params = params['params']
            self.metrics[name] = eval(cls)(**params, logger=logger, runner=runner)

    def step(self):
        for name in self.metrics.keys():
            self.metrics[name].step()


class ScalarMetrics(BaseMetric):
    def __init__(self, metrics, logger, runner,
                 update_step, log_step, update_period, log_period):
        super().__init__(logger, runner, update_step, log_step, update_period, log_period)

        self.metrics = {metric: [] for metric in metrics.keys()}

        for metric in metrics.keys():
            self.logger.define_metric(metric, step_metric=self.log_step)

        self.agg_func = {
            metric: eval(params['agg']) if type(params['agg']) is str else params['agg']
            for metric, params in metrics.items()
        }
        self.att_to_log = {
            metric: params['att']
            for metric, params in metrics.items()
        }

    def update(self):
        for name in self.metrics.keys():
            value = self.get_attr(self.att_to_log[name])
            self.metrics[name].append(value)

    def log(self, step):
        log_dict = {self.log_step: step}
        log_dict.update(self._summarize())
        self.logger.log(log_dict)
        self._reset()

    def _reset(self):
        self.metrics = {metric: [] for metric in self.metrics.keys()}

    def _summarize(self):
        return {
            key: self.agg_func[key](values)
            for key, values in self.metrics.items()
            if len(values) > 0
        }


class HeatmapMetrics(BaseMetric):
    def __init__(self, metrics, logger, runner,
                 update_step, log_step, update_period, log_period):
        super().__init__(logger, runner, update_step, log_step, update_period, log_period)
        self.logger = logger
        self.metrics = {metric: [] for metric in metrics.keys()}
        self.agg_func = {
            metric: eval(params['agg']) if type(params['agg']) is str else params['agg']
            for metric, params in metrics.items()
        }
        self.att_to_log = {
            metric: params['att']
            for metric, params in metrics.items()
        }

    def update(self):
        for name in self.metrics.keys():
            value = self.get_attr(self.att_to_log[name])
            self.metrics[name].append(value)

    def log(self, step):
        from matplotlib import pyplot as plt
        average_metrics = self._summarize()

        log_dict = {self.log_step: step}
        for key, value in average_metrics.items():
            plt.figure()
            log_dict[key] = wandb.Image(sns.heatmap(value))

        self.logger.log(log_dict)
        plt.close('all')

        self._reset()

    def _reset(self):
        self.metrics = {metric: [] for metric in self.metrics.keys()}

    def _summarize(self):
        return {
            key: self.agg_func[key](values, axis=0)
            for key, values in self.metrics.items()
            if len(values) > 0
        }


class ImageMetrics(BaseMetric):
    def __init__(self, metrics, logger, runner,
                 update_step, log_step, update_period, log_period,
                 log_fps, log_dir='/tmp'):
        super().__init__(logger, runner, update_step, log_step, update_period, log_period)

        self.metrics = {metric: [] for metric in metrics}
        self.att_to_log = {
            metric: params['att']
            for metric, params in metrics.items()
        }
        self.logger = logger
        self.log_fps = log_fps
        self.log_dir = log_dir

    def update(self):
        for name in self.metrics.keys():
            value = self.get_attr(self.att_to_log[name])
            self.metrics[name].append(value)

    def log(self, step):
        log_dict = {self.log_step: step}
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
                log_dict[metric] = wandb.Video(gif_path)
            elif len(values) == 1:
                log_dict[metric] = wandb.Image(values[0])

        self.logger.log(log_dict)
        self._reset()

    def _reset(self):
        self.metrics = {metric: [] for metric in self.metrics.keys()}


class SRStackSurprise(BaseMetric):
    def __init__(self, name, att, logger, runner,
                 update_step, log_step, update_period, log_period,
                 srs_size, history_length=5, normalize=True):
        super().__init__(logger, runner, update_step, log_step, update_period, log_period)

        self.name = name
        self.att = att
        self.srs_size = srs_size
        self.history_length = history_length + 1
        self.normalize = normalize
        self.srs = np.ones((self.history_length, srs_size))
        self.timestep = 0
        self.ages = np.arange(self.history_length)[::-1]
        self.surprises = np.zeros(self.history_length)

        self.logger.define_metric(self.name, step_metric=self.log_step)

    def update(self):
        value = self.get_attr(self.att)
        sr = value['sr']
        events = value['events']

        self.srs[self.timestep % self.history_length] = sr
        self.ages += 1
        self.ages %= self.history_length
        self.timestep += 1

        surprises = self._get_surprise(events)
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
        self._reset()

    def _reset(self):
        self.srs = np.ones((self.history_length, self.srs_size))
        self.timestep = 0
        self.ages = np.arange(self.history_length)[::-1]
        self.surprises = np.zeros(self.history_length)

    def _get_surprise(self, events):
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


class PredictionsStackSurprise(BaseMetric):
    def __init__(self, name, att, logger, runner,
                 update_step, log_step, update_period, log_period,
                 prediction_steps=5, normalize=True, mode='categorical'):
        super().__init__(logger, runner, update_step, log_step, update_period, log_period)

        self.name = name
        self.att = att
        self.prediction_steps = prediction_steps
        self.normalize = normalize
        self.mode = mode
        self.predictions = []
        self.surprises = [list() for _ in range(self.prediction_steps)]

        self.logger.define_metric(self.name, step_metric=self.log_step)

    def update(self):
        value = self.get_attr(self.att)
        predictions = value['predictions']
        events = value['events']

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
        self._reset()

    def _reset(self):
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


class Histogram(BaseMetric):
    def __init__(
            self, name, att, normalized,
            logger, runner,
            update_step, log_step, update_period, log_period
    ):
        super().__init__(logger, runner, update_step, log_step, update_period, log_period)
        self.name = name
        self.normalized = normalized
        self.att_to_log = att

        self.hist = None
        self.counts = None

    def update(self):
        value, counts = self.get_attr(self.att_to_log)

        if self.hist is None:
            self.hist = np.zeros_like(value)

        self.hist += value

        if self.normalized:
            if self.counts is None:
                self.counts = np.zeros_like(counts)

            self.counts += counts

    def log(self, step):
        from matplotlib import pyplot as plt
        if self.normalized:
            hist = np.divide(
                        self.hist, self.counts,
                        where=self.counts > 0,
                        out=np.full_like(self.hist, fill_value=np.nan)
                    )
        else:
            hist = self.hist

        plt.figure()
        self.logger.log(
            {
                self.name: wandb.Image(sns.heatmap(hist)),
                self.log_step: step
            }
        )
        plt.close('all')
        self._reset()

    def _reset(self):
        self.hist = np.zeros_like(self.hist)
        if self.normalized:
            self.counts = np.zeros_like(self.hist)


class SFDiff(BaseMetric):
    def __init__(
            self, name, att, state_att,
            difference_mode: Literal['dkl', 'mse'],
            normalization_mode: Optional[Literal['bernoulli', 'categorical']],
            base_sf: Literal['uniform', 'load'],
            base_sf_path: Optional[str],
            logger, runner,
            update_step, log_step, update_period, log_period
    ):
        super().__init__(logger, runner, update_step, log_step, update_period, log_period)
        self.name = name
        self.att_to_log = att
        self.state_att = state_att

        if base_sf == 'uniform':
            self.base_sf = None
        elif base_sf == 'load':
            # (n_true_states, sf_size)
            self.base_sf = np.load(base_sf_path)
        else:
            raise ValueError(f'No such baseline: "{base_sf}"!')

        self.difference_mode = difference_mode
        self.normalization_mode = normalization_mode

        self.values = list()

    def update(self):
        # sf: (n_vars, n_states)
        # value: (n_vars,)
        sf = self.get_attr(self.att_to_log)
        true_state = self.get_attr(self.state_att)

        if self.base_sf is None:
            base_sf = np.ones_like(sf)
        else:
            base_sf = self.base_sf[true_state].reshape(sf.shape)

        if self.normalization_mode == 'bernoulli':
            base_sf /= base_sf.max()
            sf /= sf.max()
        elif self.normalization_mode == 'categorical':
            base_sf = normalize(base_sf)
            sf = normalize(sf)

        if self.difference_mode == 'mse':
            value = np.mean(np.power(base_sf - sf, 2), axis=-1).flatten()
        elif self.difference_mode == 'dkl':
            value = np.array([np.sum(rel_entr(sf[i], base_sf[i])) for i in range(sf.shape[0])])
        else:
            raise ValueError(f'Unknown difference mode: {self.difference_mode}!')

        self.values.append(value)

    def log(self, step):
        if len(self.values) == 0:
            return

        values = np.array(self.values).mean(axis=0)
        average = np.mean(values)

        log_dict = {
                f"{self.name}_feature{i}": values[i] for i in range(len(values))
            }
        if len(values) > 1:
            log_dict[f"{self.name}_average"] = average

        log_dict[self.log_step] = step

        self.logger.log(
            log_dict
        )

        self.values.clear()


class SOMClusters(BaseMetric):
    def __init__(
            self, name, att, label_att,
            logger, runner,
            update_step, log_step, update_period, log_period,
            quality_att=None,
            quality_thresh=0.5,
            size=100, iterations=1000, sigma=0.1, learning_rate=0.1, seed=None, init='random',
            font_size=16
    ):
        super().__init__(logger, runner, update_step, log_step, update_period, log_period)
        self.name = name
        self.att_to_log = att
        self.label_att = label_att
        self.quality_att = quality_att
        self.quality_thresh = quality_thresh

        self.size = size
        self.iterations = iterations
        self.sigma = sigma
        self.learning_rate = learning_rate
        self.init = init
        self.font_size = font_size
        assert self.init in {'random', 'pca'}
        self.seed = seed

        self.patterns = list()
        self.labels = list()

    def update(self):
        pattern = self.get_attr(self.att_to_log)
        label = self.get_attr(self.label_att)

        if self.quality_att is not None:
            quality = self.get_attr(self.quality_att)
        else:
            quality = self.quality_thresh

        if (pattern is not None) and (quality >= self.quality_thresh):
            self.patterns.append(normalize(pattern).flatten())
            self.labels.append(label)

    def log(self, step):
        if len(self.patterns) <= 1:
            return

        import matplotlib.pyplot as plt

        self.patterns = np.array(self.patterns)
        self.labels = np.array(self.labels)
        classes = np.unique(self.labels)
        log_dict = dict()

        dim = int(np.sqrt(self.size))
        som = minisom.MiniSom(
            dim, dim,
            self.patterns.shape[-1],
            sigma=self.sigma,
            learning_rate=self.learning_rate,
            random_seed=self.seed,
            activation_distance=self.dkl
        )

        if self.init == 'random':
            som.random_weights_init(self.patterns)
        elif self.init == 'pca':
            som.pca_weights_init(self.patterns)

        som.train(self.patterns, self.iterations)

        activation_map = np.zeros((dim, dim, classes.size))
        fig = plt.figure(figsize=(8, 8))
        plt.imshow(som.distance_map(), cmap='Greys', alpha=0.5)
        plt.colorbar()

        for p, cls in zip(self.patterns, self.labels):
            activation_map[:, :, np.flatnonzero(classes == cls)[0]] -= som.activate(p)

            cell = som.winner(p)
            plt.text(
                cell[0] - 0.5,
                cell[1] + 0.5,
                str(cls),
                color=plt.cm.rainbow(cls / classes.size),
                alpha=0.1,
                fontdict={'weight': 'bold', 'size': self.font_size}
            )
        # plt.axis([0, som.get_weights().shape[0], 0, som.get_weights().shape[1]])

        log_dict.update(
            {
                f'{self.name}/clusters': wandb.Image(fig),
            }
        )
        plt.close('all')

        # normalize activation map
        activation_map /= self.patterns.shape[0]
        activation_map /= activation_map.sum(axis=-1).reshape((dim, dim, 1))
        # generate colormap
        colors = [plt.cm.rainbow(c / classes.size)[:-1] for c in range(classes.size)]
        color_map = (np.dot(activation_map.reshape((-1, classes.size)), colors) * 255)
        color_map = color_map.reshape((dim, dim, 3))

        for i in range(classes.size):
            log_dict.update(
                {
                    f'{self.name}/activations/class_{classes[i]}': wandb.Image(
                        sns.heatmap(activation_map[:, :, i], cmap='viridis')
                    )
                }
            )
            plt.close('all')

        log_dict.update(
            {
                f'{self.name}/soft_clusters': wandb.Image(
                    plt.imshow(color_map.astype('uint8'))
                )
            }
        )

        log_dict[self.log_step] = step

        self.logger.log(log_dict)

        self.patterns = list()
        self.labels = list()

    @staticmethod
    def dkl(x, W):
        return np.sum(rel_entr(x, W), axis=-1)


class GridworldSR(BaseMetric):
    def __init__(
            self, name, sr_att, sf_att, repr_att, state_att, state_information_att,
            logger, runner,
            update_step, log_step, update_period, log_period,
            grid_shape, max_patterns, state_detection_threshold, activity_lr, lr,
            state_information_thresh=0.5,
            norm=False,
            preparing_period=100,
            log_dir='/tmp',
            log_fps=5,
            save=False,
            save_period=100
    ):
        super().__init__(logger, runner, update_step, log_step, update_period, log_period)
        self.name = name
        self.sr_att = sr_att
        self.sf_att = sf_att
        self.repr_att = repr_att
        self.state_att = state_att
        self.state_information_att = state_information_att

        self.state_repr_shape = self.sr_shape = self.get_attr(self.repr_att).shape
        self.sf_shape = self.get_attr(self.sf_att)[0].shape
        self.pattern_size = self.state_repr_shape[0]
        self.n_states = np.prod(grid_shape)
        self.grid_shape = grid_shape
        self.norm = norm
        self.state_information_thresh = state_information_thresh
        self.log_dir = log_dir
        self.log_fps = log_fps
        self.save = save
        self.save_period = save_period

        self.preparing = True
        self.preparing_period = preparing_period
        self.preparing_step = 0
        self.decoded_patterns = []
        self.decoded_state_dkl = []

        from hima.agents.succesor_representations.striatum import Striatum
        self.memory = Striatum(
            self.pattern_size,
            (self.n_states, np.prod(self.sf_shape)),
            n_areas=2,
            max_states=max_patterns,
            state_detection_threshold=state_detection_threshold,
            activity_lr=activity_lr,
            lr=lr
        )

        self.logger.define_metric(f'{self.name}/n_states', step_metric=self.log_step)
        self.logger.define_metric(f'{self.name}/av_states_per_pos', step_metric=self.log_step)
        self.logger.define_metric(f'{self.name}/decoded_state_dkl', step_metric=self.log_step)

    def update(self):
        estimated_state = self.get_attr(self.repr_att).flatten()
        state = self.get_attr(self.state_att)
        dense_state = sparse_to_dense(state, size=self.n_states)
        state_information = self.get_attr(self.state_information_att)

        sr, steps = self.get_attr(self.sr_att)

        if self.norm:
            sr = normalize(sr)
        sr = sr.flatten()

        learn = state_information >= self.state_information_thresh

        decoded_state = self.memory.predict(estimated_state, learn=learn)

        if learn:
            self.memory.update_weights(dense_state)

            sf, _ = self.get_attr(self.sf_att)
            if self.norm:
                sf = normalize(sf)

            self.memory.predict(estimated_state, area=1, learn=False)
            self.memory.update_weights(sf.flatten(), area=1)

        self.decoded_state_dkl.append(
            rel_entr(
                dense_state,
                normalize(decoded_state.reshape(1, -1)).flatten()
            ).sum()
        )

        if not self.preparing:
            steps_bar = self._scalar_to_bar(steps)
            state_information_bar = self._scalar_to_bar(state_information)

            decoded_pattern = self.memory.predict(sr, learn=False)

            decoded_pattern = decoded_pattern.reshape(self.grid_shape)
            decoded_state = decoded_state.reshape(self.grid_shape)

            max_image_value = max(decoded_pattern.max(), 1.0)

            decoded_pattern = np.column_stack(
                [
                    decoded_pattern,
                    steps_bar * max_image_value,
                    state_information_bar * max_image_value,
                    dense_state.reshape(self.grid_shape) * max_image_value,
                    decoded_state * max_image_value
                ]
            )
            self.decoded_patterns.append(
                decoded_pattern
            )

    def log(self, step):
        log_dict = {
            f'{self.name}/n_states': len(self.memory.states_in_use), self.log_step: step,
            f'{self.name}/decoded_state_dkl': np.mean(self.decoded_state_dkl)
        }

        self.decoded_state_dkl.clear()

        if len(self.memory.states_in_use) > 0:
            weights = self.memory.weights[0][self.memory.states_in_use]
            pcounts = np.sum(weights, axis=0)

            log_dict[f'{self.name}/av_states_per_pos'] = np.mean(pcounts)

            if self.preparing_step == self.preparing_period:
                log_dict[f'{self.name}/state_per_pos'] = wandb.Image(sns.heatmap(
                    pcounts.reshape(self.grid_shape)
                ))
                plt.close('all')

        if len(self.decoded_patterns) > 0:
            sr = np.array(self.decoded_patterns)
            gif_path = os.path.join(
                self.log_dir,
                f'{self.logger.name}_{self.name}_{step}.gif'
            )
            self._save_to_gif(gif_path, sr)
            log_dict[f'{self.name}/trajectory_sr'] = wandb.Video(gif_path)

            self.decoded_patterns.clear()
            self.preparing = True
        else:
            self.preparing_step += 1
            if (self.preparing_step % self.preparing_period) == 0:
                self.preparing = False

        self.logger.log(
            log_dict
        )

        if self.save and ((self.preparing_step % self.save_period) == 0):
            self._save_memory()
    
    def _scalar_to_bar(self, value):
        value_int = int(value * self.grid_shape[0])
        value_frac = value * self.grid_shape[0] - value_int

        value_bar = np.pad(
            np.ones(value_int),
            (0, self.grid_shape[0] - value_int),
            'constant',
            constant_values=0
        )
        if value_int < self.grid_shape[0]:
            value_bar[value_int] = value_frac

        return value_bar[::-1]
            
    def _save_to_gif(self, path, array):
        values = (
                (
                        array / (array.reshape(array.shape[0], -1).max(axis=-1).reshape(-1, 1, 1))
                 ) * 255
        ).astype(np.uint8)

        gif_path = os.path.join(
            self.log_dir,
            path
        )
        # use new v3 API
        imageio.v3.imwrite(
            # mode 'L': gray 8-bit ints; duration = 1000 / fps; loop == 0: infinitely
            gif_path, values, mode='L', duration=1000 / self.log_fps, loop=0
        )

    def _save_memory(self):
        path = os.path.join(
            self.log_dir,
            f'{self.logger.name}_{self.name}_memory_{self.preparing_step}.npz'
        )
        np.savez(path, states=self.memory.weights[0], sfs=self.memory.weights[1])


class GridworldStateImage(BaseMetric):
    def __init__(
            self, name, repr_att, state_att, normalized,
            n_states,
            logger, runner,
            update_step, log_step, update_period, log_period
    ):
        super().__init__(logger, runner, update_step, log_step, update_period, log_period)
        self.name = name
        self.normalized = normalized
        self.repr_att = repr_att
        self.state_att = state_att
        self.n_states = n_states
        self.repr_shape = self.get_attr(self.repr_att).shape

        self.representations = np.zeros((n_states, *self.repr_shape))
        self.state_visits = np.zeros(n_states)

    def update(self):
        representation = self.get_attr(self.repr_att)
        state = self.get_attr(self.state_att)

        self.representations[state] += representation
        self.state_visits[state] += 1

    def log(self, step):
        from matplotlib import pyplot as plt
        if self.normalized:
            hist = np.divide(
                        self.representations, self.state_visits.reshape(-1, 1, 1),
                        where=self.state_visits.reshape(-1, 1, 1) > 0,
                        out=np.full_like(self.representations, fill_value=np.nan)
                    )
        else:
            hist = self.representations

        log_dict = dict()
        for i in range(self.n_states):
            plt.figure()
            log_dict[f"{self.name}_state_{i}"] = wandb.Image(sns.heatmap(hist[i]))
            plt.close()

        log_dict[self.log_step] = step
        self.logger.log(
            log_dict
        )
        plt.close('all')
        self._reset()

    def _reset(self):
        self.representations = np.zeros((self.n_states, *self.repr_shape))
        self.state_visits = np.zeros(self.n_states)


class ArrayMetrics(BaseMetric):
    def __init__(self, metrics, logger, runner,
                 update_step, log_step, update_period, log_period,
                 log_dir='/tmp'):
        super().__init__(logger, runner, update_step, log_step, update_period, log_period)

        self.metrics = {metric: [] for metric in metrics}
        self.att_to_log = {
            metric: params['att']
            for metric, params in metrics.items()
        }
        self.logger = logger
        self.log_dir = log_dir

    def update(self):
        for name in self.metrics.keys():
            value = self.get_attr(self.att_to_log[name])
            self.metrics[name].append(value)

    def log(self, step):
        for metric, values in self.metrics.items():
            arr_path = os.path.join(
                self.log_dir,
                f'{self.logger.name}_{metric.split("/")[-1]}_{step}.npy'
            )
            np.save(arr_path, np.array(values))
        self._reset()

    def _reset(self):
        self.metrics = {metric: [] for metric in self.metrics.keys()}


class EClusterSimilarity(BaseMetric):
    agent: ECAgent
    similarities = {"cos": cosine_similarity, "euc": euclidian_sim}
    def __init__(
            self, name, metric,
            logger, runner,
            update_step, log_step, update_period, log_period,
    ):
        """
            Cluster pairwise similarity within one observational state.
        """
        super().__init__(logger, runner, update_step, log_step, update_period, log_period)
        self.name = name
        self.agent = self.runner.agent.agent
        self.sim_func = self.similarities[metric]

    def update(self):
        pass

    def log(self, step):
        log_dict = {
            self.log_step: step
        }

        for obs_state in self.agent.obs_to_clusters:
            clusters = self.agent.obs_to_clusters[obs_state]
            if len(clusters) == 0:
                continue

            traces = list()
            for c in clusters:
                trace = [
                    self.agent.state_to_memory_trace[s]
                    for s in self.agent.cluster_to_states[c]
                ]
                trace = np.vstack(trace).mean(axis=0)
                traces.append(trace)
            traces = np.vstack(traces)
            pws = self.sim_func(traces)
            log_dict[self.name + f'/obs_state_{obs_state}'] = wandb.Image(sns.heatmap(pws))
            plt.close()

        self.logger.log(log_dict)


class EClusterPurity(BaseMetric):
    def __init__(
            self, name, state_att,
            logger, runner,
            update_step, log_step, update_period, log_period,
    ):
        """
            Relative weight of cluster's the most common label
        """
        super().__init__(logger, runner, update_step, log_step, update_period, log_period)
        self.name = name
        self.state_att = state_att
        self.state_labels = dict()
        self.logger.define_metric(f'{self.name}', step_metric=self.log_step)
        agent = self.runner.agent.agent
        agent.state_labels = self.state_labels

    def update(self):
        state = self.get_attr(self.state_att)
        estimated_state = self.runner.agent.agent.state
        self.state_labels[estimated_state] = state

    def log(self, step):
        log_dict = {
            self.log_step: step
        }

        cluster_purity = list()
        clusters = self.runner.agent.agent.cluster_to_states
        for cluster_id in clusters:
            if cluster_id == -1:
                continue
            cluster = clusters[cluster_id]
            cluster_labels = np.array([self.state_labels[s] for s in cluster])
            labels, counts = np.unique(cluster_labels, return_counts=True)
            if len(labels) == 1:
                l = labels[0]
                # don't take into account landmarks
                # they always have purity == 1
                if self.runner.environment.environment.landmarks.flatten()[l]:
                    continue
            score = np.max(counts) / counts.sum()
            cluster_purity.append(score)

        log_dict[self.name] = np.median(np.array(cluster_purity))
        self.logger.log(log_dict)

class ECTrueClusterSim(BaseMetric):
    agent: ECAgent
    similarities = {"cos": cosine_similarity, "euc": euclidian_sim}
    mode: Literal['sampled', 'average', 'sampled_averaged']

    def __init__(
            self, name, state_att, metric, mode, iterations, embedding_type,
            logger, runner,
            update_step, log_step, update_period, log_period,
    ):
        """
            Relative weight of cluster's the most common label
        """
        super().__init__(logger, runner, update_step, log_step, update_period, log_period)
        self.name = name
        self.state_att = state_att
        self.true_clusters = dict()
        self.obs_to_clusters = dict()
        self.agent = self.runner.agent.agent
        self.sim_func = self.similarities[metric]
        self.mode = mode
        self.embedding_type = embedding_type
        self.iterations = iterations

    def update(self):
        state = self.get_attr(self.state_att)
        estimated_state = self.runner.agent.agent.state
        obs = estimated_state[0]

        if state not in self.true_clusters:
            self.true_clusters[state] = set()
        if obs not in self.obs_to_clusters:
            self.obs_to_clusters[obs] = set()

        self.true_clusters[state].add(estimated_state)
        self.obs_to_clusters[obs].add(state)

    def log(self, step):
        log_dict = {
            self.log_step: step
        }

        for obs_state in self.obs_to_clusters:
            clusters = self.obs_to_clusters[obs_state]
            if len(clusters) == 0:
                continue

            if self.mode == 'sampled':
                pws = self.sampled(clusters, self.iterations)
            elif self.mode == 'averaged':
                pws = self.averaged(clusters)
            else:
                pws = self.sampled_averaged(clusters)

            log_dict[self.name + f'/obs_state_{obs_state}'] = wandb.Image(sns.heatmap(pws))
            plt.close()

        self.logger.log(log_dict)

    def sampled(self, clusters, iterations: int = 100):
        """
            Computes sampled similarity between clusters.
        """
        av_pws = np.zeros((len(clusters), len(clusters)))
        for _ in range(iterations):
            traces_x = list()
            traces_y = list()
            for c in clusters:
                states = list(self.true_clusters[c])
                s_ids = np.random.choice(
                    np.arange(len(states)),
                    size=2
                )
                s = states[s_ids[0]]
                trace = self.embedding({s})
                traces_x.append(trace)
                s = states[s_ids[1]]
                trace = self.embedding({s})
                traces_y.append(trace)
            av_pws += self.sim_func(np.vstack(traces_x), np.vstack(traces_y))
        return av_pws / iterations

    def sampled_averaged(self, clusters, iterations: int = 100):
        """
            Computes sampled similarity between clusters.
        """
        embds = list()
        for c in clusters:
            embd = self.embedding(self.true_clusters[c])
            embds.append(embd)
        av_traces = np.vstack(embds)

        av_pws = np.zeros((len(clusters), len(clusters)))
        for _ in range(iterations):
            traces = list()
            for c in clusters:
                states = list(self.true_clusters[c])
                s_id = np.random.choice(
                    np.arange(len(states))
                )
                s = states[s_id]
                trace = self.embedding({s})
                traces.append(trace)
            av_pws += self.sim_func(av_traces, np.vstack(traces))
        return av_pws / iterations

    def averaged(self, clusters):
        """
            Average similarity of clusters.
        """
        embds = list()
        for c in clusters:
            embd = self.embedding(self.true_clusters[c])
            embds.append(embd)
        traces = np.vstack(embds)
        pws = self.sim_func(traces)
        return pws

    def embedding(self, states):
        embd = list()
        if "mt" in self.embedding_type:
            trace = [
                self.agent.state_to_memory_trace[s]
                for s in states
            ]
            trace = np.vstack(trace).mean(axis=0)
            embd.append(trace)
        if "sf" in self.embedding_type:
            sf, _, _ = self.agent.generate_sf(states, early_stop=False)
            embd.append(sf)

        return np.concatenate(embd)