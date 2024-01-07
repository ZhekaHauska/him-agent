#  Copyright (c) 2023 Autonomous Non-Profit Organization "Artificial Intelligence Research
#  Institute" (AIRI); Moscow Institute of Physics and Technology (National Research University).
#  All rights reserved.
#
#  Licensed under the AGPLv3 license. See LICENSE in the project root for license information.

import os
import numpy as np

from hima.common.lazy_imports import lazy_import
from typing import Dict, Literal, Optional
from hima.modules.belief.utils import normalize
from scipy.special import rel_entr

wandb = lazy_import('wandb')
sns = lazy_import('seaborn')
imageio = lazy_import('imageio')


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

        self.logger.define_metric(self.name, step_metric=self.log_step)

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
            value = np.array([rel_entr(sf[i], base_sf[i]) for i in range(sf.shape[0])])
        else:
            raise ValueError(f'Unknown difference mode: {self.difference_mode}!')

        self.values.append(value)

    def log(self, step):
        values = np.array(self.values).mean(axis=0)
        average = np.mean(values)

        log_dict = {
                f"{self.name}_feature{i}": values[i] for i in range(len(values))
            }
        log_dict[f"{self.name}_average"] = average
        log_dict[self.log_step] = step

        self.logger.log(
            log_dict
        )

        self.values.clear()
