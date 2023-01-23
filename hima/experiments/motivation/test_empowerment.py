#  Copyright (c) 2022 Autonomous Non-Profit Organization "Artificial Intelligence Research
#  Institute" (AIRI); Moscow Institute of Physics and Technology (National Research University).
#  All rights reserved.
#
#  Licensed under the AGPLv3 license. See LICENSE in the project root for license information.

import numpy as np
import matplotlib.pyplot as plt
import wandb
from copy import deepcopy
from itertools import product

from hima.envs.biogwlab.env import BioGwLabEnvironment
from hima.envs.biogwlab.environment import Environment
from hima.envs.biogwlab.module import EntityType
from hima.envs.biogwlab.utils.state_provider import GwAgentStateProvider
from hima.envs.env import unwrap

from htm.bindings.algorithms import SpatialPooler
from htm.bindings.sdr import SDR

from hima.common.sdr import SparseSdr
from hima.common.run.runner import Runner
from hima.common.config.base import TConfig
from hima.common.plot_utils import transform_fig_to_image

from hima.modules.empowerment import Empowerment
from hima.agents.motivation.simplified_hima import Agent


def xlogx(x):
    mask = x != 0
    y = x[mask]
    return np.sum(y * np.log2(y))


def plot_valued_map(value_map, title, vmin=None, vmax=None):
    fig = plt.figure(frameon=False)
    fig.set_size_inches(5, 5)
    fig.set_dpi(300)
    fig.suptitle(title, fontsize=14)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)

    ax.imshow(value_map, cmap='copper', vmin=vmin, vmax=vmax)
    h, w = value_map.shape
    for i in range(h):
        for j in range(w):
            if value_map.mask[i, j]:
                continue
            ax.text(
                j, i, f'{value_map[i, j]: .1f}',
                va='center', ha='center', fontsize=7, c='r'
            )
    img = transform_fig_to_image(fig)
    plt.close(fig)
    return img


class SDRMetrics:
    def __init__(self, sdr_size):
        self.sdr_frequency = {}
        self.sdr_bit_frequency = {}
        self.bit_frequency = np.zeros(sdr_size)

        self.total_sdrs = 0
        self.sdr_size = sdr_size
        self.n_activations = 0

    def add(self, key, value):
        if key in self.sdr_frequency.keys():
            self.sdr_frequency[key] += 1
        else:
            self.sdr_frequency[key] = 1
            self.sdr_bit_frequency[key] = np.zeros(self.sdr_size)

        self.sdr_bit_frequency[key][value] += 1
        self.bit_frequency[value] += 1
        self.total_sdrs += 1
        self.n_activations += len(value)

    @property
    def sparsity(self):
        return self.n_activations / (self.sdr_size * self.total_sdrs)

    @property
    def entropy_stability(self):
        stability = 0
        for key in self.sdr_bit_frequency.keys():
            p = self.sdr_bit_frequency[key] / self.sdr_frequency[key]
            e = - xlogx(p) - xlogx(1 - p)
            n = np.sum(self.sdr_bit_frequency[key] != 0)
            stability += e / n
        return stability / len(self.sdr_bit_frequency)

    @property
    def sdr_entropy(self):
        p = np.zeros(len(self.sdr_frequency))
        for ind, key in enumerate(self.sdr_frequency.keys()):
            p[ind] = self.sdr_frequency[key]
        p = p / np.sum(p)
        return - xlogx(p)

    @property
    def max_sdr_entropy(self):
        n = len(self.sdr_frequency)
        return np.log2(n)

    @property
    def rel_sdr_entropy(self):
        return self.sdr_entropy / self.max_sdr_entropy

    @property
    def bit_entropy(self):
        p = self.bit_frequency / self.total_sdrs
        return - xlogx(p) - xlogx(1 - p)

    @property
    def max_bit_entropy(self):
        return self.sdr_size

    @property
    def rel_bit_entropy(self):
        return self.bit_entropy / self.max_bit_entropy

    @property
    def max_sdr_per_full_entropy(self):
        s = self.sparsity
        e = - s * np.log2(s) - (1 - s) * np.log2(1 - s)
        return e

    @property
    def redundancy(self):
        e = self.bit_entropy
        E = self.sdr_entropy
        return (e - E) / E


class SPMetrics(SDRMetrics):
    def __init__(self, size):
        super().__init__(size)
        self.sdr_codes = {}
        self.pre_stability = 0

    def reset(self):
        self.n_activations = 0
        self.pre_stability = 0
        self.total_sdrs = 0
        self.bit_frequency = np.zeros(self.sdr_size)

    def add(self, key, value):
        if key in self.sdr_frequency.keys():
            self.pre_stability += len(np.intersect1d(
                value, self.sdr_codes[key])
            ) / len(self.sdr_codes[key])
            self.sdr_codes[key] = np.copy(value)
            self.sdr_frequency[key] += 1
        else:
            self.sdr_frequency[key] = 1
            self.sdr_bit_frequency[key] = np.zeros(self.sdr_size)
            self.sdr_codes[key] = np.copy(value)

        self.sdr_bit_frequency[key][value] += 1
        self.bit_frequency[value] += 1
        self.total_sdrs += 1
        self.n_activations += len(value)

    @property
    def stability(self):
        return self.pre_stability / self.total_sdrs


class ExactEmpowerment:
    def __init__(self, environment_config: dict):
        config = deepcopy(environment_config)
        config['terminate']['early_stop'] = False
        config['terminate']['episode_max_steps'] = np.inf
        self.environment = unwrap(BioGwLabEnvironment(**config))

    def eval_state(self, position: tuple[int, int], horizon: int):
        data = set()

        for actions in product(range(self.environment.n_actions), repeat=horizon):
            self.environment.agent.position = position
            for a in actions:
                self.environment.act(a)
            data.add(self.environment.agent.position)
        num_states = len(data)
        return np.log2(num_states)


class GwEmpowermentTest(Runner):
    def __init__(self, config: TConfig, **kwargs):
        super().__init__(config, **config)

        self.seed = config['seed']
        self._rng = np.random.default_rng(self.seed)
        self.strategy = config['strategy']

        self.n_episodes = config['n_episodes']
        self.evaluate_step = config['evaluate_step']

        if self.strategy == 'uniform':
            config['environment']['terminate']['early_stop'] = False
            config['environment']['terminate']['episode_max_steps'] = np.inf
        self.environment: Environment = unwrap(BioGwLabEnvironment(**config['environment']))
        map_image = self.environment.callmethod('render_rgb')
        if isinstance(map_image, list):
            map_image = map_image[0]
        if self.logger:
            map_image = wandb.Image(map_image)
            self.logger.log({'map': map_image}, step=0)
        else:
            plt.imshow(map_image)
            plt.show()
        print(f"Environment sdr size: {self.environment.output_sdr_size}")

        state_space_size = config['state_space_size']
        self.sp = SpatialPooler(
            seed=self.seed, inputDimensions=[self.environment.output_sdr_size],
            columnDimensions=[state_space_size], **config['sp'])
        self.sp_input = SDR(self.environment.output_sdr_size)
        self.sp_output = SDR(state_space_size)

        self.emp = Empowerment(
            seed=self.seed, encode_size=state_space_size,
            sparsity=self.sp.getLocalAreaDensity(), **config['emp']
        )
        self.horizon = config['horizon']
        self.log_emp_seq = config['log_empowerment_sequence']
        self.exact_emp = ExactEmpowerment(config['environment'])
        if self.log_emp_seq:
            self.exact_emp_map = []
            for i in range(self.horizon):
                self.exact_emp_map.append(self.create_exact_empowerment_map(i+1))
        else:
            self.exact_emp_map = self.create_exact_empowerment_map(self.horizon)

        if self.strategy == 'agent':
            self.agent = Agent(
                self.seed, state_space_size, self.environment.n_actions, config['agent_config']
            )

        self.prev_state = None
        self.episode = 0
        self.steps = 0
        self.metrics = SDRMetrics(self.environment.output_sdr_size)
        self.state_metrics = SPMetrics(state_space_size)

    def get_all_observations(self) -> dict[tuple[int, int], SparseSdr]:
        height, width = self.environment.shape
        obstacle_mask = self.environment.aggregated_mask[EntityType.Obstacle]
        position_provider = GwAgentStateProvider(self.environment)
        encoding_scheme = {}

        for i in range(height):
            for j in range(width):
                if obstacle_mask[i, j]:
                    continue
                position = i, j
                position_provider.overwrite(position)
                obs = self.environment.render()
                encoding_scheme[position] = obs

        position_provider.restore()
        return encoding_scheme

    def get_masked_obstacles_map(self) -> np.ma.MaskedArray:
        ob_map = np.zeros(self.environment.shape)
        obstacle_mask = self.environment.aggregated_mask[EntityType.Obstacle]
        ob_map = np.ma.masked_where(obstacle_mask, ob_map, False)
        return ob_map

    def create_exact_empowerment_map(self, horizon: int) -> np.ma.MaskedArray:
        data = self.get_masked_obstacles_map()
        height, width = data.shape
        for i in range(height):
            for j in range(width):
                if data.mask[i, j]:
                    continue
                data[i, j] = self.exact_emp.eval_state((i, j), horizon)
        return data

    def log_metrics(self):
        observations = self.get_all_observations()
        for key in observations.keys():
            self.sp_input.sparse = observations[key]
            self.sp.compute(self.sp_input, learn=False, output=self.sp_output)
            self.state_metrics.add(key, self.sp_output.sparse)

        anomaly = np.mean(self.emp.anomalies[-self.steps:])
        iou = np.mean(self.emp.IoU[-self.steps:])
        self.log_empowerment()
        self.logger.log(
            {
                'anomaly': anomaly,
                'IoU': iou,
                'steps': self.steps,
                'obs/sdr_entropy': self.metrics.rel_sdr_entropy,
                'obs/bit_entropy': self.metrics.rel_bit_entropy,
                'obs/sparsity': self.metrics.sparsity,
                'obs/entropy_stability': self.metrics.entropy_stability,
                'obs/max_sdr_per_full_entropy': self.metrics.max_sdr_per_full_entropy,
                'state/sdr_entropy': self.state_metrics.rel_sdr_entropy,
                'state/bit_entropy': self.state_metrics.rel_bit_entropy,
                'state/sparsity': self.state_metrics.sparsity,
                'state/stability': self.state_metrics.stability,
                'state/entropy_stability': self.state_metrics.entropy_stability,
                'state/max_sdr_per_full_entropy': self.state_metrics.max_sdr_per_full_entropy,
            }, step=self.episode
        )
        if self.strategy == 'agent':
            self.logger.log({'map/anomaly': wandb.Image(plot_valued_map(
                    self.anomaly_map/self.visit_map, 'Anomaly', vmin=0, vmax=1
                )),}, step=self.episode)
        self.state_metrics.reset()

    def log_empowerment(self):
        if self.log_emp_seq:
            self.log_empowerment_sequence()
        else:
            self.log_empowerment_value()

    def log_empowerment_sequence(self):
        observations = self.get_all_observations()
        empowerment_maps = []
        for _ in range(self.horizon):
            empowerment_maps.append(self.get_masked_obstacles_map())

        for key in observations.keys():
            self.sp_input.sparse = observations[key]
            self.sp.compute(self.sp_input, learn=False, output=self.sp_output)
            for i in range(self.horizon):
                empowerment_maps[i][key] = self.emp.eval_state(self.sp_output.sparse, i + 1)

        fig, ax = plt.subplots(2, self.horizon, figsize=(3 * self.horizon, 6))
        fig.set_dpi(300)

        for i in range(self.horizon):
            ax[0][i].set_axis_off()
            ax[0][i].set_title(f'$\hat \epsilon_{i+1}$', fontsize=14)
            ax[0][i].imshow(empowerment_maps[i], cmap='copper')
            for key in observations.keys():
                ax[0][i].text(
                    key[1], key[0], f'{empowerment_maps[i][key]: .1f}',
                    va='center', ha='center', fontsize=7, c='r'
                )

        for i in range(self.horizon):
            ax[1][i].set_axis_off()
            ax[1][i].set_title(f'$\epsilon_{i+1}$', fontsize=14)
            ax[1][i].imshow(self.exact_emp_map[i], cmap='copper')
            for key in observations.keys():
                ax[1][i].text(
                    key[1], key[0], f'{self.exact_emp_map[i][key]: .1f}',
                    va='center', ha='center', fontsize=7, c='r'
                )
        plt.tight_layout()
        img = transform_fig_to_image(fig)
        plt.close(fig)
        self.logger.log({
            'map/emp_sequence': wandb.Image(img)
        },  step=self.episode)

        for i in range(self.horizon):
            dif = self.exact_emp_map[i] - empowerment_maps[i]
            self.logger.log(
                {
                    f'empowerment/mre_{i+1}': np.abs(dif / self.exact_emp_map[i]).mean(),
                    f'empowerment/min_error_{i+1}': dif.min()
                }, step=self.episode
            )

    def log_empowerment_value(self):
        observations = self.get_all_observations()
        empowerment_map = self.get_masked_obstacles_map()
        sp_outs = {}
        for key in observations.keys():
            self.sp_input.sparse = observations[key]
            self.sp.compute(self.sp_input, learn=False, output=self.sp_output)
            sp_outs[key] = self.sp_output.sparse.copy()
            empowerment_map[key] = self.emp.eval_state(self.sp_output.sparse, self.horizon)
        self.log_prediction(sp_outs)

        vmin = np.min([empowerment_map.min(), self.exact_emp_map.min()])
        vmax = np.max([empowerment_map.max(), self.exact_emp_map.max()])
        dif = self.exact_emp_map - empowerment_map
        self.logger.log({
            'map/empowerment': wandb.Image(plot_valued_map(empowerment_map, '$\hat \epsilon_4$')),
            'map/exact_emp': wandb.Image(plot_valued_map(self.exact_emp_map, '$\epsilon_4$')),
            'map/dif_emp': wandb.Image(plot_valued_map(np.abs(dif)/vmax, 'MRAE', 0, 1)),
            'empowerment/mrae': (np.abs(dif)/self.exact_emp_map).mean(),
            'empowerment/min_error': dif.min()
        }, step=self.episode)

    def log_prediction(self, states: dict[tuple, SparseSdr]):
        prediction_map = self.get_masked_obstacles_map()
        poses = np.sum(~prediction_map.mask)
        c = self._rng.integers(0, poses)
        i, j = np.nonzero(~prediction_map.mask)
        pose = (i[c], j[c])

        fig, ax = plt.subplots(1, self.horizon, figsize=(3*self.horizon, 3))
        fig.set_dpi(300)
        prediction = states[pose].copy()
        for i in range(self.horizon):
            prediction = self.emp.predict(prediction)
            for key, value in states.items():
                sim = len(np.intersect1d(prediction, value)) / len(value)
                prediction_map[key] = sim

            ax[i].set_axis_off()
            e_emp = self.exact_emp.eval_state(pose, i+1)
            a_emp = self.emp.eval_state(states[pose], i+1)
            ax[i].set_title(f'E: {2**e_emp: .0f}; A: {2**a_emp: .1f}.', fontsize=8)
            ax[i].imshow(prediction_map, vmin=0, vmax=1, cmap='copper')
            rect = plt.Rectangle((pose[1]-0.5, pose[0]-0.5), 1, 1, ec='r', fill=False)
            ax[i].add_patch(rect)
            for key in states.keys():
                ax[i].text(
                    key[1], key[0], f'{prediction_map[key]: .1f}',
                    va='center', ha='center', fontsize=7, c='r'
                )

        plt.tight_layout()
        img = transform_fig_to_image(fig)
        plt.close(fig)
        self.logger.log({
            'map/prediction': wandb.Image(img)
        },  step=self.episode)

    def run(self):
        if self.strategy == 'uniform':
            self.run_uniform()
        elif self.strategy == 'agent':
            self.run_agent()
        else:
            raise ValueError(f'Undefined strategy type: {self.strategy}')

    def run_uniform(self):
        height, width = self.environment.shape
        obstacle_mask = self.environment.aggregated_mask[EntityType.Obstacle]

        for self.episode in range(1, self.n_episodes + 1):
            for i in range(height):
                for j in range(width):
                    if obstacle_mask[i, j]:
                        continue
                    position = i, j
                    for a in range(self.environment.n_actions):
                        self.environment.agent.position = position
                        self.sp_input.sparse = self.environment.render()
                        self.metrics.add(self.environment.agent.position, self.sp_input.sparse)
                        self.sp.compute(self.sp_input, learn=True, output=self.sp_output)
                        sdr_0 = self.sp_output.sparse.copy()

                        self.environment.act(a)
                        self.sp_input.sparse = self.environment.render()
                        self.metrics.add(self.environment.agent.position, self.sp_input.sparse)
                        self.sp.compute(self.sp_input, learn=True, output=self.sp_output)
                        sdr_1 = self.sp_output.sparse.copy()
                        self.emp.learn(sdr_0, sdr_1)
                        self.steps += 1
            self.log_metrics()
            self.steps = 0

    def run_agent(self):
        self.episode = 0
        self.steps = 0

        self.anomaly_map = np.ma.zeros(self.environment.shape)
        self.anomaly_map[:, :] = np.ma.masked
        self.visit_map = np.ma.zeros(self.environment.shape)
        self.visit_map[:, :] = np.ma.masked

        while True:

            reward, obs, is_first = self.environment.observe()
            self.metrics.add(self.environment.agent.position, obs)
            self.sp_input.sparse = obs
            self.sp.compute(self.sp_input, learn=True, output=self.sp_output)
            if is_first:
                self.prev_state = np.copy(self.sp_output.sparse)
            else:
                self.emp.learn(self.prev_state, self.sp_output.sparse)
                self.prev_state = np.copy(self.sp_output.sparse)
                if self.visit_map.mask[self.environment.agent.position]:
                    self.anomaly_map[self.environment.agent.position] = self.emp.anomalies[-1]
                    self.visit_map[self.environment.agent.position] = 1
                else:
                    self.anomaly_map[self.environment.agent.position] += self.emp.anomalies[-1]
                    self.visit_map[self.environment.agent.position] += 1

            if is_first:
                if self.episode != 0 and self.logger and self.episode % self.evaluate_step == 0:
                    self.log_metrics()
                self.episode += 1
                self.steps = 0
                if self.episode > self.n_episodes:
                    break
            else:
                self.steps += 1

            # action = self._rng.integers(0, self.environment.n_actions)
            action = self.agent.act(self.sp_output.sparse, reward, is_first)
            self.environment.act(action)
