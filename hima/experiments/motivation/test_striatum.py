#  Copyright (c) 2022 Autonomous Non-Profit Organization "Artificial Intelligence Research
#  Institute" (AIRI); Moscow Institute of Physics and Technology (National Research University).
#  All rights reserved.
#
#  Licensed under the AGPLv3 license. See LICENSE in the project root for license information.

import numpy as np
import matplotlib.pyplot as plt
import wandb

from hima.envs.biogwlab.env import BioGwLabEnvironment
from hima.envs.biogwlab.environment import Environment
from hima.envs.env import unwrap

from hima.common.run_utils import Runner
from hima.common.config_utils import TConfig
from hima.common.plot_utils import transform_fig_to_image
from hima.agents.motivation.agent import Agent
from collections import Counter
import matplotlib.colors as mcolors
from hima.common.sdr import SparseSdr


class VizSDR:
    def __init__(self, num_vars: int, sdr_size: int):
        self.data = [Counter() for _ in range(num_vars)]
        self.total_inputs = 0
        self.num_vars = num_vars
        self.sdr_size = sdr_size

    def update(self, sdr: SparseSdr, sdr_var: int):
        self.data[sdr_var].update(sdr)
        self.total_inputs += 1

    def plot_hist(self):
        fig, ax = plt.subplots()
        fig.set_dpi(500)

        columns = np.arange(self.sdr_size)
        colors = [*mcolors.BASE_COLORS.keys()]
        if self.num_vars > len(colors):
            raise ValueError('Too many sdr variations')
        y_offset = np.zeros(self.sdr_size)
        data = np.zeros(self.sdr_size)
        norm = self.total_inputs
        for var in range(self.num_vars):
            index = np.array([*self.data[var].keys()], dtype=int)
            data[index] = [self.data[var][s] / norm for s in index]
            ax.bar(columns, data, 1, bottom=y_offset, color=colors[var])
            y_offset = y_offset + data
            data.fill(0)

        img = transform_fig_to_image(fig)
        plt.close(fig)
        return img


class GwStriatumTest(Runner):
    def __init__(self, config: TConfig, **kwargs):
        super().__init__(config, **config)

        self.seed = config['seed']
        self.n_episodes = config['n_episodes']
        self._rng = np.random.default_rng(self.seed)
        self.environment: Environment = unwrap(BioGwLabEnvironment(**config['environment']))

        self.task = -1
        self.task_queue = config['task_queue']
        self.tasks = config['tasks']
        self.change_step = config['change_step']
        self.change_task()

        self.motiv_size = config['motiv_size']
        self.motiv_dim = config['motiv_dim']
        self.agent = Agent(
            self.environment.output_sdr_size, self.environment.n_actions,
            self.motiv_dim, config['agent_config']
        )

        self.episode = 0
        self.steps = 0
        self.total_steps = 0
        self.q_map = np.ma.zeros((*self.environment.shape, self.environment.n_actions))
        self.q_map[:, :] = np.ma.masked
        self.v_map = np.ma.zeros(self.environment.shape)
        self.v_map[:, :] = np.ma.masked
        self.preactivation_stat = VizSDR(len(self.tasks), self.agent.striatum.action_size)
        self.motiv_stat = VizSDR(len(self.tasks), self.agent.motiv_dim)

    def run(self):
        while True:
            t = self.task_queue[self.task]
            motiv = np.arange(t * self.motiv_size, (t + 1) * self.motiv_size)
            _, obs, _ = self.environment.observe()
            a = self.agent.act(obs, motiv)
            self.preactivation_stat.update(self.agent.striatum.preactivation_field, t)
            self.motiv_stat.update(motiv, t)
            self.v_map[self.environment.agent.position] = self.agent.get_value()
            self.q_map[
                self.environment.agent.position[0],
                self.environment.agent.position[1],
                :
            ] = self.agent.get_probs()

            self.environment.act(a)
            self.steps += 1
            self.total_steps += 1
            reward, _, _ = self.environment.observe()
            self.agent.update(reward)

            if self.environment.is_terminal():
                self.environment.act(a)
                self.agent.reset()
                self.episode += 1
                self.draw_map()
                if self.episode % self.change_step == 0:
                    self.change_task()
                if self.logger:
                    # self.log_metrics()
                    self.log_sdr_statistics()
                    self.logger.log({
                        'steps': self.steps,
                        'total_steps': self.total_steps 
                    }, step=self.episode)
                self.steps = 0
                if self.episode == self.n_episodes:
                    break

    def draw_map(self):
        map_image = self.environment.callmethod('render_rgb')
        if isinstance(map_image, list):
            map_image = map_image[0]
        if self.logger:
            map_image = wandb.Image(map_image)
            self.logger.log({'map': map_image}, step=self.episode)

    def change_task(self):
        if self.task < len(self.task_queue) - 1:
            self.task += 1
        t = self.tasks[self.task_queue[self.task]]
        self.environment.modules['food'].generator.positions = [t]
        self.environment.callmethod('reset')

    def log_metrics(self):
        base_vectors = np.array([
            [1, 0],
            [0, 1],
            [-1, 0],
            [0, -1]
        ]) / 2

        fig = plt.figure(frameon=False)
        fig.set_size_inches(10, 10)
        fig.set_dpi(300)
        ax = plt.Axes(fig, [0., 0., 1., 1.])
        ax.set_axis_off()
        fig.add_axes(ax)
        ax.imshow(self.q_map.max(axis=-1))

        h, w, _ = self.q_map.shape
        for i in range(h):
            for j in range(w):
                y, x = i, j
                values = self.q_map[i, j]
                m = np.argmax(values)
                d = values.reshape((-1, 1)) * base_vectors
                for dx, dy in d:
                    if dx > 1e-2 or dy > 1e-2:
                        ax.arrow(x, y, dx, dy, width=0.05, color='red', length_includes_head=True)
                ax.arrow(x, y, *(d[m]), width=0.05, color='black', length_includes_head=True)

        img = transform_fig_to_image(fig)
        plt.close(fig)
        self.logger.log(
            {
                'maps/q_map': wandb.Image(img),
                'maps/v_map': wandb.Image(plt.imshow(self.v_map))
            }, step=self.episode
        )

    def log_sdr_statistics(self):
        img = self.preactivation_stat.plot_hist()
        motiv_img = self.motiv_stat.plot_hist()
        wandb.log({
            'preactivation_hist': wandb.Image(img),
            'motiv_hist': wandb.Image(motiv_img),
            'motiv_weights': wandb.Image(
                plt.imshow(
                    self.agent.striatum.motiv_weights.dopa_weights.synapse_values.values.reshape((
                        self.agent.striatum.motiv_weights.output_size,
                        self.agent.striatum.motiv_weights.potential_size
                    ))
                )
            )
        }, step=self.episode)

