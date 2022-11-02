#  Copyright (c) 2022 Autonomous Non-Profit Organization "Artificial Intelligence Research
#  Institute" (AIRI); Moscow Institute of Physics and Technology (National Research University).
#  All rights reserved.
#
#  Licensed under the AGPLv3 license. See LICENSE in the project root for license information.


import os.path
import pathlib
import random
from functools import partial

import imageio
import matplotlib.pyplot as plt
import wandb
import numpy as np

from hima.common.run.runner import Runner
from hima.common.plot_utils import transform_fig_to_image
from hima.common.config import TConfig
from hima.envs.biogwlab.env import BioGwLabEnvironment
from hima.envs.env import unwrap
from hima.envs.biogwlab.environment import Environment
from hima.envs.biogwlab.module import EntityType
from hima.common.sdr import SparseSdr
from hima.envs.biogwlab.utils.state_provider import GwAgentStateProvider


def draw_masked_observation(concatenated_observation: np.ndarray, sdr_sizes: list[tuple[str, int]]):
    n_sdrs = len(sdr_sizes)
    fig, ax = plt.subplots(1, n_sdrs, figsize=(n_sdrs, 2))
    fig.set_dpi(200)
    start = 0
    obs = concatenated_observation
    vmin = np.min(obs)
    vmax = np.max(obs)
    for i, (name, sdr_size) in enumerate(sdr_sizes):
        # all sdrs should be initially squares
        size = int(np.sqrt(sdr_size))
        finish = start + sdr_size
        ax[i].set_axis_off()
        ax[i].imshow(obs[start: finish].reshape((size, size)), vmin=vmin, vmax=vmax)
        ax[i].set_title(name)
        start = finish
    fig.suptitle(f'Max: {vmax: .2e}; Min: {vmin: .2e}')
    plt.tight_layout()
    img = transform_fig_to_image(fig)
    plt.close(fig)
    return img


class GwExhaustibleResource(Runner):
    def __init__(self, config: TConfig, **kwargs):
        super().__init__(config, **config)

        self.gif_schedule = config['gif_schedule']
        self.animation_fps = config['animation_fps']

        self.scenario = config['scenario']
        self._rng = random.Random(self.scenario['seed'])
        self.n_episodes = self.scenario['n_episodes']
        self.env_config = config['environment']
        self.env_config['seed'] = self.scenario['seed']
        self.environment: Environment = unwrap(BioGwLabEnvironment(**self.env_config))
        self.agent = resolve_agent(
            config['agent'],
            obs_dim=self.environment.output_sdr_size,
            action_dim=self.environment.n_actions,
            config=config['agent_config']
        )

        # episode
        self.episode = 0
        self.steps = 0
        self.total_reward = 0

        # log
        self.animation = False

        # goal
        self.goal = 1
        self.steps_per_goal = 0
        self.goal_reached = False

        # task
        self.task = 1
        self.steps_per_task = 0
        self.task_complete = False
        self.map_change_indicator = 0
        self.goals_per_task = self.scenario['goals_per_task']

        # level
        self.level = 1
        self.level_complete = False
        self.tasks_per_level = self.scenario['tasks_per_level']

        # experiment
        self.steps_total = 0
        self.steps_cumulative = 0
        self.all_steps = 0
        self.early_stop = False

        self.path_to_store_logs = config['path_to_store_logs']
        pathlib.Path(self.path_to_store_logs).mkdir(parents=True, exist_ok=True)
        self.get_path = partial(os.path.join, self.path_to_store_logs)

    def run(self):
        log_enabled = self.logger is not None

        if log_enabled:
            self.define_logging_metrics()

        self.reset_task(**self.scenario)
        if log_enabled:
            self.draw_map()

        need_reset = False
        while True:
            reward, obs, is_first = self.environment.observe()
            if self.environment.is_terminal() and self.environment.items_collected > 0:
                # detect that the goal is reached
                self.goal_reached = True
                if self.goal % self.goals_per_task == 0:
                    self.task_complete = True
                    need_reset = True
                    if self.task % self.tasks_per_level == 0:
                        self.level_complete = True

            if is_first:
                self.finish_episode()

                if self.finish_condition():
                    break

                self.init_new_episode()
                if need_reset:
                    need_reset = False
                    self.reset_task(**self.scenario)
                    if log_enabled:
                        self.draw_map()
                    reward, obs, is_first = self.environment.observe()

            self.total_reward += reward
            if log_enabled and self.animation:
                self.draw_animation_frame()

            current_action = self.agent.act(obs, reward, is_first)
            self.environment.act(current_action)
            self.steps += 1

        if log_enabled:
            self.log_goal_complete()
            self.log_task_complete()
            self.logger.log({"total_steps": self.steps_total}, step=self.episode)

    def define_logging_metrics(self):
        self.logger.define_metric("task")
        self.logger.define_metric("main_metrics/steps_per_task", step_metric="task")
        self.logger.define_metric("main_metrics/t_*", step_metric="task")

        self.logger.define_metric("goal")
        self.logger.define_metric("main_metrics/g_*", step_metric="goal")

    def finish_episode(self):
        self.steps_per_goal += self.steps
        self.steps_per_task += self.steps
        self.steps_total += self.steps
        self.all_steps += self.steps
        self.steps_cumulative += self.steps

        self.map_change_indicator = 1 if self.task_complete else 0

        log_enabled = self.logger is not None
        if log_enabled and self.episode > 0:
            self.log_episode_complete()

            if self.animation:
                # log all saved frames for the finished episode
                self.animation = False
                self.log_gif_animation()

            if self.goal_reached:
                self.log_goal_complete()
            if self.task_complete:
                self.log_task_complete()

    def log_episode_complete(self):
        self.logger.log(
            {
                'main_metrics/steps': self.steps,
                'reward': self.total_reward,
                'episode': self.episode,
                'main_metrics/level': self.level,
                'main_metrics/total_terminals': self.goal,
                'main_metrics/steps_cumulative': self.steps_cumulative,
                'main_metrics/total_steps': self.steps_total,
                'main_metrics/map_change_indicator': self.map_change_indicator,
                'main_metrics/all_steps': self.all_steps,
            },
            step=self.episode
        )

    def log_gif_animation(self):
        # TODO: replace with in-memory storage
        with imageio.get_writer(
                self.get_path(f'{self.logger.id}_episode_{self.episode}.gif'),
                mode='I',
                fps=self.animation_fps
        ) as writer:
            for i in range(self.steps):
                image = imageio.imread(
                    self.get_path(
                        f'{self.logger.id}_episode_{self.episode}_step_{i}.png'
                    )
                )
                writer.append_data(image)
        # noinspection PyTypeChecker
        gif_video = wandb.Video(
            self.get_path(f'{self.logger.id}_episode_{self.episode}.gif'),
            fps=self.animation_fps,
            format='gif'
        )
        self.logger.log({f'behavior_samples/animation': gif_video}, step=self.episode)

    def log_goal_complete(self):
        self.logger.log(
            {
                'goal': self.goal,
                'main_metrics/g_goal_steps': self.steps_per_goal,
                'main_metrics/g_task_steps': self.steps_per_task,
                'main_metrics/g_total_steps': self.steps_total,
                'main_metrics/g_episode': self.episode,
            },
            step=self.episode
        )

    def log_task_complete(self):
        self.log_agent()
        self.logger.log(
            {
                'task': self.task,
                'main_metrics/steps_per_task': self.steps_per_task,
                'main_metrics/t_task_steps': self.steps_per_task,
                'main_metrics/t_total_steps': self.steps_total
            },
            step=self.episode
        )

    def init_new_episode(self):
        self.episode += 1
        self.steps = 0
        self.total_reward = 0

        if self.goal_reached:
            self.goal += 1
            self.goal_reached = False
            self.steps_per_goal = 0

        if self.task_complete:
            self.task += 1
            self.task_complete = False
            self.steps_per_task = 0

        if self.level_complete:
            self.level += 1
            self.level_complete = False

        if self.gif_schedule > 0 and (self.episode == 1 or self.episode % self.gif_schedule == 0):
            self.animation = True

    def draw_animation_frame(self):
        pic = self.environment.callmethod('render_rgb')
        if isinstance(pic, list):
            pic = pic[0]

        plt.imsave(
            os.path.join(
                self.path_to_store_logs,
                f'{self.logger.id}_episode_{self.episode}_step_{self.steps}.png'
            ), pic.astype('uint8')
        )
        plt.close()

    def finish_condition(self):
        case1 = self.episode == self.n_episodes
        case2 = self.level == 3 and self.level_complete
        return case1 or case2

    @staticmethod
    def room_ranges(room, width, wall_thickness):
        if room == 1:
            row_range = [0, width - 1]
            col_range = [0, width - 1]
        elif room == 2:
            row_range = [0, width - 1]
            col_range = [width + wall_thickness, width * 2 + wall_thickness - 1]
        elif room == 3:
            row_range = [width + wall_thickness, 2 * width + wall_thickness - 1]
            col_range = [0, width - 1]
        elif room == 4:
            row_range = [width + wall_thickness, 2 * width + wall_thickness - 1]
            col_range = [width + wall_thickness, width * 2 + wall_thickness - 1]
        else:
            raise ValueError(f'Room should be in 1..4')
        return row_range, col_range

    def reset_task(
            self, agent_fixed_positions=None, food_fixed_positions=None,
            door_positions=None, wall_thickness=1, **kwargs
    ):
        """
        Room and Door numbers:
        #############
        #     #     #
        #  1  2  2  #
        #     #     #
        ###1#####3###
        #     #     #
        #  3  4  4  #
        #     #     #
        #############

        Parameters
        ----------
        agent_fixed_positions
        food_fixed_positions
        door_positions
        wall_thickness

        Returns
        -------
        """
        adjacent_rooms = {1: [2, 3], 2: [1, 4], 3: [1, 4], 4: [2, 3]}
        adjacent_doors = {1: [1, 2], 2: [2, 3], 3: [1, 4], 4: [3, 4]}

        if self.level == 1:
            agent_room = self._rng.randint(1, 4)
            food_room = None
            food_door = self._rng.sample(adjacent_doors[agent_room], k=1)[0]
        elif self.level == 2:
            agent_room = self._rng.randint(1, 4)
            food_room = self._rng.sample(adjacent_rooms[agent_room], k=1)[0]
            food_door = None
        else:
            agent_room, food_room = self._rng.sample(list(range(1, 5)), k=2)
            food_door = None

        room_width = (self.env_config['shape_xy'][0] - wall_thickness) // 2
        if agent_fixed_positions is not None:
            agent_pos = tuple(agent_fixed_positions[agent_room - 1])
        else:
            row_range, col_range = self.room_ranges(agent_room, room_width, wall_thickness)
            row = self._rng.randint(*row_range)
            col = self._rng.randint(*col_range)
            agent_pos = (row, col)

        if food_door:
            food_pos = tuple(door_positions[food_door - 1])
        elif food_fixed_positions is not None:
            food_pos = tuple(food_fixed_positions[food_room - 1])
        else:
            row_range, col_range = self.room_ranges(food_room, room_width, wall_thickness)
            row = self._rng.randint(*row_range)
            col = self._rng.randint(*col_range)
            food_pos = (row, col)

        self.set_agent_positions([agent_pos])
        self.set_food_positions([food_pos])
        self.environment.callmethod('reset')

    def set_agent_positions(self, positions, rand=False, sample_size=1):
        if rand:
            positions = self._rng.sample(positions, sample_size)
        positions = [
            self.environment.renderer.shape.shift_relative_to_corner(pos) for pos in positions
        ]
        # noinspection PyUnresolvedReferences
        self.environment.modules['agent'].positions = positions

    def set_food_positions(self, positions, rand=False, sample_size=1):
        if rand:
            positions = self._rng.sample(positions, sample_size)
        positions = [
            self.environment.renderer.shape.shift_relative_to_corner(pos) for pos in positions
        ]
        # noinspection PyUnresolvedReferences
        self.environment.modules['food'].generator.positions = positions

    def draw_map(self):
        map_image = self.environment.callmethod('render_rgb')
        if isinstance(map_image, list):
            map_image = map_image[0]
        self.logger.log({'maps/map': wandb.Image(map_image)}, step=self.episode)

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

    def log_agent(self):
        # TODO: check agent properties
        observations = self.get_all_observations()
        value_map = np.zeros(self.environment.shape)
        obstacle_mask = self.environment.aggregated_mask[EntityType.Obstacle]
        value_map = np.ma.masked_where(obstacle_mask, value_map, False)
        q_map = np.zeros(self.environment.shape)
        q_map = np.ma.masked_where(obstacle_mask, q_map, False)
        q_values = {}

        for position, obs in observations.items():
            value_map[position] = self.agent.get_amg_value(obs)
            q_values[position] = self.agent.get_q_values(obs)
            q_map[position] = np.max(q_values[position])

        fig = plt.figure(frameon=False)
        fig.set_size_inches(5, 5)
        fig.set_dpi(300)
        ax = plt.Axes(fig, [0., 0., 1., 1.])
        ax.set_axis_off()
        fig.add_axes(ax)
        self.logger.log({
            'maps/value_map': wandb.Image(ax.imshow(value_map))
        }, step=self.episode)
        plt.close(fig)

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
        ax.imshow(q_map)

        for position, values in q_values.items():
            y, x = position
            d = values.reshape((-1, 1)) * base_vectors
            m = np.argmax(values)
            for dx, dy in d:
                if dx > 1e-2 or dy > 1e-2:
                    ax.arrow(x, y, dx, dy, width=0.05, color='red', length_includes_head=True)
            ax.arrow(x, y, *(d[m]), width=0.05, color='black', length_includes_head=True)
        img = transform_fig_to_image(fig)
        plt.close(fig)
        self.logger.log({
            'maps/q_map': wandb.Image(img)
        }, step=self.episode)

        amg_obs = draw_masked_observation(
            self.agent.amg.get_masked_values(),
            self.environment.renderer.rendering_sdr_sizes
        )
        self.logger.log(
            {
                'maps/amg': wandb.Image(amg_obs)
            }, step=self.episode
        )


def resolve_agent(name, **config):
    agent = None
    if 'motiv' in name:
        from hima.agents.motivation.agent import Agent
        agent = Agent(**config)
    else:
        AttributeError(f'Unknown agent: {name}')

    return agent
