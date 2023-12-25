#  Copyright (c) 2023 Autonomous Non-Profit Organization "Artificial Intelligence Research
#  Institute" (AIRI); Moscow Institute of Physics and Technology (National Research University).
#  All rights reserved.
#
#  Licensed under the AGPLv3 license. See LICENSE in the project root for license information.
from __future__ import annotations

import numpy as np
from hima.common.metrics import MetricsRack


class BaseAgent:
    initial_action: int | None
    state_value: float

    def observe(self, events, action):
        raise NotImplementedError

    def sample_action(self):
        raise NotImplementedError

    def reinforce(self, reward):
        raise NotImplementedError

    def reset(self):
        raise NotImplementedError


class BaseEnvironment:
    raw_obs_shape: (int, int)
    actions: tuple
    n_actions: int

    def obs(self):
        raise NotImplementedError

    def act(self, action):
        raise NotImplementedError

    def step(self):
        raise NotImplementedError

    def reset(self):
        raise NotImplementedError

    def change_setup(self, setup):
        raise NotImplementedError

    def close(self):
        raise NotImplementedError


class BaseRunner:
    agent: BaseAgent
    environment: BaseEnvironment
    metrics_rack: MetricsRack | None

    def __init__(self, logger, conf):
        """
        config_structure:
        run:
            n_episodes
            update_start
            max_steps
            reward_free
            action_inertia
            frame_skip
            strategies
            setups
            setup_period
        env:
            ,,,
        agent:
            ...
        metrics:
            ...
        """
        self.logger = logger
        self.seed = conf['run'].get('seed')
        self._rng = np.random.default_rng(self.seed)

        self.n_episodes = conf['run']['n_episodes']
        self.max_steps = conf['run']['max_steps']

        self.update_start = conf['run'].get('update_start', 0)
        self.reward_free = conf['run'].get('reward_free', 0)
        self.action_inertia = conf['run'].get('action_inertia', 1)
        self.frame_skip = conf['run'].get('frame_skip', 0)
        self.strategies = conf['run'].get('strategies', None)

        assert self.frame_skip >= 0

        self.setups = conf['run']['setups']
        self.setup_period = conf['run'].get('setup_period', None)
        self.current_setup_id = 0

        if self.setup_period is None:
            self.setup_period = [self.n_episodes // len(self.setups)] * len(self.setups)
        elif type(self.setup_period) is int:
            period = self.setup_period
            self.setup_period = [period] * len(self.setups)

        assert len(self.setups) == len(self.setup_period)

        env_conf = conf['env']
        env_conf['seed'] = self.seed
        agent_conf = conf['agent']
        agent_conf['seed'] = self.seed

        self.environment = self.make_environment(conf['env_type'], env_conf, self.setups[0])

        agent_conf['raw_obs_shape'] = self.environment.raw_obs_shape
        agent_conf['n_actions'] = self.environment.n_actions
        self.agent = self.make_agent(conf['agent_type'], agent_conf)

        metrics_conf = conf.get('metrics', None)
        if metrics_conf is not None and self.logger is not None:
            self.metrics_rack = MetricsRack(
                self.logger,
                self,
                **conf['metrics']
            )
        else:
            self.metrics_rack = None

        self.steps = 0
        self.episodes = 0
        self.setup_episodes = 0
        self.strategy = None
        self.action_step = 0
        self.running = True
        self.action = self.agent.initial_action
        self.reward = 0
        self.events = None
        self.obs = None

    @staticmethod
    def make_environment(env_type, conf, setup):
        raise NotImplementedError

    @staticmethod
    def make_agent(agent_type, conf):
        raise NotImplementedError

    def prepare_episode(self):
        self.steps = 0
        self.running = True
        self.action = self.agent.initial_action

        # change setup
        if self.setup_episodes >= self.setup_period[self.current_setup_id]:
            self.current_setup_id += 1
            self.current_setup_id = self.current_setup_id % len(self.setups)
            self.environment.change_setup(self.setups[self.current_setup_id])
            self.setup_episodes = 0

        self.environment.reset()
        self.agent.reset()

        self.strategy = None
        self.action_step = 0

    def run(self):
        self.episodes = 0
        self.setup_episodes = 0
        self.current_setup_id = 0

        for i in range(self.n_episodes):
            self.prepare_episode()

            while self.running:
                self.reward = 0
                self.obs = None
                for frame in range(self.frame_skip + 1):
                    self.environment.act(self.action)
                    self.environment.step()
                    self.obs, self.reward, is_terminal = self.environment.obs()

                    self.running = not is_terminal
                    if is_terminal:
                        break

                    if self.action is None:
                        break

                # observe events_t and action_{t-1}
                self.agent.observe(self.obs, self.action)
                self.agent.reinforce(self.reward)

                if self.running:
                    if self.strategies is not None:
                        if self.steps == 0:
                            self.strategy = self.strategies[self.agent.sample_action()]

                        if (self.steps % self.action_inertia) == 0:
                            if self.action_step < len(self.strategy):
                                self.action = self.strategy[self.action_step]
                            else:
                                self.running = False
                            self.action_step += 1
                    else:
                        if (self.steps % self.action_inertia) == 0:
                            if self.setup_episodes < self.reward_free:
                                self.action = self._rng.integers(self.environment.n_actions)
                            else:
                                self.action = self.agent.sample_action()

                self.steps += 1

                if self.steps >= self.max_steps:
                    self.running = False

                if not self.running:
                    self.episodes += 1
                    self.setup_episodes += 1

                if (self.metrics_rack is not None) and (self.steps >= self.update_start):
                    self.metrics_rack.step()
        else:
            self.environment.close()
