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

from hima.agents.rnd.agent import RndAgent
from hima.modules.td_lambda import TDLambda


class GwTDLambdaTest(Runner):
    def __init__(self, config: TConfig, **kwargs):
        super().__init__(config, **config)

        self.seed = config['seed']
        self.n_episodes = config['n_episodes']
        self.environment: Environment = unwrap(BioGwLabEnvironment(**config['environment']))
        map_image = self.environment.callmethod('render_rgb')
        if isinstance(map_image, list):
            map_image = map_image[0]
        if self.logger:
            map_image = wandb.Image(map_image)
            self.logger.log({'map': map_image}, step=0)

        self.agent = RndAgent(self.environment.n_actions, self.seed)
        self.td_lambda = TDLambda(self.seed, self.environment.output_sdr_size, **config['TDLambda'])
        self.episode = 0

        self.value_map = np.ma.zeros(self.environment.shape)
        self.value_map[:, :] = np.ma.masked

    def run(self):
        while True:
            _, obs, _ = self.environment.observe()
            self.value_map[self.environment.agent.position] = self.td_lambda.get_value(obs)
            action = self.agent.act()
            self.environment.act(action)
            reward, next_obs, _ = self.environment.observe()
            if self.environment.is_terminal():
                self.environment.act(action)
                self.td_lambda.update(obs, reward, [])
                self.td_lambda.reset()
                self.episode += 1
                if self.logger:
                    value_image = wandb.Image(plt.imshow(self.value_map))
                    self.logger.log({'values': value_image}, step=self.episode)
                if self.episode == self.n_episodes:
                    break
            else:
                self.td_lambda.update(obs, reward, next_obs)

