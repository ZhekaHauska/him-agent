#  Copyright (c) 2022 Autonomous Non-Profit Organization "Artificial Intelligence Research
#  Institute" (AIRI); Moscow Institute of Physics and Technology (National Research University).
#  All rights reserved.
#
#  Licensed under the AGPLv3 license. See LICENSE in the project root for license information.

from hima.envs.biogwlab.environment import Environment
from hima.envs.env import Wrapper


class AgentPositionProvider(Wrapper):
    root_env: Environment

    def get_info(self) -> dict:
        info = super(AgentPositionProvider, self).get_info()
        info['agent_position'] = self.root_env.agent.position
        return info


