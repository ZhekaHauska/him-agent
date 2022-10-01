#  Copyright (c) 2022 Autonomous Non-Profit Organization "Artificial Intelligence Research
#  Institute" (AIRI); Moscow Institute of Physics and Technology (National Research University).
#  All rights reserved.
#
#  Licensed under the AGPLv3 license. See LICENSE in the project root for license information.
from typing import Union


def resolve_run_setup(config: dict, run_setup_config: Union[dict, str], experiment_type: str):
    if isinstance(run_setup_config, str):
        run_setup_config = config['run_setups'][run_setup_config]

    if experiment_type == 'policies':
        from hima.experiments.temporal_pooling.test_on_policies import RunSetup
        return RunSetup(**run_setup_config)
    elif experiment_type == 'observations':
        from hima.experiments.temporal_pooling.test_on_states import RunSetup
        return RunSetup(**run_setup_config)
    else:
        KeyError(f'Experiment type {experiment_type} is not supported')
