#  Copyright (c) 2022 Autonomous Non-Profit Organization "Artificial Intelligence Research
#  Institute" (AIRI); Moscow Institute of Physics and Technology (National Research University).
#  All rights reserved.
#
#  Licensed under the AGPLv3 license. See LICENSE in the project root for license information.

from hima.common.run.entrypoint import run_experiment, get_run_command_arg_parser
from hima.experiments.temporal_pooling.test_on_obs_layered import ObservationsLayeredExperiment
from hima.experiments.temporal_pooling._depr.test_on_policies import PoliciesExperiment
from hima.experiments.temporal_pooling._depr.test_on_states import ObservationsExperiment


if __name__ == '__main__':
    run_experiment(
        run_command_parser=get_run_command_arg_parser(),
        experiment_runner_registry={
            'tp.policy': PoliciesExperiment,
            'tp.observations': ObservationsExperiment,
            'tp.layered': ObservationsLayeredExperiment,
        }
    )
