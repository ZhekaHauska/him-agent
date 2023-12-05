#  Copyright (c) 2022 Autonomous Non-Profit Organization "Artificial Intelligence Research
#  Institute" (AIRI); Moscow Institute of Physics and Technology (National Research University).
#  All rights reserved.
#
#  Licensed under the AGPLv3 license. See LICENSE in the project root for license information.

from hima.common.run.entrypoint import run_experiment, default_run_arg_parser
from hima.experiments.motivation.test_empowerment import GwEmpowermentTest
from hima.experiments.motivation.test_striatum import GwStriatumTest
from hima.experiments.motivation.test_td_lambda import GwTDLambdaTest


if __name__ == '__main__':
    run_experiment(
        arg_parser=default_run_arg_parser(),
        experiment_runner_registry={
            'motivation.emp': GwEmpowermentTest,
            'motivation.str': GwStriatumTest,
            'motivation.tdlambda': GwTDLambdaTest
        }
    )
