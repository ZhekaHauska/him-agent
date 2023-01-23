#  Copyright (c) 2022 Autonomous Non-Profit Organization "Artificial Intelligence Research
#  Institute" (AIRI); Moscow Institute of Physics and Technology (National Research University).
#  All rights reserved.
#
#  Licensed under the AGPLv3 license. See LICENSE in the project root for license information.
from __future__ import annotations

import os
from argparse import ArgumentParser
from dataclasses import dataclass
from pathlib import Path

from hima.common.config.base import override_config, TConfig, read_config, TKeyPathValue
from hima.common.config.global_config import GlobalConfig
from hima.common.config.types import TTypeResolver
from hima.common.run.argparse import parse_arg_list
from hima.common.run.wandb import set_wandb_entity


# TODO:
#   - pass log folder root with the default behavior: make temp folder with standard procedure
#   - make experiment runner registry lazy import

@dataclass
class RunParams:
    config: TConfig
    config_path: Path
    config_overrides: list[TKeyPathValue]

    type_resolver: TTypeResolver


def run_experiment(
        *, arg_parser: ArgumentParser, experiment_runner_registry: TTypeResolver
) -> None:
    """
    THE MAIN entry point for starting a program.
        1) resolves run args
        2) resolves whether it is a single run or a wandb sweep
        3) reads config
        4) sets any execution params
        5) resolves who will run this experiment — a runner
        6) passes execution handling to the runner.
    """
    args, unknown_args = arg_parser.parse_known_args()

    if args.wandb_entity:
        set_wandb_entity(args.wandb_entity)

    if not args.multithread:
        # prevent math parallelization as it usually only slows things down for us
        set_single_threaded_math()

    config_path = Path(args.config_filepath)
    config = read_config(config_path)
    config_overrides = parse_arg_list(unknown_args)

    run_params = RunParams(
        config=config, config_path=config_path,
        config_overrides=config_overrides,
        type_resolver=experiment_runner_registry
    )

    if args.wandb_sweep:
        from hima.common.run.sweep import run_sweep
        run_sweep(
            sweep_id=args.wandb_sweep_id,
            n_agents=args.n_sweep_agents,
            sweep_run_params=run_params,
            run_arg_parser=arg_parser,
        )
    else:
        # single run
        run_single_run_experiment(run_params)


def run_single_run_experiment(run_params: RunParams) -> None:
    global_config = GlobalConfig(
        config=run_params.config, config_path=run_params.config_path,
        type_resolver=run_params.type_resolver
    )
    # `config` here is the single object shared between all holders, so we can safely override it
    override_config(global_config.config, run_params.config_overrides)

    # resolve config in case it references other config files [on the root level]
    # NB: passing the copy to the resolver keeps it clean
    resolved_config = global_config.config_resolver.resolve(
        global_config.config.copy(), config_type=dict
    )

    # we make a copy of the resolved config to make runner args with the references to the config
    # itself appended, while leaving it untouched (it also prevent us passing self-referencing
    # config to wandb.init(), which is prohibited)
    runner_args = resolved_config.copy()
    runner_args.update(
        config=resolved_config,
        config_path=global_config.config_path,
    )

    # if runner is a callback function, run happens on resolve
    # otherwise, we expect it to be an object with an explicit `run` method that should be called
    runner = global_config.object_resolver.resolve(runner_args)
    if runner is not None:
        runner.run()


def set_single_threaded_math():
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'


def default_run_arg_parser() -> ArgumentParser:
    """
    Returns default run command parser.

    Instead of creating a new one for your specific purposes, you can create a default one
    and then extend it by adding new arguments.
    """
    parser = ArgumentParser()
    parser.add_argument('-c', '--config', dest='config_filepath', required=True)
    parser.add_argument('-e', '--entity', dest='wandb_entity', required=False, default=None)
    parser.add_argument('--sweep', dest='wandb_sweep', action='store_true', default=False)
    parser.add_argument('--sweep_id', dest='wandb_sweep_id', default=None)
    parser.add_argument('-n', '--n_sweep_agents', type=int, default=None)

    parser.add_argument('--multithread', dest='multithread', action='store_true', default=False)
    return parser
