#  Copyright (c) 2022 Autonomous Non-Profit Organization "Artificial Intelligence Research
#  Institute" (AIRI); Moscow Institute of Physics and Technology (National Research University).
#  All rights reserved.
#
#  Licensed under the AGPLv3 license. See LICENSE in the project root for license information.
from __future__ import annotations

import os
from argparse import ArgumentParser
from pathlib import Path
from typing import Type

from ruamel import yaml

from hima.common.config import (
    extracted_type, TConfig, override_config
)
from hima.common.run.argparse import parse_arg_list
from hima.common.run.runner import Runner

TExperimentRunnerRegistry = dict[str, Type[Runner]]


# TODO:
#   - pass log folder root with the default behavior: make temp folder with standard procedure
#   - make experiment runner registry lazy import


def run_experiment(
        run_command_parser: ArgumentParser,
        experiment_runner_registry: TExperimentRunnerRegistry
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
    args, unknown_args = run_command_parser.parse_known_args()

    config_path = Path(args.config_filepath)
    experiment_root = config_path.parent

    config = read_config(config_path)
    config_overrides = parse_arg_list(unknown_args)

    if args.wandb_entity:
        # overwrite wandb entity for the run
        os.environ['WANDB_ENTITY'] = args.wandb_entity

    if not args.multithread:
        # prevent math parallelization as it usually only slows things down for us
        set_single_threaded_math()

    if args.wandb_sweep:
        from hima.common.run.sweep import Sweep
        Sweep(
            sweep_id=args.wandb_sweep_id,
            config=config,
            n_agents=args.n_sweep_agents,
            experiment_runner_registry=experiment_runner_registry,
            experiment_root=experiment_root,
            shared_config_overrides=config_overrides,
            run_arg_parser=run_command_parser,
        ).run()
    else:
        # single run
        override_config(config, config_overrides)
        runner = resolve_experiment_runner(config, experiment_runner_registry)
        runner.run()


def set_single_threaded_math():
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'


def get_run_command_arg_parser() -> ArgumentParser:
    parser = ArgumentParser()
    parser.add_argument('-c', '--config', dest='config_filepath', required=True)
    parser.add_argument('-e', '--entity', dest='wandb_entity', required=False, default=None)
    parser.add_argument('--sweep', dest='wandb_sweep', action='store_true', default=False)
    parser.add_argument('--sweep_id', dest='wandb_sweep_id', default=None)
    parser.add_argument('-n', '--n_sweep_agents', type=int, default=None)

    parser.add_argument('--multithread', dest='multithread', action='store_true', default=False)
    return parser


def resolve_experiment_runner(
        config: TConfig,
        experiment_runner_registry: TExperimentRunnerRegistry
) -> Runner:
    config, experiment_type = extracted_type(config)
    runner_cls = experiment_runner_registry.get(experiment_type, None)

    assert runner_cls, f'Experiment runner type "{experiment_type}" is not supported'
    return runner_cls(config, **config)


def read_config(filepath: str | Path) -> TConfig:
    if not isinstance(filepath, Path):
        filepath = Path(filepath)

    with filepath.open('r') as config_io:
        return yaml.load(config_io, Loader=yaml.Loader)
