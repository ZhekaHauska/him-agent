#  Copyright (c) 2022 Autonomous Non-Profit Organization "Artificial Intelligence Research
#  Institute" (AIRI); Moscow Institute of Physics and Technology (National Research University).
#  All rights reserved.
#
#  Licensed under the AGPLv3 license. See LICENSE in the project root for license information.
from argparse import ArgumentParser
from copy import deepcopy
from multiprocessing import Process
from typing import Callable, Any

import wandb
from wandb.sdk.wandb_run import Run

from hima.common.config_utils import (
    TConfig, TConfigOverrideKV, extracted, read_config, parse_arg,
    override_config
)
from hima.common.utils import isnone

TRunEntryPoint = Callable[[TConfig], None]


class Runner:
    config: TConfig
    logger: Run

    def __init__(
            self, config: TConfig,
            log: bool = False, project: str = None,
            **unpacked_config: Any
    ):
        self.config = config

        if log:
            self.logger = wandb.init(project=project)
            # we have to pass the config with update instead of init because of sweep runs
            self.logger.config.update(self.config)

    def run(self) -> None:
        ...


class Sweep:
    id: str
    project: str
    config: dict
    n_agents: int

    # sweep runs' shared config
    run_entry_point: TRunEntryPoint
    shared_run_config: dict
    shared_run_config_overrides: list[TConfigOverrideKV]

    run_command_arg_parser: ArgumentParser

    def __init__(
            self, config: dict, n_agents: int,
            single_run_entry_point: TRunEntryPoint,
            shared_config_overrides: list[TConfigOverrideKV],
            run_arg_parser: ArgumentParser = None,
    ):
        self.run_command_arg_parser = run_arg_parser

        config, run_command_args, wandb_project = extracted(config, 'command', 'project')
        self.config = config
        self.n_agents = isnone(n_agents, 1)
        self.project = wandb_project
        self.run_entry_point = single_run_entry_point

        shared_config_filepath = self._extract_agents_shared_config_filepath(run_command_args)
        self.shared_run_config = read_config(shared_config_filepath)
        self.shared_run_config_overrides = shared_config_overrides

        self.id = wandb.sweep(self.config, project=wandb_project)

    def run(self):
        print(f'==> Sweep {self.id}')

        agent_processes = []
        for _ in range(self.n_agents):
            p = Process(
                target=wandb.agent,
                kwargs={
                    'sweep_id': self.id,
                    'function': self._wandb_agent_entry_point
                }
            )
            p.start()
            agent_processes.append(p)

        for p in agent_processes:
            p.join()

        print(f'<== Sweep {self.id}')

    def _wandb_agent_entry_point(self) -> None:
        # BE CAREFUL: this method is expected to be run in parallel — DO NOT mutate `self` here

        # we know here that it's a sweep-induced run and can expect single sweep run config to be
        # passed via wandb.config, hence we take it and apply all overrides:
        # while concatenating overrides, the order DOES matter: run params, then args
        run = wandb.init()
        sweep_overrides = list(map(parse_arg, run.config.items()))
        config_overrides = sweep_overrides + self.shared_run_config_overrides

        # it's important to take COPY of the shared config to prevent mutating `self` state
        config = deepcopy(self.shared_run_config)
        override_config(config, config_overrides)

        # start single run
        self.run_entry_point(config)

    def _extract_agents_shared_config_filepath(self, run_args):
        parser = self.run_command_arg_parser or get_run_command_arg_parser()
        args, _ = parser.parse_known_args(run_args)
        return args.config_filepath


def get_run_command_arg_parser() -> ArgumentParser:
    parser = ArgumentParser()
    # todo: add examples
    # todo: remove --sweep ?
    parser.add_argument('-c', '--config', dest='config_filepath', required=True)
    parser.add_argument('-e', '--entity', dest='wandb_entity', required=False, default=None)
    parser.add_argument('--sweep', dest='wandb_sweep', action='store_true', default=False)
    parser.add_argument('-n', '--n_sweep_agents', type=int, default=None)
    return parser


def run_experiment(run_command_parser: ArgumentParser, run_entry_point: TRunEntryPoint) -> None:
    args, unknown_args = run_command_parser.parse_known_args()

    config = read_config(args.config_filepath)
    config_overrides = list(map(parse_arg, unknown_args))

    if args.wandb_entity:
        # overwrite wandb entity for the run
        import os
        os.environ['WANDB_ENTITY'] = args.wandb_entity

    if args.wandb_sweep:
        Sweep(
            config=config,
            n_agents=args.n_sweep_agents,
            single_run_entry_point=run_entry_point,
            shared_config_overrides=config_overrides,
            run_arg_parser=run_command_parser,
        ).run()
    else:
        override_config(config, config_overrides)
        run_entry_point(config)
