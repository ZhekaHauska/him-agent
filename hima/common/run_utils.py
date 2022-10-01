#  Copyright (c) 2022 Autonomous Non-Profit Organization "Artificial Intelligence Research
#  Institute" (AIRI); Moscow Institute of Physics and Technology (National Research University).
#  All rights reserved.
#
#  Licensed under the AGPLv3 license. See LICENSE in the project root for license information.
import os
import sys
import traceback
from argparse import ArgumentParser
from copy import deepcopy
from multiprocessing import Process
from typing import Callable, Any, Optional, Type

import wandb
from matplotlib import pyplot as plt
from wandb.sdk.wandb_run import Run

from hima.common.config_utils import (
    TConfig, TConfigOverrideKV, extracted, read_config, parse_arg,
    override_config, extracted_type
)
from hima.common.utils import isnone

TRunEntryPoint = Callable[[TConfig], None]
TExperimentRunnerRegistry = dict[str, Type['Runner']]


class Runner:
    config: TConfig
    logger: Optional[Run]

    def __init__(
            self, config: TConfig,
            log: bool = False, project: str = None,
            **unpacked_config: Any
    ):
        self.config = config

        self.logger = None
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

    experiment_runner_registry: TExperimentRunnerRegistry

    # sweep runs' shared config
    shared_run_config: dict
    shared_run_config_overrides: list[TConfigOverrideKV]

    def __init__(
            self, sweep_id: str, config: dict, n_agents: int,
            experiment_runner_registry: TExperimentRunnerRegistry,
            shared_config_overrides: list[TConfigOverrideKV],
            run_arg_parser: ArgumentParser = None,
    ):
        config, run_command_args, wandb_project = extracted(config, 'command', 'project')
        self.config = config
        self.n_agents = isnone(n_agents, 1)
        self.project = wandb_project
        self.experiment_runner_registry = experiment_runner_registry

        shared_config_filepath = self._extract_agents_shared_config_filepath(
            parser=run_arg_parser or get_run_command_arg_parser(),
            run_command_args=run_command_args
        )
        self.shared_run_config = read_config(shared_config_filepath)
        self.shared_run_config_overrides = shared_config_overrides

        # on Linux machines there's some kind of problem with running sweeps in threads?
        # see https://github.com/wandb/client/issues/1409#issuecomment-870174971
        # and https://github.com/wandb/client/issues/3045#issuecomment-1010435868
        os.environ['WANDB_START_METHOD'] = 'thread'

        if sweep_id is None:
            self.id = wandb.sweep(self.config, project=wandb_project)
        else:
            self.id = sweep_id

    def run(self):
        print(f'==> Sweep {self.id}')

        # TODO: test error handling - we want to terminate [on any error]
        #  a) the whole sweep
        #  b) a single agent
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
        # noinspection PyBroadException
        try:
            self._run_provided_config()
        except Exception as _:
            # catch it only to print traces to the terminal as wandb doesn't do it in Agents!
            print(traceback.print_exc(), file=sys.stderr)
            # finish explicitly with error code (NB: I tend to think it's not necessary here)
            wandb.finish(1)
            # re-raise after printing so wandb catch it
            raise

    def _run_provided_config(self) -> None:
        # BE CAREFUL: this method is expected to be run in parallel — DO NOT mutate `self` here

        # see comments inside func
        turn_off_gui_for_matplotlib()

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
        runner = resolve_experiment_runner(config, self.experiment_runner_registry)
        runner.run()

    @staticmethod
    def _extract_agents_shared_config_filepath(parser: ArgumentParser, run_command_args):
        # there are several ways to extract config filepath based on different conventions
        # we use parser as the most simplistic and automated,
        # but we could introduce strict positional convention or parse with hands

        args, _ = parser.parse_known_args(run_command_args)
        return args.config_filepath


def turn_off_gui_for_matplotlib():
    # Matplotlib tries to spawn GUI which is prohibited for sub-processes meaning
    # you will encounter kernel core errors. To prevent it we tell matplotlib to
    # not touch GUI at all in each of the spawned sub-processes.
    plt.switch_backend('Agg')


def set_single_threaded_math():
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'


def get_run_command_arg_parser() -> ArgumentParser:
    parser = ArgumentParser()
    # todo: add examples
    # todo: remove --sweep ?
    parser.add_argument('-c', '--config', dest='config_filepath', required=True)
    parser.add_argument('-e', '--entity', dest='wandb_entity', required=False, default=None)
    parser.add_argument('--sweep', dest='wandb_sweep', action='store_true', default=False)
    parser.add_argument('--sweep_id', dest='wandb_sweep_id', default=None)
    parser.add_argument('-n', '--n_sweep_agents', type=int, default=None)
    return parser


def run_experiment(
        run_command_parser: ArgumentParser,
        experiment_runner_registry: TExperimentRunnerRegistry
) -> None:
    args, unknown_args = run_command_parser.parse_known_args()

    config = read_config(args.config_filepath)
    config_overrides = list(map(parse_arg, unknown_args))

    if args.wandb_entity:
        # overwrite wandb entity for the run
        os.environ['WANDB_ENTITY'] = args.wandb_entity

    # prevent math parallelization as it usually only slows things down for us
    set_single_threaded_math()

    if args.wandb_sweep:
        Sweep(
            sweep_id=args.wandb_sweep_id,
            config=config,
            n_agents=args.n_sweep_agents,
            experiment_runner_registry=experiment_runner_registry,
            shared_config_overrides=config_overrides,
            run_arg_parser=run_command_parser,
        ).run()
    else:
        override_config(config, config_overrides)
        runner = resolve_experiment_runner(config, experiment_runner_registry)
        runner.run()


def resolve_experiment_runner(
        config: TConfig,
        experiment_runner_registry: TExperimentRunnerRegistry
) -> Runner:
    config, experiment_type = extracted_type(config)
    runner = experiment_runner_registry.get(experiment_type, None)

    assert runner, f'Experiment runner type "{experiment_type}" is not supported'
    return runner(config, **config)
