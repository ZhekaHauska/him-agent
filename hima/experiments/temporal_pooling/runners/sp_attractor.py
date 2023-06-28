#  Copyright (c) 2023 Autonomous Non-Profit Organization "Artificial Intelligence Research
#  Institute" (AIRI); Moscow Institute of Physics and Technology (National Research University).
#  All rights reserved.
#
#  Licensed under the AGPLv3 license. See LICENSE in the project root for license information.

import wandb
import os

from hima.experiments.temporal_pooling.test_attractor_mnist import SpAttractorMnistExperiment
from hima.experiments.temporal_pooling.test_attractor_rbits import SpAttractorRightBitsExperiment


def main(config_path):
    import sys
    import yaml
    import ast

    if len(sys.argv) > 1:
        config_path = sys.argv[1]

    config = dict()

    with open(config_path, 'r') as file:
        config['run'] = yaml.load(file, Loader=yaml.Loader)

    with open(config['run']['attractor_conf'], 'r') as file:
        config['attractor'] = yaml.load(file, Loader=yaml.Loader)

    encoder_conf = config['run'].get('encoder_conf', None)
    if encoder_conf is not None:
        with open(encoder_conf, 'r') as file:
            config['encoder'] = yaml.load(file, Loader=yaml.Loader)

    env_conf = config['run'].get('env_conf', None)
    if env_conf is not None:
        with open(env_conf, 'r') as file:
            config['env'] = yaml.load(file, Loader=yaml.Loader)

    for arg in sys.argv[2:]:
        key, value = arg.split('=')

        try:
            value = ast.literal_eval(value)
        except ValueError:
            ...

        key = key.lstrip('-')
        if key.endswith('.'):
            # a trick that allow distinguishing sweep params from config params
            # by adding a suffix `.` to sweep param - now we should ignore it
            key = key[:-1]
        tokens = key.split('.')
        c = config
        for k in tokens[:-1]:
            if not k:
                # a trick that allow distinguishing sweep params from config params
                # by inserting additional dots `.` to sweep param - we just ignore it
                continue
            if 0 in c:
                k = int(k)
            c = c[k]
        c[tokens[-1]] = value

    if config['run']['log']:
        logger = wandb.init(
            project=config['run']['project_name'],
            entity=os.environ.get('WANDB_ENTITY'),
            config=config
        )
    else:
        logger = None

    if config['run']['experiment'] == 'mnist':
        runner = SpAttractorMnistExperiment(logger, config)
    elif config['run']['experiment'] == 'rbits':
        runner = SpAttractorRightBitsExperiment(logger, config)
    else:
        raise NotImplementedError(f'There is no such experiment {config["run"]["experiment"]}')
    runner.run()


if __name__ == '__main__':
    default_config = 'configs/runner/sp_attractor.yaml'
    main(os.environ.get('RUN_CONF', default_config))
