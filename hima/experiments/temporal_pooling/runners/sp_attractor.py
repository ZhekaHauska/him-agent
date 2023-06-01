#  Copyright (c) 2023 Autonomous Non-Profit Organization "Artificial Intelligence Research
#  Institute" (AIRI); Moscow Institute of Physics and Technology (National Research University).
#  All rights reserved.
#
#  Licensed under the AGPLv3 license. See LICENSE in the project root for license information.
import numpy as np

from hima.experiments.temporal_pooling.stp.sp import SpatialPooler
from hima.common.sds import Sds
from hima.envs.mnist import MNISTEnv

import wandb
import os


class SPAttractorRunner:
    def __init__(self, logger, conf):
        self.seed = conf['run']['seed']
        self.logger = logger

        self.env = MNISTEnv(seed=self.seed)

        input_sds = Sds(shape=self.env.obs_shape, sparsity=1.0)

        if conf.get('encoder') is not None:
            conf['encoder']['seed'] = self.seed
            conf['encoder']['feedforward_sds'] = input_sds
            conf['encoder']['output_sds'] = Sds(conf['encoder']['output_sds'])
            self.encoder = SpatialPooler(
                **conf['encoder']
            )

            conf['attractor']['adapt_to_ff_sparsity'] = False
            attractor_sds = self.encoder.output_sds
        else:
            conf['attractor']['adapt_to_ff_sparsity'] = True
            attractor_sds = input_sds
            self.encoder = None

        conf['attractor']['feedforward_sds'] = attractor_sds
        conf['attractor']['output_sds'] = Sds(conf['attractor']['output_sds'])
        conf['attractor']['seed'] = self.seed
        self.attractor = SpatialPooler(
            **conf['attractor']
        )

        if conf['run'].get('max_steps') is not None:
            self.max_steps = conf['run']['max_steps']
        else:
            self.max_steps = self.env.size

        self.n_episodes = conf['run']['n_episodes']
        self.log_update_rate = conf['run'].get('update_rate')

    def run(self):
        for i in range(self.n_episodes):
            steps = 0
            att_entropy = []
            enc_entropy = []

            self.env.reset()

            while True:
                obs = self.env.obs()
                thresh = obs.mean()
                obs = np.flatnonzero(obs >= thresh)

                if self.encoder is not None:
                    obs = self.encoder.compute(obs, learn=True)
                    enc_entropy.append(self.encoder.output_entropy())

                self.attractor.compute(obs, learn=True)
                att_entropy.append(self.attractor.output_entropy())

                steps += 1
                if steps >= self.max_steps:
                    break

            if self.logger is not None:
                self.logger.log(
                    {'main_metrics/attractor_entropy': np.array(att_entropy).mean()},
                    step=i
                )
                if self.encoder is not None:
                    self.logger.log(
                        {'main_metrics/encoder_entropy': np.array(enc_entropy).mean()},
                        step=i
                    )

                if (self.log_update_rate is not None) and (i % self.log_update_rate == 0):
                    ...


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

    runner = SPAttractorRunner(logger, config)
    runner.run()


if __name__ == '__main__':
    default_config = 'configs/runner/sp_attractor.yaml'
    main(os.environ.get('RUN_CONF', default_config))
