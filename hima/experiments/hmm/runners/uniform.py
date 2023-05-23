#  Copyright (c) 2023 Autonomous Non-Profit Organization "Artificial Intelligence Research
#  Institute" (AIRI); Moscow Institute of Physics and Technology (National Research University).
#  All rights reserved.
#
#  Licensed under the AGPLv3 license. See LICENSE in the project root for license information.

from hima.modules.htm.spatial_pooler import SPDecoder, HtmSpatialPooler, SPEnsemble
from htm.bindings.sdr import SDR
from hima.experiments.hmm.runners.utils import get_surprise

try:
    from pinball import Pinball
except ModuleNotFoundError:
    Pinball = None

import numpy as np
import matplotlib.pyplot as plt
import wandb
import yaml
import os
import sys
import ast
from copy import copy


class PinballTest:
    def __init__(self, logger, conf):
        self.seed = conf['run']['seed']

        conf['env']['seed'] = self.seed
        conf['env']['exe_path'] = os.environ.get('PINBALL_EXE', None)
        conf['env']['config_path'] = os.path.join(
            os.environ.get('PINBALL_ROOT', None),
            'configs',
            f"{conf['run']['setup']}.json"
        )

        self.env = Pinball(**conf['env'])

        obs = self.env.obs()
        self.obs_shape = (obs.shape[0], obs.shape[1])

        self.encoder_type = conf['run']['encoder']
        sp_conf = conf.get('sp', None)

        if self.encoder_type == 'one_sp':
            assert sp_conf is not None
            sp_conf['seed'] = self.seed
            self.encoder = HtmSpatialPooler(
                self.obs_shape,
                **sp_conf
            )
            self.obs_shape = self.encoder.getColumnDimensions()
            self.sp_input = SDR(self.encoder.getInputDimensions())
            self.sp_output = SDR(self.encoder.getColumnDimensions())

            self.decoder = SPDecoder(self.encoder)

            self.n_obs_vars = self.obs_shape[0] * self.obs_shape[1]
            self.n_obs_states = 1

            self.surprise_mode = 'bernoulli'

        elif self.encoder_type == 'sp_ensemble':
            assert sp_conf is not None
            sp_conf['seed'] = self.seed
            sp_conf['inputDimensions'] = list(self.obs_shape)
            n_sp = sp_conf.pop('n_sp')
            self.encoder = SPEnsemble(
                n_sp,
                **sp_conf
            )
            shape = self.encoder.sps[0].getColumnDimensions()
            self.obs_shape = (shape[0]*self.encoder.n_sp, shape[1])
            self.sp_input = SDR(self.encoder.getNumInputs())
            self.sp_output = SDR(self.encoder.getNumColumns())

            self.decoder = SPDecoder(self.encoder)

            self.n_obs_vars = self.encoder.n_sp
            self.n_obs_states = self.encoder.sps[0].getNumColumns()

            self.surprise_mode = 'categorical'
        else:
            self.encoder = None
            self.sp_input = None
            self.sp_output = None
            self.decoder = None

            self.n_obs_vars = self.obs_shape[0] * self.obs_shape[1]
            self.n_obs_states = 1

            self.surprise_mode = 'bernoulli'

        self.actions = conf['run']['actions']
        self.positions = conf['run']['positions']
        self.prediction_steps = conf['run']['prediction_steps']
        self.n_episodes = conf['run']['n_episodes']
        self.log_update_rate = conf['run']['update_rate']
        self.max_steps = conf['run']['max_steps']
        self.save_model = conf['run']['save_model']
        self.log_fps = conf['run']['log_gif_fps']
        self.mc_iterations = conf['run']['mc_iterations']
        self.normalize_surprise = conf['run']['normalize_surprise']

        self._rng = np.random.default_rng(self.seed)

        self.logger = logger

        if self.logger is not None:
            self.env.reset(self.positions[0])
            self.env.step()
            self.logger.log(
                {
                    'setting': wandb.Image(
                        plt.imshow(self.env.obs())
                    )
                },
                step=0
            )

    def run(self):
        total_surprise = 0
        total_surprise_decoder = 0

        for i in range(self.n_episodes):
            surprises = []
            surprises_decoder = []

            obs_probs_stack = []
            hidden_probs_stack = []
            n_step_surprise_obs = [list() for _ in range(self.prediction_steps)]
            n_step_surprise_hid = [list() for _ in range(self.prediction_steps)]

            steps = 0

            init_i = self._rng.integers(0, len(self.actions), 1)
            action = self.actions[init_i[0]]
            position = self.positions[init_i[0]]
            self.env.reset(position)
            self.env.act(action)

            self.env.step()
            prev_im = self.preprocess(self.env.obs())

            while True:
                self.env.step()
                raw_im = self.preprocess(self.env.obs())
                thresh = raw_im.mean()
                diff = np.abs(raw_im - prev_im) >= thresh
                prev_im = raw_im.copy()

                obs_state = np.flatnonzero(diff)

                if self.encoder is not None:
                    self.sp_input.sparse = obs_state
                    self.encoder.compute(self.sp_input, True, self.sp_output)
                    obs_state = self.sp_output.sparse

                column_probs = np.ones(self.obs_shape).flatten()
                column_probs /= len(column_probs)

                if steps > 0:
                    # metrics
                    # 1. surprise
                    surprise = get_surprise(
                        column_probs,
                        obs_state,
                        mode=self.surprise_mode,
                        normalize=self.normalize_surprise
                    )
                    surprises.append(surprise)
                    total_surprise += surprise

                    if self.decoder is not None:
                        decoded_probs = self.decoder.decode(column_probs, update=True)

                        surprise_decoder = get_surprise(
                            decoded_probs,
                            self.sp_input.sparse,
                            normalize=self.normalize_surprise
                        )

                        surprises_decoder.append(surprise_decoder)
                        total_surprise_decoder += surprise_decoder

                    # 2. image
                    if (i % self.log_update_rate) == 0:
                        obs_probs = []
                        hidden_probs = []

                        if self.decoder is not None:
                            hidden_prediction = column_probs.reshape(self.obs_shape)
                            decoded_probs = self.decoder.decode(column_probs, update=True)
                            decoded_probs = decoded_probs.reshape(self.encoder.getInputDimensions())
                        else:
                            decoded_probs = column_probs.reshape(self.obs_shape)
                            hidden_prediction = None

                        obs_probs.append(decoded_probs.copy())
                        hidden_probs.append(hidden_prediction.copy())

                        if self.prediction_steps > 1:
                            for j in range(self.prediction_steps - 1):
                                if self.decoder is not None:
                                    hidden_prediction = column_probs.reshape(self.obs_shape)
                                    decoded_probs = self.decoder.decode(column_probs)
                                    decoded_probs = decoded_probs.reshape(
                                        self.encoder.getInputDimensions()
                                    )
                                else:
                                    decoded_probs = column_probs.reshape(self.obs_shape)
                                    hidden_prediction = None

                                obs_probs.append(decoded_probs.copy())
                                hidden_probs.append(hidden_prediction.copy())

                        obs_probs_stack.append(copy(obs_probs))
                        hidden_probs_stack.append(copy(hidden_probs))

                        # remove empty lists
                        obs_probs_stack = [x for x in obs_probs_stack if len(x) > 0]
                        hidden_probs_stack = [x for x in hidden_probs_stack if len(x) > 0]

                        pred_horizon = [self.prediction_steps - len(x) for x in obs_probs_stack]
                        current_predictions_obs = [x.pop(0) for x in obs_probs_stack]
                        current_predictions_hid = [x.pop(0) for x in hidden_probs_stack]

                        for p_obs, p_hid, s in zip(
                                current_predictions_obs, current_predictions_hid, pred_horizon
                        ):
                            surp_obs = get_surprise(
                                p_obs.flatten(),
                                self.sp_input.sparse,
                                normalize=self.normalize_surprise
                            )
                            surp_hid = get_surprise(
                                p_hid.flatten(),
                                self.sp_output.sparse,
                                mode=self.surprise_mode,
                                normalize=self.normalize_surprise
                            )
                            n_step_surprise_obs[s].append(surp_obs)
                            n_step_surprise_hid[s].append(surp_hid)

                steps += 1

                if steps >= self.max_steps:
                    break

            if self.logger is not None:
                self.logger.log(
                    {
                        'main_metrics/surprise': np.array(surprises).mean(),
                        'main_metrics/total_surprise': total_surprise,
                        'main_metrics/steps': steps,
                    }, step=i
                )

                if self.decoder is not None:
                    self.logger.log(
                        {
                            'main_metrics/surprise_decoder': np.array(surprises_decoder).mean(),
                            'main_metrics/total_surprise_decoder': total_surprise_decoder,
                        }, step=i
                    )

                if (self.log_update_rate is not None) and (i % self.log_update_rate == 0):
                    n_step_surprises_hid = {
                        f'n_step_hidden/surprise_step_{s + 1}': np.mean(x) for s, x in
                        enumerate(n_step_surprise_hid)
                    }
                    n_step_surprises_obs = {
                        f'n_step_raw/surprise_step_{s + 1}': np.mean(x) for s, x in
                        enumerate(n_step_surprise_obs)
                    }

                    self.logger.log(
                        n_step_surprises_obs,
                        step=i
                    )

                    self.logger.log(
                        n_step_surprises_hid,
                        step=i
                    )

    @staticmethod
    def preprocess(image):
        gray_im = image.sum(axis=-1)
        gray_im /= gray_im.max()

        return gray_im


def main(config_path):
    if len(sys.argv) > 1:
        config_path = sys.argv[1]

    config = dict()

    with open(config_path, 'r') as file:
        config['run'] = yaml.load(file, Loader=yaml.Loader)

    with open(config['run']['env_conf'], 'r') as file:
        config['env'] = yaml.load(file, Loader=yaml.Loader)
    sp_conf = config['run'].get('sp_conf', None)
    if sp_conf is not None:
        with open(sp_conf, 'r') as file:
            config['sp'] = yaml.load(file, Loader=yaml.Loader)

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

    if config['run']['seed'] is None:
        config['run']['seed'] = np.random.randint(0, np.iinfo(np.int32).max)

    if config['run']['log']:
        logger = wandb.init(
            project=config['run']['project_name'], entity=os.environ['WANDB_ENTITY'],
            config=config
        )
    else:
        logger = None

    experiment = config['run']['experiment']

    if experiment == 'pinball':
        runner = PinballTest(logger, config)
    else:
        raise ValueError

    runner.run()


if __name__ == '__main__':
    default_config = 'configs/runner/uniform/pinball.yaml'
    main(os.environ.get('RUN_CONF', default_config))
