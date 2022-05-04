#  Copyright (c) 2022 Autonomous Non-Profit Organization "Artificial Intelligence Research
#  Institute" (AIRI); Moscow Institute of Physics and Technology (National Research University).
#  All rights reserved.
#
#  Licensed under the AGPLv3 license. See LICENSE in the project root for license information.

import numpy as np
from hima.envs.mpg import MarkovProcessGrammar
from hima.common.sdr_encoders import IntBucketEncoder
from hima.modules.htm.belief_tm import NaiveBayesTM

import wandb
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator


def main():
    # process from "Deep Predictive Learning in Neocortex and Pulvinar" O'Reilly 2021
    transitions = [
        [0, 1, 0, 0, 0, 0, 0, 0],
        [0, 0, 0.5, 0.5, 0, 0, 0, 0],
        [0, 0, 0.5, 0, 0, 0.5, 0, 0],
        [0, 0, 0, 0.5, 0.5, 0, 0, 0],
        [0, 0, 0.5, 0, 0, 0, 0.5, 0],
        [0, 0, 0, 0, 0.5, 0, 0.5, 0],
        [0, 0, 0, 0, 0, 0, 0, 1],
    ]

    letters = [
        [0, 'B', 0, 0, 0, 0, 0, 0],
        [0, 0, 'P', 'T', 0, 0, 0, 0],
        [0, 0, 'T', 0, 0, 'V', 0, 0],
        [0, 0, 0, 'S', 'X', 0, 0, 0],
        [0, 0, 'X', 0, 0, 0, 'S', 0],
        [0, 0, 0, 0, 'P', 0, 'V', 0],
        [0, 0, 0, 0, 0, 0, 0, 'E'],
    ]

    char_to_number = {'B': 0, 'P': 1, 'T': 2, 'V': 3, 'S': 4, 'X': 5, 'E': 6}
    numer_to_char = {value: key for key, value in char_to_number.items()}

    log = True
    update_rate = 100
    bucket_size = 4
    seed = 6989
    encoder = IntBucketEncoder(len(char_to_number), bucket_size)

    mpg = MarkovProcessGrammar(
        8,
        transitions,
        letters,
        initial_state=0,
        terminal_state=7,
        autoreset=False,
        seed=seed
    )
    print(f'n_columns: {encoder.output_sdr_size}')
    tm = NaiveBayesTM(
        encoder.output_sdr_size,
        cells_per_column=10,
        max_segments_per_cell=6,
        max_receptive_field_size=-1,
        w_lr=0.01,
        w_punish=0.0,
        nu_lr=0.01,
        b_lr=0.01,
        init_w=1.0,
        init_nu=0.0,
        init_b=1.0,
        seed=seed
    )

    if log:
        logger = wandb.init(
            project='test_belief_tm', entity='hauska',
            config=dict(
                seed=seed,
                bucket_size=bucket_size
            )
        )
    else:
        logger = None

    density = np.zeros((8, 7))
    lr = 0.02

    hist_dist = np.zeros((8, 7))

    for i in range(1001):
        mpg.reset()
        tm.reset()

        word = []
        surprise = []
        anomaly = []
        confidence = []

        while True:
            letter = mpg.next_state()

            if letter:
                word.append(letter)
            else:
                break

            tm.set_active_columns(encoder.encode(char_to_number[letter]))
            tm.activate_cells(learn=True)

            surprise.append(min(200.0, tm.surprise))
            anomaly.append(tm.anomaly)
            confidence.append(tm.confidence)

            tm.activate_dendrites()
            tm.predict_cells()

            letter_dist = np.prod(tm.column_probs.reshape((-1, bucket_size)).T, axis=0)
            density[mpg.current_state] += lr * (letter_dist - density[mpg.current_state])

            pred_columns_dense = np.zeros(tm.n_columns)
            pred_columns_dense[tm.predicted_columns] = 1
            predicted_letters = np.prod(pred_columns_dense.reshape((-1, bucket_size)).T, axis=0)
            hist_dist[mpg.current_state] += lr * (predicted_letters - hist_dist[mpg.current_state])

        if logger is not None:
            logger.log({
                'surprise': np.array(surprise)[1:].mean(),
                'anomaly': np.array(anomaly)[1:].mean(),
                'confidence': np.array(confidence)[1:].mean()
            }, step=i)

            if i % update_rate == 0:
                def format_fn(tick_val, tick_pos):
                    if int(tick_val) in numer_to_char.keys():
                        return numer_to_char[int(tick_val)]
                    else:
                        return ''

                fig, ax = plt.subplots(2, sharex=True)
                ax[0].xaxis.set_major_formatter(format_fn)
                ax[0].xaxis.set_major_locator(MaxNLocator(integer=True))
                for x in range(density.shape[0]):
                    ax[0].plot(density[x], label=f'state{x}', linewidth=2, marker='o')
                ax[0].grid()

                ax[1].xaxis.set_major_formatter(format_fn)
                ax[1].xaxis.set_major_locator(MaxNLocator(integer=True))
                for x in range(hist_dist.shape[0]):
                    ax[1].plot(hist_dist[x], linewidth=2, marker='o')
                ax[1].grid()

                fig.legend(loc=7)

                logger.log({f'letter_predictions': wandb.Image(fig)}, step=i)

                plt.close(fig)

    ...


if __name__ == '__main__':
    main()
