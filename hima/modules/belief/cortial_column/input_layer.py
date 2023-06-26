#  Copyright (c) 2023 Autonomous Non-Profit Organization "Artificial Intelligence Research
#  Institute" (AIRI); Moscow Institute of Physics and Technology (National Research University).
#  All rights reserved.
#
#  Licensed under the AGPLv3 license. See LICENSE in the project root for license information.
from hima.modules.belief.cortial_column.layer import Layer


class Encoder(Layer):
    def __init__(
            self,
            *args,
            **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.encoded_observation = None

    def encode(
            self,
            observation,
            hidden_prediction=None,
            decoded_observation=None,
            learn=True
    ):
        """
            observation is a dense pattern or messages
        """
        self.set_context_messages(observation)
        self.predict()
        self.encoded_observation = self.prediction_columns.copy()

        encoded_state = self._sample_cells(
            self.encoded_observation.reshape(
                self.n_hidden_vars, -1
            )
        )

        if learn:
            # self reinforce
            self.observe(encoded_state)

            if hidden_prediction is not None or decoded_observation is not None:
                obs = self._sample_cells(
                    hidden_prediction.reshape(
                        self.n_hidden_vars, -1
                    )
                )
                # align encoder with transition matrix
                if hidden_prediction is not None:
                    self.set_context_messages(observation)
                    self.predict()
                    self.observe(obs)

                # adjust encoder to decoder
                if decoded_observation is not None:
                    self.set_context_messages(decoded_observation)
                    self.predict()
                    self.observe(obs)

        return encoded_state


class Decoder(Layer):
    def __init__(
            self,
            *args,
            **kwargs
    ):
        super().__init__(*args, **kwargs)

    def decode(self, hidden_messages, observation=None, learn=True):
        self.set_context_messages(hidden_messages)
        self.predict()
        decoded_observation = self.prediction_columns.copy()

        if learn:
            assert observation is not None
            self.observe(observation)

        return decoded_observation
