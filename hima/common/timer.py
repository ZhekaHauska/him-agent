#  Copyright (c) 2022 Autonomous Non-Profit Organization "Artificial Intelligence Research
#  Institute" (AIRI); Moscow Institute of Physics and Technology (National Research University).
#  All rights reserved.
#
#  Licensed under the AGPLv3 license. See LICENSE in the project root for license information.
import timeit

timer = timeit.default_timer


def print_with_timestamp(text: str, start_time: float):
    elapsed_sec = timer() - start_time
    if elapsed_sec < 1:
        time_format = '5.3f'
    elif elapsed_sec < 10:
        time_format = '5.2f'
    elif elapsed_sec < 1000:
        time_format = '5.1f'
    else:
        time_format = '5.0f'
    print(f'[{elapsed_sec:{time_format}}] {text}')
