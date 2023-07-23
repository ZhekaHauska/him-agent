#  Copyright (c) 2022 Autonomous Non-Profit Organization "Artificial Intelligence Research
#  Institute" (AIRI); Moscow Institute of Physics and Technology (National Research University).
#  All rights reserved.
#
#  Licensed under the AGPLv3 license. See LICENSE in the project root for license information.
"""
    Helper script for running copies of the same command in several processes.
    It's often used for starting wandb agents.
"""

if __name__ == '__main__':
    import subprocess
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--n_processes", type=int, default=1)
    parser.add_argument("-c", "--command", type=str, default='')

    args = parser.parse_args()
    procs = list()
    n_processes = args.n_processes
    command = args.command
    tokens = command.split()
    for core in range(n_processes):
        procs.append(subprocess.Popen(tokens))

    for p in procs:
        p.wait()
