#  Copyright (c) 2022 Autonomous Non-Profit Organization "Artificial Intelligence Research
#  Institute" (AIRI); Moscow Institute of Physics and Technology (National Research University).
#  All rights reserved.
#
#  Licensed under the AGPLv3 license. See LICENSE in the project root for license information.


def multiplicative_loss(smae, pmf_coverage):
    # Google it: y = 0.25 * ((1 - x) / (1.4 * x))^0.5 + 0.75
    # == 1 around 0.41 pmf coverage — it's a target value
    pmf_weight = (1 - pmf_coverage) / (1.4 * pmf_coverage)
    # smooth with sqrt and shift it up
    pmf_weight = 0.25 * (pmf_weight ** 0.55) + 0.75

    # == 1 at smae = 0.08. At ~0.2 SMAE we get almost garbage
    smae_weight = (smae / 0.06)**1.5

    return pmf_weight * smae_weight
