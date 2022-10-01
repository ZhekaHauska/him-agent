#  Copyright (c) 2022 Autonomous Non-Profit Organization "Artificial Intelligence Research
#  Institute" (AIRI); Moscow Institute of Physics and Technology (National Research University).
#  All rights reserved.
#
#  Licensed under the AGPLv3 license. See LICENSE in the project root for license information.

import matplotlib.pyplot as plt
from PIL import Image
import numpy as np


def heatmap(image, cmap=None):
    cm = plt.get_cmap(cmap)
    colored = cm(image)
    return Image.fromarray((colored[:, :, :3] * 255).astype(np.uint8))
