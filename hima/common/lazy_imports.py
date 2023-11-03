#  Copyright (c) 2022 Autonomous Non-Profit Organization "Artificial Intelligence Research
#  Institute" (AIRI); Moscow Institute of Physics and Technology (National Research University).
#  All rights reserved.
#
#  Licensed under the AGPLv3 license. See LICENSE in the project root for license information.

import importlib.util
import sys


def lazy_import(name, package=None):
    module = sys.modules.get(name, None)
    if module is not None:
        return module

    spec = importlib.util.find_spec(name, package=package)
    loader = importlib.util.LazyLoader(spec.loader)
    spec.loader = loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    loader.exec_module(module)
    return module

# Here I collect the libraries which are particularly slow at import
# and should be lazy loaded if possible:
#
#   - [FIXED] matplotlib: bare matplotlib and matplotlib.pyplot are fast.
#       NOTE: DO NOT use lazy loading for matplotlib as it cannot be lazily loaded :( either
#       make it eagerly or make local imports.
#
#       IMPORTANT: Still do not forget using our common func `turn_off_gui_for_matplotlib` to
#       turn on a headless mode, when you use plt solely for wandb plotting (not for popping GUI
#       window). Do it either at the start of the script or when you're sure you will be using plt.

#   - [FIXED] wandb: was very slow (at least on M1 Mac) ~4-8sec. They probably removed a new
#       version import-time check... Thank God!
#       You can still lazy import it for a good habit training :)
#
#   - Seaborn: ~1-1.5sec. Still slow import, use lazy import for declaration.
#   - [FIXED] Torch: was also ~1.5sec overhead. Now seems to have been reduced to sub-sec.
#   - [FIXED] pandas
