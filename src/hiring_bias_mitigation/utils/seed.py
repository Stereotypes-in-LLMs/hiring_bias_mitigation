"""Seeding. Every stage takes `seed` from its config so a run is reproducible end to end.

Note what this does NOT buy you: with a sampling decoder, seeding makes the *sequence* of
samples reproducible, not the decision deterministic. Section 7 of the audit paper lists
"single sampled run" as a limitation precisely because seeding does not separate model bias
from sampling variance -- only repeated sampling does (`n_samples` in the audit config).
"""

import os
import random

import numpy as np


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass
