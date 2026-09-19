"""Paper figures, built from scored runs with Altair.

Two complete sets -- English and Ukrainian -- from one set of builders, so the pair cannot
drift. Nothing here loads a model: everything is derived from `eval/results/*.json`, which is
why figures can be rebuilt at any point in a sweep and re-rendered after it without GPU.
"""

from .data import attribute_level, cell_level, load, run_level, with_baseline
from .labels import LANGUAGES, strategy, t
from .theme import PALETTE, check_font, register

__all__ = [
    "LANGUAGES", "PALETTE", "attribute_level", "cell_level", "check_font",
    "load", "register", "run_level", "strategy", "t", "with_baseline",
]
