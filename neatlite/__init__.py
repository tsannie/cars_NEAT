"""A small, dependency-free NEAT implementation.

Written from the Stanley & Miikkulainen paper rather than wrapping the
``neat-python`` package, so every moving part of the algorithm is visible and
hackable.
"""

from .config import NeatConfig
from .genome import Genome, InnovationTracker
from .network import FeedForwardNetwork
from .population import Population, load_genome, save_genome

__all__ = [
    "NeatConfig",
    "Genome",
    "InnovationTracker",
    "FeedForwardNetwork",
    "Population",
    "save_genome",
    "load_genome",
]
