"""Configuration for the NEAT engine.

Everything the algorithm needs is in one dataclass so an experiment can be
tweaked from Python instead of an .ini file.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass
class NeatConfig:
    # --- topology -------------------------------------------------------
    num_inputs: int
    num_outputs: int
    num_hidden: int = 0
    # "full"    : every input wired to every output
    # "partial" : each input/output pair wired with probability initial_conn_prob
    # "none"    : start with no connection at all
    initial_connection: str = "full"
    initial_conn_prob: float = 0.5

    # --- population -----------------------------------------------------
    pop_size: int = 120
    elitism: int = 2  # genomes copied untouched from each big enough species
    survival_threshold: float = 0.3  # top fraction of a species allowed to breed
    min_species_size: int = 2
    crossover_prob: float = 0.75  # otherwise the child is a mutated clone

    # --- weights & biases ------------------------------------------------
    weight_init_stdev: float = 1.0
    weight_mutate_rate: float = 0.8
    weight_mutate_power: float = 0.5
    weight_replace_rate: float = 0.1
    weight_max: float = 8.0

    bias_init_stdev: float = 1.0
    bias_mutate_rate: float = 0.7
    bias_mutate_power: float = 0.5
    bias_replace_rate: float = 0.1
    bias_max: float = 8.0

    # --- structural mutation ---------------------------------------------
    conn_add_prob: float = 0.5
    conn_delete_prob: float = 0.2
    node_add_prob: float = 0.2
    node_delete_prob: float = 0.1
    enabled_mutate_rate: float = 0.01

    # --- speciation -------------------------------------------------------
    compat_disjoint_coeff: float = 1.0
    compat_weight_coeff: float = 0.5
    compat_threshold: float = 3.0
    # The threshold drifts to keep roughly this many species alive. 0 disables it.
    target_species: int = 8
    compat_threshold_step: float = 0.1
    compat_threshold_min: float = 0.5

    max_stagnation: int = 20
    species_elitism: int = 2  # never kill the N best species for stagnation
    reset_on_extinction: bool = True

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "NeatConfig":
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in data.items() if k in known})
