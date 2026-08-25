"""Population, speciation and reproduction -- the NEAT loop itself.

The three ideas from the paper that make this work, and that a plain genetic
algorithm on neural nets does not have:

  * innovation numbers, so two genomes can be crossed over meaningfully;
  * speciation, so a new topology is compared against similar genomes instead of
    being wiped out by an older, better tuned one;
  * fitness sharing, so a species that gets big does not take over the pool.
"""

from __future__ import annotations

import json
import random
from statistics import mean

from .config import NeatConfig
from .genome import Genome, InnovationTracker


class Species:
    def __init__(self, key: int, representative: Genome, generation: int):
        self.key = key
        self.representative = representative
        self.members: list[Genome] = []
        self.created = generation
        self.last_improved = generation
        self.best_fitness = -float("inf")
        self.adjusted_fitness = 0.0

    def update(self, generation: int) -> None:
        best = max(g.fitness for g in self.members)
        if best > self.best_fitness:
            self.best_fitness = best
            self.last_improved = generation


class Population:
    def __init__(self, cfg: NeatConfig, seed: int | None = None):
        self.cfg = cfg
        self.rng = random.Random(seed)
        self.tracker = InnovationTracker(cfg.num_inputs, cfg.num_outputs)
        self.generation = 0
        self.compat_threshold = cfg.compat_threshold

        self._next_genome_key = 0
        self._next_species_key = 0

        self.genomes = [self._new_genome() for _ in range(cfg.pop_size)]
        self.species: list[Species] = []
        self.best_genome: Genome | None = None
        self.history: list[dict] = []

    # ---------------------------------------------------------------- helpers
    def _new_genome(self) -> Genome:
        g = Genome.new(self._next_genome_key, self.cfg, self.tracker, self.rng)
        self._next_genome_key += 1
        return g

    def _next_key(self) -> int:
        key = self._next_genome_key
        self._next_genome_key += 1
        return key

    # ------------------------------------------------------------------- run
    def run(self, eval_fitness, generations: int, on_generation=None):
        """``eval_fitness(genomes, config)`` must set ``genome.fitness``."""
        for _ in range(generations):
            eval_fitness(self.genomes, self.cfg)

            gen_best = max(self.genomes, key=lambda g: g.fitness)
            if self.best_genome is None or gen_best.fitness > self.best_genome.fitness:
                self.best_genome = gen_best.copy()

            self.speciate()
            for sp in self.species:
                sp.update(self.generation)

            stats = self._stats(gen_best)
            self.history.append(stats)
            if on_generation is not None and on_generation(self, stats) is False:
                break

            self.genomes = self.reproduce()
            self.generation += 1

        return self.best_genome

    def _stats(self, gen_best: Genome) -> dict:
        nodes, conns = gen_best.size()
        return {
            "generation": self.generation,
            "best": gen_best.fitness,
            "mean": mean(g.fitness for g in self.genomes),
            "all_time_best": self.best_genome.fitness if self.best_genome else 0.0,
            "species": len(self.species),
            "threshold": self.compat_threshold,
            "nodes": nodes,
            "conns": conns,
        }

    # ------------------------------------------------------------ speciation
    def speciate(self) -> None:
        cfg = self.cfg
        unspeciated = set(range(len(self.genomes)))
        members: dict[int, list[Genome]] = {}

        # Each surviving species carries over, represented by whichever member of
        # the new population sits closest to its old representative. A genome can
        # only stand for one species, so it leaves the pool once it is picked.
        for sp in self.species:
            if not unspeciated:
                continue
            i = min(unspeciated, key=lambda i: self.genomes[i].distance(sp.representative, cfg))
            unspeciated.discard(i)
            sp.representative = self.genomes[i]
            members[sp.key] = [self.genomes[i]]

        for i in sorted(unspeciated):
            genome = self.genomes[i]
            home = None
            best_dist = None
            for sp in self.species:
                d = genome.distance(sp.representative, cfg)
                if d < self.compat_threshold and (best_dist is None or d < best_dist):
                    home, best_dist = sp, d
            if home is None:
                # Too different from everything: it founds a species of its own,
                # which the next genomes can then join.
                home = Species(self._next_species_key, genome, self.generation)
                self._next_species_key += 1
                self.species.append(home)
            members.setdefault(home.key, []).append(genome)

        for sp in self.species:
            sp.members = members.get(sp.key, [])
        self.species = [sp for sp in self.species if sp.members]

        # Nudge the threshold so the number of species stays in a workable range.
        if cfg.target_species > 0:
            if len(self.species) > cfg.target_species:
                self.compat_threshold += cfg.compat_threshold_step
            elif len(self.species) < cfg.target_species:
                self.compat_threshold = max(
                    cfg.compat_threshold_min,
                    self.compat_threshold - cfg.compat_threshold_step,
                )

    # ---------------------------------------------------------- reproduction
    def _drop_stagnant(self) -> list[Species]:
        cfg = self.cfg
        ranked = sorted(self.species, key=lambda s: s.best_fitness, reverse=True)
        alive = []
        for i, sp in enumerate(ranked):
            stagnant = (self.generation - sp.last_improved) >= cfg.max_stagnation
            if stagnant and i >= cfg.species_elitism:
                continue
            alive.append(sp)
        return alive

    def reproduce(self) -> list[Genome]:
        cfg = self.cfg
        self.species = self._drop_stagnant()

        if not self.species:
            if cfg.reset_on_extinction:
                self.species = []
                return [self._new_genome() for _ in range(cfg.pop_size)]
            raise RuntimeError("all species went extinct")

        # --- fitness sharing -------------------------------------------------
        # Fitness is normalised to [0, 1] over the whole population, then
        # averaged inside each species. A species therefore gets offspring for
        # being good *on average*, not for being numerous.
        all_fit = [g.fitness for g in self.genomes]
        f_min, f_max = min(all_fit), max(all_fit)
        span = max(f_max - f_min, 1e-9)
        for sp in self.species:
            sp.adjusted_fitness = mean((g.fitness - f_min) / span for g in sp.members)

        adj_sum = sum(sp.adjusted_fitness for sp in self.species) or 1e-9
        spawns = []
        for sp in self.species:
            share = sp.adjusted_fitness / adj_sum
            spawns.append(max(cfg.min_species_size, int(round(share * cfg.pop_size))))

        # Round the allocation back onto pop_size.
        while sum(spawns) > cfg.pop_size:
            i = max(range(len(spawns)), key=lambda i: spawns[i])
            if spawns[i] <= cfg.min_species_size:
                break
            spawns[i] -= 1
        while sum(spawns) < cfg.pop_size:
            i = max(range(len(self.species)), key=lambda i: self.species[i].adjusted_fitness)
            spawns[i] += 1

        # --- breed ------------------------------------------------------------
        new_genomes: list[Genome] = []
        for sp, n_spawn in zip(self.species, spawns):
            members = sorted(sp.members, key=lambda g: g.fitness, reverse=True)

            for elite in members[: cfg.elitism]:
                if len(new_genomes) >= cfg.pop_size:
                    break
                new_genomes.append(elite.copy())
                n_spawn -= 1

            n_survivors = max(2, int(round(cfg.survival_threshold * len(members))))
            pool = members[:n_survivors]

            for _ in range(max(0, n_spawn)):
                if len(new_genomes) >= cfg.pop_size:
                    break
                p1 = self.rng.choice(pool)
                if len(pool) > 1 and self.rng.random() < cfg.crossover_prob:
                    p2 = self.rng.choice(pool)
                    if p2.fitness > p1.fitness:
                        p1, p2 = p2, p1
                    child = Genome.crossover(p1, p2, self._next_key(), self.rng)
                else:
                    child = p1.copy()
                    child.key = self._next_key()
                child.mutate(cfg, self.tracker, self.rng)
                child.fitness = 0.0
                new_genomes.append(child)

            sp.representative = self.rng.choice(members)

        while len(new_genomes) < cfg.pop_size:
            new_genomes.append(self._new_genome())

        return new_genomes[: cfg.pop_size]


# --------------------------------------------------------------------- io
def save_genome(genome: Genome, cfg: NeatConfig, path: str, extra: dict | None = None) -> None:
    payload = {"genome": genome.to_dict(), "config": cfg.to_dict(), "extra": extra or {}}
    with open(path, "w") as fh:
        json.dump(payload, fh, indent=1)


def load_genome(path: str) -> tuple[Genome, NeatConfig, dict]:
    with open(path) as fh:
        payload = json.load(fh)
    return (
        Genome.from_dict(payload["genome"]),
        NeatConfig.from_dict(payload["config"]),
        payload.get("extra", {}),
    )
