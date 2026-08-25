#!/usr/bin/env python3
"""Is any circuit passable without ever braking?

The claim "you cannot get round flat out" is not something a single hand-written
driver can settle: a pure-pursuit pilot crashes because *it* steers badly, and an
evolved one can find a smooth wide line that a naive controller never would. So
the check is to evolve the steering itself, twice:

  * throttle nailed to +1, only steering evolves -- how far can a car that never
    brakes get?
  * throttle under the network's control -- the control, showing the circuit is
    actually drivable.

A circuit passes if the first number stays well under a lap while the second
clears it. (An open track counts as cleared at 0.98 of its length, where the
finish line is.)

    python tools/flat_out_check.py [--generations 45] [--pop 100] [track ...]
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from neatlite import FeedForwardNetwork, NeatConfig, Population  # noqa: E402
from sim.builtin_tracks import BUILTIN, resolve  # noqa: E402
from sim.car import NUM_INPUTS, NUM_OUTPUTS, Fleet  # noqa: E402
from sim.evolve import Episode  # noqa: E402


def evolve(track, generations: int, pop_size: int, forced_throttle=None, seed: int = 0):
    episode = Episode.for_track(track)
    cfg = NeatConfig(num_inputs=NUM_INPUTS, num_outputs=NUM_OUTPUTS, pop_size=pop_size)
    population = Population(cfg, seed=seed)
    target = None if track.closed else track.finish_distance

    def evaluate(genomes, config):
        nets = [FeedForwardNetwork.create(g, config) for g in genomes]
        fleet = Fleet(track, len(genomes))
        t = 0.0
        while t < episode.max_time and fleet.any_alive():
            idx, obs = fleet.observe()
            rows = obs.tolist()
            out = np.array([nets[i].activate(rows[j]) for j, i in enumerate(idx)])
            throttle = (np.full(idx.size, forced_throttle) if forced_throttle is not None
                        else out[:, 1])
            fleet.step(idx, out[:, 0], throttle, episode.dt, target)
            t += episode.dt
        for i, genome in enumerate(genomes):
            genome.fitness = float(fleet.distance[i])

    population.run(evaluate, generations)
    return population.best_genome.fitness / track.length


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("tracks", nargs="*", default=sorted(BUILTIN))
    parser.add_argument("--generations", "-g", type=int, default=45)
    parser.add_argument("--pop", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", help="also write the results to this JSON file")
    args = parser.parse_args(argv)

    print(f"{args.generations} generations, pop {args.pop}\n")
    print(f"{'track':12s} {'throttle nailed to +1':>22s} {'throttle free':>15s}   verdict")
    failures = 0
    results = {}
    for name in args.tracks:
        track = resolve(name)
        forced = evolve(track, args.generations, args.pop, 1.0, args.seed)
        free = evolve(track, args.generations, args.pop, None, args.seed)
        # An open track is "done" at the finish line, which sits at 0.98 of its
        # length, so a closed-lap threshold would mark a finisher as a failure.
        done = 1.0 if track.closed else 0.98
        ok = forced < done <= free
        failures += not ok
        results[name] = {"flat_out_laps": forced, "free_laps": free, "ok": ok}
        print(f"{name:12s} {forced:17.2f} laps {free:10.2f} laps   "
              f"{'ok' if ok else 'FLAT OUT GETS ROUND'}", flush=True)
        if args.out:  # written as we go, so a long run can be watched
            with open(args.out, "w") as fh:
                json.dump({"generations": args.generations, "pop": args.pop,
                           "seed": args.seed, "tracks": results}, fh, indent=1)
    print("\nall circuits require braking" if not failures
          else f"\n{failures} circuit(s) passable without braking")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
