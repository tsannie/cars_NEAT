#!/usr/bin/env python3
"""cars_NEAT -- cars that learn to drive, with NEAT implemented from scratch.

    python main.py train --track grand_prix
    python main.py watch --model models/grand_prix_best.json
    python main.py edit  --out tracks/mytrack.json
    python main.py play  --track curvy
    python main.py tracks
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np

from neatlite import NeatConfig, Population, load_genome, save_genome
from sim.builtin_tracks import BUILTIN, ensure_builtin, resolve
from sim.car import MAX_SPEED, NUM_INPUTS, NUM_OUTPUTS, Fleet, handling_fingerprint
from sim.evolve import Episode, Evaluator, drive_solo
from sim.records import RecordBook, describe, format_lap, timing_lines
from sim.render import Renderer

MODELS_DIR = "models"
TRACKS_DIR = "tracks"
CHASE_SECONDS = 1.2     # road the chase camera keeps ahead of you, at top speed


# --------------------------------------------------------------------- train
def cmd_train(args) -> int:
    track = resolve(args.track, TRACKS_DIR)
    print(f"track  {track}")

    cfg = NeatConfig(
        num_inputs=NUM_INPUTS,
        num_outputs=NUM_OUTPUTS,
        pop_size=args.pop,
        num_hidden=args.hidden,
        target_species=args.species,
        elitism=args.elitism,
    )
    if args.time:
        episode = Episode(max_time=args.time, lap_bonus=args.lap_bonus)
        why = "requested"
    else:
        episode = Episode.for_track(track, laps=args.laps, lap_bonus=args.lap_bonus)
        why = f"~{args.laps:.1f} laps at cruising speed"
    print(f"episode  {episode.max_time:.0f}s per generation "
          f"({why}, {int(episode.max_time / episode.dt)} steps)")

    records = RecordBook()
    print(f"lap      par {format_lap(episode.par_lap(track)).strip()}   "
          f"record {describe(records.best(track))}")

    renderer = None if args.no_render else Renderer(track, caption=f"cars_NEAT - training {track.name}")
    evaluator = Evaluator(track, episode, renderer=renderer, seed=args.seed, records=records)
    pop = Population(cfg, seed=args.seed)

    os.makedirs(MODELS_DIR, exist_ok=True)
    out = args.out or os.path.join(MODELS_DIR, f"{track.name}_best.json")

    started = time.time()
    best_seen = -float("inf")

    # A quick exploratory run must not overwrite the result of a long one. Only
    # compare against a previous model trained on the same track for the same
    # duration, since fitness means nothing across different settings.
    if os.path.exists(out) and not args.overwrite:
        try:
            _, _, previous = load_genome(out)
            same_setup = (
                previous.get("track", {}).get("points") == track.to_dict()["points"]
                and abs(float(previous.get("max_time", -1)) - episode.max_time) < 1e-6
                # Change the handling and old scores stop meaning anything, so
                # the guard has to step aside rather than block every new run.
                and previous.get("physics") == handling_fingerprint()
                # Same for how a drive is scored: a pace bonus changes what the
                # number is worth even when the car and the circuit are the same.
                and previous.get("scoring") == episode.scoring()
            )
            if same_setup and "fitness" in previous:
                best_seen = float(previous["fitness"])
                print(f"keeping  {out} (scores {best_seen:.0f}); it is only replaced "
                      f"by a better run -- pass --overwrite to ignore it")
        except (OSError, ValueError, KeyError):
            pass

    # Fitness carries a pace bonus as well as distance, so it is no longer a lap
    # count in disguise: laps are reported from the ground actually covered.
    def as_laps(distance):
        return f"{distance / track.length:.2f} laps" if track.closed else f"{distance:.0f} px"

    def on_generation(population, stats):
        nonlocal best_seen
        print(
            f"gen {stats['generation']:4d}  best {stats['best']:9.0f}  "
            f"mean {stats['mean']:8.0f}  all-time {stats['all_time_best']:9.0f}  "
            f"lap {format_lap(evaluator.best_lap).strip():>8s}  "
            f"furthest {as_laps(evaluator.best_run)}  species {stats['species']:3d}  "
            f"net {stats['nodes']}n/{stats['conns']}c  {time.time() - started:6.0f}s",
            flush=True,
        )
        evaluator.stats_line = (
            f"pop {cfg.pop_size}  species {stats['species']}  "
            f"elites kept {min(cfg.elitism, cfg.pop_size) * stats['species']}"
        )
        if stats["all_time_best"] > best_seen:
            best_seen = stats["all_time_best"]
            save_genome(
                population.best_genome, cfg, out,
                extra={"track": track.to_dict(), "generation": stats["generation"],
                       "fitness": stats["all_time_best"], "max_time": episode.max_time,
                       "physics": handling_fingerprint(), "scoring": episode.scoring(),
                       "best_lap": evaluator.best_lap},
            )
        if renderer is not None and renderer.quit_requested:
            return False
        return None

    try:
        pop.run(evaluator, args.generations, on_generation=on_generation)
    except KeyboardInterrupt:
        print("\ninterrupted")

    if pop.best_genome is not None:
        print(f"\nbest fitness {pop.best_genome.fitness:.0f}  "
              f"furthest {as_laps(evaluator.best_run)}  "
              f"fastest lap {format_lap(evaluator.best_lap).strip()} -> {out}")
        print(f"record   {track.name}: {describe(records.best(track))}")
        if renderer is not None and not renderer.quit_requested and not args.no_replay:
            renderer.quit_requested = False
            renderer.enabled = True
            drive_solo(track, pop.best_genome, cfg, renderer, episode, records=records,
                       label=f"best of {pop.generation} generations",
                       detail=f"gen {pop.generation}")
    if renderer is not None:
        renderer.close()
    return 0


# --------------------------------------------------------------------- watch
def cmd_watch(args) -> int:
    path = args.model
    if not os.path.exists(path):
        print(f"no model at {path}", file=sys.stderr)
        return 1
    genome, cfg, extra = load_genome(path)

    if args.track:
        track = resolve(args.track, TRACKS_DIR)
    elif "track" in extra:
        from sim.track import Track
        track = Track.from_dict(extra["track"])
    else:
        track = resolve("grand_prix", TRACKS_DIR)

    print(f"{path}: fitness {extra.get('fitness', genome.fitness):.0f} "
          f"from generation {extra.get('generation', '?')}  {genome}")
    if extra.get("best_lap"):
        print(f"trained fastest lap {format_lap(extra['best_lap']).strip()}")

    records = RecordBook()
    print(f"record   {track.name}: {describe(records.best(track))}")

    renderer = Renderer(track, caption=f"cars_NEAT - {os.path.basename(path)}")
    if args.time:
        episode = Episode(max_time=args.time)
    elif "max_time" in extra:
        episode = Episode(max_time=float(extra["max_time"]))  # replay it as trained
    else:
        episode = Episode.for_track(track)
    while not renderer.quit_requested:
        drive_solo(track, genome, cfg, renderer, episode, records=records,
                   label=os.path.basename(path),
                   detail=f"gen {extra.get('generation', '?')}")
        if args.once:
            break
    print(f"record   {track.name}: {describe(records.best(track))}")
    renderer.close()
    return 0


# ---------------------------------------------------------------------- play
def cmd_play(args) -> int:
    import pygame

    track = resolve(args.track, TRACKS_DIR)
    renderer = Renderer(track, caption="cars_NEAT - manual")
    # [L] follows the car. On a circuit that fits on screen that is just a
    # recentring, but endurance is 4500px across: the car is 7px, the corners
    # are unreadable, and driving it means chasing a dot. Following there zooms
    # to the framing the small circuits already give you -- the car is centred,
    # so half the span is the road ahead.
    renderer.chase_span = 2 * MAX_SPEED * CHASE_SECONDS
    # No stall timeout here: sitting still, reversing or turning round is a
    # perfectly reasonable thing for a person to do, and being teleported back to
    # the line for it is baffling.
    def new_fleet():
        return Fleet(track, 1, retire_stalled=False)

    records = RecordBook()
    record = records.best(track)

    fleet = new_fleet()
    dt = 1.0 / 60.0
    renderer.fps_cap = 60
    t = 0.0
    crashes = 0
    best = 0.0
    notice = 0.0
    # The clock survives a restart: the point of driving it yourself is to chase
    # the machine's time, and that is a best-of-the-session, not of one attempt.
    best_lap = float("inf")
    timed = float("inf")   # the lap the clock last stopped on
    lap_notice = 0.0
    lap_banner = None

    while not renderer.quit_requested:
        renderer.pump()
        keys = pygame.key.get_pressed()
        steer = (keys[pygame.K_RIGHT] or keys[pygame.K_d]) - (keys[pygame.K_LEFT] or keys[pygame.K_a])
        throttle = (keys[pygame.K_UP] or keys[pygame.K_w]) - (keys[pygame.K_DOWN] or keys[pygame.K_s])

        if not fleet.alive[0]:
            crashes += 1
            notice = 1.6
            best = max(best, float(fleet.distance[0]))
            fleet = new_fleet()
            t = 0.0
        elif keys[pygame.K_r]:
            best = max(best, float(fleet.distance[0]))
            fleet = new_fleet()
            t = 0.0

        idx, _ = fleet.observe()
        fleet.step(idx, np.array([float(steer)]), np.array([float(throttle)]), dt)
        t += dt
        notice = max(0.0, notice - dt)
        lap_notice = max(0.0, lap_notice - dt)

        lap = float(fleet.last_lap[0])
        if lap != timed:            # came over the line, or a restart cleared the clock
            timed = lap
            if lap < float("inf"):
                best_lap = min(best_lap, lap)
                beat = records.submit(track, lap, by="you", detail="manual drive")
                if beat:
                    record = records.best(track)
                title = "NEW LAP RECORD" if beat else f"lap {int(fleet.laps[0])}"
                lap_banner = (f"{title}   {format_lap(lap).strip()}", "hot" if beat else None)
                lap_notice = 2.5

        load = float(fleet.grip_load[0])
        lines = [
            "manual drive",
            f"dist    {fleet.distance[0]:7.0f} px ({fleet.distance[0] / track.length:.2f} lap)",
            f"speed   {fleet.speed[0]:7.0f} px/s",
            f"tyres   {load * 100:6.0f} %" + ("  UNDERSTEER" if load > 1.0 else ""),
        ]
        lines += timing_lines(float(fleet.lap_time[0]), best_lap, record, label="best")
        lines += [
            f"t       {t:5.1f}s   best {best:.0f} px   crashes {crashes}",
            "arrows / WASD   [R] restart   [ESC] quit",
            f"[L] follow {'on' if renderer.follow else 'off'}   wheel zoom   "
            f"drag pan   [F] whole circuit",
        ]

        banner = None
        if lap_notice > 0.0:
            banner = lap_banner
        elif notice > 0.0:
            banner = "off the track - back to the line"
        renderer.draw(fleet, lines, leader=0, banner=banner)
    print(f"fastest lap {format_lap(best_lap).strip()}")
    print(f"record   {track.name}: {describe(records.best(track))}")
    renderer.close()
    return 0


# ---------------------------------------------------------------------- edit
def cmd_edit(args) -> int:
    from sim.editor import edit_track

    base = resolve(args.load, TRACKS_DIR) if args.load else None
    out = args.out
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    track = edit_track(out, base)
    if track is None:
        print("cancelled")
        return 1
    print(f"saved {track} -> {out}")
    print(f"train on it with:  python main.py train --track {out}")
    return 0


# -------------------------------------------------------------------- tracks
def cmd_tracks(args) -> int:
    ensure_builtin(TRACKS_DIR)
    records = RecordBook()

    if args.forget:
        gone = records.forget(args.forget)
        print(f"{'forgot' if gone else 'no record for'} {args.forget}")
        return 0

    def row(label: str, track) -> None:
        record = records.best(track)
        shape = "lap " if track.closed else "run "
        print(f"  {label:12s} {track.length:6.0f} px  {shape}{describe(record)}")

    print(f"  {'circuit':12s} {'length':>9s}  fastest")
    for name in sorted(BUILTIN):
        row(name, resolve(name, TRACKS_DIR))
    others = sorted(
        f for f in os.listdir(TRACKS_DIR)
        if f.endswith(".json") and f[:-5] not in BUILTIN
    )
    for f in others:
        row(f[:-5], resolve(f, TRACKS_DIR))
    print(f"\n  times live in {records.path}; a redrawn circuit or a change to the "
          "car's handling retires the old one")
    return 0


# ----------------------------------------------------------------------- cli
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="command", required=True)

    t = sub.add_parser("train", help="evolve a population of drivers")
    t.add_argument("--track", default="grand_prix", help="builtin name or path to a .json track")
    t.add_argument("--generations", "-g", type=int, default=150)
    t.add_argument("--pop", type=int, default=120, help="population size")
    t.add_argument("--time", type=float,
                   help="simulated seconds per generation (default: scaled to the track)")
    t.add_argument("--laps", type=float, default=2.5,
                   help="laps a generation should last, when --time is not given")
    t.add_argument("--hidden", type=int, default=0, help="hidden nodes in the seed genomes")
    t.add_argument("--species", type=int, default=8, help="species count the threshold aims for")
    t.add_argument("--lap-bonus", type=float, default=180.0,
                   help="fitness per second a lap comes in under par; 0 scores distance only")
    t.add_argument("--elitism", type=int, default=2,
                   help="genomes carried over untouched per species; raising it keeps "
                        "far more of the pack alive but explores less")
    t.add_argument("--seed", type=int, default=0)
    t.add_argument("--no-render", action="store_true", help="train without a window (fastest)")
    t.add_argument("--no-replay", action="store_true", help="skip the replay of the winner")
    t.add_argument("--overwrite", action="store_true",
                   help="save over an existing model even if it scored better")
    t.add_argument("--out", help="where to save the best genome")
    t.set_defaults(func=cmd_train)

    w = sub.add_parser("watch", help="replay a saved genome")
    w.add_argument("--model", "-m", default=os.path.join(MODELS_DIR, "grand_prix_best.json"))
    w.add_argument("--track", help="override the track the model was trained on")
    w.add_argument("--time", type=float,
                   help="seconds to drive (default: the duration it was trained with)")
    w.add_argument("--once", action="store_true")
    w.set_defaults(func=cmd_watch)

    pl = sub.add_parser("play", help="drive the car yourself")
    pl.add_argument("--track", default="grand_prix")
    pl.set_defaults(func=cmd_play)

    e = sub.add_parser("edit", help="draw a track with the mouse")
    e.add_argument("--out", "-o", default=os.path.join(TRACKS_DIR, "custom.json"))
    e.add_argument("--load", "-l", help="start from an existing track")
    e.set_defaults(func=cmd_edit)

    ls = sub.add_parser("tracks", help="list the circuits and their lap records")
    ls.add_argument("--forget", metavar="NAME", help="drop the lap record of a circuit")
    ls.set_defaults(func=cmd_tracks)
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
