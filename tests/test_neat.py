"""Checks that the from-scratch NEAT engine and the simulation actually work.

Run with:  .venv/bin/python -m pytest tests -q
or just:   .venv/bin/python tests/test_neat.py
"""

from __future__ import annotations

import math
import os
import random
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from neatlite import FeedForwardNetwork, NeatConfig, Population  # noqa: E402
from neatlite.genome import Genome, InnovationTracker, creates_cycle  # noqa: E402
from sim.builtin_tracks import BUILTIN, resolve  # noqa: E402
from sim.editor import Camera as EditorCamera  # noqa: E402
from sim.render import Camera as RenderCamera  # noqa: E402
from sim.track import Track  # noqa: E402
from sim.car import (  # noqa: E402
    BRAKE_GRIP, MAX_GRIP, MAX_SPEED, NUM_INPUTS, NUM_OUTPUTS, STALL_MARGIN,
    STALL_TIMEOUT, TURN_RATE, Fleet,
)
from sim.pilot import ReferencePilot, drive  # noqa: E402
from sim.evolve import Episode, Evaluator  # noqa: E402
from sim.records import RecordBook, format_lap  # noqa: E402

XOR = [((0.0, 0.0), 0.0), ((0.0, 1.0), 1.0), ((1.0, 0.0), 1.0), ((1.0, 1.0), 0.0)]


def _xor_fitness(genomes, cfg):
    for g in genomes:
        net = FeedForwardNetwork.create(g, cfg)
        g.fitness = 4.0 - sum((net.activate(xi)[0] - xo) ** 2 for xi, xo in XOR)


def test_xor_is_solved():
    """The classic NEAT benchmark: it needs a hidden node to solve XOR at all."""
    cfg = NeatConfig(num_inputs=2, num_outputs=1, pop_size=150, target_species=6)
    for seed in range(2):
        pop = Population(cfg, seed=seed)
        pop.run(_xor_fitness, 150, on_generation=lambda p, s: False if s["best"] > 3.9 else None)
        assert pop.best_genome.fitness > 3.9, f"seed {seed} scored {pop.best_genome.fitness}"


def test_networks_stay_feed_forward():
    """Heavy structural mutation must never close a loop."""
    cfg = NeatConfig(num_inputs=3, num_outputs=2, conn_add_prob=0.9, node_add_prob=0.6,
                     conn_delete_prob=0.3, node_delete_prob=0.2)
    tracker = InnovationTracker(cfg.num_inputs, cfg.num_outputs)
    rng = random.Random(0)
    g = Genome.new(0, cfg, tracker, rng)
    for step in range(400):
        g.mutate(cfg, tracker, rng)
        if step % 20:  # the full check is quadratic in the number of genes
            continue
        for src, dst in g.conns:
            others = {k for k in g.conns if k != (src, dst)}
            assert not creates_cycle(others, src, dst), f"cycle via {src}->{dst}"
    assert FeedForwardNetwork.create(g, cfg).activate([0.1, 0.2, 0.3]) is not None


def test_crossover_keeps_the_fitter_topology():
    cfg = NeatConfig(num_inputs=2, num_outputs=1)
    tracker = InnovationTracker(cfg.num_inputs, cfg.num_outputs)
    rng = random.Random(1)
    a = Genome.new(0, cfg, tracker, rng)
    b = Genome.new(1, cfg, tracker, rng)
    for _ in range(30):
        a.mutate(cfg, tracker, rng)
    child = Genome.crossover(a, b, 2, rng)
    assert set(child.conns) == set(a.conns)
    assert set(child.nodes) == set(a.nodes)


def test_genome_survives_a_round_trip():
    cfg = NeatConfig(num_inputs=4, num_outputs=2)
    tracker = InnovationTracker(cfg.num_inputs, cfg.num_outputs)
    rng = random.Random(2)
    g = Genome.new(0, cfg, tracker, rng)
    for _ in range(50):
        g.mutate(cfg, tracker, rng)
    clone = Genome.from_dict(g.to_dict())
    x = [0.3, -0.2, 0.8, 0.1]
    a = FeedForwardNetwork.create(g, cfg).activate(x)
    b = FeedForwardNetwork.create(clone, cfg).activate(x)
    assert a == b


def test_track_geometry_and_sensors():
    for name in BUILTIN:
        track = resolve(name)
        assert track.length > 500
        assert track.on_track(track.start_pos[None, :])[0], f"{name}: start is off track"
        # No corner may be tighter than half the width: past that the offset
        # walls fold back through the tarmac and open a shortcut.
        assert not track.tight_corners().any(), (
            f"{name}: min turn radius {track.turn_radius().min():.0f} "
            f"< half width {track.width / 2:.0f}"
        )
        # With that guaranteed, both walls really are solid barriers.
        for wall, outward in ((track.left, 1.0), (track.right, -1.0)):
            probe = wall + track.normal * outward * 4.0
            assert not track.on_track(probe).any(), f"{name}: wall is not solid"

    track = resolve("oval")
    origin = np.repeat(track.start_pos[None, :], 4, axis=0)
    ang = track.start_angle + np.array([0.0, math.pi / 2, math.pi, -math.pi / 2])
    dirs = np.stack([np.cos(ang), np.sin(ang)], axis=1)
    d = track.raycast(origin, dirs, 400.0)
    assert np.all(d > 0)
    # Sideways rays hit a wall roughly half a track-width away.
    assert abs(d[1] - track.width / 2) < 12, d
    assert abs(d[3] - track.width / 2) < 12, d


def test_large_track_builds_quickly_and_correctly():
    """A big map must stay cheap to bake, and the grid must match the geometry.

    The grid used to be built by asking every cell which centre-line sample was
    nearest, which is cells x samples and took seconds on a large circuit.
    """
    ang = np.linspace(0, 2 * math.pi, 16, endpoint=False)
    radius = 1.0 + 0.18 * np.sin(3 * ang)
    pts = np.stack([np.cos(ang) * 1800 * radius + 2200,
                    np.sin(ang) * 1150 * radius + 1400], axis=1)

    start = time.perf_counter()
    track = Track(pts, width=150.0, name="big")
    elapsed = time.perf_counter() - start
    assert track.length > 9000, track.length
    assert elapsed < 1.0, f"building a {track.length:.0f}px track took {elapsed:.2f}s"

    # The drivable area is "within half a width of the centre line" -- check the
    # baked grid says the same thing, away from the half-cell boundary.
    rng = np.random.default_rng(0)
    lo, hi = track.center.min(axis=0) - 300, track.center.max(axis=0) + 300
    probe = rng.uniform(lo, hi, size=(4000, 2))
    d = np.sqrt(((probe[:, None, :] - track.center[None, :, :]) ** 2).sum(axis=2)).min(axis=1)
    clear = np.abs(d - track.width / 2) > track.cell  # skip cells straddling the edge
    assert (track.on_track(probe)[clear] == (d[clear] <= track.width / 2)).all()


def test_cameras_round_trip_and_anchor_zoom():
    """Pan/zoom must not drift, or points would crawl away as you edit."""
    for camera in (EditorCamera((1280, 720)), RenderCamera(resolve("oval"), (1280, 720))):
        world = np.array([[-1500.0, 2300.0], [0.0, 0.0], [8000.0, -400.0]])

        camera.pan((137.0, -42.0))
        camera.zoom_at((900.0, 500.0), 0.3)
        back = camera.to_world(camera.to_screen(world))
        assert np.abs(back - world).max() < 1e-6, np.abs(back - world).max()

        # Zooming must keep whatever sits under the cursor exactly under it.
        cursor = (412.0, 631.0)
        before = camera.to_world(cursor)
        camera.zoom_at(cursor, 1.15)
        assert np.abs(camera.to_world(cursor) - before).max() < 1e-6

    # Fitting brings the whole track inside the window.
    track = resolve("endurance")
    camera = RenderCamera(track, (1280, 720))
    camera.zoom_at((0.0, 0.0), 4.0)
    camera.fit_track(track)
    screen = camera.to_screen(np.vstack([track.left, track.right]))
    assert (screen >= 0).all() and (screen[:, 0] <= 1280).all() and (screen[:, 1] <= 720).all()


def test_every_circuit_needs_braking():
    """Flat out must not be a viable strategy, and braking must be enough.

    Both halves matter. The first is the point of the tyre model: with no grip
    limit the throttle output was pinned at +1 for a whole lap and the network
    only ever learned to steer. The second guards the obvious overcorrection --
    it is easy to make a circuit that simply cannot be driven.

    This uses the reference pilot, which is deterministic and fast. It is a
    necessary condition, not a proof: an *evolved* steering policy is a stronger
    flat-out driver than pure pursuit, so `tools/flat_out_check.py` re-runs the
    question by evolving the steering itself.
    """
    for name in BUILTIN:
        track = resolve(name)
        episode = Episode.for_track(track)

        braking = drive(track, Fleet(track, 1), episode.dt, episode.max_time)
        assert braking["alive"] or braking["finished"], (
            f"{name}: not drivable even when braking ({braking['laps']:.2f} laps)"
        )
        assert braking["braking_fraction"] > 0.1, (
            f"{name}: the reference pilot barely brakes ({braking['braking_fraction']:.0%})"
        )

        flat = drive(track, Fleet(track, 1), episode.dt, episode.max_time, flat_out=True)
        assert not (flat["alive"] or flat["finished"]), (
            f"{name}: got round flat out ({flat['laps']:.2f} laps)"
        )


def test_the_tyres_cap_what_the_driver_asks_for():
    """Steering hard at speed must understeer rather than obey."""
    track = resolve("oval")
    fleet = Fleet(track, 1)
    fleet.speed[0] = MAX_SPEED

    heading = fleet.angle[0]
    fleet.step(np.array([0]), np.array([1.0]), np.array([1.0]), 1 / 30)
    achieved = abs(fleet.angle[0] - heading) * 30.0  # rad/s actually delivered

    assert fleet.grip_load[0] > 1.0, "full lock at top speed should overload the tyres"
    assert achieved < TURN_RATE * 0.5, (
        f"asked for {TURN_RATE:.1f} rad/s at {MAX_SPEED:.0f} px/s and got {achieved:.1f}"
    )

    # The same input at low speed is within the budget, so it is obeyed in full.
    slow = Fleet(track, 1)
    slow.speed[0] = 60.0
    heading = slow.angle[0]
    slow.step(np.array([0]), np.array([1.0]), np.array([0.0]), 1 / 30)
    assert slow.grip_load[0] <= 1.0
    assert abs(slow.angle[0] - heading) * 30.0 > TURN_RATE * 0.3


def test_the_stall_rule_retires_the_stuck_not_the_slow():
    """Progress is measured since the clock last reset, not per step.

    Comparing against the previous step means demanding STALL_MARGIN of progress
    within one dt -- a hidden minimum speed of 360 px/s at 30Hz, which quietly
    retired any car that slowed down for a corner while sitting perfectly on the
    track. It fought directly against the braking the tyre model exists to force.
    """
    track = resolve("oval")
    episode = Episode.for_track(track)

    # A deliberately slow lap: well under the old rule's implied floor, but
    # progressing steadily and never off the tarmac.
    crawling = drive(track, Fleet(track, 1), episode.dt, episode.max_time, pace=0.35)
    pace = crawling["distance"] / max(crawling["time"], 1e-9)
    assert pace < STALL_MARGIN / episode.dt, (
        f"this lap is not slow enough to be a test ({pace:.0f} px/s)"
    )
    assert crawling["alive"] or crawling["finished"], (
        f"a car averaging {pace:.0f} px/s was retired after {crawling['time']:.1f}s"
    )

    stuck = Fleet(track, 1)
    for _ in range(int(2 * STALL_TIMEOUT * 30)):
        stuck.step(np.array([0]), np.zeros(1), np.zeros(1), 1 / 30)
    assert not stuck.any_alive(), "a car that never moved should be retired"
    assert not stuck.crashed[0], "it was retired for stalling, not for crashing"


def test_sitting_still_is_only_fatal_while_training():
    """The stall timeout must not apply to a person holding no keys.

    Retiring a car that has stopped progressing keeps a generation from burning
    its clock on one that spun and stopped. Applied to manual driving it teleports
    the player back to the line after 2.5s of standing still, on an empty track,
    with no explanation.
    """
    track = resolve("grand_prix")
    controls = (np.array([0]), np.zeros(1), np.zeros(1))

    training = Fleet(track, 1)
    for _ in range(400):  # ~6.7s at 60Hz
        if not training.any_alive():
            break
        training.step(*controls, 1 / 60)
    assert not training.any_alive(), "a stuck genome should be retired"
    assert not training.crashed[0], "it was retired for stalling, not for crashing"

    manual = Fleet(track, 1, retire_stalled=False)
    for _ in range(400):
        manual.step(*controls, 1 / 60)
    assert manual.any_alive(), "a stationary car on clear tarmac must stay alive"
    assert manual.track.on_track(manual.pos)[0]


def test_the_brakes_are_stronger_than_the_cornering_grip():
    """Slowing down and turning draw on separate budgets.

    A single shared budget means the brakes can only be strengthened by handing
    grip back to the corners, which is the opposite of what was wanted. Splitting
    them also matches the car: weight pitches forward under braking and loads the
    tyres that do the stopping.
    """
    track = resolve("oval")

    straight = Fleet(track, 1)
    straight.speed[0] = MAX_SPEED
    before = float(straight.speed[0])
    straight.step(np.array([0]), np.zeros(1), np.array([-1.0]), 1 / 30)
    stopping = (before - float(straight.speed[0])) * 30.0  # px/s^2, drag included

    assert stopping > MAX_GRIP, f"full brakes gave only {stopping:.0f}, under the {MAX_GRIP:.0f} of grip"
    assert stopping < BRAKE_GRIP * 1.3, f"{stopping:.0f} is more than the tyres have"

    # Cornering is still held to the smaller budget.
    turning = Fleet(track, 1)
    turning.speed[0] = MAX_SPEED
    heading = float(turning.angle[0])
    turning.step(np.array([0]), np.array([1.0]), np.zeros(1), 1 / 30)
    delivered = abs(float(turning.angle[0]) - heading) * 30.0
    assert delivered * MAX_SPEED < MAX_GRIP * 1.05, "cornering escaped its budget"


def test_a_car_driving_straight_leaves_the_track():
    track = resolve("grand_prix")
    fleet = Fleet(track, 8)
    for _ in range(200):
        idx, _ = fleet.observe()
        if idx.size == 0:
            break
        fleet.step(idx, np.zeros(idx.size), np.ones(idx.size), 1 / 30)
    assert not fleet.any_alive(), "a car going flat out in a straight line should crash"


def test_evaluation_is_reproducible():
    """Elitism only means something if the same genome scores the same twice."""
    track = resolve("oval")
    cfg = NeatConfig(num_inputs=NUM_INPUTS, num_outputs=NUM_OUTPUTS, pop_size=12)
    tracker = InnovationTracker(NUM_INPUTS, NUM_OUTPUTS)
    rng = random.Random(7)
    genomes = [Genome.new(i, cfg, tracker, rng) for i in range(12)]
    ev = Evaluator(track, Episode(max_time=6.0))
    ev(genomes, cfg)
    first = [g.fitness for g in genomes]
    ev(genomes, cfg)
    assert [g.fitness for g in genomes] == first


def _reference_laps(track, dt: float, seconds: float, pace: float = 0.95):
    """Drive the reference pilot round and hand back every lap it timed."""
    fleet = Fleet(track, 1, retire_stalled=False)
    pilot = ReferencePilot(track, pace=pace)
    laps, counted = [], 0
    for _ in range(int(seconds / dt)):
        if not fleet.alive[0]:
            break
        fleet.observe()
        steer, throttle = pilot.control(fleet.pos[0], fleet.angle[0], fleet.speed[0])
        fleet.step(np.array([0]), np.array([steer]), np.array([throttle]), dt)
        if fleet.laps[0] > counted:
            counted = int(fleet.laps[0])
            laps.append(float(fleet.last_lap[0]))
    return fleet, laps


def test_the_clock_agrees_with_the_distance_covered():
    """Laps counted, the running chrono and the ground covered must be one story."""
    track = resolve("oval")
    fleet, laps = _reference_laps(track, 1 / 30, 40.0)

    assert len(laps) >= 3, f"only {len(laps)} laps in 40s"
    assert int(fleet.laps[0]) == len(laps) == int(fleet.distance[0] // track.length)
    assert abs(min(laps) - float(fleet.best_lap[0])) < 1e-9
    assert abs(laps[-1] - float(fleet.last_lap[0])) < 1e-9

    # The chrono is the part of the run that is not in a completed lap.
    on_the_clock = float(fleet.lap_time[0])
    assert abs(float(fleet.time_alive[0]) - sum(laps) - on_the_clock) < 1e-6

    # A lap cannot be quicker than the car is capable of.
    assert float(fleet.best_lap[0]) > track.length / MAX_SPEED


def test_a_lap_time_does_not_depend_on_the_simulation_step():
    """The clock has to measure the driving, not the frame rate.

    Timing a crossing to the step it was noticed on costs a whole step -- 33ms at
    30Hz, more than the gap between two genuinely different drivers, and it would
    move every lap time when the step changed. So the instant is read off the
    car's own position against the start plane instead.
    """
    track = resolve("grand_prix")
    coarse = _reference_laps(track, 1 / 30, 60.0)[1]
    fine = _reference_laps(track, 1 / 120, 60.0)[1]
    assert coarse and fine

    # Ignore the standing start; compare settled laps.
    slow, quick = min(coarse[1:]), min(fine[1:])
    assert abs(slow - quick) < 0.05, f"{slow:.4f} at 30Hz vs {quick:.4f} at 120Hz"
    # And they are not simply rounded to the step, which is what that would hide.
    assert min(abs(slow / (1 / 30) - round(slow / (1 / 30))), 1) > 1e-6


def test_reversing_over_the_line_does_not_invent_a_lap():
    """Backing over the line and coming round again is not a flying lap.

    Without the guard it is the cheapest lap record on the board: cross, reverse
    a metre, cross again, and the clock reads a tenth of a second.
    """
    track = resolve("oval")
    fleet = Fleet(track, 1, retire_stalled=False)
    idx = np.array([0])

    # Nudge it forward over the line, then back, then forward again.
    for throttle, steps in ((1.0, 40), (-1.0, 80), (1.0, 60)):
        for _ in range(steps):
            fleet.step(idx, np.zeros(1), np.array([throttle]), 1 / 30)

    assert not math.isfinite(float(fleet.best_lap[0])), (
        f"credited a {fleet.best_lap[0]:.3f}s lap for shuffling over the line"
    )


def test_a_quicker_lap_is_worth_more_than_the_same_distance():
    """The point of the pace bonus: mileage alone cannot rank two equal distances.

    Two cars can end an episode on exactly the same distance, one of them having
    strung together a clean quick lap and the other having scrabbled round at a
    steady jog. Without the pace term they score identically and the population
    has no reason to prefer the first.
    """
    track = resolve("oval")
    evaluator = Evaluator(track, Episode(max_time=30.0))
    par = evaluator.episode.par_lap(track)
    fleet = Fleet(track, 2)
    fleet.distance[:] = track.length * 3

    fleet.best_lap[0], fleet.best_lap[1] = par - 2.5, par - 1.0
    quick, steady = evaluator.score(fleet, 0), evaluator.score(fleet, 1)
    assert quick > steady, (quick, steady)
    assert abs((quick - steady) - 1.5 * evaluator.episode.lap_bonus) < 1e-9

    # It is a bonus for genuine pace, not a tax: a lap slower than par simply
    # earns nothing, and a car that never completed one is scored on distance.
    fleet.best_lap[1] = par + 3.0
    assert evaluator.score(fleet, 1) == float(fleet.distance[1])
    fleet.best_lap[1] = math.inf
    assert evaluator.score(fleet, 1) == float(fleet.distance[1])


def test_records_outlive_the_run_but_not_the_circuit():
    """A record is only a record while the question it answers is the same one."""
    import tempfile

    track = resolve("oval")
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "records.json")
        book = RecordBook(path)
        assert book.best(track) is None

        assert book.submit(track, 21.5, by="ai", detail="gen 3")
        assert not book.submit(track, 22.0), "a slower lap must not take the record"
        assert book.submit(track, 20.25, by="you")
        assert not book.submit(track, float("inf")), "a car that never finished a lap"

        again = RecordBook(path)                      # it survives the process
        assert abs(float(again.best(track)["lap"]) - 20.25) < 1e-9
        assert again.best(track)["by"] == "you"

        # Redraw the circuit under the same name and the old time stops counting.
        moved = Track(track.control_points * 1.5, width=track.width, name=track.name)
        assert again.best(moved) is None
        assert again.submit(moved, 30.0)
        assert again.forget(track.name) and not again.best(track)


def test_lap_times_read_like_a_timing_screen():
    assert format_lap(83.4567).strip() == "1:23.457"
    assert format_lap(21.9).strip() == "21.900"
    assert format_lap(float("inf")).strip() == "--.---"
    assert format_lap(9.5).strip() == "9.500"


def test_population_learns_to_drive():
    """A short run must clearly beat generation zero."""
    track = resolve("oval")
    cfg = NeatConfig(num_inputs=NUM_INPUTS, num_outputs=NUM_OUTPUTS, pop_size=40)
    pop = Population(cfg, seed=0)
    history = []
    pop.run(Evaluator(track, Episode(max_time=10.0)), 10,
            on_generation=lambda p, s: history.append(s["mean"]))
    assert history[-1] > 3 * max(history[0], 1.0), history


if __name__ == "__main__":
    failures = 0
    for name, fn in sorted(globals().items()):
        if not name.startswith("test_") or not callable(fn):
            continue
        try:
            fn()
            print(f"PASS  {name}")
        except AssertionError as exc:
            failures += 1
            print(f"FAIL  {name}: {exc}")
    print("\nall good" if not failures else f"\n{failures} failure(s)")
    raise SystemExit(1 if failures else 0)
