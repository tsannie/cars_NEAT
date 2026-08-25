"""Fitness evaluation: run one generation of cars on the track."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from neatlite import FeedForwardNetwork

from .car import Fleet
from .records import format_lap, timing_lines


# What a competent-but-not-perfect driver averages, in px/s. Used only to turn
# "how many laps should a generation last" into a number of seconds.
CRUISING_SPEED = 250.0
DEFAULT_LAPS = 2.5
# Drawn frames a "someone just went quicker" message stays up for.
BANNER_FRAMES = 90


@dataclass
class Episode:
    dt: float = 1.0 / 30.0
    max_time: float = 30.0       # simulated seconds per generation
    spread: float = 0.0          # lateral scatter on the grid (cosmetic only)
    speed_bonus: float = 180.0   # fitness per second saved on an open track
    lap_bonus: float = 180.0     # fitness per second under a par lap, on a circuit

    @classmethod
    def for_track(cls, track, laps: float = DEFAULT_LAPS, **kwargs) -> "Episode":
        """Scale the generation length to the circuit.

        A fixed 30s is about three laps of a 3000px track but barely one lap of a
        9800px one, which leaves a big map with almost no room to tell a good
        driver from a lucky one. Deriving the time from the track length keeps
        the *task* the same whatever its size.
        """
        seconds = laps * track.length / CRUISING_SPEED
        return cls(max_time=max(8.0, round(seconds)), **kwargs)

    def par_lap(self, track) -> float:
        """The lap a competent driver would set, in seconds.

        This is the bar the pace bonus pays out against, not a target: a good
        genome goes comfortably under it, a bad one never gets there at all.
        """
        return track.finish_distance / CRUISING_SPEED

    def scoring(self) -> dict:
        """What a fitness number means, so two of them are only compared fairly.

        A saved model records the score it earned; change how scoring works and
        that number stops being comparable to a new run's, exactly as a change to
        the car's handling would.
        """
        return {"dt": self.dt, "speed_bonus": self.speed_bonus,
                "lap_bonus": self.lap_bonus, "cruising_speed": CRUISING_SPEED}


class Evaluator:
    """Callable passed to ``Population.run``."""

    def __init__(self, track, episode: Episode | None = None, renderer=None, seed: int = 0,
                 records=None):
        self.track = track
        self.episode = episode or Episode()
        self.renderer = renderer
        self.rng = np.random.default_rng(seed)
        self.generation = 0
        self.last_fleet: Fleet | None = None
        self.stats_line = ""
        self.best_run = 0.0   # furthest anything has got in a *complete* episode

        # The stopwatch side: the quickest lap this run has produced, and the
        # quickest anything has ever produced on this circuit.
        self.records = records
        self.lap_record = records.best(track) if records is not None else None
        self.best_lap = math.inf
        self.best_lap_gen = None
        self._flash = 0     # frames left showing "someone just went quicker"

    @property
    def target_distance(self):
        """Where the episode ends -- ``None`` on a circuit.

        On a closed track the objective is simply "cover as much ground as you
        can in ``max_time`` seconds": no ceiling, so there is always something to
        gain by cornering better. An open track has an actual finish line, so
        cars that reach it stop and are rewarded for the time they saved.
        """
        if self.track.closed:
            return None
        return self.track.finish_distance

    def __call__(self, genomes, cfg) -> None:
        ep = self.episode
        nets = [FeedForwardNetwork.create(g, cfg) for g in genomes]
        fleet = Fleet(self.track, len(genomes), spread=ep.spread, rng=self.rng)
        self.last_fleet = fleet

        target = self.target_distance
        t = 0.0
        frame = 0
        r = self.renderer
        if r is not None:
            r.skip_generation = False

        while t < ep.max_time and fleet.any_alive():
            idx, obs = fleet.observe()
            rows = obs.tolist()
            out = np.empty((idx.size, 2))
            for j, i in enumerate(idx):
                out[j] = nets[i].activate(rows[j])

            fleet.step(idx, out[:, 0], out[:, 1], ep.dt, target)
            t += ep.dt
            frame += 1

            if r is not None:
                r.pump()
                if r.quit_requested or r.skip_generation:
                    break
                if r.enabled and frame % r.steps_per_frame == 0:
                    leader = self._leader(fleet)
                    self._note_laps(fleet)
                    r.draw(fleet, self._hud(t, fleet, leader), leader=leader,
                           banner=self._banner())

        for i, g in enumerate(genomes):
            g.fitness = self.score(fleet, i)
        self.best_run = max(self.best_run, float(fleet.distance.max()))
        self._note_laps(fleet)
        self._submit_lap()

        self.generation += 1

    # -------------------------------------------------------------- fitness
    def score(self, fleet: Fleet, i: int) -> float:
        """What one car's drive was worth.

        Distance is the base: dense, informative on every frame, and impossible
        to game by sitting still. On top of it comes pace -- the seconds the car
        saved against a par lap -- because mileage alone cannot tell a quick
        driver from a busy one. Two cars can cover the same ground in the same
        time, one of them stringing together clean fast laps and the other
        scrabbling round barely under control; the second is a dead end, and
        without the lap term it scores exactly as well as the first.

        An open track pays the same idea out through ``speed_bonus`` at the flag,
        so the two never both apply.
        """
        ep = self.episode
        score = float(fleet.distance[i])
        if fleet.finished[i]:
            score += max(0.0, ep.max_time - float(fleet.finish_time[i])) * ep.speed_bonus
        elif self.track.closed:
            lap = float(fleet.best_lap[i])
            if math.isfinite(lap):
                score += max(0.0, ep.par_lap(self.track) - lap) * ep.lap_bonus
        return score

    # --------------------------------------------------------- the stopwatch
    def _note_laps(self, fleet: Fleet) -> None:
        """Pick up any lap that has just beaten the best of this run."""
        quickest = float(fleet.best_lap.min())
        if quickest < self.best_lap:
            self.best_lap = quickest
            self.best_lap_gen = self.generation
            self._flash = BANNER_FRAMES

    def _submit_lap(self) -> None:
        """Offer the run's best lap to the record book, once a generation."""
        if self.records is None or not math.isfinite(self.best_lap):
            return
        if self.records.submit(self.track, self.best_lap, by="ai",
                               detail=f"gen {self.best_lap_gen}"):
            self.lap_record = self.records.best(self.track)

    def _banner(self):
        if self._flash <= 0:
            return None
        self._flash -= 1
        beat = self.lap_record is None or self.best_lap < float(self.lap_record["lap"])
        text = f"{'NEW LAP RECORD' if beat else 'fastest lap'}   {format_lap(self.best_lap).strip()}"
        return (text, "hot") if beat else text

    # ------------------------------------------------------------------ hud
    def _leader(self, fleet: Fleet):
        alive = np.flatnonzero(fleet.alive)
        if alive.size == 0:
            return None
        return int(alive[np.argmax(fleet.distance[alive])])

    def _hud(self, t: float, fleet: Fleet, leader: int | None = None):
        best = float(fleet.distance.max())
        alive = fleet.alive
        over = float((fleet.grip_load[alive] > 1.0).mean() * 100) if alive.any() else 0.0
        lines = [
            f"gen {self.generation}   t {t:5.1f}s / {self.episode.max_time:.0f}s",
            f"alive   {int(alive.sum()):3d} / {fleet.n}",
            f"leader  {best:7.0f} px  ({best / self.track.length:.2f} lap) so far",
        ]
        chrono = float(fleet.lap_time[leader]) if leader is not None else math.inf
        lines += timing_lines(chrono, self.best_lap, self.lap_record)
        # The best run is a whole episode; the leader is only part-way through
        # one. Comparing the two raw numbers makes a healthy generation look like
        # a collapse, so show where the best run stood at this same moment.
        if self.best_run > 0.0 and t > 0.5:
            on_pace = self.best_run * t / self.episode.max_time
            lines.append(f"pace    {best / max(on_pace, 1e-9) * 100:3.0f} % of the best "
                         f"run ({self.best_run / self.track.length:.2f} lap)")
        lines.append(f"over the limit  {over:3.0f} % of the pack")
        if self.stats_line:
            lines.append(self.stats_line)
        if self.renderer is not None:
            lines.append(self.renderer.controls_hint())
        return lines


def drive_solo(track, genome, cfg, renderer, episode: Episode | None = None,
               label: str = "", records=None, detail: str = "") -> float:
    """Watch one genome drive. Returns the distance it covered.

    Whatever it does with the stopwatch counts: a replay is the same car on the
    same circuit as training, so a lap set here belongs in the record book.
    """
    ep = episode or Episode()
    net = FeedForwardNetwork.create(genome, cfg)
    fleet = Fleet(track, 1, spread=0.0)
    target = None if track.closed else track.finish_distance
    record = records.best(track) if records is not None else None
    last_lap_seen = float(fleet.last_lap[0])   # inf until it crosses the line
    flash = 0

    t = 0.0
    while t < ep.max_time * 2 and fleet.any_alive():
        idx, obs = fleet.observe()
        out = net.activate(obs[0].tolist())
        fleet.step(idx, np.array([out[0]]), np.array([out[1]]), ep.dt, target)
        t += ep.dt

        if fleet.last_lap[0] != last_lap_seen:
            last_lap_seen = float(fleet.last_lap[0])
            flash = BANNER_FRAMES

        renderer.pump()
        if renderer.quit_requested:
            break
        lines = [
            label or "replay",
            f"t       {t:5.1f}s   lap {int(fleet.laps[0]) + 1}",
            f"dist    {fleet.distance[0]:7.0f} px  ({fleet.distance[0] / track.length:.2f} lap)",
            f"speed   {fleet.speed[0]:7.0f} px/s",
            f"tyres   {fleet.grip_load[0] * 100:6.0f} %"
            + ("  UNDERSTEER" if fleet.grip_load[0] > 1.0 else ""),
        ]
        lines += timing_lines(float(fleet.lap_time[0]), float(fleet.best_lap[0]), record,
                              label="best")
        lines.append("wheel zoom  drag pan  [F] fit  [L] follow  [ESC] quit")
        banner = None
        if flash > 0:
            flash -= 1
            beat = record is None or last_lap_seen < float(record["lap"])
            title = "NEW LAP RECORD" if beat else f"lap {int(fleet.laps[0])}"
            banner = f"{title}   {format_lap(last_lap_seen).strip()}"
            if beat:
                banner = (banner, "hot")
        renderer.draw(fleet, lines, leader=0, banner=banner)

    if records is not None:
        records.submit(track, float(fleet.best_lap[0]), by="ai", detail=detail or label)
    return float(fleet.distance[0])
