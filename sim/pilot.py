"""A hand-written driver, used as a yardstick.

This one is not evolved: it reads the track geometry directly. It exists for two
reasons.

  * It says what a *good* lap looks like, so an evolved network's score means
    something ("3.7 laps" is only impressive next to a reference).
  * It proves things about a circuit that a single evolved genome cannot. If the
    reference cannot get round flat out, no amount of steering skill will --
    braking is genuinely required. If it *can* get round while braking, the
    circuit is drivable and the task is fair.
"""

from __future__ import annotations

import math

import numpy as np

from .car import BRAKE_GRIP, MAX_GRIP, MAX_SPEED, TURN_RATE

# Braking and cornering share one budget, so a speed profile that assumes the
# tyres are 100% busy stopping leaves nothing to steer with and is unusable in
# practice. Spend only part of the budget on slowing down.
BRAKE_SHARE = 0.65

# Full pedal once the car is this far off its target speed, in px/s.
THROTTLE_GAIN = 1.0 / 12.0


class ReferencePilot:
    """Pure pursuit for steering, a grip-limited speed profile for the pedals."""

    def __init__(self, track, lookahead: float = 0.35, flat_out: bool = False,
                 pace: float = 0.95, min_lookahead: float = 34.0,
                 max_lookahead: float = 120.0):
        self.track = track
        # Seconds of travel to aim ahead by. A fixed distance aims past the apex
        # of a tight corner and the car cuts straight into the inside wall, so it
        # has to shrink with speed.
        self.lookahead = lookahead
        self.min_lookahead = min_lookahead
        self.max_lookahead = max_lookahead
        self.flat_out = flat_out
        self.pace = pace
        self.speed_profile = self._build_speed_profile()

    def _build_speed_profile(self) -> np.ndarray:
        """Fastest speed at each point that still leaves room to brake.

        Forward: what the corner itself allows, sqrt(grip * radius). Backward:
        walk the track in reverse and pull each entry speed down until it is
        reachable from the corner that follows it.
        """
        track = self.track
        radius = np.maximum(track.turn_radius(), 1.0)
        limit = np.minimum(np.sqrt(MAX_GRIP * radius), MAX_SPEED)
        # The corner the car is *entering* matters as much as the one it is on,
        # and the radius estimate is noisy, so take the worst of a short window.
        pad = 3
        stacked = np.stack([np.roll(limit, -k) for k in range(-pad, pad + 1)])
        limit = stacked.min(axis=0) if track.closed else limit

        step = np.linalg.norm(np.roll(track.center, -1, axis=0) - track.center, axis=1)
        n = len(limit)
        speed = limit.copy()
        # Two laps of the backward pass so a closed circuit converges across the
        # start line as well.
        for _ in range(2 if track.closed else 1):
            for k in range(n - 1, -1, -1):
                nxt = (k + 1) % n if track.closed else min(k + 1, n - 1)
                reachable = math.sqrt(speed[nxt] ** 2 + 2 * BRAKE_GRIP * BRAKE_SHARE * step[k])
                speed[k] = min(speed[k], reachable)
        return speed

    def _nearest(self, pos) -> int:
        d2 = ((self.track.center - pos) ** 2).sum(axis=1)
        return int(np.argmin(d2))

    def control(self, pos, angle: float, speed: float):
        track = self.track
        here = self._nearest(pos)
        distance = float(np.clip(abs(speed) * self.lookahead,
                                 self.min_lookahead, self.max_lookahead))
        ahead = max(1, int(distance / track.SPACING))
        n = len(track.center)
        target_i = (here + ahead) % n if track.closed else min(here + ahead, n - 1)

        # --- steering: aim at a point further down the centre line ------------
        to_target = track.center[target_i] - np.asarray(pos, dtype=float)
        error = math.atan2(to_target[1], to_target[0]) - angle
        error = (error + math.pi) % (2 * math.pi) - math.pi
        # Full lock is TURN_RATE rad/s, so this is "how much of full lock do I
        # need to erase the heading error in about a fifth of a second".
        steer = float(np.clip(error / (TURN_RATE * 0.2), -1.0, 1.0))

        # --- pedals: chase the speed profile ----------------------------------
        if self.flat_out:
            return steer, 1.0
        # The profile already answers "how fast may I be *here* so that I can
        # still slow down for everything downstream", so it is read at the car's
        # own position. Reading it at the aim point instead makes the car
        # accelerate again the moment the far side of a corner opens up.
        target = self.speed_profile[here] * self.pace
        # Proportional, not on/off. A bang-bang pedal overshoots its target by a
        # whole step's worth of acceleration, so the lap it drives depends on the
        # frame rate -- which makes this pilot useless as a fixed yardstick.
        throttle = float(np.clip((target - speed) * THROTTLE_GAIN, -1.0, 1.0))
        return steer, throttle


def drive(track, fleet, dt: float, max_time: float, flat_out: bool = False,
          pace: float = 0.98, index: int = 0):
    """Run a single car of ``fleet`` under the reference pilot. Returns a report."""
    pilot = ReferencePilot(track, flat_out=flat_out, pace=pace)
    target = None if track.closed else track.finish_distance
    t = 0.0
    top_speed = 0.0
    braking = 0
    steps = 0
    while fleet.alive[index] and t < max_time:
        idx = np.array([index])
        fleet.observe()
        steer, throttle = pilot.control(fleet.pos[index], fleet.angle[index],
                                        fleet.speed[index])
        fleet.step(idx, np.array([steer]), np.array([throttle]), dt, target)
        top_speed = max(top_speed, float(fleet.speed[index]))
        braking += throttle < 0
        steps += 1
        t += dt
    return {
        "distance": float(fleet.distance[index]),
        "laps": float(fleet.distance[index]) / track.length,
        "alive": bool(fleet.alive[index]),
        "time": t,
        "top_speed": top_speed,
        "braking_fraction": braking / max(steps, 1),
        "finished": bool(fleet.finished[index]),
    }
