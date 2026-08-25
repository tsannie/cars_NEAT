"""The car fleet.

Every car in a generation is simulated in the same numpy arrays, so one
population step is a handful of vectorised operations rather than a Python loop
over 120 objects. The only per-car Python work left is the network activation.
"""

from __future__ import annotations

import math

import numpy as np

# --- body ------------------------------------------------------------------
CAR_LENGTH = 34.0
CAR_WIDTH = 18.0

# --- handling ---------------------------------------------------------------
MAX_SPEED = 400.0        # px/s
MAX_REVERSE = -90.0
ACCEL = 170.0            # px/s^2, engine-limited -- low enough that getting the
                         # power down out of a corner is worth something
BRAKE = 520.0            # px/s^2 asked for; the tyres cap what actually happens
# Deliberately low. With heavy drag, lifting off already scrubbed ~190px/s^2 and
# a car that overcooked a corner slowed itself down before it could run wide --
# the driver never had to touch the brakes. At 0.15 coasting only sheds ~50px/s^2,
# so shedding 174px/s to make the tightest corner of grand_prix takes 3.4s of
# coasting against 0.4s of braking. Braking is now the only real option.
DRAG = 0.15              # fraction of speed bled off per second
TURN_RATE = 3.4          # rad/s at full lock
TURN_SPEED_REF = 110.0   # below this speed the car turns proportionally less

# The whole point of this number: one tyre budget shared by turning, accelerating
# and braking. Cornering demands v^2/R, so the fastest a car can take a corner is
# sqrt(MAX_GRIP * R) -- 140 px/s in the tightest corner of grand_prix against a
# 400 px/s top speed. Flat out is no longer an option: it just understeers into
# the wall.
#
# Grip and sensor range are one decision, not two. Braking also spends grip, so
# taking grip away lengthens every braking zone: hauling 400 down to 140 px/s
# costs 360px at the pace the reference pilot brakes. If the car cannot see that
# far, the corner arrives before it can be seen -- not a harder problem, just an
# unfair one. RAY_RANGE below is sized from this number, with margin.
MAX_GRIP = 300.0         # px/s^2, total

# Braking gets its own, larger budget. Under braking the mass pitches forward and
# loads the front tyres, which is where nearly all the stopping happens -- a car
# stops harder than it corners. Keeping the two separate is also what lets the
# brakes be strengthened without handing the cornering back the grip we took away.
BRAKE_GRIP = 380.0       # px/s^2, longitudinal, while slowing down

# --- sensors ----------------------------------------------------------------
# Bunched towards the front: the braking decision hangs on what is straight
# ahead, and an evenly spread fan gives that the same single number as the view
# out of the side window. Measured worth 5.5% of lap time on grand_prix -- more
# than doubling the population, and cheaper.
RAY_ANGLES = np.radians([-90.0, -60.0, -40.0, -22.0, -10.0, 0.0,
                         10.0, 22.0, 40.0, 60.0, 90.0])
NUM_RAYS = len(RAY_ANGLES)
RAY_RANGE = 450.0        # far enough to see a braking zone coming; see MAX_GRIP

NUM_INPUTS = NUM_RAYS + 1   # rays + normalised speed
NUM_OUTPUTS = 2             # steering, throttle

# --- episode rules -----------------------------------------------------------
STALL_TIMEOUT = 2.5      # seconds without progress before a car is retired
STALL_MARGIN = 12.0      # px of progress that counts as "still moving"


def handling_fingerprint() -> dict:
    """The constants that define what "fitness 11000" means.

    A saved genome's score is only comparable to a new run's if the car behaves
    the same way, so this travels with the model.
    """
    return {
        "max_speed": MAX_SPEED, "accel": ACCEL, "brake": BRAKE, "drag": DRAG,
        "turn_rate": TURN_RATE, "turn_speed_ref": TURN_SPEED_REF,
        "max_grip": MAX_GRIP, "brake_grip": BRAKE_GRIP,
        "rays": NUM_RAYS, "ray_range": RAY_RANGE,
    }


class Fleet:
    """``n`` cars driving the same track at the same time."""

    def __init__(self, track, n: int, spread: float = 0.0, rng: np.random.Generator | None = None,
                 retire_stalled: bool = True):
        # ``spread`` is purely cosmetic: cars do not collide with each other, and
        # keeping it at 0 makes a genome's score exactly reproducible, which is
        # what elitism needs to be meaningful.
        self.track = track
        self.n = n
        # Retiring a car that has stopped making progress keeps a generation from
        # burning its clock on one that spun and sat there. It is a training
        # device only: a human sitting still at the line is not a stuck genome.
        self.retire_stalled = retire_stalled
        rng = rng or np.random.default_rng(0)

        self.pos = np.repeat(track.start_pos[None, :], n, axis=0).astype(np.float64)
        self.angle = np.full(n, track.start_angle, dtype=np.float64)
        if spread > 0.0:
            # A little lateral scatter so identical genomes do not overlap exactly.
            normal = np.array([-math.sin(track.start_angle), math.cos(track.start_angle)])
            offset = rng.uniform(-spread, spread, n)[:, None] * normal[None, :]
            self.pos += offset

        self.speed = np.zeros(n)
        self.alive = np.ones(n, dtype=bool)
        self.finished = np.zeros(n, dtype=bool)
        self.crashed = np.zeros(n, dtype=bool)   # left the tarmac on the last step

        self.distance = np.zeros(n)          # signed arc length covered
        # Where each car stood the last time it was credited with progress. The
        # stall clock is measured from here, not from the previous step.
        self.stall_mark = np.zeros(n)
        self.stall_time = np.zeros(n)
        self.time_alive = np.zeros(n)
        self.finish_time = np.full(n, np.inf)
        self._last_progress = track.progress_at(self.pos).astype(np.float64)

        # ------ the stopwatch ------------------------------------------------
        # Every car carries its own clock, started at the line and stopped each
        # time it runs over it again. ``lap_clean`` guards the one way a bogus
        # time can appear: reversing back over the line and coming round again
        # would otherwise be timed as a very quick lap.
        self.lap_length = float(track.finish_distance)
        self.laps = np.zeros(n, dtype=np.int64)
        self.lap_started = np.zeros(n)       # ``time_alive`` the current lap began at
        self.lap_clean = np.ones(n, dtype=bool)
        self.last_lap = np.full(n, np.inf)
        self.best_lap = np.full(n, np.inf)

        self.last_rays = np.full((n, NUM_RAYS), RAY_RANGE)
        # How much of the tyre budget the driver asked for: > 1 means understeer.
        self.grip_load = np.zeros(n)
        # What the driver last asked for, kept so the view can show the inputs
        # and outputs of whichever car is being watched.
        self.last_steer = np.zeros(n)
        self.last_throttle = np.zeros(n)

    @property
    def lap_time(self) -> np.ndarray:
        """Seconds showing on each car's clock for the lap in progress.

        A retired car stops being stepped, so its clock freezes where it died
        rather than running on.
        """
        # A crossing is timed to a fraction of a step, which can land just after
        # the frame it was spotted on; the clock still starts at zero, not below.
        return np.maximum(self.time_alive - self.lap_started, 0.0)

    # ------------------------------------------------------------- sensors
    def observe(self):
        """Returns ``(indices, inputs)`` for the cars still driving."""
        idx = np.flatnonzero(self.alive)
        if idx.size == 0:
            return idx, np.zeros((0, NUM_INPUTS))

        ang = self.angle[idx][:, None] + RAY_ANGLES[None, :]
        dirs = np.stack([np.cos(ang), np.sin(ang)], axis=-1).reshape(-1, 2)
        origins = np.repeat(self.pos[idx], NUM_RAYS, axis=0)

        dist = self.track.raycast(origins, dirs, RAY_RANGE).reshape(idx.size, NUM_RAYS)
        self.last_rays[idx] = dist

        inputs = np.empty((idx.size, NUM_INPUTS))
        inputs[:, :NUM_RAYS] = dist / RAY_RANGE
        inputs[:, NUM_RAYS] = self.speed[idx] / MAX_SPEED
        return idx, inputs

    # ------------------------------------------------------------- dynamics
    def step(self, idx: np.ndarray, steer: np.ndarray, throttle: np.ndarray, dt: float,
             target_distance: float | None = None) -> None:
        if idx.size == 0:
            return

        steer = np.clip(steer, -1.0, 1.0)
        throttle = np.clip(throttle, -1.0, 1.0)
        self.last_steer[idx] = steer
        self.last_throttle[idx] = throttle

        speed = self.speed[idx]

        # --- what the driver is asking the tyres for -------------------------
        a_long = np.where(throttle >= 0.0, throttle * ACCEL, throttle * BRAKE)
        # A stationary car cannot steer; authority ramps in with speed.
        authority = np.clip(np.abs(speed) / TURN_SPEED_REF, 0.0, 1.0)
        yaw_cmd = steer * TURN_RATE * authority * np.sign(speed)
        a_lat = np.abs(speed * yaw_cmd)

        # --- friction circle -------------------------------------------------
        # Longitudinal and lateral draw on the same budget, so braking into a
        # corner costs grip that is then missing to turn with. Asking for more
        # than the tyres have does not stop the car turning -- it just turns
        # less than asked, which is understeer, and it runs wide.
        budget = np.where(a_long < 0.0, BRAKE_GRIP, MAX_GRIP)
        demand = np.hypot(a_long, a_lat)
        overload = demand / budget
        scale = np.where(overload > 1.0, 1.0 / np.maximum(overload, 1e-9), 1.0)
        a_long = a_long * scale
        yaw = yaw_cmd * scale
        self.grip_load[idx] = overload

        # Integrate properly rather than with plain Euler. A first-order step
        # makes a lap time depend on the frame rate -- 0.13s over a 12s lap at
        # 30Hz, which is more than the gap between two genuinely different
        # drivers. Drag is linear, so it has a closed form over the step, and the
        # position uses the mid-step heading and the mean speed.
        before = speed
        decay = math.exp(-DRAG * dt)
        speed = speed * decay + (a_long / DRAG) * (1.0 - decay)
        speed = np.clip(speed, MAX_REVERSE, MAX_SPEED)

        heading = self.angle[idx]
        angle = heading + yaw * dt
        mid = heading + yaw * (dt * 0.5)
        travelled = (before + speed) * 0.5 * dt

        pos = self.pos[idx] + np.stack([np.cos(mid), np.sin(mid)], axis=1) * travelled[:, None]

        self.speed[idx] = speed
        self.angle[idx] = angle
        self.pos[idx] = pos

        # --- collision: any corner of the body off the tarmac ----------------
        crashed = ~self._corners_on_track(pos, angle).all(axis=1)

        # --- progress along the lap ------------------------------------------
        prog = self.track.progress_at(pos).astype(np.float64)
        delta = prog - self._last_progress[idx]
        half = self.track.length / 2.0
        delta = np.where(delta < -half, delta + self.track.length, delta)
        delta = np.where(delta > half, delta - self.track.length, delta)
        self._last_progress[idx] = prog
        before = self.distance[idx]
        self.distance[idx] = before + delta

        # --- retire cars that stopped making progress -------------------------
        # "Has it covered STALL_MARGIN since the clock last reset", not "did it
        # cover STALL_MARGIN in this one step" -- the latter is a hidden minimum
        # speed of STALL_MARGIN/dt (360 px/s at 30Hz), which quietly retired any
        # car that slowed down for a corner.
        improved = self.distance[idx] > self.stall_mark[idx] + STALL_MARGIN
        self.stall_mark[idx] = np.where(improved, self.distance[idx], self.stall_mark[idx])
        self.stall_time[idx] = np.where(improved, 0.0, self.stall_time[idx] + dt)
        stalled = (self.stall_time[idx] > STALL_TIMEOUT if self.retire_stalled
                   else np.zeros(idx.size, dtype=bool))

        self.time_alive[idx] += dt
        self._time_the_line(idx, before, delta, dt)

        done = np.zeros(idx.size, dtype=bool)
        if target_distance is not None:
            done = self.distance[idx] >= target_distance
            self.finished[idx[done]] = True
            self.finish_time[idx[done]] = self.time_alive[idx[done]]

        self.crashed[:] = False
        self.crashed[idx[crashed]] = True
        self.alive[idx[crashed | stalled | done]] = False

    def _time_the_line(self, idx: np.ndarray, before: np.ndarray, delta: np.ndarray,
                       dt: float) -> None:
        """Stop and restart the clock of every car that ran over the start line."""
        line = self.lap_length
        after = before + delta
        was = np.floor(before / line).astype(np.int64)
        now = np.floor(after / line).astype(np.int64)
        moved = np.flatnonzero(now != was)
        if moved.size == 0:
            return

        cars = idx[moved]
        # Whichever way it was crossed, the line in play is the higher of the two
        # lap boundaries.
        boundary = np.maximum(was[moved], now[moved]) * line
        crossed_at = self._crossing_time(cars, before[moved], delta[moved], boundary, dt)

        forward = now[moved] > was[moved]
        timed = forward & self.lap_clean[cars]
        lap = np.where(timed, crossed_at - self.lap_started[cars], np.inf)

        self.last_lap[cars] = np.where(timed, lap, self.last_lap[cars])
        self.best_lap[cars] = np.minimum(self.best_lap[cars], lap)
        self.laps[cars] = now[moved]
        self.lap_started[cars] = crossed_at
        self.lap_clean[cars] = forward

    def _crossing_time(self, cars: np.ndarray, before: np.ndarray, delta: np.ndarray,
                       boundary: np.ndarray, dt: float) -> np.ndarray:
        """*When* inside the step the line went by.

        Rounding a crossing to the step it was spotted on puts a full step of
        noise on every lap time -- 33 ms at 30 Hz, which is more than the gap
        between two genuinely different drivers.

        On a circuit the line is a real place, the plane through the start point,
        so the instant is read off the car itself: how far past that plane it
        sits, over how fast it is going through it. The lap-progress grid only
        knows where the car is to the nearest 8 px sample, but the car's own
        position does not, and both ends of a lap are timed the same way, so what
        the sampling does to one cancels in the difference. An open track's flag
        is not the start plane, so there the step is split on progress instead.
        """
        end = self.time_alive[cars]
        if self.track.closed:
            along = self.track.start_tangent
            heading = np.stack([np.cos(self.angle[cars]), np.sin(self.angle[cars])], axis=1)
            through = self.speed[cars] * (heading @ along)     # px/s across the line
            past = (self.pos[cars] - self.track.start_pos) @ along
            # A car barely crawling over the line has no meaningful crossing
            # instant, so the correction is never allowed past a whole step.
            safe = np.where(np.abs(through) > 1e-6, through, np.inf)
            return end - np.clip(past / safe, -dt, dt)

        fraction = np.clip((boundary - before) / delta, 0.0, 1.0)
        return end - (1.0 - fraction) * dt

    def _corners_on_track(self, pos: np.ndarray, angle: np.ndarray) -> np.ndarray:
        c, s = np.cos(angle), np.sin(angle)
        fwd = np.stack([c, s], axis=1) * (CAR_LENGTH / 2.0)
        side = np.stack([-s, c], axis=1) * (CAR_WIDTH / 2.0)
        corners = np.stack(
            [pos + fwd + side, pos + fwd - side, pos - fwd + side, pos - fwd - side],
            axis=1,
        )
        return self.track.on_track(corners)

    # --------------------------------------------------------------- helpers
    def any_alive(self) -> bool:
        return bool(self.alive.any())

    def body_polygon(self, i: int) -> np.ndarray:
        a = self.angle[i]
        c, s = math.cos(a), math.sin(a)
        fwd = np.array([c, s]) * (CAR_LENGTH / 2.0)
        side = np.array([-s, c]) * (CAR_WIDTH / 2.0)
        p = self.pos[i]
        return np.stack([p + fwd + side, p + fwd - side, p - fwd - side, p - fwd + side])
