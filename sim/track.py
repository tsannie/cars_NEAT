"""Track geometry, collision and progress measurement.

A track is just a centre line plus a width. Everything the simulation needs is
baked once into two small grids:

  ``on_track``  -- boolean, "is this cell drivable"
  ``progress``  -- float,   "how far along the lap is this cell"

Collision is then a single array lookup, and the distance a car has travelled
along the circuit is read straight off the second grid. That second grid is what
gives NEAT a dense fitness gradient instead of the near-flat ``1/distance`` the
first version used.
"""

from __future__ import annotations

import json
import math
import os

import numpy as np


# Cells beyond this force a coarser grid, to bound memory on very large maps.
MAX_GRID_CELLS = 4_000_000
# How far outside the tarmac the lap-progress field stays meaningful.
PROGRESS_MARGIN = 48.0
# An open track counts as run at 98% of its length, so a car is not asked to
# thread the last centimetre of the centre line to be credited with the trip.
FINISH_FRACTION = 0.98


def catmull_rom(points: np.ndarray, closed: bool, samples_per_span: int = 24) -> np.ndarray:
    """Smooth a handful of clicked points into a flowing curve."""
    n = len(points)
    if n < 3:
        return points.astype(float)

    if closed:
        p = np.vstack([points[-1], points, points[0], points[1]])
        spans = n
    else:
        p = np.vstack([points[0], points, points[-1]])
        spans = n - 1

    t = np.linspace(0.0, 1.0, samples_per_span, endpoint=False)[:, None]
    out = []
    for i in range(spans):
        p0, p1, p2, p3 = p[i], p[i + 1], p[i + 2], p[i + 3]
        out.append(
            0.5
            * (
                (2 * p1)
                + (-p0 + p2) * t
                + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t**2
                + (-p0 + 3 * p1 - 3 * p2 + p3) * t**3
            )
        )
    curve = np.vstack(out)
    if not closed:
        curve = np.vstack([curve, points[-1]])
    return curve


def resample(curve: np.ndarray, spacing: float, closed: bool) -> np.ndarray:
    """Re-space a polyline so consecutive points are ``spacing`` apart."""
    pts = np.vstack([curve, curve[0]]) if closed else curve
    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    total = cum[-1]
    if total < spacing:
        return curve
    count = max(3, int(total / spacing))
    targets = np.linspace(0.0, total, count, endpoint=not closed)
    x = np.interp(targets, cum, pts[:, 0])
    y = np.interp(targets, cum, pts[:, 1])
    return np.stack([x, y], axis=1)


def turn_radius(center: np.ndarray, closed: bool) -> np.ndarray:
    """Local radius of curvature at every sample of a centre line.

    A corner tighter than half the track width makes the offset walls fold back
    through the tarmac, which silently opens a shortcut. Both the editor and the
    tests use this to catch that.
    """
    nxt = np.roll(center, -1, axis=0)
    prv = np.roll(center, 1, axis=0)
    if not closed:
        nxt[-1] = center[-1]
        prv[0] = center[0]
    tang = nxt - prv
    tang = tang / np.maximum(np.linalg.norm(tang, axis=1, keepdims=True), 1e-9)

    ang = np.unwrap(np.arctan2(tang[:, 1], tang[:, 0]))
    swept = np.gradient(ang)
    step = np.linalg.norm(np.roll(center, -1, axis=0) - center, axis=1)
    if not closed:
        step[-1] = step[-2] if len(step) > 1 else 1.0
    return step / np.maximum(np.abs(swept), 1e-9)


class Track:
    SPACING = 8.0  # distance between centre-line samples

    def __init__(self, points, width=120.0, closed=True, name="track", cell=4.0, margin=80.0):
        self.control_points = np.asarray(points, dtype=float).reshape(-1, 2)
        self.width = float(width)
        self.closed = bool(closed)
        self.name = name
        self.cell = float(cell)
        self.margin = float(margin)
        self._build()

    # ------------------------------------------------------------------ build
    def _build(self) -> None:
        curve = catmull_rom(self.control_points, self.closed)
        self.center = resample(curve, self.SPACING, self.closed)
        n = len(self.center)

        # Tangent by central difference, normal is the tangent rotated 90 deg.
        nxt = np.roll(self.center, -1, axis=0)
        prv = np.roll(self.center, 1, axis=0)
        if not self.closed:
            nxt[-1] = self.center[-1]
            prv[0] = self.center[0]
        tang = nxt - prv
        norms = np.linalg.norm(tang, axis=1, keepdims=True)
        self.tangent = tang / np.maximum(norms, 1e-9)
        self.normal = np.stack([-self.tangent[:, 1], self.tangent[:, 0]], axis=1)

        half = self.width / 2.0
        self.left = self.center + self.normal * half
        self.right = self.center - self.normal * half

        # Arc length of every sample -- this is what "progress" means.
        step = np.linalg.norm(np.diff(self.center, axis=0), axis=1)
        self.arc = np.concatenate([[0.0], np.cumsum(step)])
        if self.closed:
            self.length = float(self.arc[-1] + np.linalg.norm(self.center[0] - self.center[-1]))
        else:
            self.length = float(self.arc[-1])

        self.start_pos = self.center[0].copy()
        self.start_tangent = self.tangent[0].copy()
        self.start_angle = float(math.atan2(self.tangent[0, 1], self.tangent[0, 0]))

        self._build_grids()

    @property
    def finish_distance(self) -> float:
        """How far one full run of the circuit is.

        A lap on a closed track, the distance to the flag on an open one. It is
        what the lap clock is measured over and where an open track ends.
        """
        return self.length if self.closed else self.length * FINISH_FRACTION

    def turn_radius(self) -> np.ndarray:
        return turn_radius(self.center, self.closed)

    def tight_corners(self) -> np.ndarray:
        """Mask of samples where the corner is too tight for the track width."""
        return self.turn_radius() < self.width / 2.0

    def _build_grids(self) -> None:
        """Bake the drivable area and the lap-progress field into two grids.

        Rather than asking every cell which centre-line sample is nearest -- which
        costs cells x samples and blows up on a big circuit -- each sample stamps
        a disc into its own neighbourhood, keeping whichever sample turned out to
        be closest. That is linear in the length of the track, so a 10x bigger
        map costs 10x more, not 100x.
        """
        pad = self.width / 2.0 + self.margin
        lo = self.center.min(axis=0) - pad
        hi = self.center.max(axis=0) + pad
        self.origin = lo
        self.size = hi - lo

        # Keep the grid within a sane memory budget by coarsening huge maps.
        cell = self.cell
        while (math.ceil(self.size[0] / cell) * math.ceil(self.size[1] / cell)) > MAX_GRID_CELLS:
            cell *= 2.0
        self.cell = cell

        w = int(math.ceil(self.size[0] / cell))
        h = int(math.ceil(self.size[1] / cell))
        self.grid_shape = (h, w)

        xs = self.origin[0] + (np.arange(w) + 0.5) * cell
        ys = self.origin[1] + (np.arange(h) + 0.5) * cell

        half = self.width / 2.0
        # Progress is stamped a little wider than the tarmac so a car that has
        # just clipped a wall still reads a sensible lap position on the step it
        # dies, instead of a garbage jump.
        reach = half + PROGRESS_MARGIN

        best = np.full((h, w), np.inf, dtype=np.float32)
        prog = np.zeros((h, w), dtype=np.float32)

        for i in range(len(self.center)):
            px, py = self.center[i]
            x0 = max(0, int((px - reach - self.origin[0]) / cell))
            x1 = min(w, int((px + reach - self.origin[0]) / cell) + 1)
            y0 = max(0, int((py - reach - self.origin[1]) / cell))
            y1 = min(h, int((py + reach - self.origin[1]) / cell) + 1)
            if x0 >= x1 or y0 >= y1:
                continue

            dx = xs[x0:x1] - px
            dy = ys[y0:y1] - py
            d2 = dy[:, None] ** 2 + dx[None, :] ** 2

            window = best[y0:y1, x0:x1]
            closer = d2 < window
            window[closer] = d2[closer]
            prog[y0:y1, x0:x1][closer] = self.arc[i]

        self.grid_on = best <= half * half
        self.grid_progress = prog

    # ------------------------------------------------------------- lookups
    def _cells(self, pts: np.ndarray):
        rel = (pts - self.origin) / self.cell
        gx = rel[..., 0].astype(np.int32)
        gy = rel[..., 1].astype(np.int32)
        h, w = self.grid_shape
        inside = (gx >= 0) & (gx < w) & (gy >= 0) & (gy < h)
        np.clip(gx, 0, w - 1, out=gx)
        np.clip(gy, 0, h - 1, out=gy)
        return gx, gy, inside

    def on_track(self, pts: np.ndarray) -> np.ndarray:
        gx, gy, inside = self._cells(pts)
        return self.grid_on[gy, gx] & inside

    def progress_at(self, pts: np.ndarray) -> np.ndarray:
        gx, gy, _ = self._cells(pts)
        return self.grid_progress[gy, gx]

    def raycast(self, origins: np.ndarray, dirs: np.ndarray, max_range: float,
                coarse_step: float = 6.0, refine: int = 3) -> np.ndarray:
        """Distance to the first wall for a batch of rays.

        March every ray at once on the occupancy grid, then bisect a few times
        around the first off-track sample. Cost does not depend on how
        complicated the track is, only on the range.
        """
        n_steps = max(2, int(max_range / coarse_step))
        ts = np.linspace(coarse_step, max_range, n_steps, dtype=np.float32)

        pts = origins[:, None, :] + dirs[:, None, :] * ts[None, :, None]
        off = ~self.on_track(pts)

        any_off = off.any(axis=1)
        first = np.argmax(off, axis=1)

        hi = np.where(any_off, ts[first], max_range)
        lo = np.where(any_off & (first > 0), ts[np.maximum(first - 1, 0)], 0.0)
        lo = np.where(any_off, lo, max_range)

        for _ in range(refine):
            mid = (lo + hi) / 2.0
            probe = origins + dirs * mid[:, None]
            hit = ~self.on_track(probe)
            hi = np.where(hit, mid, hi)
            lo = np.where(hit, lo, mid)

        return np.where(any_off, hi, max_range).astype(np.float32)

    # ------------------------------------------------------------------- io
    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "width": self.width,
            "closed": self.closed,
            "points": self.control_points.tolist(),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Track":
        return cls(
            data["points"],
            width=data.get("width", 120.0),
            closed=data.get("closed", True),
            name=data.get("name", "track"),
        )

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w") as fh:
            json.dump(self.to_dict(), fh, indent=1)

    @classmethod
    def load(cls, path: str) -> "Track":
        with open(path) as fh:
            return cls.from_dict(json.load(fh))

    def __repr__(self) -> str:
        return (
            f"<Track {self.name!r} len={self.length:.0f}px width={self.width:.0f} "
            f"closed={self.closed} grid={self.grid_shape}>"
        )
