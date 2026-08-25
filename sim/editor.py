"""Mouse-driven track editor with a pan/zoom canvas.

Click a few points to lay out the centre line; the curve through them is
smoothed, so a handful of clicks gives a drivable circuit. The canvas is
unbounded -- the window is a viewport onto world coordinates -- so a track is
not limited to what fits on screen.

The track is saved as a small JSON file (the control points, the width, and
whether it loops), which means a track is a few lines of text rather than a
bitmap.
"""

from __future__ import annotations

import os

import numpy as np
import pygame

from .track import Track, catmull_rom, resample, turn_radius

BG = (24, 26, 32)
GRID = (34, 37, 45)
GRID_MAJOR = (44, 48, 58)
TARMAC = (58, 62, 72)
KERB = (120, 128, 145)
POINT = (255, 196, 62)
POINT_HOT = (255, 120, 90)
TIGHT = (236, 66, 66)
GUIDE = (78, 84, 96)
TEXT = (226, 230, 238)
DIM = (140, 148, 162)
START_LINE = (235, 235, 240)

GRAB_RADIUS = 14  # screen pixels
MIN_WIDTH = 60.0
MAX_WIDTH = 400.0
WIDTH_STEP = 5.0
KEY_PAN_SPEED = 700.0  # screen px per second

# World spacings the background grid is allowed to use, in px.
GRID_STEPS = (25, 50, 100, 250, 500, 1000, 2500, 5000, 10000)

HELP = [
    "left click       add a point   (or drag an existing one)",
    "right click      remove the point under the cursor",
    "wheel            zoom on the cursor",
    "middle drag      pan          (or arrow keys)",
    "shift + wheel    track width  (or [ and ])",
    "F                fit the whole track in view",
    "C                open / closed loop      G   grid",
    "H                hide these controls",
    "Z                undo last point         R   clear",
    "ENTER or S       save and quit           ESC cancel",
]
HELP_HINT = "H   show controls"


class Camera:
    """Viewport onto world coordinates: screen = world * scale + offset."""

    MIN_SCALE = 0.04
    MAX_SCALE = 4.0

    def __init__(self, size):
        self.size = np.asarray(size, dtype=float)
        self.scale = 1.0
        self.offset = np.zeros(2)

    def to_screen(self, pts) -> np.ndarray:
        return np.asarray(pts, dtype=float) * self.scale + self.offset

    def to_world(self, pts) -> np.ndarray:
        return (np.asarray(pts, dtype=float) - self.offset) / self.scale

    def zoom_at(self, screen_pos, factor: float) -> None:
        """Zoom while keeping whatever is under the cursor pinned there."""
        anchor = self.to_world(screen_pos)
        self.scale = float(np.clip(self.scale * factor, self.MIN_SCALE, self.MAX_SCALE))
        self.offset = np.asarray(screen_pos, dtype=float) - anchor * self.scale

    def pan(self, screen_delta) -> None:
        self.offset += np.asarray(screen_delta, dtype=float)

    def fit(self, pts, pad: float = 0.0, margin: int = 90) -> None:
        pts = np.asarray(pts, dtype=float).reshape(-1, 2)
        if len(pts) == 0:
            return
        lo, hi = pts.min(axis=0) - pad, pts.max(axis=0) + pad
        span = np.maximum(hi - lo, 1.0)
        usable = np.maximum(self.size - 2 * margin, 100.0)
        self.scale = float(np.clip(min(usable / span), self.MIN_SCALE, self.MAX_SCALE))
        self.offset = self.size / 2.0 - (lo + hi) / 2.0 * self.scale


class TrackEditor:
    def __init__(self, size=(1280, 720), track: Track | None = None):
        pygame.init()
        pygame.display.set_caption("cars_NEAT - track editor")
        self.size = size
        self.screen = pygame.display.set_mode(size, pygame.RESIZABLE)
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("dejavusansmono,consolas,monospace", 14)
        self.font_big = pygame.font.SysFont("dejavusansmono,consolas,monospace", 18, bold=True)

        self.camera = Camera(size)
        self.show_grid = True
        self.show_help = True
        self.dragging: int | None = None
        self.panning = False

        if track is not None:
            self.points = [tuple(p) for p in track.control_points]
            self.width = track.width
            self.closed = track.closed
            self.camera.fit(track.control_points, pad=self.width)
        else:
            self.points = []
            self.width = 130.0
            self.closed = True

    # ------------------------------------------------------------------ loop
    def run(self) -> Track | None:
        """Returns the finished track, or ``None`` if the user bailed out."""
        while True:
            for event in pygame.event.get():
                result = self._handle(event)
                if result is not False:
                    pygame.quit()
                    return result
            self._key_pan(self.clock.get_time() / 1000.0)
            self._draw()
            self.clock.tick(60)

    def _handle(self, event):
        """``False`` means "keep going"; anything else ends the editor."""
        if event.type == pygame.QUIT:
            return None

        if event.type == pygame.VIDEORESIZE:
            self.size = (event.w, event.h)
            self.screen = pygame.display.set_mode(self.size, pygame.RESIZABLE)
            self.camera.size = np.array(self.size, dtype=float)

        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                return None
            if event.key in (pygame.K_RETURN, pygame.K_KP_ENTER, pygame.K_s):
                if len(self.points) >= 3:
                    return self.build()
            elif event.key == pygame.K_c:
                self.closed = not self.closed
            elif event.key == pygame.K_g:
                self.show_grid = not self.show_grid
            elif event.key == pygame.K_h:
                self.show_help = not self.show_help
            elif event.key == pygame.K_f:
                self.fit_view()
            elif event.key in (pygame.K_z, pygame.K_BACKSPACE):
                if self.points:
                    self.points.pop()
            elif event.key == pygame.K_r:
                self.points = []
            elif event.key == pygame.K_LEFTBRACKET:
                self._resize(-WIDTH_STEP)
            elif event.key == pygame.K_RIGHTBRACKET:
                self._resize(+WIDTH_STEP)

        elif event.type == pygame.MOUSEBUTTONDOWN:
            pos = event.pos  # the click's own position, not wherever the cursor is now
            if event.button == 1:
                hit = self._pick(pos)
                if hit is None:
                    self.points.append(tuple(self.camera.to_world(pos)))
                else:
                    self.dragging = hit
            elif event.button == 2:
                self.panning = True
            elif event.button == 3:
                hit = self._pick(pos)
                if hit is not None:
                    self.points.pop(hit)
            elif event.button in (4, 5):
                step = 1 if event.button == 4 else -1
                if pygame.key.get_mods() & pygame.KMOD_SHIFT:
                    self._resize(step * WIDTH_STEP)
                else:
                    self.camera.zoom_at(pos, 1.15**step)

        elif event.type == pygame.MOUSEBUTTONUP:
            if event.button == 1:
                self.dragging = None
            elif event.button == 2:
                self.panning = False

        elif event.type == pygame.MOUSEMOTION:
            if self.panning:
                self.camera.pan(event.rel)
            elif self.dragging is not None:
                self.points[self.dragging] = tuple(self.camera.to_world(event.pos))

        return False

    def _key_pan(self, dt: float) -> None:
        keys = pygame.key.get_pressed()
        dx = (keys[pygame.K_LEFT] - keys[pygame.K_RIGHT]) * KEY_PAN_SPEED * dt
        dy = (keys[pygame.K_UP] - keys[pygame.K_DOWN]) * KEY_PAN_SPEED * dt
        if dx or dy:
            self.camera.pan((dx, dy))

    def _resize(self, delta: float) -> None:
        self.width = float(np.clip(self.width + delta, MIN_WIDTH, MAX_WIDTH))

    def _pick(self, screen_pos):
        """Index of the control point under the cursor, in screen pixels."""
        if not self.points:
            return None
        screen = self.camera.to_screen(self.points)
        d2 = ((screen - np.asarray(screen_pos, dtype=float)) ** 2).sum(axis=1)
        i = int(np.argmin(d2))
        return i if d2[i] <= GRAB_RADIUS**2 else None

    def fit_view(self) -> None:
        if self.points:
            self.camera.fit(self.points, pad=self.width)

    def build(self) -> Track:
        return Track(np.array(self.points), width=self.width, closed=self.closed, name="custom")

    # --------------------------------------------------------------- preview
    def _curve(self):
        """Same smoothing the real Track uses, minus the (slow) grid build."""
        if len(self.points) < 3:
            return None
        pts = np.array(self.points, dtype=float)
        return resample(catmull_rom(pts, self.closed), Track.SPACING, self.closed)

    def _draw(self):
        self.screen.fill(BG)
        if self.show_grid:
            self._draw_grid()

        center = self._curve()
        tight = None
        if center is not None:
            tight = self._draw_track(center)
        elif len(self.points) >= 2:
            guide = self.camera.to_screen(self.points)
            pygame.draw.lines(self.screen, GUIDE, False,
                              [(float(a), float(b)) for a, b in guide], 1)

        hover = self._pick(pygame.mouse.get_pos())
        for i, p in enumerate(self.camera.to_screen(self.points) if self.points else []):
            color = POINT_HOT if i == hover else POINT
            pygame.draw.circle(self.screen, color, (int(p[0]), int(p[1])), 6)
            pygame.draw.circle(self.screen, BG, (int(p[0]), int(p[1])), 3)

        self._hud(center, tight)
        pygame.display.flip()

    def _draw_grid(self) -> None:
        """A world grid whose spacing adapts so it stays readable at any zoom."""
        scale = self.camera.scale
        step = next((s for s in GRID_STEPS if s * scale >= 60), GRID_STEPS[-1])
        w, h = self.size
        lo = self.camera.to_world((0, 0))
        hi = self.camera.to_world((w, h))

        for axis, limit in ((0, w), (1, h)):
            first = np.floor(lo[axis] / step) * step
            count = int((hi[axis] - first) / step) + 2
            for k in range(count):
                world = first + k * step
                pos = world * scale + self.camera.offset[axis]
                if not -1 <= pos <= limit + 1:
                    continue
                color = GRID_MAJOR if abs(world) < 1e-6 or (world / step) % 5 == 0 else GRID
                if axis == 0:
                    pygame.draw.line(self.screen, color, (pos, 0), (pos, h))
                else:
                    pygame.draw.line(self.screen, color, (0, pos), (w, pos))

    def _draw_track(self, center: np.ndarray) -> np.ndarray:
        cam = self.camera
        radius = max(1, int(self.width / 2 * cam.scale))
        screen = cam.to_screen(center)

        # Only draw the discs the viewport can actually see. Consecutive samples
        # overlap heavily, so a stride keeps the cost down on a long track
        # without opening gaps (the bulge it leaves is about 1px).
        stride = max(1, int(self.width / 4 / Track.SPACING))
        w, h = self.size
        for p in screen[::stride]:
            if -radius <= p[0] <= w + radius and -radius <= p[1] <= h + radius:
                pygame.draw.circle(self.screen, TARMAC, (int(p[0]), int(p[1])), radius)

        nxt = np.roll(center, -1, axis=0)
        prv = np.roll(center, 1, axis=0)
        if not self.closed:
            nxt[-1], prv[0] = center[-1], center[0]
        tang = nxt - prv
        tang /= np.maximum(np.linalg.norm(tang, axis=1, keepdims=True), 1e-9)
        normal = np.stack([-tang[:, 1], tang[:, 0]], axis=1)

        for sign in (1, -1):
            wall = cam.to_screen(center + normal * sign * (self.width / 2))
            pygame.draw.lines(self.screen, KERB, self.closed,
                              [(float(a), float(b)) for a, b in wall], 2)

        # A corner tighter than half the width makes the walls fold back through
        # the tarmac, quietly opening a shortcut. Flag it while the track is
        # still being drawn rather than after training on it.
        tight = turn_radius(center, self.closed) < self.width / 2
        for p in screen[tight]:
            pygame.draw.circle(self.screen, TIGHT, (int(p[0]), int(p[1])), 4)

        a = cam.to_screen(center[0] + normal[0] * (self.width / 2))
        b = cam.to_screen(center[0] - normal[0] * (self.width / 2))
        pygame.draw.line(self.screen, START_LINE, a, b, 3)
        tip = cam.to_screen(center[0] + tang[0] * max(46.0, self.width * 0.4))
        pygame.draw.line(self.screen, START_LINE, cam.to_screen(center[0]), tip, 2)
        pygame.draw.circle(self.screen, START_LINE, (int(tip[0]), int(tip[1])), 4)
        return tight

    def _hud(self, center, tight):
        loop = "closed loop" if self.closed else "open (start -> finish)"
        length = 0.0
        extent = "-"
        if center is not None:
            length = float(np.linalg.norm(np.diff(center, axis=0), axis=1).sum())
            span = center.max(axis=0) - center.min(axis=0)
            extent = f"{span[0]:.0f}x{span[1]:.0f}"
        cursor = self.camera.to_world(pygame.mouse.get_pos())

        head = f"{len(self.points)} points   width {self.width:.0f}px   {loop}"
        sub = (f"length {length:.0f}px   extent {extent}   "
               f"zoom {self.camera.scale * 100:.0f}%   "
               f"cursor {cursor[0]:.0f},{cursor[1]:.0f}")

        surfaces = [self.font_big.render(head, True, TEXT),
                    self.font.render(sub, True, DIM)]
        surfaces += [self.font.render(t, True, DIM)
                     for t in (HELP if self.show_help else [HELP_HINT])]
        if len(self.points) < 3:
            surfaces.append(self.font.render("need at least 3 points", True, POINT_HOT))
        elif tight is not None and tight.any():
            surfaces.append(self.font.render(
                "corner too tight for this width (red) - spread the points or narrow the track",
                True, TIGHT))

        w = max(s.get_width() for s in surfaces) + 24
        h = sum(s.get_height() + 3 for s in surfaces) + 20
        panel = pygame.Surface((w, h), pygame.SRCALPHA)
        panel.fill((16, 18, 22, 210))
        self.screen.blit(panel, (12, 12))
        y = 22
        for s in surfaces:
            self.screen.blit(s, (24, y))
            y += s.get_height() + 3


def edit_track(path: str, base: Track | None = None) -> Track | None:
    track = TrackEditor(track=base).run()
    if track is None:
        return None
    track.name = os.path.splitext(os.path.basename(path))[0]
    track.save(path)
    return track
