"""Pygame view of a training run (or of a single car driving)."""

from __future__ import annotations

import colorsys
import math

import numpy as np
import pygame

from . import car as carmod
from .records import format_lap

BG = (24, 26, 32)
TARMAC = (58, 62, 72)
KERB = (120, 128, 145)
CENTER_LINE = (78, 84, 96)
START_LINE = (235, 235, 240)
TEXT = (226, 230, 238)
DIM = (140, 148, 162)
# Hues are stepped by the golden angle so that neighbouring indices - which is
# what a fresh generation hands out - never land on the same colour.
HUE_STEP = 0.6180339887
HUE_START = 0.58        # start on the blue the fleet used to be
CAR_SAT = 0.62
CAR_VAL = 0.95
LEADER = (255, 196, 62)
OVERLOAD = (236, 66, 66)
BAR_BG = (44, 48, 58)
BAR_IN = (98, 168, 228)
BAR_OUT = (255, 196, 62)
BAR_BRAKE = (236, 120, 90)
RULE = (58, 63, 74)

TELEMETRY_WIDTH = 214
# Timing-screen convention: purple is the fastest lap anyone has set.
PURPLE = (190, 120, 235)
GREEN = (98, 210, 138)
RAY = (255, 196, 62, 90)
PANEL = (16, 18, 22)

# A HUD line may be a plain string, or ``(text, tag)`` to ask for a colour. The
# simulation talks in tags rather than RGB so it never has to import the view.
TAGS = {"hot": PURPLE, "good": GREEN, "bright": TEXT}


def car_color(i: int) -> tuple[int, int, int]:
    """A stable, distinct colour for car ``i``."""
    r, g, b = colorsys.hsv_to_rgb((HUE_START + i * HUE_STEP) % 1.0, CAR_SAT, CAR_VAL)
    return int(r * 255), int(g * 255), int(b * 255)


def _faded(color, amount: float = 0.72):
    """Pull a colour towards the background, for cars that are out."""
    return tuple(int(c + (b - c) * amount) for c, b in zip(color, BG))


class Camera:
    """World (track) coordinates -> screen pixels, with zoom and pan."""

    MIN_SCALE = 0.02
    MAX_SCALE = 6.0

    def __init__(self, track, screen_size, pad=70):
        self.size = np.asarray(screen_size, dtype=float)
        self.scale = 1.0
        self.offset = np.zeros(2)
        self.fit_track(track, pad)

    def fit_track(self, track, pad=70) -> None:
        lo = np.minimum(track.left.min(axis=0), track.right.min(axis=0))
        hi = np.maximum(track.left.max(axis=0), track.right.max(axis=0))
        span = np.maximum(hi - lo, 1.0)
        usable = np.maximum(self.size - 2 * pad, 100.0)
        self.scale = float(np.clip(min(usable / span), self.MIN_SCALE, self.MAX_SCALE))
        self.offset = self.size / 2.0 - (lo + hi) / 2.0 * self.scale

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

    def center_on(self, world_pos) -> None:
        self.offset = self.size / 2.0 - np.asarray(world_pos, dtype=float) * self.scale


class Renderer:
    def __init__(self, track, size=(1280, 720), caption="cars_NEAT"):
        pygame.init()
        pygame.display.set_caption(caption)
        self.size = size
        self.screen = pygame.display.set_mode(size, pygame.RESIZABLE)
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("dejavusansmono,consolas,monospace", 15)
        self.font_big = pygame.font.SysFont("dejavusansmono,consolas,monospace", 22, bold=True)
        self.font_small = pygame.font.SysFont("dejavusansmono,consolas,monospace", 12)
        self.set_track(track)

        self.enabled = True        # False = headless turbo
        self.follow = False        # keep the leader centred
        # Most world pixels the view should span while following, for a mode
        # that wants [L] to be a proper chase camera. Left None the key only
        # recentres, which is all a training run watched whole-circuit needs.
        self.chase_span: float | None = None
        self.show_telemetry = True
        self.steps_per_frame = 1
        self._track_dirty = False
        self.fps_cap = 60
        self.quit_requested = False
        self.skip_generation = False
        self._colors: list[tuple[tuple[int, int, int], tuple[int, int, int]]] = []

    # ------------------------------------------------------------------ track
    def set_track(self, track) -> None:
        self.track = track
        # Which wall points fold back onto the tarmac depends only on the
        # geometry, so work it out once instead of on every redraw.
        probe = max(4.0, track.cell * 1.5)
        self._folded = {
            1.0: track.on_track(track.left + track.normal * probe),
            -1.0: track.on_track(track.right - track.normal * probe),
        }
        self.camera = Camera(track, self.size)
        self.track_surface = self._render_track()
        self._track_dirty = False

    def zoom_in_to(self, span: float) -> None:
        """Zoom in until at most ``span`` world pixels fill the shorter screen
        axis. A view already tighter than that is left alone -- pulling back
        from a close-up is not what asking for one means.
        """
        scale = float(np.clip(min(self.size) / max(span, 1.0),
                              Camera.MIN_SCALE, Camera.MAX_SCALE))
        if scale > self.camera.scale:
            self.camera.scale = scale
            self._track_dirty = True

    def _render_track(self) -> pygame.Surface:
        surf = getattr(self, "_track_buffer", None)
        if surf is None or surf.get_size() != tuple(self.size):
            surf = pygame.Surface(self.size)
            self._track_buffer = surf
        surf.fill(BG)
        cam = self.camera
        radius = max(2, int(self.track.width / 2 * cam.scale))

        pts = cam.to_screen(self.track.center)
        # The tarmac is the union of discs along the centre line. Consecutive
        # samples overlap heavily, so a stride keeps a long track cheap to
        # redraw without opening gaps (it leaves a bulge of about 1px).
        stride = max(1, int(self.track.width / 4 / 8.0))
        w, h = self.size
        for p in pts[::stride]:
            if -radius <= p[0] <= w + radius and -radius <= p[1] <= h + radius:
                pygame.draw.circle(surf, TARMAC, (int(p[0]), int(p[1])), radius)

        for wall, outward in ((self.track.left, 1.0), (self.track.right, -1.0)):
            self._kerb(surf, wall, outward)

        for p in pts[::6]:
            if 0 <= p[0] <= w and 0 <= p[1] <= h:
                pygame.draw.circle(surf, CENTER_LINE, (int(p[0]), int(p[1])), 1)

        # Start / finish line
        a = cam.to_screen(self.track.left[0])
        b = cam.to_screen(self.track.right[0])
        pygame.draw.line(surf, START_LINE, a, b, 3)
        return surf

    def _kerb(self, surf, wall: np.ndarray, outward: float) -> None:
        """Draw a wall outline, skipping the bits that fold back onto the track.

        Where a corner is tighter than half the track width the offset curve
        self-intersects, and a naive polyline would draw a barrier across open
        tarmac. The drivable area itself comes from the occupancy grid and is
        unaffected, so it is enough to drop the outline points that sit on it.
        """
        folded = self._folded[outward]
        pts = [(float(a), float(b)) for a, b in self.camera.to_screen(wall)]

        if not folded.any():
            pygame.draw.lines(surf, KERB, self.track.closed, pts, 2)
            return

        run = []
        for p, bad in zip(pts, folded):
            if bad:
                if len(run) > 1:
                    pygame.draw.lines(surf, KERB, False, run, 2)
                run = []
            else:
                run.append(p)
        if len(run) > 1:
            pygame.draw.lines(surf, KERB, False, run, 2)

    # ------------------------------------------------------------------ input
    def pump(self) -> None:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.quit_requested = True
            elif event.type == pygame.VIDEORESIZE:
                self.size = (event.w, event.h)
                self.screen = pygame.display.set_mode(self.size, pygame.RESIZABLE)
                self.camera.size = np.array(self.size, dtype=float)
                self._track_dirty = True
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button in (4, 5):
                    # Zooming keeps follow on: the wheel is how you get closer to
                    # the car you are already tracking.
                    self.camera.zoom_at(event.pos, 1.15 ** (1 if event.button == 4 else -1))
                    self._track_dirty = True
                elif event.button in (1, 2):
                    self._dragging = True
            elif event.type == pygame.MOUSEBUTTONUP and event.button in (1, 2):
                self._dragging = False
            elif event.type == pygame.MOUSEMOTION and getattr(self, "_dragging", False):
                self.camera.pan(event.rel)
                self.follow = False
                self._track_dirty = True
            elif event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_ESCAPE, pygame.K_q):
                    self.quit_requested = True
                elif event.key == pygame.K_h:
                    self.enabled = not self.enabled
                elif event.key == pygame.K_SPACE:
                    self.skip_generation = True
                elif event.key in (pygame.K_PLUS, pygame.K_EQUALS, pygame.K_KP_PLUS):
                    self.steps_per_frame = min(64, self.steps_per_frame * 2)
                elif event.key in (pygame.K_MINUS, pygame.K_KP_MINUS):
                    self.steps_per_frame = max(1, self.steps_per_frame // 2)
                elif event.key == pygame.K_f:
                    self.camera.fit_track(self.track)
                    self.follow = False
                    self._track_dirty = True
                elif event.key == pygame.K_l:
                    self.follow = not self.follow
                    # On a circuit too big to read at full-screen scale a chase
                    # camera has to zoom as well as recentre, or centring on the
                    # car only slides the same picture sideways. Letting go of it
                    # hands the whole circuit back rather than leaving the view
                    # parked on one corner.
                    if self.chase_span:
                        if self.follow:
                            self.zoom_in_to(self.chase_span)
                        else:
                            self.camera.fit_track(self.track)
                            self._track_dirty = True
                elif event.key == pygame.K_t:
                    self.show_telemetry = not self.show_telemetry

    # ----------------------------------------------------------------- drawing
    def draw(self, fleet, hud_lines=(), leader: int | None = None, show_rays=True,
             banner=None) -> None:
        """One frame. ``banner`` is a short message across the top, plain or
        ``(text, tag)``; ``hud_lines`` are the panel, each line the same."""
        if not self.enabled:
            return

        if self.follow and leader is not None:
            self.camera.center_on(fleet.pos[leader])
            self._track_dirty = True
        if self._track_dirty:
            self.track_surface = self._render_track()
            self._track_dirty = False

        self.screen.blit(self.track_surface, (0, 0))
        cam = self.camera
        self._ensure_colors(fleet.n)

        alive_idx = np.flatnonzero(fleet.alive)
        dead_idx = np.flatnonzero(~fleet.alive)

        for i in dead_idx:
            self._car(i, fleet, self._colors[i][1], 0)
        for i in alive_idx:
            if i != leader:
                self._car(i, fleet, self._colors[i][0], 0)

        if leader is not None and fleet.alive[leader]:
            if show_rays:
                self._rays(fleet, leader)
            # Red when the driver is asking the tyres for more than they have.
            over = float(np.clip(fleet.grip_load[leader] - 1.0, 0.0, 1.0))
            color = tuple(int(a + (b - a) * over) for a, b in zip(LEADER, OVERLOAD))
            self._car(leader, fleet, color, 2)

        self._hud(hud_lines)
        if self.show_telemetry and leader is not None:
            self._telemetry(fleet, leader)
        if banner:
            text, tag = banner if isinstance(banner, tuple) else (banner, None)
            self.message(text, TAGS.get(tag, TEXT))
        pygame.display.flip()
        if self.fps_cap:
            self.clock.tick(self.fps_cap)

    def _ensure_colors(self, n: int) -> None:
        """Grow the (live, dead) colour table to cover a fleet of ``n`` cars."""
        for i in range(len(self._colors), n):
            live = car_color(i)
            self._colors.append((live, _faded(live)))

    def _car(self, i, fleet, color, outline) -> None:
        poly = self.camera.to_screen(fleet.body_polygon(i))
        pts = [(float(x), float(y)) for x, y in poly]
        pygame.draw.polygon(self.screen, color, pts)
        if outline:
            pygame.draw.polygon(self.screen, (255, 255, 255), pts, outline)

    def _rays(self, fleet, i) -> None:
        origin = fleet.pos[i]
        base = fleet.angle[i]
        for k, off in enumerate(carmod.RAY_ANGLES):
            d = fleet.last_rays[i, k]
            end = origin + np.array([math.cos(base + off), math.sin(base + off)]) * d
            a = self.camera.to_screen(origin)
            b = self.camera.to_screen(end)
            pygame.draw.line(self.screen, (255, 196, 62), a, b, 1)
            pygame.draw.circle(self.screen, (255, 120, 90), (int(b[0]), int(b[1])), 3)

    def _hud(self, lines) -> None:
        if not lines:
            return
        pad = 10
        surfaces = []
        for i, line in enumerate(lines):
            text, tag = line if isinstance(line, tuple) else (line, None)
            colour = TAGS.get(tag, TEXT if i == 0 else DIM)
            surfaces.append(self.font.render(text, True, colour))
        w = max(s.get_width() for s in surfaces) + 2 * pad
        h = sum(s.get_height() + 2 for s in surfaces) + 2 * pad
        panel = pygame.Surface((w, h), pygame.SRCALPHA)
        panel.fill((*PANEL, 205))
        self.screen.blit(panel, (12, 12))
        y = 12 + pad
        for s in surfaces:
            self.screen.blit(s, (12 + pad, y))
            y += s.get_height() + 2

    # ------------------------------------------------------------- telemetry
    def _bar(self, x, y, width, value, color, centered=False, height=7) -> None:
        """One horizontal gauge. ``centered`` draws from the middle, for a
        signed value like steering."""
        pygame.draw.rect(self.screen, BAR_BG, (x, y, width, height))
        value = float(np.clip(value, -1.0, 1.0))
        if centered:
            mid = x + width / 2
            span = abs(value) * width / 2
            left = mid if value >= 0 else mid - span
            pygame.draw.rect(self.screen, color, (left, y, max(1.0, span), height))
            pygame.draw.line(self.screen, RULE, (mid, y - 1), (mid, y + height))
        else:
            pygame.draw.rect(self.screen, color, (x, y, max(1.0, value * width), height))

    def _telemetry(self, fleet, i: int) -> None:
        """What the car being watched is reading and doing, top right.

        The seven ray lengths and the speed are literally the eight numbers the
        network is handed each step, and steer/throttle are the two it answers
        with -- so this panel is the whole conversation.
        """
        rows = carmod.NUM_RAYS + 8
        line_h = 15
        pad = 10
        width = TELEMETRY_WIDTH
        height = rows * line_h + 2 * pad
        x0 = self.size[0] - width - 12
        y0 = 12

        # Nearly opaque on purpose: this panel is read as numbers, and a sensor
        # ray crossing behind a translucent one looks exactly like a strikethrough.
        panel = pygame.Surface((width, height), pygame.SRCALPHA)
        panel.fill((*PANEL, 238))
        self.screen.blit(panel, (x0, y0))

        x = x0 + pad
        y = y0 + pad
        inner = width - 2 * pad
        label_w = 40
        bar_w = inner - label_w - 44

        def text(s, tx, ty, color=DIM, font=None):
            self.screen.blit((font or self.font_small).render(s, True, color), (tx, ty))

        alive = bool(fleet.alive[i])
        head = f"car {i}" + ("" if alive else "  (out)")
        text(head, x, y, TEXT if alive else OVERLOAD, self.font)
        text(f"lap {int(fleet.laps[i]) + 1}", x + inner - 46, y, TEXT, self.font)
        y += line_h + 3

        chrono = float(fleet.lap_time[i])
        last = float(fleet.last_lap[i])
        text(f"this lap {format_lap(chrono).strip():>9s}", x, y)
        y += line_h
        text(f"last lap {format_lap(last).strip():>9s}", x, y)
        y += line_h + 4

        pygame.draw.line(self.screen, RULE, (x, y), (x + inner, y))
        y += 5
        text("sensors  (network input)", x, y, TEXT)
        y += line_h

        for k, angle in enumerate(carmod.RAY_ANGLES):
            value = float(fleet.last_rays[i, k]) / carmod.RAY_RANGE
            degrees = math.degrees(angle)
            text(f"{degrees:+4.0f}" if degrees else "   0", x, y)
            self._bar(x + label_w, y + 3, bar_w, value, BAR_IN)
            text(f"{value:5.2f}", x + label_w + bar_w + 6, y)
            y += line_h

        speed = float(fleet.speed[i])
        text("spd", x, y)
        self._bar(x + label_w, y + 3, bar_w, speed / carmod.MAX_SPEED, BAR_IN)
        text(f"{speed:5.0f}", x + label_w + bar_w + 6, y)
        y += line_h + 4

        pygame.draw.line(self.screen, RULE, (x, y), (x + inner, y))
        y += 5
        text("controls  (network output)", x, y, TEXT)
        y += line_h

        steer = float(fleet.last_steer[i])
        text("steer", x, y)
        self._bar(x + label_w, y + 3, bar_w, steer, BAR_OUT, centered=True)
        text(f"{steer:+5.2f}", x + label_w + bar_w + 6, y)
        y += line_h

        throttle = float(fleet.last_throttle[i])
        text("thr", x, y)
        self._bar(x + label_w, y + 3, bar_w, throttle,
                  BAR_OUT if throttle >= 0 else BAR_BRAKE, centered=True)
        text(f"{throttle:+5.2f}", x + label_w + bar_w + 6, y)
        y += line_h

        load = float(fleet.grip_load[i])
        over = load > 1.0
        text("tyres", x, y, OVERLOAD if over else DIM)
        # The bar runs to 200% of the budget with a tick at 100%, so asking for
        # more grip than the tyres have reads as passing the mark rather than as
        # a bar that is simply full.
        self._bar(x + label_w, y + 3, bar_w, min(load / 2.0, 1.0),
                  OVERLOAD if over else BAR_OUT)
        tick = x + label_w + bar_w / 2
        pygame.draw.line(self.screen, TEXT, (tick, y + 2), (tick, y + 11))
        text(f"{load * 100:4.0f}%", x + label_w + bar_w + 6, y,
             OVERLOAD if over else DIM)

    def controls_hint(self) -> str:
        follow = "on" if self.follow else "off"
        return (f"x{self.steps_per_frame}  [H] hide  [SPACE] skip  [+/-] speed   "
                f"wheel zoom  drag pan  [F] fit  [L] follow {follow}  [T] telemetry")

    def message(self, text: str, colour=TEXT) -> None:
        """A short line across the screen -- a lap time, a crash notice.

        Along the bottom rather than the top: the HUD panel is as wide as its
        longest line, which on a training run reaches well past the middle, and a
        centred message at the top lands on top of it.
        """
        s = self.font_big.render(text, True, colour)
        rect = s.get_rect(center=(self.size[0] // 2, self.size[1] - 44))
        panel = pygame.Surface((rect.width + 24, rect.height + 12), pygame.SRCALPHA)
        panel.fill((*PANEL, 215))
        self.screen.blit(panel, (rect.x - 12, rect.y - 6))
        self.screen.blit(s, rect)

    def close(self) -> None:
        pygame.quit()
