"""A few ready-made circuits so training can start without drawing one."""

from __future__ import annotations

import os

import numpy as np

from .track import Track


def _ring(n, rx, ry, cx=760.0, cy=440.0, modulation=None):
    a = np.linspace(0.0, 2 * np.pi, n, endpoint=False)
    r = np.ones(n) if modulation is None else modulation(a)
    return np.stack([cx + np.cos(a) * rx * r, cy + np.sin(a) * ry * r], axis=1)


def oval() -> Track:
    return Track(_ring(12, 460, 260), width=140.0, name="oval")


def curvy() -> Track:
    pts = _ring(18, 430, 300, modulation=lambda a: 1.0 + 0.16 * np.sin(3 * a))
    return Track(pts, width=120.0, name="curvy")


def grand_prix() -> Track:
    """Long straights, a fast right-hander, and one properly slow corner.

    Control points are kept ~200px apart so no corner ends up tighter than half
    the track width -- see ``Track.tight_corners``.
    """
    pts = [
        (300, 180), (760, 140), (1180, 200), (1350, 380), (1290, 580),
        (1080, 720), (830, 730), (620, 640), (430, 750), (230, 620), (180, 420),
    ]
    return Track(pts, width=110.0, name="grand_prix")


def snake() -> Track:
    """Open track: start at one end, finish at the other."""
    x = np.linspace(160, 1360, 12)
    y = 440 + 240 * np.sin(np.linspace(0, 2.6 * np.pi, 12))
    return Track(np.stack([x, y], axis=1), width=130.0, closed=False, name="snake")


def endurance() -> Track:
    """A big circuit -- roughly 4x the length of the others.

    Useful for checking that the pan/zoom views and the grid builder behave on
    something far larger than the window. Its corners are deliberately as tight,
    relative to its width, as the small tracks: a big open circuit would just be
    a flat-out lap and would not exercise braking at all.
    """
    pts = _ring(18, 1750, 1120, cx=2100, cy=1350,
                modulation=lambda a: 1.0 + 0.42 * np.sin(3 * a) + 0.10 * np.cos(7 * a + 0.7))
    return Track(pts, width=120.0, name="endurance")


BUILTIN = {
    "oval": oval,
    "curvy": curvy,
    "grand_prix": grand_prix,
    "snake": snake,
    "endurance": endurance,
}


def ensure_builtin(directory: str = "tracks") -> None:
    os.makedirs(directory, exist_ok=True)
    for name, factory in BUILTIN.items():
        path = os.path.join(directory, f"{name}.json")
        if not os.path.exists(path):
            factory().save(path)


def resolve(name_or_path: str, directory: str = "tracks") -> Track:
    """Accept a builtin name, a bare file name, or a path."""
    ensure_builtin(directory)
    if os.path.exists(name_or_path):
        return Track.load(name_or_path)
    candidate = os.path.join(directory, name_or_path)
    if os.path.exists(candidate):
        return Track.load(candidate)
    candidate = os.path.join(directory, f"{name_or_path}.json")
    if os.path.exists(candidate):
        return Track.load(candidate)
    if name_or_path in BUILTIN:
        return BUILTIN[name_or_path]()
    raise FileNotFoundError(
        f"unknown track {name_or_path!r} -- available: {', '.join(sorted(BUILTIN))} "
        f"or any .json in {directory}/"
    )
