"""The lap records, kept between runs.

A time only means something next to the ones it is compared with, so each record
carries a fingerprint of the two things that decide what a lap is worth: the
shape of the circuit and the handling of the car. Redraw the track or change the
physics and the old time is retired rather than quietly compared against a
different problem.
"""

from __future__ import annotations

import datetime
import hashlib
import json
import math
import os

from .car import handling_fingerprint

RECORDS_PATH = os.path.join("models", "records.json")

NO_TIME = "  --.---"


def format_lap(seconds) -> str:
    """A lap time the way a timing screen writes it: ``1:23.456`` or ``23.456``."""
    if seconds is None or not math.isfinite(float(seconds)) or float(seconds) < 0.0:
        return NO_TIME
    minutes, rest = divmod(float(seconds), 60.0)
    text = f"{int(minutes)}:{rest:06.3f}" if minutes >= 1 else f"{rest:.3f}"
    return text.rjust(len(NO_TIME))


def format_gap(seconds) -> str:
    """A signed gap to the record, ``-0.412`` / ``+1.088``."""
    if seconds is None or not math.isfinite(float(seconds)):
        return ""
    return f"{float(seconds):+.3f}"


def fingerprint(track) -> str:
    """Identifies the *shape* driven, not the file it was loaded from."""
    payload = json.dumps(
        {"points": track.to_dict()["points"], "width": track.width, "closed": track.closed},
        sort_keys=True,
    )
    return hashlib.sha1(payload.encode()).hexdigest()[:12]


def describe(entry) -> str:
    """``21.903  ai  gen 47`` -- the time and who set it."""
    if entry is None:
        return f"{NO_TIME}  (nobody yet)"
    who = " ".join(p for p in (entry.get("by", ""), entry.get("detail", "")) if p)
    return f"{format_lap(entry.get('lap'))}  {who}".rstrip()


class RecordBook:
    """The fastest lap ever driven on each circuit, on disk."""

    def __init__(self, path: str = RECORDS_PATH):
        self.path = path
        self.entries = self._read()

    def _read(self) -> dict:
        try:
            with open(self.path) as fh:
                data = json.load(fh)
        except (OSError, ValueError):
            return {}
        return data if isinstance(data, dict) else {}

    def _write(self) -> None:
        """Replace the file in one go.

        Two training runs on two circuits share this file quite happily -- each
        re-reads before it writes -- but a half-written file is another matter,
        so the new one is put in place whole.
        """
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        tmp = self.path + ".tmp"
        with open(tmp, "w") as fh:
            json.dump(self.entries, fh, indent=1, sort_keys=True)
        os.replace(tmp, self.path)

    def best(self, track):
        """The standing record for ``track``, or ``None`` if none still counts."""
        entry = self.entries.get(track.name)
        if not entry:
            return None
        stale = (entry.get("track") != fingerprint(track)
                 or entry.get("physics") != handling_fingerprint())
        return None if stale else entry

    def best_lap(self, track) -> float:
        entry = self.best(track)
        return float(entry["lap"]) if entry else math.inf

    def submit(self, track, seconds, by: str = "ai", detail: str = "") -> bool:
        """Write ``seconds`` down if it beats the standing time. True if it did."""
        if seconds is None or not math.isfinite(float(seconds)) or float(seconds) <= 0.0:
            return False
        # Re-read first: a training run on another circuit may have written to the
        # same file since this book was loaded, and it must not be clobbered.
        self.entries = self._read()
        current = self.best(track)
        if current is not None and float(current["lap"]) <= float(seconds):
            return False
        self.entries[track.name] = {
            "lap": round(float(seconds), 4),
            "by": by,
            "detail": detail,
            "date": datetime.date.today().isoformat(),
            "length": round(float(track.length), 1),
            "track": fingerprint(track),
            "physics": handling_fingerprint(),
        }
        self._write()
        return True

    def forget(self, name: str) -> bool:
        self.entries = self._read()
        if name not in self.entries:
            return False
        del self.entries[name]
        self._write()
        return True


def timing_lines(chrono, best_lap, record, label: str = "fastest") -> list:
    """The three lines every mode puts on screen: this lap, the best, the record.

    A line is plain text, or ``(text, tag)`` where the view is asked to colour it.

    Keeping them identical in training, replay and manual driving is the point:
    the same lap on the same circuit reads the same number whoever is at the
    wheel, which is what makes "am I quicker than the machine" answerable.
    """
    lines = [f"lap     {format_lap(chrono)}"]

    fastest = f"{label:7s} {format_lap(best_lap)}"
    reference = float(record["lap"]) if record else None
    quickest = math.isfinite(float(best_lap)) and (reference is None or best_lap <= reference)
    if reference is not None and math.isfinite(float(best_lap)):
        fastest += f"  {format_gap(float(best_lap) - reference)}"
    # Purple for the quickest lap anyone has driven here, as a timing screen does.
    lines.append((fastest, "hot") if quickest else fastest)

    lines.append(f"record  {describe(record)}")
    return lines
