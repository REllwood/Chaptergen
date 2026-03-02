"""Parse plain-text and markdown transcript files into timed segments.

Supports two formats:
  1. Timestamped lines:  ``HH:MM:SS`` or ``MM:SS`` prefix followed by text.
  2. Un-timestamped lines: each non-empty line becomes a segment with
     ``start_seconds=0`` so the LLM can still infer topic boundaries.
"""

from __future__ import annotations

import re
from pathlib import Path

from chaptergen.models import Segment

_TS_PATTERN = re.compile(
    r"^[\[\(]?"
    r"(?:(?P<h>\d{1,2}):)?"
    r"(?P<m>\d{1,2}):(?P<s>\d{2})"
    r"[\]\)]?\s*[-–—:]?\s*"
)


def _ts_to_seconds(h: str | None, m: str, s: str) -> float:
    return int(h or 0) * 3600 + int(m) * 60 + int(s)


def parse_transcript(path: Path) -> list[Segment]:
    """Return a list of :class:`Segment` from a ``.txt`` or ``.md`` file."""
    text = path.read_text(encoding="utf-8")
    segments: list[Segment] = []
    has_timestamps = False

    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue

        match = _TS_PATTERN.match(line)
        if match:
            has_timestamps = True
            secs = _ts_to_seconds(match.group("h"), match.group("m"), match.group("s"))
            body = line[match.end():].strip()
            if body:
                segments.append(Segment(start_seconds=secs, text=body))
        else:
            segments.append(Segment(start_seconds=0.0, text=line))

    if not has_timestamps:
        total = len(segments)
        if total > 0:
            segments = [
                Segment(start_seconds=0.0, text=seg.text)
                for seg in segments
            ]

    return segments
