"""Parse plain-text and markdown transcript files into timed segments.

Supports two formats:
  1. Timestamped lines:  ``HH:MM:SS`` or ``MM:SS`` prefix followed by text,
     or a timestamp on its own line with the text on the following lines
     (the layout YouTube's "Show transcript" panel copies as).
  2. Un-timestamped lines: each line carries on from the most recent
     timestamp, or starts at 0 if the file has no timestamps at all.
"""

from __future__ import annotations

import re
from pathlib import Path

from chaptergen.models import Segment

_TS_PATTERN = re.compile(
    r"^[\[\(]?"
    r"(?:(?P<h>\d{1,2}):)?"
    r"(?P<m>\d{1,2}):(?P<s>\d{2})(?!\d)"
    r"[\]\)]?\s*[-–—:]?\s*"
)


def _ts_to_seconds(h: str | None, m: str, s: str) -> float:
    return int(h or 0) * 3600 + int(m) * 60 + int(s)


def parse_transcript(path: Path) -> list[Segment]:
    """Return a list of :class:`Segment` from a ``.txt`` or ``.md`` file."""
    text = path.read_text(encoding="utf-8")
    segments: list[Segment] = []
    current_start = 0.0

    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue

        match = _TS_PATTERN.match(line)
        if match:
            current_start = _ts_to_seconds(match.group("h"), match.group("m"), match.group("s"))
            line = line[match.end():].strip()
            if not line:
                continue

        segments.append(Segment(start_seconds=current_start, text=line))

    return segments
