"""Parse plain-text and markdown transcript files into timed segments.

Supports two formats:
  1. Timestamped lines:  ``HH:MM:SS`` or ``MM:SS`` prefix followed by text,
     or a timestamp on its own line with the text on the following lines
     (the layout YouTube's "Show transcript" panel copies as).
  2. Un-timestamped lines: each line carries on from the most recent
     timestamp, or starts at 0 if the file has no timestamps at all (see
     :func:`estimate_timings` for filling those in).
"""

from __future__ import annotations

import re
from pathlib import Path

from chaptergen.models import Segment
from chaptergen.parsers.textfile import read_text

_TS_PATTERN = re.compile(
    r"^[\[\(]?"
    r"(?:(?P<h>\d{1,2}):)?"
    r"(?P<m>\d{1,2}):(?P<s>\d{2})(?!\d)"
    r"[\]\)]?\s*[-–—:]?\s*"
)


# Typical conversational speaking pace, used when a transcript has no timestamps
WORDS_PER_MINUTE = 150


def _ts_to_seconds(h: str | None, m: str, s: str) -> float:
    return int(h or 0) * 3600 + int(m) * 60 + int(s)


def parse_transcript(path: Path) -> list[Segment]:
    """Return a list of :class:`Segment` from a ``.txt`` or ``.md`` file."""
    text = read_text(path)
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


def estimate_timings(segments: list[Segment], duration_seconds: float | None = None) -> list[Segment]:
    """Spread untimed segments across the video by word count.

    Assumes a steady speaking pace: :data:`WORDS_PER_MINUTE`, or whatever pace
    fits ``duration_seconds`` when the video length is known.
    """
    total_words = sum(len(seg.text.split()) for seg in segments)
    if not total_words:
        return segments

    seconds_per_word = duration_seconds / total_words if duration_seconds else 60 / WORDS_PER_MINUTE
    timed: list[Segment] = []
    words_so_far = 0
    for seg in segments:
        timed.append(Segment(start_seconds=round(words_so_far * seconds_per_word), text=seg.text))
        words_so_far += len(seg.text.split())
    return timed
