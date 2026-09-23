"""Unified loader that dispatches to the correct parser based on file extension."""

from __future__ import annotations

from pathlib import Path

from chaptergen.models import Segment
from chaptergen.parsers.subtitles import parse_srt, parse_vtt
from chaptergen.parsers.transcript import estimate_timings, parse_transcript

__all__ = ["SUPPORTED_EXTENSIONS", "estimate_timings", "load_segments"]

_PARSERS = {
    ".txt": parse_transcript,
    ".md": parse_transcript,
    ".srt": parse_srt,
    ".vtt": parse_vtt,
}

SUPPORTED_EXTENSIONS = frozenset(_PARSERS)


def load_segments(path: Path) -> list[Segment]:
    """Load segments from any supported transcript/subtitle file."""
    ext = path.suffix.lower()
    parser = _PARSERS.get(ext)
    if parser is None:
        raise ValueError(f"Unsupported file extension: {ext}")
    return parser(path)
