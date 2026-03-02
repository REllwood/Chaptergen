"""Unified loader that dispatches to the correct parser based on file extension."""

from __future__ import annotations

from pathlib import Path

from chaptergen.models import Segment
from chaptergen.parsers.subtitles import parse_srt, parse_vtt
from chaptergen.parsers.transcript import parse_transcript


def load_segments(path: Path) -> list[Segment]:
    """Load segments from any supported transcript/subtitle file."""
    ext = path.suffix.lower()
    if ext in {".txt", ".md"}:
        return parse_transcript(path)
    if ext == ".srt":
        return parse_srt(path)
    if ext == ".vtt":
        return parse_vtt(path)
    raise ValueError(f"Unsupported file extension: {ext}")
