"""Parse .srt and .vtt subtitle files into timed segments."""

from __future__ import annotations

import re
from pathlib import Path

from chaptergen.models import Segment
from chaptergen.parsers.textfile import read_text

_SRT_TS = re.compile(
    r"(?P<h>\d{2}):(?P<m>\d{2}):(?P<s>\d{2})[,.](?P<ms>\d{3})"
)

_VTT_TS = re.compile(
    r"(?:(?P<h>\d{2}):)?(?P<m>\d{2}):(?P<s>\d{2})[.](?P<ms>\d{3})"
)

_TAG_STRIP = re.compile(r"<[^>]+>")


def _ts_to_seconds(h: str | None, m: str, s: str, ms: str) -> float:
    return int(h or 0) * 3600 + int(m) * 60 + int(s) + int(ms) / 1000


def _clean_text(text: str) -> str:
    text = _TAG_STRIP.sub("", text)
    return " ".join(text.split())


def parse_srt(path: Path) -> list[Segment]:
    """Parse a SubRip (.srt) file."""
    return _parse_subtitle_file(path, _SRT_TS)


def parse_vtt(path: Path) -> list[Segment]:
    """Parse a WebVTT (.vtt) file."""
    return _parse_subtitle_file(path, _VTT_TS)


def _parse_subtitle_file(path: Path, ts_pattern: re.Pattern[str]) -> list[Segment]:
    raw = read_text(path)
    blocks = re.split(r"\n\s*\n", raw)
    segments: list[Segment] = []

    for block in blocks:
        lines = block.strip().splitlines()
        if not lines:
            continue

        start_seconds: float | None = None
        text_lines: list[str] = []

        for line in lines:
            line = line.strip()
            if not line:
                continue

            ts_match = ts_pattern.match(line)
            if ts_match and "-->" in line:
                start_seconds = _ts_to_seconds(
                    ts_match.group("h"),
                    ts_match.group("m"),
                    ts_match.group("s"),
                    ts_match.group("ms"),
                )
                continue

            if line.isdigit():
                continue

            if line.upper().startswith("WEBVTT"):
                continue

            if "::" in line and line.strip().isupper():
                continue

            text_lines.append(line)

        if start_seconds is not None and text_lines:
            cleaned = _clean_text(" ".join(text_lines))
            if cleaned:
                segments.append(Segment(start_seconds=start_seconds, text=cleaned))

    return segments
