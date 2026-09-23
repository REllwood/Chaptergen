"""Parse .srt and .vtt subtitle files into timed segments."""

from __future__ import annotations

import html
import re
from pathlib import Path

from chaptergen.models import Segment
from chaptergen.parsers.textfile import read_text

_SRT_TS = re.compile(
    r"(?P<h>\d{1,3}):(?P<m>\d{2}):(?P<s>\d{2})[,.](?P<ms>\d{3})"
)

_VTT_TS = re.compile(
    r"(?:(?P<h>\d+):)?(?P<m>\d{2}):(?P<s>\d{2})[.](?P<ms>\d{3})"
)

_TAG_STRIP = re.compile(r"<[^>]+>")


def _ts_to_seconds(h: str | None, m: str, s: str, ms: str) -> float:
    return int(h or 0) * 3600 + int(m) * 60 + int(s) + int(ms) / 1000


def _clean_text(text: str, *, unescape: bool = False) -> str:
    text = _TAG_STRIP.sub("", text)
    if unescape:
        text = html.unescape(text)
    return " ".join(text.split())


def parse_srt(path: Path) -> list[Segment]:
    """Parse a SubRip (.srt) file."""
    return _parse_cues(read_text(path), _SRT_TS)


def parse_vtt(path: Path) -> list[Segment]:
    """Parse a WebVTT (.vtt) file."""
    # WebVTT escapes &, < and > in cue text as HTML entities
    return _parse_cues(read_text(path), _VTT_TS, unescape=True)


def _parse_cues(raw: str, ts_pattern: re.Pattern[str], *, unescape: bool = False) -> list[Segment]:
    """Collect the text that follows each cue timing line.

    Lines before a timing line (SRT cue numbers, VTT cue identifiers, the
    WEBVTT header, NOTE/STYLE/REGION blocks) are ignored, and a blank line
    ends the current cue. Lines repeating the previous caption line are
    dropped, which undoes the "rolling" layout of YouTube's auto-generated
    captions where each cue starts with the line before it.
    """
    cues: list[tuple[float, list[str]]] = []
    in_cue = False

    for raw_line in raw.splitlines():
        line = raw_line.strip()
        if not line:
            # YouTube's auto-captions put a single-space line before the text,
            # so a whitespace-only line doesn't end a cue that has no text yet
            if not (raw_line and in_cue and not cues[-1][1]):
                in_cue = False
            continue

        ts_match = ts_pattern.match(line)
        if ts_match and "-->" in line:
            if in_cue and cues[-1][1] and cues[-1][1][-1].isdigit():
                cues[-1][1].pop()  # next cue's number, when the blank line between cues is missing
            start_seconds = _ts_to_seconds(
                ts_match.group("h"),
                ts_match.group("m"),
                ts_match.group("s"),
                ts_match.group("ms"),
            )
            cues.append((start_seconds, []))
            in_cue = True
        elif in_cue:
            cues[-1][1].append(line)

    segments: list[Segment] = []
    previous_line = None
    for start_seconds, text_lines in cues:
        kept: list[str] = []
        for text_line in text_lines:
            cleaned = _clean_text(text_line, unescape=unescape)
            if not cleaned or cleaned == previous_line:
                continue
            kept.append(cleaned)
            previous_line = cleaned
        if kept:
            segments.append(Segment(start_seconds=start_seconds, text=" ".join(kept)))

    return segments
