"""Validation, timestamp normalisation, and minimum-gap enforcement."""

from __future__ import annotations

import json
import math
import re
from itertools import pairwise

from chaptergen.models import Chapter
from chaptergen.timestamps import parse_timestamp

# YouTube only shows chapters when the list has at least this many...
YOUTUBE_MIN_CHAPTERS = 3
# ...and none is shorter than this
YOUTUBE_MIN_CHAPTER_SECONDS = 10

# Models sometimes rename the start-time key; accept the common variants
_START_KEYS = ("start_seconds", "start", "timestamp", "time")

# Reasoning models (qwen3, deepseek-r1, ...) prefix their answer with a thinking block
_THINK_BLOCK = re.compile(r"<(think|thinking)>.*?</\1>", re.DOTALL | re.IGNORECASE)
# ...sometimes with only the closing tag, when the opening one came from the chat template
_THINK_UNOPENED = re.compile(r"^.*</(?:think|thinking)>", re.DOTALL | re.IGNORECASE)


def parse_chapters_json(raw: str) -> list[Chapter]:
    """Extract and parse a JSON array of chapters from raw model output.

    Tolerates thinking blocks, markdown code fences and surrounding prose.
    Entries without a usable start time or title are skipped rather than
    failing the whole list.
    """
    data = _load_json_array(raw)

    chapters: list[Chapter] = []
    for item in data:
        if not isinstance(item, dict):
            continue
        start = _to_seconds(next((item[k] for k in _START_KEYS if k in item), None))
        title = item.get("title")
        if start is None or title is None or not str(title).strip():
            continue
        chapters.append(Chapter(start_seconds=start, title=str(title).strip()))

    if not chapters:
        raise ValueError("No valid chapters found in model response.")

    return chapters


def _to_seconds(value: object) -> float | None:
    """Coerce a model-supplied start time to seconds, or ``None`` if it's unusable."""
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        seconds = float(value)
    elif isinstance(value, str):
        try:
            seconds = parse_timestamp(value)
        except ValueError:
            return None
    else:
        return None
    if not math.isfinite(seconds) or seconds < 0:
        return None
    return seconds


def enforce_rules(
    chapters: list[Chapter],
    *,
    min_gap_seconds: int = 30,
    max_chapters: int | None = None,
    latest_start_seconds: float | None = None,
) -> list[Chapter]:
    """Apply YouTube-friendly business rules to chapter list.

    Chapters starting after ``latest_start_seconds`` (past the end of the
    video or transcript) are dropped.
    """
    chapters = sorted(chapters, key=lambda c: c.start_seconds)
    if latest_start_seconds is not None:
        chapters = [c for c in chapters if c.start_seconds <= latest_start_seconds]

    # Ensure the first chapter starts at 0:00. One that starts just after 0:00 is
    # moved there; inserting an "Introduction" instead would leave it too close
    # to survive the minimum gap, losing the model's real first chapter.
    if not chapters:
        chapters = [Chapter(start_seconds=0, title="Introduction")]
    elif chapters[0].start_seconds != 0:
        if chapters[0].start_seconds < max(min_gap_seconds, YOUTUBE_MIN_CHAPTER_SECONDS):
            chapters[0] = Chapter(start_seconds=0, title=chapters[0].title)
        else:
            chapters.insert(0, Chapter(start_seconds=0, title="Introduction"))

    # Deduplicate by timestamp (keep first occurrence)
    seen: set[float] = set()
    unique: list[Chapter] = []
    for ch in chapters:
        if ch.start_seconds not in seen:
            seen.add(ch.start_seconds)
            unique.append(ch)
    chapters = unique

    # Enforce minimum gap between chapters
    if min_gap_seconds > 0 and len(chapters) > 1:
        filtered = [chapters[0]]
        for ch in chapters[1:]:
            if ch.start_seconds - filtered[-1].start_seconds >= min_gap_seconds:
                filtered.append(ch)
        chapters = filtered

    # Enforce the chapter limit by repeatedly dropping the boundary that makes
    # the shortest chapter, which keeps the remaining chapters evenly spread
    if max_chapters is not None:
        while len(chapters) > max(max_chapters, 1):
            shortest = min(
                range(1, len(chapters)),
                key=lambda i: chapters[i].start_seconds - chapters[i - 1].start_seconds,
            )
            del chapters[shortest]

    return chapters


def youtube_problems(chapters: list[Chapter]) -> list[str]:
    """Reasons YouTube would ignore this chapter list, if any."""
    problems: list[str] = []
    if len(chapters) < YOUTUBE_MIN_CHAPTERS:
        problems.append(
            f"YouTube only shows chapters when there are at least {YOUTUBE_MIN_CHAPTERS}, "
            f"and this list has {len(chapters)}. Try a lower --min-gap or a different model."
        )
    if any(b.start_seconds - a.start_seconds < YOUTUBE_MIN_CHAPTER_SECONDS for a, b in pairwise(chapters)):
        problems.append(
            f"Some chapters are shorter than YouTube's {YOUTUBE_MIN_CHAPTER_SECONDS}-second minimum, "
            "so YouTube won't show them. Raise --min-gap to at least 10."
        )
    return problems


def _load_json_array(raw: str) -> list:
    """Find and decode the chapter array in raw model output.

    Tries to decode a JSON value at every ``[`` and returns the first array
    that contains objects, so brackets in surrounding prose don't matter.
    """
    text = _THINK_UNOPENED.sub("", _THINK_BLOCK.sub("", raw))
    text = re.sub(r"```(?:json)?\s*", "", text).replace("```", "")

    decoder = json.JSONDecoder()
    first_error: json.JSONDecodeError | None = None
    fallback: list | None = None
    for match in re.finditer(r"\[", text):
        try:
            value, _ = decoder.raw_decode(text, match.start())
        except json.JSONDecodeError as exc:
            first_error = first_error or exc
            continue
        if isinstance(value, list):
            if any(isinstance(item, dict) for item in value):
                return value
            if fallback is None:
                fallback = value

    if fallback is not None:
        return fallback
    if first_error is not None:
        raise ValueError(f"Model did not return valid JSON: {first_error}") from first_error
    raise ValueError("No JSON array found in model response.")
