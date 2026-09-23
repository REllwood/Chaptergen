"""Validation, timestamp normalisation, and minimum-gap enforcement."""

from __future__ import annotations

import json
import math
import re

from chaptergen.models import Chapter
from chaptergen.timestamps import parse_timestamp

# YouTube ignores chapter lists containing a chapter shorter than this
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
) -> list[Chapter]:
    """Apply YouTube-friendly business rules to chapter list."""
    chapters = sorted(chapters, key=lambda c: c.start_seconds)

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

    # YouTube requires at least 3 chapters (including 0:00) for the feature to activate
    if len(chapters) < 3:
        pass  # Return what we have — better than fabricating chapters

    return chapters


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
