"""Validation, timestamp normalisation, and minimum-gap enforcement."""

from __future__ import annotations

import json
import re

from chaptergen.models import Chapter


def parse_chapters_json(raw: str) -> list[Chapter]:
    """Extract and parse a JSON array of chapters from raw model output.

    Tolerates markdown code fences and leading/trailing junk.
    """
    cleaned = _extract_json_array(raw)

    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Model did not return valid JSON: {exc}") from exc

    if not isinstance(data, list):
        raise ValueError("Expected a JSON array of chapter objects.")

    chapters: list[Chapter] = []
    for item in data:
        if not isinstance(item, dict):
            continue
        start = item.get("start_seconds")
        title = item.get("title")
        if start is None or title is None:
            continue
        chapters.append(Chapter(start_seconds=float(start), title=str(title).strip()))

    if not chapters:
        raise ValueError("No valid chapters found in model response.")

    return chapters


def enforce_rules(
    chapters: list[Chapter],
    *,
    min_gap_seconds: int = 30,
) -> list[Chapter]:
    """Apply YouTube-friendly business rules to chapter list."""
    chapters = sorted(chapters, key=lambda c: c.start_seconds)

    # Ensure first chapter starts at 0:00
    if not chapters or chapters[0].start_seconds != 0:
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


def _extract_json_array(raw: str) -> str:
    """Strip markdown fences and find the JSON array."""
    # Remove markdown code fences
    raw = re.sub(r"```(?:json)?\s*", "", raw)
    raw = raw.replace("```", "")

    # Find the outermost [ ... ]
    start = raw.find("[")
    end = raw.rfind("]")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("No JSON array found in model response.")

    return raw[start:end + 1]
