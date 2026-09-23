"""Deterministic prompt templates for chapter generation."""

from __future__ import annotations

from chaptergen.models import Segment

SYSTEM_PROMPT = """\
You are an expert YouTube editor who creates concise, descriptive video chapters.

RULES:
- Output ONLY a valid JSON array. No markdown, no explanation, no extra text.
- Each element must be an object with exactly two keys: "start_seconds" (number) and "title" (string).
- The first chapter MUST have "start_seconds": 0.
- Titles should be 2-6 words: descriptive, engaging, and suitable for a YouTube chapter list.
- Chapters should mark genuine topic shifts, not every sentence.
- Return chapters sorted by start_seconds ascending.

Example output:
[{"start_seconds": 0, "title": "Introduction"}, {"start_seconds": 124, "title": "Setting Up the Project"}]
"""

REPAIR_PROMPT = """\
Your previous response couldn't be used: {error}
Reply with ONLY the JSON array of chapters for the transcript above.
Each object must have "start_seconds" (number) and "title" (string). No other text."""


def build_user_prompt(
    segments: list[Segment],
    *,
    max_chapters: int | None = None,
    video_duration_hint: float | None = None,
) -> str:
    """Build the user prompt containing the transcript segments."""
    parts: list[str] = ["Here is the video transcript with timestamps:\n"]

    for seg in segments:
        ts = _format_ts(seg.start_seconds)
        parts.append(f"[{ts}] {seg.text}")

    parts.append("")

    if video_duration_hint:
        parts.append(f"Total video duration: approximately {_format_ts(video_duration_hint)}.")

    if max_chapters:
        parts.append(f"Generate at most {max_chapters} chapters.")
    else:
        parts.append("Generate an appropriate number of chapters for this content.")

    parts.append("\nRespond with ONLY the JSON array.")

    return "\n".join(parts)


def _format_ts(seconds: float) -> str:
    h = int(seconds) // 3600
    m = (int(seconds) % 3600) // 60
    s = int(seconds) % 60
    if h > 0:
        return f"{h}:{m:02d}:{s:02d}"
    return f"{m}:{s:02d}"
