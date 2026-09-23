"""End-to-end chapter generation pipeline: prompt -> LLM -> validate -> post-process."""

from __future__ import annotations

from rich.console import Console
from rich.markup import escape

from chaptergen.config import ProviderConfig
from chaptergen.models import GenerationResult, Segment
from chaptergen.postprocess import YOUTUBE_MIN_CHAPTER_SECONDS, enforce_rules, parse_chapters_json
from chaptergen.prompts.chapters import REPAIR_PROMPT, SYSTEM_PROMPT, build_user_prompt
from chaptergen.providers.base import LLMProvider

_console = Console(stderr=True)

MAX_RETRIES = 1

# Consecutive segments starting within this many seconds are sent as one line
MERGE_WINDOW_SECONDS = 15


def generate_chapters(
    *,
    segments: list[Segment],
    provider: LLMProvider,
    config: ProviderConfig,
    max_chapters: int | None = None,
    min_gap_seconds: int = 30,
    duration_seconds: float | None = None,
) -> GenerationResult:
    """Run the full generation pipeline and return validated chapters."""
    # A chapter can't start after the video ends (and the last one needs room to be
    # long enough for YouTube). Without the real length, anything after the last
    # transcript line is invented.
    if duration_seconds:
        latest_start = duration_seconds - YOUTUBE_MIN_CHAPTER_SECONDS
    else:
        latest_start = _estimate_duration(segments)

    segments = condense_segments(segments)
    duration_hint = duration_seconds or _estimate_duration(segments)

    user_prompt = build_user_prompt(
        segments,
        max_chapters=max_chapters,
        video_duration_hint=duration_hint,
    )

    _console.print("[dim]Sending transcript to model…[/dim]")
    raw = provider.complete(SYSTEM_PROMPT, user_prompt, temperature=config.temperature)

    chapters = _parse_with_retry(provider, user_prompt, raw, config.temperature)

    chapters = enforce_rules(
        chapters,
        min_gap_seconds=min_gap_seconds,
        max_chapters=max_chapters,
        latest_start_seconds=latest_start,
    )

    _console.print(f"[dim]Generated {len(chapters)} chapters[/dim]")

    return GenerationResult(
        chapters=chapters,
        provider=config.provider,
        model=config.model,
    )


def _parse_with_retry(provider: LLMProvider, user_prompt: str, raw: str, temperature: float | None) -> list:
    """Parse model output; on failure, ask the model once to correct it.

    The repair request continues the original conversation, so the model still
    has the rules and the transcript, and is told what was wrong.
    """
    try:
        return parse_chapters_json(raw)
    except ValueError as exc:
        # Python unbinds the ``as`` name when the except block ends, so keep our own reference
        first_err = exc
        _console.print(f"[yellow]First parse failed ({escape(str(first_err))}), retrying with repair prompt…[/yellow]")

    history = [
        {"role": "user", "content": user_prompt},
        {"role": "assistant", "content": raw},
    ]
    raw_retry = provider.complete(
        SYSTEM_PROMPT,
        REPAIR_PROMPT.format(error=first_err),
        temperature=temperature,
        history=history,
    )

    try:
        return parse_chapters_json(raw_retry)
    except ValueError as second_err:
        raise ValueError(
            f"Model failed to produce valid chapter JSON after retry. "
            f"First error: {first_err}. Second error: {second_err}"
        ) from second_err


def condense_segments(segments: list[Segment], window_seconds: float = MERGE_WINDOW_SECONDS) -> list[Segment]:
    """Merge consecutive segments into lines covering roughly ``window_seconds`` each.

    Subtitle files often have a cue every few seconds; sending each with its own
    timestamp wastes context without helping chapter placement.
    """
    merged: list[Segment] = []
    for seg in segments:
        if merged and seg.start_seconds - merged[-1].start_seconds < window_seconds:
            merged[-1] = Segment(start_seconds=merged[-1].start_seconds, text=f"{merged[-1].text} {seg.text}")
        else:
            merged.append(seg)
    return merged


def _estimate_duration(segments: list[Segment]) -> float | None:
    """Best-effort duration estimate from the last timestamp."""
    if not segments:
        return None
    max_ts = max(seg.start_seconds for seg in segments)
    return max_ts if max_ts > 0 else None
