"""End-to-end chapter generation pipeline: prompt -> LLM -> validate -> post-process."""

from __future__ import annotations

from rich.console import Console

from chaptergen.config import ProviderConfig
from chaptergen.models import GenerationResult, Segment
from chaptergen.postprocess import enforce_rules, parse_chapters_json
from chaptergen.prompts.chapters import REPAIR_PROMPT, SYSTEM_PROMPT, build_user_prompt
from chaptergen.providers.base import LLMProvider

_console = Console(stderr=True)

MAX_RETRIES = 1


def generate_chapters(
    *,
    segments: list[Segment],
    provider: LLMProvider,
    config: ProviderConfig,
    max_chapters: int | None = None,
    min_gap_seconds: int = 30,
) -> GenerationResult:
    """Run the full generation pipeline and return validated chapters."""
    duration_hint = _estimate_duration(segments)

    user_prompt = build_user_prompt(
        segments,
        max_chapters=max_chapters,
        video_duration_hint=duration_hint,
    )

    _console.print("[dim]Sending transcript to model…[/dim]")
    raw = provider.complete(SYSTEM_PROMPT, user_prompt, temperature=config.temperature)

    chapters = _parse_with_retry(provider, raw, config.temperature)

    chapters = enforce_rules(chapters, min_gap_seconds=min_gap_seconds)

    _console.print(f"[dim]Generated {len(chapters)} chapters[/dim]")

    return GenerationResult(
        chapters=chapters,
        provider=config.provider,
        model=config.model,
    )


def _parse_with_retry(provider: LLMProvider, raw: str, temperature: float) -> list:
    """Parse model output; on failure, send a repair prompt once."""
    try:
        return parse_chapters_json(raw)
    except ValueError as first_err:
        _console.print(f"[yellow]First parse failed ({first_err}), retrying with repair prompt…[/yellow]")

    repair_user = f"Your previous output:\n{raw}\n\n{REPAIR_PROMPT}"
    raw_retry = provider.complete(REPAIR_PROMPT, repair_user, temperature=temperature)

    try:
        return parse_chapters_json(raw_retry)
    except ValueError as second_err:
        raise ValueError(
            f"Model failed to produce valid chapter JSON after retry. "
            f"First error: {first_err}. Second error: {second_err}"
        ) from second_err


def _estimate_duration(segments: list[Segment]) -> float | None:
    """Best-effort duration estimate from the last timestamp."""
    if not segments:
        return None
    max_ts = max(seg.start_seconds for seg in segments)
    return max_ts if max_ts > 0 else None
