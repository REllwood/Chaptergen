"""Output renderers — chapters list, YouTube description block, and JSON."""

from __future__ import annotations

import json

from chaptergen.models import Chapter, GenerationResult


def render(result: GenerationResult, *, fmt: str = "chapters") -> str:
    """Render a :class:`GenerationResult` into the requested format."""
    renderers = {
        "chapters": _render_chapters,
        "youtube": _render_youtube,
        "json": _render_json,
    }
    renderer = renderers.get(fmt.lower())
    if renderer is None:
        raise ValueError(f"Unknown format '{fmt}'. Supported: {', '.join(sorted(renderers))}")
    return renderer(result)


def _render_chapters(result: GenerationResult) -> str:
    lines = [f"{_format_ts(ch.start_seconds)} {ch.title}" for ch in result.chapters]
    return "\n".join(lines)


def _render_youtube(result: GenerationResult) -> str:
    """Ready-to-paste YouTube description block with chapters header."""
    parts: list[str] = []
    parts.append("Chapters:")
    for ch in result.chapters:
        parts.append(f"{_format_ts(ch.start_seconds)} {ch.title}")
    return "\n".join(parts)


def _render_json(result: GenerationResult) -> str:
    data = {
        "provider": result.provider,
        "model": result.model,
        "chapters": [
            {"timestamp": _format_ts(ch.start_seconds), "start_seconds": ch.start_seconds, "title": ch.title}
            for ch in result.chapters
        ],
    }
    return json.dumps(data, indent=2, ensure_ascii=False)


def _format_ts(seconds: float) -> str:
    """Format seconds into HH:MM:SS or MM:SS for YouTube."""
    total = int(seconds)
    h = total // 3600
    m = (total % 3600) // 60
    s = total % 60
    if h > 0:
        return f"{h}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"
