"""Shared data models used across the pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Segment:
    """A timed piece of transcript text."""

    start_seconds: float
    text: str


@dataclass(frozen=True)
class Chapter:
    """A single generated chapter."""

    start_seconds: float
    title: str


@dataclass
class GenerationResult:
    """Complete output of the chapter generation pipeline."""

    chapters: list[Chapter] = field(default_factory=list)
    provider: str = ""
    model: str = ""
