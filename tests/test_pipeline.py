"""Tests for the end-to-end generation pipeline."""

import json

import pytest

from chaptergen.config import ProviderConfig
from chaptergen.models import Segment
from chaptergen.pipeline import condense_segments, generate_chapters
from chaptergen.providers.base import LLMProvider

CONFIG = ProviderConfig(provider="ollama", model="test-model")

SEGMENTS = [
    Segment(start_seconds=0, text="Welcome to the video"),
    Segment(start_seconds=120, text="Setting things up"),
    Segment(start_seconds=300, text="Writing the code"),
]

VALID_OUTPUT = json.dumps([
    {"start_seconds": 0, "title": "Intro"},
    {"start_seconds": 120, "title": "Setup"},
    {"start_seconds": 300, "title": "Coding"},
])


class FakeProvider(LLMProvider):
    """Returns canned responses in order and records every call."""

    def __init__(self, *responses: str) -> None:
        self._responses = list(responses)
        self.calls: list[tuple[str, str]] = []

    def complete(self, system_prompt: str, user_prompt: str, *, temperature: float = 0.0) -> str:
        self.calls.append((system_prompt, user_prompt))
        return self._responses.pop(0)

    def health_check(self) -> tuple[bool, str]:
        return True, "ok"


class TestGenerateChapters:

    def test_valid_output_first_time(self):
        provider = FakeProvider(VALID_OUTPUT)
        result = generate_chapters(segments=SEGMENTS, provider=provider, config=CONFIG)
        assert [c.title for c in result.chapters] == ["Intro", "Setup", "Coding"]
        assert result.provider == "ollama"
        assert result.model == "test-model"
        assert len(provider.calls) == 1

    def test_retries_with_repair_prompt(self):
        provider = FakeProvider("Sorry, here you go: not json", VALID_OUTPUT)
        result = generate_chapters(segments=SEGMENTS, provider=provider, config=CONFIG)
        assert len(result.chapters) == 3
        assert len(provider.calls) == 2
        assert "not json" in provider.calls[1][1]

    def test_clear_error_when_retry_also_fails(self):
        provider = FakeProvider("not json", "still not json")
        with pytest.raises(ValueError, match="after retry") as exc_info:
            generate_chapters(segments=SEGMENTS, provider=provider, config=CONFIG)
        message = str(exc_info.value)
        assert "First error" in message
        assert "Second error" in message

    def test_max_chapters_enforced(self):
        many = json.dumps([{"start_seconds": t, "title": f"Chapter {t}"} for t in range(0, 1200, 60)])
        provider = FakeProvider(many)
        result = generate_chapters(segments=SEGMENTS, provider=provider, config=CONFIG, max_chapters=5)
        assert len(result.chapters) == 5
        assert result.chapters[0].start_seconds == 0

    def test_chapters_after_transcript_end_dropped(self):
        output = json.dumps([
            {"start_seconds": 0, "title": "Intro"},
            {"start_seconds": 120, "title": "Setup"},
            {"start_seconds": 900, "title": "Invented"},
        ])
        result = generate_chapters(segments=SEGMENTS, provider=FakeProvider(output), config=CONFIG)
        assert [c.title for c in result.chapters] == ["Intro", "Setup"]

    def test_duration_allows_chapters_after_last_line(self):
        output = json.dumps([
            {"start_seconds": 0, "title": "Intro"},
            {"start_seconds": 400, "title": "Outro"},
            {"start_seconds": 595, "title": "Too close to the end"},
        ])
        result = generate_chapters(
            segments=SEGMENTS, provider=FakeProvider(output), config=CONFIG, duration_seconds=600,
        )
        assert [c.title for c in result.chapters] == ["Intro", "Outro"]


class TestCondenseSegments:

    def test_merges_within_window(self):
        segs = [Segment(start_seconds=t, text=str(t)) for t in [0, 3, 6, 16, 19, 40]]
        merged = condense_segments(segs, window_seconds=15)
        assert [(s.start_seconds, s.text) for s in merged] == [(0, "0 3 6"), (16, "16 19"), (40, "40")]

    def test_sparse_segments_unchanged(self):
        assert condense_segments(SEGMENTS) == SEGMENTS

    def test_prompt_uses_merged_lines(self):
        segs = [Segment(start_seconds=t, text=f"cue {t}") for t in [0, 2, 4, 30]]
        provider = FakeProvider(VALID_OUTPUT)
        generate_chapters(segments=segs, provider=provider, config=CONFIG)
        prompt = provider.calls[0][1]
        assert "cue 0 cue 2 cue 4" in prompt
        assert "cue 30" in prompt
