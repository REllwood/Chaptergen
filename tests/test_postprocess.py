"""Tests for chapter post-processing and JSON parsing."""

import json

import pytest

from chaptergen.models import Chapter
from chaptergen.postprocess import enforce_rules, parse_chapters_json


class TestParseChaptersJson:

    def test_valid_json(self):
        raw = json.dumps([
            {"start_seconds": 0, "title": "Intro"},
            {"start_seconds": 120, "title": "Main"},
        ])
        chapters = parse_chapters_json(raw)
        assert len(chapters) == 2
        assert chapters[0].title == "Intro"
        assert chapters[1].start_seconds == 120.0

    def test_json_with_code_fences(self):
        raw = '```json\n[{"start_seconds": 0, "title": "Start"}]\n```'
        chapters = parse_chapters_json(raw)
        assert len(chapters) == 1

    def test_json_with_surrounding_text(self):
        raw = 'Here are the chapters:\n[{"start_seconds": 0, "title": "Start"}]\nDone!'
        chapters = parse_chapters_json(raw)
        assert len(chapters) == 1

    def test_invalid_json_raises(self):
        with pytest.raises(ValueError, match="No JSON array"):
            parse_chapters_json("not json at all")

    def test_no_array_raises(self):
        with pytest.raises(ValueError, match="No valid chapters"):
            parse_chapters_json('{"chapters": []}')

    def test_skips_malformed_entries(self):
        raw = json.dumps([
            {"start_seconds": 0, "title": "Good"},
            {"bad_key": 10},
            {"start_seconds": 60, "title": "Also good"},
        ])
        chapters = parse_chapters_json(raw)
        assert len(chapters) == 2

    def test_empty_array_raises(self):
        with pytest.raises(ValueError, match="No valid chapters"):
            parse_chapters_json("[]")


class TestEnforceRules:

    def test_adds_intro_if_missing(self):
        chapters = [Chapter(start_seconds=60, title="Not Intro")]
        result = enforce_rules(chapters)
        assert result[0].start_seconds == 0
        assert result[0].title == "Introduction"

    def test_keeps_existing_zero_chapter(self):
        chapters = [Chapter(start_seconds=0, title="My Intro"), Chapter(start_seconds=60, title="Next")]
        result = enforce_rules(chapters)
        assert result[0].title == "My Intro"

    def test_sorts_by_timestamp(self):
        chapters = [
            Chapter(start_seconds=300, title="C"),
            Chapter(start_seconds=0, title="A"),
            Chapter(start_seconds=120, title="B"),
        ]
        result = enforce_rules(chapters)
        assert [c.start_seconds for c in result] == [0, 120, 300]

    def test_deduplicates_timestamps(self):
        chapters = [
            Chapter(start_seconds=0, title="First"),
            Chapter(start_seconds=0, title="Duplicate"),
            Chapter(start_seconds=60, title="Next"),
        ]
        result = enforce_rules(chapters)
        assert len([c for c in result if c.start_seconds == 0]) == 1
        assert result[0].title == "First"

    def test_minimum_gap_enforcement(self):
        chapters = [
            Chapter(start_seconds=0, title="Intro"),
            Chapter(start_seconds=10, title="Too close"),
            Chapter(start_seconds=15, title="Still too close"),
            Chapter(start_seconds=60, title="Far enough"),
        ]
        result = enforce_rules(chapters, min_gap_seconds=30)
        assert len(result) == 2
        assert result[0].start_seconds == 0
        assert result[1].start_seconds == 60

    def test_empty_input(self):
        result = enforce_rules([])
        assert len(result) == 1
        assert result[0].title == "Introduction"
