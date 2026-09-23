"""Tests for chapter post-processing and JSON parsing."""

import json

import pytest

from chaptergen.models import Chapter
from chaptergen.postprocess import enforce_rules, parse_chapters_json
from chaptergen.timestamps import parse_timestamp


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

    def test_string_timestamps_accepted(self):
        raw = json.dumps([
            {"start_seconds": "0:00", "title": "Intro"},
            {"start_seconds": "2:15", "title": "Setup"},
            {"start_seconds": "1:02:03", "title": "Late"},
            {"start_seconds": "95", "title": "Plain seconds"},
        ])
        assert [c.start_seconds for c in parse_chapters_json(raw)] == [0, 135, 3723, 95]

    def test_alternative_start_key(self):
        raw = json.dumps([{"timestamp": "05:30", "title": "Renamed key"}])
        assert parse_chapters_json(raw)[0].start_seconds == 330

    def test_one_bad_entry_does_not_sink_the_rest(self):
        raw = (
            '[{"start_seconds": 0, "title": "Good"},'
            ' {"start_seconds": "soon", "title": "Unparseable"},'
            ' {"start_seconds": NaN, "title": "Not a number"},'
            ' {"start_seconds": Infinity, "title": "Infinite"},'
            ' {"start_seconds": -5, "title": "Negative"},'
            ' {"start_seconds": true, "title": "Boolean"},'
            ' {"start_seconds": [1], "title": "List"},'
            ' {"start_seconds": 30, "title": "   "},'
            ' {"start_seconds": 40, "title": null},'
            ' {"start_seconds": 60, "title": "Also good"}]'
        )
        assert [c.title for c in parse_chapters_json(raw)] == ["Good", "Also good"]

    def test_think_block_with_brackets(self):
        raw = '<think>Maybe [0, 120] works? Or [intro]...</think>\n[{"start_seconds": 0, "title": "Intro"}]'
        assert [c.title for c in parse_chapters_json(raw)] == ["Intro"]

    def test_think_block_with_only_closing_tag(self):
        raw = 'Chapters could be [a] or [b].</think>\n\n```json\n[{"start_seconds": 0, "title": "Intro"}]\n```'
        assert [c.title for c in parse_chapters_json(raw)] == ["Intro"]

    def test_brackets_in_preamble(self):
        raw = 'Here is the [JSON] you asked for:\n[{"start_seconds": 0, "title": "Intro"}]\nHope that helps [smile]'
        assert [c.title for c in parse_chapters_json(raw)] == ["Intro"]

    def test_wrapped_in_object(self):
        raw = '{"chapters": [{"start_seconds": 0, "title": "Intro"}, {"start_seconds": 60, "title": "Next"}]}'
        assert len(parse_chapters_json(raw)) == 2

    def test_truncated_output_reports_invalid_json(self):
        raw = '[{"start_seconds": 0, "title": "Intro"}, {"start_sec'
        with pytest.raises(ValueError, match="valid JSON"):
            parse_chapters_json(raw)


class TestParseTimestamp:

    @pytest.mark.parametrize(("value", "expected"), [
        ("0:00", 0), ("2:15", 135), ("02:15", 135), ("1:02:03", 3723),
        ("12:30.5", 750.5), ("95", 95), (" 7 ", 7),
    ])
    def test_valid(self, value, expected):
        assert parse_timestamp(value) == expected

    @pytest.mark.parametrize("value", ["", "soon", "nan", "inf", "-5", "1:2:3:4", "2:15pm"])
    def test_invalid(self, value):
        with pytest.raises(ValueError):
            parse_timestamp(value)


class TestEnforceRules:

    def test_adds_intro_if_missing(self):
        chapters = [Chapter(start_seconds=60, title="Not Intro")]
        result = enforce_rules(chapters)
        assert result[0].start_seconds == 0
        assert result[0].title == "Introduction"

    def test_early_first_chapter_moved_to_zero(self):
        chapters = [
            Chapter(start_seconds=5, title="Welcome and Overview"),
            Chapter(start_seconds=120, title="Setup"),
            Chapter(start_seconds=300, title="Demo"),
        ]
        result = enforce_rules(chapters)
        assert [(c.start_seconds, c.title) for c in result] == [
            (0, "Welcome and Overview"), (120, "Setup"), (300, "Demo"),
        ]

    def test_early_first_chapter_moved_even_without_min_gap(self):
        # An "Introduction" 0:00-0:05 would be shorter than YouTube allows
        result = enforce_rules([Chapter(start_seconds=5, title="Welcome")], min_gap_seconds=0)
        assert [(c.start_seconds, c.title) for c in result] == [(0, "Welcome")]

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
