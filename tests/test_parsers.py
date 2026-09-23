"""Tests for transcript and subtitle parsers."""

from pathlib import Path

import pytest

from chaptergen.models import Segment
from chaptergen.parsers import load_segments
from chaptergen.parsers.subtitles import parse_srt, parse_vtt
from chaptergen.parsers.transcript import estimate_timings, parse_transcript

FIXTURES = Path(__file__).parent / "fixtures"


class TestTranscriptParser:

    def test_timestamped_txt(self):
        segs = parse_transcript(FIXTURES / "sample.txt")
        assert len(segs) == 7
        assert segs[0].start_seconds == 0.0
        assert segs[0].text == "Welcome to the video"
        assert segs[2].start_seconds == 135.0  # 2:15
        assert segs[-1].start_seconds == 860.0  # 14:20

    def test_no_timestamps(self):
        segs = parse_transcript(FIXTURES / "sample_no_ts.txt")
        assert len(segs) == 7
        assert all(s.start_seconds == 0.0 for s in segs)
        assert "CLI tools" in segs[0].text

    def test_bracketed_timestamps(self, tmp_path):
        f = tmp_path / "ts.txt"
        f.write_text("[0:00] Intro\n[1:30] Main topic\n[5:00] Outro\n")
        segs = parse_transcript(f)
        assert len(segs) == 3
        assert segs[1].start_seconds == 90.0

    def test_hms_timestamps(self, tmp_path):
        f = tmp_path / "hms.txt"
        f.write_text("1:02:30 Long video chapter\n2:00:00 Another chapter\n")
        segs = parse_transcript(f)
        assert segs[0].start_seconds == 3750.0
        assert segs[1].start_seconds == 7200.0

    def test_untimed_lines_continue_previous_timestamp(self, tmp_path):
        f = tmp_path / "mixed.txt"
        f.write_text("0:00 Intro\n10:00 Main topic starts\nand this line continues it\n12:00 Wrap up\n")
        segs = parse_transcript(f)
        assert [s.start_seconds for s in segs] == [0, 600, 600, 720]
        assert segs[2].text == "and this line continues it"

    def test_timestamp_on_its_own_line(self, tmp_path):
        # Layout produced by copying YouTube's "Show transcript" panel
        f = tmp_path / "youtube.txt"
        f.write_text("0:00\nwelcome to the video\n0:45\ntoday we look at parsing\n1:02:03\nand we're done\n")
        segs = parse_transcript(f)
        assert [(s.start_seconds, s.text) for s in segs] == [
            (0, "welcome to the video"),
            (45, "today we look at parsing"),
            (3723, "and we're done"),
        ]

    def test_text_before_first_timestamp_starts_at_zero(self, tmp_path):
        f = tmp_path / "preamble.txt"
        f.write_text("Episode 12 transcript\n1:30 First topic\n")
        segs = parse_transcript(f)
        assert [s.start_seconds for s in segs] == [0, 90]

    def test_digits_after_seconds_are_not_a_timestamp(self, tmp_path):
        f = tmp_path / "numbers.txt"
        f.write_text("1:234 is not a timestamp\n")
        segs = parse_transcript(f)
        assert segs[0].start_seconds == 0
        assert segs[0].text == "1:234 is not a timestamp"

    def test_empty_file(self, tmp_path):
        f = tmp_path / "empty.txt"
        f.write_text("")
        segs = parse_transcript(f)
        assert segs == []


class TestEstimateTimings:

    def test_default_speaking_pace(self):
        segs = [Segment(start_seconds=0, text="word " * 150) for _ in range(3)]
        assert [s.start_seconds for s in estimate_timings(segs)] == [0, 60, 120]

    def test_scaled_to_duration(self):
        segs = [Segment(start_seconds=0, text="one two three four five six seven eight nine ten") for _ in range(2)]
        assert [s.start_seconds for s in estimate_timings(segs, duration_seconds=100)] == [0, 50]

    def test_keeps_text(self):
        segs = [Segment(start_seconds=0, text="hello there"), Segment(start_seconds=0, text="general kenobi")]
        assert [s.text for s in estimate_timings(segs)] == ["hello there", "general kenobi"]


class TestSrtParser:

    def test_parse_srt(self):
        segs = parse_srt(FIXTURES / "sample.srt")
        assert len(segs) == 6
        assert segs[0].start_seconds == 0.0
        assert segs[0].text == "Welcome to this tutorial on Python."
        assert segs[2].start_seconds == 45.0
        # Multi-line subtitle gets joined
        assert "setting up" in segs[2].text.lower()
        assert "our project structure" in segs[2].text.lower()

    def test_timestamps_are_sorted(self):
        segs = parse_srt(FIXTURES / "sample.srt")
        times = [s.start_seconds for s in segs]
        assert times == sorted(times)


class TestVttParser:

    def test_parse_vtt(self):
        segs = parse_vtt(FIXTURES / "sample.vtt")
        assert len(segs) == 6
        assert segs[0].start_seconds == 0.0
        assert segs[0].text == "Welcome to this tutorial on Python."

    def test_webvtt_header_excluded(self):
        segs = parse_vtt(FIXTURES / "sample.vtt")
        for seg in segs:
            assert "WEBVTT" not in seg.text


class TestCueParsing:

    def test_named_vtt_cue_identifiers_are_not_text(self, tmp_path):
        f = tmp_path / "ids.vtt"
        f.write_text("WEBVTT\n\nintro-cue\n00:00.000 --> 00:05.000\nHello there\n\n2\n00:06.000 --> 00:09.000\nSecond cue\n")
        assert [s.text for s in parse_vtt(f)] == ["Hello there", "Second cue"]

    def test_spoken_number_line_is_kept(self, tmp_path):
        f = tmp_path / "numbers.srt"
        f.write_text("1\n00:00:00,000 --> 00:00:02,000\nThe year was\n2024\n\n2\n00:00:02,500 --> 00:00:04,000\n42\n")
        assert [s.text for s in parse_srt(f)] == ["The year was 2024", "42"]

    def test_vtt_note_and_style_blocks_ignored(self, tmp_path):
        f = tmp_path / "blocks.vtt"
        f.write_text(
            "WEBVTT\nKind: captions\nLanguage: en\n\n"
            "STYLE\n::cue { color: yellow }\n\n"
            "NOTE This is a comment\nspanning two lines\n\n"
            "00:00.000 --> 00:05.000 align:start position:0%\nHello there\n"
        )
        assert [s.text for s in parse_vtt(f)] == ["Hello there"]

    def test_vtt_entities_and_tags(self, tmp_path):
        f = tmp_path / "entities.vtt"
        f.write_text("WEBVTT\n\n00:00.000 --> 00:05.000\n<v Roger>Tom &amp; Jerry &lt;3</v> <00:00:01.000><c>really</c>\n")
        assert parse_vtt(f)[0].text == "Tom & Jerry <3 really"

    def test_srt_cues_without_blank_line_between(self, tmp_path):
        f = tmp_path / "squashed.srt"
        f.write_text("1\n00:00:00,000 --> 00:00:02,000\nFirst\n00:00:03,000 --> 00:00:05,000\nSecond\n")
        assert [(s.start_seconds, s.text) for s in parse_srt(f)] == [(0, "First"), (3, "Second")]

    def test_srt_single_digit_hours(self, tmp_path):
        f = tmp_path / "hours.srt"
        f.write_text("1\n1:02:03,500 --> 1:02:05,000\nLate in the video\n")
        assert parse_srt(f)[0].start_seconds == 3723.5

    def test_youtube_auto_captions(self):
        # Rolling layout from YouTube's auto-generated captions (e.g. yt-dlp --write-auto-subs)
        segs = parse_vtt(FIXTURES / "youtube_auto.vtt")
        assert [(s.start_seconds, s.text) for s in segs] == [
            (0.16, "hey everyone welcome back"),
            (2.32, "to the channel"),
            (5.04, "today we're building"),
        ]

    def test_srt_missing_blank_line_before_cue_number(self, tmp_path):
        f = tmp_path / "squashed.srt"
        f.write_text("1\n00:00:00,000 --> 00:00:02,000\nFirst\n2\n00:00:03,000 --> 00:00:05,000\nSecond\n")
        assert [s.text for s in parse_srt(f)] == ["First", "Second"]

    def test_whitespace_separator_lines_in_srt(self, tmp_path):
        f = tmp_path / "spaces.srt"
        f.write_text("1\n00:00:00,000 --> 00:00:02,000\nFirst\n  \n2\n00:00:03,000 --> 00:00:05,000\nSecond\n")
        assert [s.text for s in parse_srt(f)] == ["First", "Second"]


class TestLoadSegments:

    def test_dispatches_txt(self):
        segs = load_segments(FIXTURES / "sample.txt")
        assert len(segs) > 0

    def test_dispatches_srt(self):
        segs = load_segments(FIXTURES / "sample.srt")
        assert len(segs) > 0

    def test_dispatches_vtt(self):
        segs = load_segments(FIXTURES / "sample.vtt")
        assert len(segs) > 0

    def test_unsupported_extension(self, tmp_path):
        f = tmp_path / "bad.csv"
        f.write_text("data")
        try:
            load_segments(f)
            assert False, "Should have raised"
        except ValueError:
            pass


class TestFileEncodings:

    def test_utf8_bom_srt(self, tmp_path):
        f = tmp_path / "bom.srt"
        f.write_bytes("\ufeff1\n00:00:00,000 --> 00:00:05,000\nHello there\n".encode())
        segs = parse_srt(f)
        assert segs[0].text == "Hello there"

    def test_utf8_bom_vtt(self, tmp_path):
        f = tmp_path / "bom.vtt"
        f.write_bytes("\ufeffWEBVTT\n\n00:00.000 --> 00:05.000\nHello there\n".encode())
        segs = parse_vtt(f)
        assert [s.text for s in segs] == ["Hello there"]

    def test_utf8_bom_txt(self, tmp_path):
        f = tmp_path / "bom.txt"
        f.write_bytes("\ufeff0:30 Hello there\n".encode())
        segs = parse_transcript(f)
        assert (segs[0].start_seconds, segs[0].text) == (30, "Hello there")

    def test_windows_1252_srt(self, tmp_path):
        f = tmp_path / "latin.srt"
        f.write_bytes("1\r\n00:00:00,000 --> 00:00:05,000\r\nCafé time\r\n".encode("cp1252"))
        segs = parse_srt(f)
        assert segs[0].text == "Café time"

    def test_utf16_txt(self, tmp_path):
        f = tmp_path / "notepad.txt"
        f.write_bytes("0:00 Intro\r\n1:00 Café chat\r\n".encode("utf-16"))
        segs = parse_transcript(f)
        assert [s.text for s in segs] == ["Intro", "Café chat"]

    def test_crlf_srt(self, tmp_path):
        f = tmp_path / "crlf.srt"
        f.write_bytes(b"1\r\n00:00:00,000 --> 00:00:05,000\r\nFirst\r\n\r\n2\r\n00:00:06,000 --> 00:00:09,000\r\nSecond\r\n")
        segs = parse_srt(f)
        assert [(s.start_seconds, s.text) for s in segs] == [(0, "First"), (6, "Second")]

    def test_undecodable_file_raises_value_error(self, tmp_path):
        f = tmp_path / "binary.txt"
        f.write_bytes(b"\x81\x8d\x8f\x90\x9d\xff")
        with pytest.raises(ValueError, match="Re-save it as UTF-8"):
            load_segments(f)
