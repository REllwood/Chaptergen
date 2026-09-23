"""Tests for transcript and subtitle parsers."""

from pathlib import Path

from chaptergen.parsers import load_segments
from chaptergen.parsers.subtitles import parse_srt, parse_vtt
from chaptergen.parsers.transcript import parse_transcript

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
