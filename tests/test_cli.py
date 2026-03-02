"""Tests for CLI argument handling and output rendering."""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

from click.testing import CliRunner

from chaptergen.cli import main
from chaptergen.models import Chapter, GenerationResult
from chaptergen.output import render

FIXTURES = Path(__file__).parent / "fixtures"


class TestOutputRenderers:

    RESULT = GenerationResult(
        chapters=[
            Chapter(start_seconds=0, title="Introduction"),
            Chapter(start_seconds=135, title="Setting Up"),
            Chapter(start_seconds=330, title="Core Logic"),
            Chapter(start_seconds=3661, title="Long Video Part"),
        ],
        provider="ollama",
        model="llama3.1",
    )

    def test_chapters_format(self):
        out = render(self.RESULT, fmt="chapters")
        lines = out.strip().split("\n")
        assert lines[0] == "00:00 Introduction"
        assert lines[1] == "02:15 Setting Up"
        assert lines[2] == "05:30 Core Logic"
        assert lines[3] == "1:01:01 Long Video Part"

    def test_youtube_format(self):
        out = render(self.RESULT, fmt="youtube")
        lines = out.strip().split("\n")
        assert lines[0] == "Chapters:"
        assert lines[1] == "00:00 Introduction"

    def test_json_format(self):
        out = render(self.RESULT, fmt="json")
        data = json.loads(out)
        assert data["provider"] == "ollama"
        assert data["model"] == "llama3.1"
        assert len(data["chapters"]) == 4
        assert data["chapters"][0]["timestamp"] == "00:00"
        assert data["chapters"][0]["start_seconds"] == 0

    def test_unknown_format_raises(self):
        try:
            render(self.RESULT, fmt="xml")
            assert False, "Should have raised"
        except ValueError:
            pass


class TestCLI:

    def test_version(self):
        runner = CliRunner()
        result = runner.invoke(main, ["--version"])
        assert result.exit_code == 0
        assert "chaptergen" in result.output

    def test_generate_missing_input(self):
        runner = CliRunner()
        result = runner.invoke(main, ["generate"])
        assert result.exit_code != 0

    def test_generate_unsupported_file(self, tmp_path):
        f = tmp_path / "data.csv"
        f.write_text("x,y\n1,2\n")
        runner = CliRunner()
        result = runner.invoke(main, ["generate", "--input", str(f)])
        assert result.exit_code != 0

    @patch("chaptergen.cli.get_provider")
    @patch("chaptergen.cli.generate_chapters")
    def test_generate_success(self, mock_gen, mock_get_prov, tmp_path):
        mock_gen.return_value = GenerationResult(
            chapters=[
                Chapter(start_seconds=0, title="Intro"),
                Chapter(start_seconds=60, title="Main"),
                Chapter(start_seconds=300, title="Outro"),
            ],
            provider="ollama",
            model="llama3.1",
        )
        mock_get_prov.return_value = MagicMock()

        runner = CliRunner()
        result = runner.invoke(main, [
            "generate",
            "--input", str(FIXTURES / "sample.txt"),
            "--provider", "ollama",
        ])
        assert result.exit_code == 0
        assert "00:00 Intro" in result.output

    @patch("chaptergen.cli.get_provider")
    @patch("chaptergen.cli.generate_chapters")
    def test_generate_to_file(self, mock_gen, mock_get_prov, tmp_path):
        mock_gen.return_value = GenerationResult(
            chapters=[
                Chapter(start_seconds=0, title="Intro"),
                Chapter(start_seconds=60, title="Main"),
            ],
            provider="ollama",
            model="llama3.1",
        )
        mock_get_prov.return_value = MagicMock()

        out_file = tmp_path / "chapters.txt"
        runner = CliRunner()
        result = runner.invoke(main, [
            "generate",
            "--input", str(FIXTURES / "sample.txt"),
            "--output", str(out_file),
        ])
        assert result.exit_code == 0
        assert out_file.exists()
        content = out_file.read_text()
        assert "00:00 Intro" in content
