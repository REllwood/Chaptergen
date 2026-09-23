"""CLI entrypoint — argument parsing and command orchestration."""

from __future__ import annotations

import sys
from pathlib import Path

import click
from rich.console import Console
from rich.markup import escape

from chaptergen import __version__
from chaptergen.config import resolve_config
from chaptergen.output import render
from chaptergen.parsers import load_segments
from chaptergen.pipeline import generate_chapters
from chaptergen.providers import get_provider

console = Console(stderr=True)

SUPPORTED_EXTENSIONS = {".txt", ".md", ".srt", ".vtt"}


@click.group()
@click.version_option(version=__version__, prog_name="chaptergen")
def main() -> None:
    """YouTube Chapter Generator — auto-generate timestamps/chapters from transcripts."""


@main.command()
@click.option("--input", "-i", "input_path", required=True, type=click.Path(exists=True, path_type=Path), help="Path to transcript file (.txt, .md, .srt, .vtt)")
@click.option("--provider", "-p", default=None, help="LLM provider: ollama (default), openai")
@click.option("--model", "-m", default=None, help="Model name (e.g. llama3.1, gpt-4o-mini)")
@click.option("--api-key", default=None, help="API key (prefer --api-key-env instead)")
@click.option("--api-key-env", default=None, help="Env var name holding the API key (e.g. OPENAI_API_KEY)")
@click.option("--base-url", default=None, help="Override base URL for the provider")
@click.option("--temperature", default=0.0, type=float, show_default=True, help="Sampling temperature")
@click.option("--format", "-f", "fmt", type=click.Choice(["chapters", "youtube", "json"], case_sensitive=False), default="chapters", show_default=True, help="Output format")
@click.option("--output", "-o", "output_path", default=None, type=click.Path(path_type=Path), help="Write output to file instead of stdout")
@click.option("--max-chapters", default=None, type=int, help="Suggest a maximum number of chapters to the LLM")
@click.option("--min-gap", default=30, type=int, show_default=True, help="Minimum seconds between chapters")
def generate(
    input_path: Path,
    provider: str | None,
    model: str | None,
    api_key: str | None,
    api_key_env: str | None,
    base_url: str | None,
    temperature: float,
    fmt: str,
    output_path: Path | None,
    max_chapters: int | None,
    min_gap: int,
) -> None:
    """Generate chapters from a transcript file."""
    ext = input_path.suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        console.print(f"[red]Unsupported file type '{escape(ext)}'. Supported: {', '.join(sorted(SUPPORTED_EXTENSIONS))}[/red]")
        sys.exit(1)

    try:
        cfg = resolve_config(
            provider=provider,
            model=model,
            api_key=api_key,
            api_key_env=api_key_env,
            base_url=base_url,
            temperature=temperature,
        )
    except ValueError as exc:
        console.print(f"[red]Configuration error: {escape(str(exc))}[/red]")
        sys.exit(1)

    console.print(f"[dim]Provider:[/dim] {escape(cfg.provider)}  [dim]Model:[/dim] {escape(cfg.model)}")

    segments = load_segments(input_path)
    if not segments:
        console.print("[red]No transcript segments found in input file.[/red]")
        sys.exit(1)

    console.print(f"[dim]Parsed {len(segments)} segments from {escape(input_path.name)}[/dim]")

    llm = get_provider(cfg)

    try:
        result = generate_chapters(
            segments=segments,
            provider=llm,
            config=cfg,
            max_chapters=max_chapters,
            min_gap_seconds=min_gap,
        )
    except Exception as exc:
        console.print(f"[red]Generation failed: {escape(str(exc))}[/red]")
        sys.exit(1)

    output_text = render(result, fmt=fmt)

    if output_path:
        output_path.write_text(output_text, encoding="utf-8")
        console.print(f"[green]Wrote {len(result.chapters)} chapters to {escape(str(output_path))}[/green]")
    else:
        # Plain echo: Rich would treat titles as markup, swap :emoji: codes and hard-wrap long lines
        click.echo(output_text)


@main.command("check")
@click.option("--provider", "-p", default="ollama", help="Provider to check")
@click.option("--model", "-m", default=None, help="Model to verify availability")
@click.option("--base-url", default=None, help="Override base URL")
@click.option("--api-key", default=None, help="API key for cloud provider")
@click.option("--api-key-env", default=None, help="Env var holding API key")
def check_provider(
    provider: str,
    model: str | None,
    base_url: str | None,
    api_key: str | None,
    api_key_env: str | None,
) -> None:
    """Check provider connectivity and model availability."""
    try:
        cfg = resolve_config(
            provider=provider,
            model=model,
            api_key=api_key,
            api_key_env=api_key_env,
            base_url=base_url,
        )
    except ValueError as exc:
        console.print(f"[red]{escape(str(exc))}[/red]")
        sys.exit(1)

    llm = get_provider(cfg)
    ok, msg = llm.health_check()
    if ok:
        console.print(f"[green]{escape(msg)}[/green]")
    else:
        console.print(f"[red]{escape(msg)}[/red]")
        sys.exit(1)
